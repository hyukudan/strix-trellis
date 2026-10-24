#include "tp_splitk.hpp"
#include "hadamard.hpp"
#include "packed_decoder.hpp"
#include "tp_activation.hpp"
#include <cstdio>

namespace strix_trellis::tp_splitk {
using Pair = _Float16 __attribute__((ext_vector_type(2)));
template<int Input,int Rows> union SharedStorage {
    _Float16 cache[Rows*tp_split_cache_h128(Input)*128];
    int is_last;
};

// No restrict/const qualification of P or counters: the completing CTA reads
// data published by other CTAs in this same dispatch.
template<int M>
__device__ __forceinline__ void reduce_values(const TpSplitLaunch& a, int ob, int row, float (&v)[4]) {
    const auto& d = a.member[M];
    const auto& w = d.weights;
    const int lane = threadIdx.x & 31;
    const int splits = tp_split_device_count(w.input,w.output);
    float scale[4],bias[4];
    #pragma unroll
    for(int j=0;j<4;++j) {
        const int col=ob*128+lane+32*j;
        scale[j]=__half2float(__ushort_as_half(w.svh[col]));
        bias[j]=w.bias ? w.bias[col] : 0.0f;
    }
    // Weight metadata is immutable and disjoint from diagnostics/output.
    // Issue all epilogue reads before the fold so their latency can overlap
    // independent P loads and H128, rather than serializing each column.
    asm volatile("" ::: "memory");
    __builtin_amdgcn_sched_barrier(0);
    for (int base = 0; base < splits; base += 4) {
        float pending[4][4];
        #pragma unroll
        for(int u=0;u<4;++u) {
            #pragma unroll
            for(int j=0;j<4;++j)
                pending[u][j]=a.partial[M][(size_t(base+u)*d.rows+row)*w.output+ob*128+lane+32*j];
        }
        // S is always 4,16,32. Expose independent memory operations across
        // four planes, then keep exactly the legacy ascending-s additions.
        // The compiler boundary prevents sinking loads into the serial fold;
        // it emits no GPU barrier and does not replace the agent acquire.
        asm volatile("" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        #pragma unroll
        for(int u=0;u<4;++u) {
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            v[j] += pending[u][j];
        }
    }
    hadamard128(v);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = ob * 128 + lane + 32*j;
        v[j] *= scale[j];
        if (w.bias) v[j] += bias[j];
        if (a.diagnostic[M]) a.diagnostic[M][size_t(row) * w.output + col] = v[j];
    }
}
template<int M>
__device__ __forceinline__ void store_values(const TpSplitLaunch& a, int ob, int row, float (&v)[4]) {
    const auto& d = a.member[M];
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const size_t at = size_t(row) * d.y_row_stride + ob * 128 + lane + 32*j;
        float value = v[j] * d.alpha;
        if (d.beta != 0) value += d.beta * d.y[at];
        d.y[at] = value;
    }
}
template<int Rows>
__device__ __forceinline__ void finish(const TpSplitLaunch& a, int member, int ob, int first) {
    const int row = int(threadIdx.x) / 32 + first;
    // The validated Rows=8 entry can only have R=8 and one row block.
    // This also supplies the full-row K4 specialization requested by phase 3.
    if constexpr(Rows<8)
        if (int(threadIdx.x) / 32 >= Rows || row >= a.member[0].rows) return;
    float v[4] = {};
    if (a.swiglu) {
        float g[4] = {};
        reduce_values<0>(a,ob,row,g);
        reduce_values<1>(a,ob,row,v);
        #pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = tp_stable_swiglu(g[j],v[j]);
        store_values<1>(a,ob,row,v);
    } else if (member == 0) {
        reduce_values<0>(a,ob,row,v); store_values<0>(a,ob,row,v);
    } else {
        reduce_values<1>(a,ob,row,v); store_values<1>(a,ob,row,v);
    }
}

template<int Input, int Bits, int Rows, int Splits, bool Two>
__device__ __forceinline__ void produce(TpSplitLaunch a, int member, int ob, int split,
    int first, SharedStorage<Input,Rows>& shared) {
    const auto d = a.member[member];
    const auto w = d.weights;
    const int lane = threadIdx.x & 31;
    const int wave = __builtin_amdgcn_readfirstlane(int(threadIdx.x) / 32);
    const int tile_n = __builtin_amdgcn_readfirstlane(ob * 8 + wave);
    constexpr int CacheH=tp_split_cache_h128(Input);
    constexpr int Tiles=(Input/16)/Splits;
    static_assert((Input/16)%Splits==0 && (Tiles==10 || Tiles==20 || Tiles==64));
    // These measured partitions have a constant H128 span, including their
    // boundary overlap (20 tiles -> 3 H128, 10 -> 2, 64 -> 8). CPU tests prove
    // this for every split. Avoid runtime integer division in the prologue.
    constexpr int nh=(Tiles+7)/8;
    const auto part = tp_split_partition(Input, split, Splits);
    for (int job = wave; job < Rows * nh; job += 8) {
        const int row = job / nh, h = job % nh;
        float value[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = (part.first_h128 + h) * 128 + lane + 32*j;
            const float x = (Rows==8 || first + row < d.rows) ? d.x[size_t(first + row) * d.x_row_stride + col] : 0;
            value[j] = float(_Float16(materialize_fp32(x * __half2float(__ushort_as_half(w.suh[col])))));
        }
        hadamard128(value);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const _Float16 v = _Float16(value[j]);
            shared.cache[(row * CacheH + h) * 128 + lane + 32*j] = v;
            if (a.activation && member == 0 && ob == 0 && (Rows==8 || first + row < d.rows))
                a.activation[((size_t(split) * d.rows + first + row) * CacheH + h) * 128 + lane + 32*j] = __builtin_bit_cast(uint16_t, v);
        }
    }
    __syncthreads();
    const _Float16* cache=shared.cache;
    float sum[Rows][2] = {};
    constexpr int U = Tiles % 4 ? 2 : 4;
    #pragma unroll 1
    for (int base = 0; base < Tiles; base += U) {
        using Reader = packed_decode::SaddrLane<Bits>;
        const uint32_t* p = w.trellis + (size_t(part.begin_tile + base) * (w.output / 16) + tile_n) * (Bits * 8);
        const size_t stride = size_t(w.output / 16) * Bits * 8;
        // Construct all readers before consuming any of their states.
        Reader readers[4] = {Reader(p), Reader(p + stride),
            Reader(p + (U == 4 ? 2 : 0) * stride), Reader(p + (U == 4 ? 3 : 0) * stride)};
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            const int local = (part.begin_tile + base + u - part.first_h128 * 8) * 16;
            if constexpr(Rows<=4) {
                // Small tiles fit the unconstrained U-wide schedule. Retain
                // its measured overlap instead of forcing a boundary per K16.
                Pair input[Rows][2];
                #pragma unroll
                for(int r=0;r<Rows;++r) {
                    __builtin_memcpy(&input[r][0],cache+r*CacheH*128+local+2*(lane&3),4);
                    __builtin_memcpy(&input[r][1],cache+r*CacheH*128+local+2*(lane&3)+8,4);
                }
                #pragma unroll
                for(int pair=0;pair<4;++pair) {
                    uint32_t s0,s1;
                    if(pair==0){s0=readers[u].template state<0>();s1=readers[u].template state<1>();}
                    else if(pair==1){s0=readers[u].template state<2>();s1=readers[u].template state<3>();}
                    else if(pair==2){s0=readers[u].template state<4>();s1=readers[u].template state<5>();}
                    else{s0=readers[u].template state<6>();s1=readers[u].template state<7>();}
                    const Pair weight=__builtin_bit_cast(Pair,packed_decode::mul1_half2(s0,s1));
                    #pragma unroll
                    for(int r=0;r<Rows;++r)
                        sum[r][pair/2]=__builtin_amdgcn_fdot2(weight,input[r][pair&1],sum[r][pair/2],false);
                }
            }else{
            // Consume the same activation pair for low/high output halves.
            // Reordering independent accumulators preserves p0->p1 and p2->p3
            // within each sum while halving the activation register lifetime.
            #pragma unroll
            for (int channel_pair = 0; channel_pair < 2; ++channel_pair) {
                Pair low, high;
                if (channel_pair == 0) {
                    low = __builtin_bit_cast(Pair, packed_decode::mul1_half2(
                        readers[u].template state<0>(),readers[u].template state<1>()));
                    high = __builtin_bit_cast(Pair, packed_decode::mul1_half2(
                        readers[u].template state<4>(),readers[u].template state<5>()));
                } else {
                    low = __builtin_bit_cast(Pair, packed_decode::mul1_half2(
                        readers[u].template state<2>(),readers[u].template state<3>()));
                    high = __builtin_bit_cast(Pair, packed_decode::mul1_half2(
                        readers[u].template state<6>(),readers[u].template state<7>()));
                }
                #pragma unroll
                for (int r = 0; r < Rows; ++r) {
                    Pair input;
                    __builtin_memcpy(&input,cache+r*CacheH*128+local+2*(lane&3)+channel_pair*8,4);
                    sum[r][0] = __builtin_amdgcn_fdot2(low,input,sum[r][0],false);
                    sum[r][1] = __builtin_amdgcn_fdot2(high,input,sum[r][1],false);
                }
            }
            // Compiler scheduling boundary only (no GPU/CTA barrier). Keep
            // later tiles' LDS operands from all becoming live at once; the
            // U packed readers remain prefetched in registers.
            asm volatile("" ::: "memory");
            __builtin_amdgcn_sched_barrier(0);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < Rows; ++r) {
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            float v = sum[r][half];
            v += __shfl_down(v, 2, 4); v += __shfl_down(v, 1, 4);
            if (!(lane & 3) && (Rows==8 || first + r < d.rows))
                a.partial[member][(size_t(split) * d.rows + first + r) * w.output + tile_n * 16 + lane / 4 + 8 * half] = v;
        }
    }
    if constexpr(!Two) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        __syncthreads();
        const int nb = w.output / 128;
        const int total_nb = a.swiglu ? nb : a.member[0].weights.output / 128 + (a.members == 2 ? a.member[1].weights.output / 128 : 0);
        const int ci = (first / Rows) * total_nb + ob + (!a.swiglu && member ? a.member[0].weights.output / 128 : 0);
        if (!threadIdx.x) {
            const auto old = __hip_atomic_fetch_add(a.counters + ci, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
            // All cache consumers have completed before the preceding CTA
            // barrier. Reuse its storage for this scalar instead of crossing
            // an LDS allocation boundary by four bytes (notably o_b at 16KiB).
            shared.is_last = old == unsigned((a.swiglu ? 2 : 1) * Splits - 1);
        }
        __syncthreads();
        if (!shared.is_last) return;
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        finish<Rows>(a, member, ob, first);
        __syncthreads();
        if (!threadIdx.x) __hip_atomic_store(a.counters + ci, 0u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}

template<int Input, int Rows, bool Two>
__global__ __launch_bounds__(256) void split_kernel(TpSplitLaunch a) {
    __shared__ SharedStorage<Input,Rows> shared;
    const int first = int(blockIdx.y) * Rows;
    const int n0 = a.member[0].weights.output / 128;
    const int s0 = tp_split_device_count(Input,a.member[0].weights.output);
    unsigned physical = blockIdx.x;
    if (a.schedule_rotation) physical = (physical + a.schedule_rotation) % gridDim.x;
    const int m = int(physical) < n0 * s0 ? 0 : 1;
    const auto w = a.member[m].weights;
    const int flat = int(physical) - (m ? n0 * s0 : 0);
    const int ob = flat % (w.output / 128), s = flat / (w.output / 128);
    #define BODY(B,S) produce<Input,B,Rows,S,Two>(a,m,ob,s,first,shared)
    if constexpr(Input!=5120) { if(w.bits==4) BODY(4,4); else BODY(5,4); }
    else if (w.output == 512) { if (w.bits == 4) BODY(4,32); else BODY(5,32); }
    else { if (w.bits == 4) BODY(4,16); else BODY(5,16); }
    #undef BODY
}
template<int Rows>
__global__ __launch_bounds__(256) void reduce_kernel(TpSplitLaunch a) {
    const int nb = a.member[0].weights.output / 128;
    const int m = !a.swiglu && int(blockIdx.x) >= nb ? 1 : 0;
    finish<Rows>(a, m, int(blockIdx.x) - (m ? nb : 0), int(blockIdx.y) * Rows);
}
template<int Input>
const void* input_function(int rows, bool two, bool reduce) {
    #define CASE(R) case R: return reduce ? reinterpret_cast<const void*>(reduce_kernel<R>) : two ? reinterpret_cast<const void*>(split_kernel<Input,R,true>) : reinterpret_cast<const void*>(split_kernel<Input,R,false>)
    switch (rows) { CASE(1); CASE(2); CASE(3); CASE(4); CASE(5); CASE(6); CASE(7); CASE(8); default: return nullptr; }
    #undef CASE
}
const void* function(int input,int rows,bool two,bool reduce=false) {
    if(input==5120)return input_function<5120>(rows,two,reduce);
    if(input==1280)return input_function<1280>(rows,two,reduce);
    if(input==4096)return input_function<4096>(rows,two,reduce);
    return nullptr;
}
} // namespace strix_trellis::tp_splitk
namespace strix_trellis {
hipError_t query_tp_split_resources(const TpSplitLaunch& a, size_t shared, bool* supported) {
    if (!supported) return hipErrorInvalidValue;
    *supported = false;
    for (int phase = 0; phase < (a.two_kernel ? 2 : 1); ++phase) {
        const void* f = tp_splitk::function(a.member[0].weights.input,a.member[0].rows_per_block, a.two_kernel, phase != 0);
        if (!f) return hipErrorInvalidValue;
        hipFuncAttributes at{};
        const auto e = hipFuncGetAttributes(&at, f);
        if (e != hipSuccess) return e;
        std::printf("TP_RESOURCE strategy=splitk input=%d phase=%d tile=%d vgpr=%d lds=%zu scratch=%zu max_threads=%d\n",
            a.member[0].weights.input, phase, a.member[0].rows_per_block, at.numRegs, at.sharedSizeBytes, at.localSizeBytes, at.maxThreadsPerBlock);
        const int maximum_regs = a.member[0].rows_per_block <= 4 ? 96 : 128;
        if (at.maxThreadsPerBlock < 256 || at.sharedSizeBytes > shared || at.localSizeBytes ||
            at.numRegs > maximum_regs || at.sharedSizeBytes > size_t(tp_split_cache_h128(a.member[0].weights.input)*128*a.member[0].rows_per_block*2+4)) return hipSuccess;
    }
    *supported = true;
    return hipSuccess;
}
hipError_t launch_tp_split_trusted(const TpSplitLaunch& a, hipStream_t stream) {
    const int tile = a.member[0].rows_per_block;
    int blocks = 0, reductions = 0;
    for (int m = 0; m < a.members; ++m) {
        const auto& w = a.member[m].weights;
        blocks += w.output / 128 * tp_split_count(w.input, w.output);
        reductions += w.output / 128;
    }
    void* args[] = {const_cast<TpSplitLaunch*>(&a)};
    const dim3 row_grid(1, (a.member[0].rows + tile - 1) / tile);
    auto e = hipLaunchKernel(tp_splitk::function(a.member[0].weights.input,tile, a.two_kernel), dim3(blocks, row_grid.y), dim3(256), args, 0, stream);
    if (e != hipSuccess || !a.two_kernel) return e;
    // Never fall back after partial enqueue. A failed second enqueue propagates.
    return hipLaunchKernel(tp_splitk::function(a.member[0].weights.input,tile, true, true),
        dim3(a.swiglu ? a.member[0].weights.output / 128 : reductions, row_grid.y), dim3(256), args, 0, stream);
}
} // namespace strix_trellis
