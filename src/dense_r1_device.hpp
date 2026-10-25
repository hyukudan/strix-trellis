#pragma once
// Independent exact-order dense projection. Physical output tiling adds CTAs
// without cutting any of the declared logical K accumulation chains.
#include "dense_r1.hpp"
#include "hadamard.hpp"
#include "packed_decoder.hpp"
#include <cstdio>

namespace strix_trellis::dense_r1 {
using Pair = _Float16 __attribute__((ext_vector_type(2)));

template<int Input, int Splits, int Rows> union Storage {
    // A split can straddle an H128 boundary on either side. Its actual span
    // is computed below; this bound also covers shared_down's 18 K16 tiles.
    _Float16 activation[Rows * dense_r1_cache_h128<Input,Splits>() * 128];
    int last;
};

template<int Rows, int Waves, int Splits>
__device__ __forceinline__ void finish(const DenseR1Launch& a, int ob, int first) {
    const auto& d = a.projection;
    const auto& w = d.weights;
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x / 32;
    const int total_rows=Rows==8?8:d.rows;
    for (int r = wave; r < Rows && (Rows==8 || first + r < d.rows); r += Waves) {
        float scale[4],bias[4];
        #pragma unroll
        for(int j=0;j<4;++j) {
            const int col=ob*128+lane+32*j;
            scale[j]=__half2float(__ushort_as_half(w.svh[col]));
            bias[j]=w.bias?w.bias[col]:0;
        }
        asm volatile("" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        float v[4] = {};
        // Arrival order never enters this fold. Include the +0 origin.
        #pragma unroll 1
        for (int base = 0; base < Splits; base += 4) {
            float pending[4][4];
            #pragma unroll
            for(int p=0;p<4;++p) {
                #pragma unroll
                for(int j=0;j<4;++j)
                    pending[p][j]=a.partial[(size_t(base+p)*total_rows+first+r)*w.output+ob*128+lane+32*j];
            }
            // Publish four independent VMEM planes to LLVM before the serial
            // fold. The additions still run in the declared S order.
            asm volatile("" ::: "memory");
            __builtin_amdgcn_sched_barrier(0);
            #pragma unroll
            for(int p=0;p<4;++p) {
                #pragma unroll
                for(int j=0;j<4;++j)v[j]+=pending[p][j];
            }
        }
        hadamard128(v);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = ob * 128 + lane + 32*j;
            const size_t at = size_t(first + r) * d.y_row_stride + col;
            float value = v[j] * scale[j];
            if (w.bias) value += bias[j];
            value *= d.alpha;
            if (d.beta != 0) value += d.beta * d.y[at];
            d.y[at] = value;
        }
    }
}

// One CTA of one projection: (split, nt, row_block) is the CTA's logical
// coordinate. The single and grouped kernels below differ only in how they
// select the launch member and that coordinate.
template<int Input, int Splits, int Bits, int Rows, int Columns, int U, bool HalfWave, bool Pipe = false>
__device__ __forceinline__ void projection_body(const DenseR1Launch& a, Storage<Input, Splits, Rows>& shared,
                                                int split, int nt, int row_block) {
    constexpr bool N8 = HalfWave || Columns == 8;
    constexpr int Waves = N8 ? Columns / 8 : Columns / 16;
    constexpr int Tiles = Input / 16 / Splits;
    constexpr int CacheH = dense_r1_cache_h128<Input,Splits>();
    static_assert(Tiles * Splits == Input / 16 && Tiles % 2 == 0);
    static_assert(dense_r1_uniform_h128<Input,Splits>());
    const auto d = a.projection;
    const auto w = d.weights;
    const int lane = threadIdx.x & 31;
    const int wave = __builtin_amdgcn_readfirstlane(int(threadIdx.x) / 32);
    const int ob = nt / (128 / Columns);
    const int tile_n = __builtin_amdgcn_readfirstlane(N8 ? (nt * Waves + wave) / 2 : nt * Waves + wave);
    const int output_half = __builtin_amdgcn_readfirstlane((nt * Waves + wave) & 1);
    const int first = row_block * Rows;
    const int total_rows=Rows==8?8:d.rows;
    const int begin = split * Tiles;
    const int first_h = begin / 8;
    using Reader = packed_decode::SaddrLane<Bits>;
    struct Batch { Reader reader[4]; };
    struct Requested { typename Reader::Raw raw[4]; };
    const size_t stride = size_t(w.output / 16) * Bits * 8;
    // U=1/2/4 are independently measured schedules. No asynchronous load
    // is hidden from LLVM; it owns all VMEM dependencies and waitcnts.
    const auto load = [&](int base) {
        const uint32_t* p = w.trellis + size_t(begin + base) * stride + size_t(tile_n) * Bits * 8;
        return Batch{{Reader(p), Reader(p + (U > 1 ? stride : 0)),
            Reader(p + (U > 2 && base + 2 < Tiles ? 2*stride : 0)),
            Reader(p + (U > 2 && base + 3 < Tiles ? 3*stride : 0))}};
    };
    // Pipe: the first weight batch is requested before the activation staging and
    // every next batch before the current one is consumed (same arithmetic order);
    // it is kept as loaded words and combined only when consumed.
    const auto request = [&](int base) {
        const uint32_t* p = w.trellis + size_t(begin + base) * stride + size_t(tile_n) * Bits * 8;
        return Requested{{Reader::issue(p), Reader::issue(p + (U > 1 ? stride : 0)),
            Reader::issue(p + (U > 2 && base + 2 < Tiles ? 2*stride : 0)),
            Reader::issue(p + (U > 2 && base + 3 < Tiles ? 3*stride : 0))}};
    };
    Requested next{};
    if constexpr (Pipe) next = request(0);
    // The admitted partition family has a constant H span. Avoid runtime
    // division by a variable span in every activation-staging iteration.
    constexpr int nh = CacheH;
    // Stage each complete input H128 exactly once per CTA, before decode.
    // Work is distributed over all waves, even when Rows is only one.
    for (int job = wave; job < Rows * nh; job += Waves) {
        const int r = job / nh;
        const int h = job % nh;
        float v[4], x[4], s[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = (first_h + h) * 128 + lane + 32*j;
            x[j] = (Rows==8 || first + r < d.rows) ? d.x[size_t(first + r) * d.x_row_stride + col] : 0;
            s[j] = __half2float(__ushort_as_half(w.suh[col]));
        }
        // all loads in flight before the first RN32 boundary (an asm volatile that orders them)
        #pragma unroll
        for (int j = 0; j < 4; ++j) v[j] = float(_Float16(materialize_fp32(x[j] * s[j])));
        hadamard128(v);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const _Float16 v16 = _Float16(v[j]);
            shared.activation[(r * CacheH + h) * 128 + lane + 32*j] = v16;
            // H128 overlap is read-only mathematically; only its first owner
            // publishes diagnostics to avoid overlapping GPU writes.
            if (a.activation && nt == 0 && (Rows==8 || first + r < d.rows) &&
                ((first_h + h) * 8 >= begin || split == 0))
                a.activation[size_t(first+r)*Input+(first_h+h)*128+lane+32*j] = __builtin_bit_cast(uint16_t, v16);
        }
    }
    __syncthreads();
    float sum[Rows][2] = {};
    #pragma unroll 1
    for (int base = 0; base < Tiles; base += U) {
        Batch current;
        if constexpr (Pipe) {
            #pragma unroll
            for (int u = 0; u < 4; ++u) current.reader[u] = Reader(next.raw[u]);
            if (base + U < Tiles) next = request(base + U);
        } else {
            current = load(base);
        }
        const auto& reader = current.reader;
        #pragma unroll
        for (int u = 0; u < U; ++u) if (base + u < Tiles) {
            const int local = (begin + base + u - first_h * 8) * 16 + 2 * (lane & 3);
            if constexpr(Rows<=4) {
                // Expose both activation pairs for every row before decode.
                // The packed hashes hide LDS latency for these small tiles.
                Pair input[Rows][2];
                #pragma unroll
                for(int r=0;r<Rows;++r) {
                    __builtin_memcpy(&input[r][0],shared.activation+r*CacheH*128+local,4);
                    __builtin_memcpy(&input[r][1],shared.activation+r*CacheH*128+local+8,4);
                }
                if constexpr(N8) {
                    const uint32_t window=dense_r1_half_window<Bits>(reader[u].high,reader[u].low,output_half);
                    #pragma unroll
                    for(int pair=0;pair<2;++pair) {
                        const uint32_t s0=(window>>(16-2*pair*Bits))&65535u;
                        const uint32_t s1=(window>>(16-(2*pair+1)*Bits))&65535u;
                        const Pair weight=__builtin_bit_cast(Pair,packed_decode::mul1_half2(s0,s1));
                        #pragma unroll
                        for(int r=0;r<Rows;++r)
                            sum[r][0]=__builtin_amdgcn_fdot2(weight,input[r][pair],sum[r][0],false);
                    }
                } else {
                    #pragma unroll
                    for(int pair=0;pair<4;++pair) {
                        uint32_t s0,s1;
                        if(pair==0){s0=reader[u].template state<0>();s1=reader[u].template state<1>();}
                        else if(pair==1){s0=reader[u].template state<2>();s1=reader[u].template state<3>();}
                        else if(pair==2){s0=reader[u].template state<4>();s1=reader[u].template state<5>();}
                        else{s0=reader[u].template state<6>();s1=reader[u].template state<7>();}
                        const Pair weight=__builtin_bit_cast(Pair,packed_decode::mul1_half2(s0,s1));
                        #pragma unroll
                        for(int r=0;r<Rows;++r)
                            sum[r][pair/2]=__builtin_amdgcn_fdot2(weight,input[r][pair&1],sum[r][pair/2],false);
                    }
                }
            } else {
            // Low/high output chains share the activation load. Each chain
            // still sees p0,p1 (or p2,p3) and K16 in exactly ascending order.
            #pragma unroll
            for (int channel = 0; channel < 2; ++channel) {
                if constexpr(N8) {
                    // Four consecutive states fit in one 32-bit window for
                    // K4/K5. Shift to the selected output half once, avoiding
                    // decoding the other eight outputs. This doubles waves
                    // for narrow S4 shapes without reassociating K sums.
                    const uint32_t window=dense_r1_half_window<Bits>(reader[u].high,reader[u].low,output_half);
                    const uint32_t s0=(window >> (16-2*channel*Bits)) & 65535u;
                    const uint32_t s1=(window >> (16-(2*channel+1)*Bits)) & 65535u;
                    const Pair weight=__builtin_bit_cast(Pair,packed_decode::mul1_half2(s0,s1));
                    #pragma unroll
                    for(int r=0;r<Rows;++r) {
                        Pair x;
                        __builtin_memcpy(&x,shared.activation+r*CacheH*128+local+channel*8,4);
                        sum[r][0]=__builtin_amdgcn_fdot2(weight,x,sum[r][0],false);
                    }
                } else {
                Pair low, high;
                if (channel == 0) {
                    low = __builtin_bit_cast(Pair, packed_decode::mul1_half2(reader[u].template state<0>(), reader[u].template state<1>()));
                    high = __builtin_bit_cast(Pair, packed_decode::mul1_half2(reader[u].template state<4>(), reader[u].template state<5>()));
                } else {
                    low = __builtin_bit_cast(Pair, packed_decode::mul1_half2(reader[u].template state<2>(), reader[u].template state<3>()));
                    high = __builtin_bit_cast(Pair, packed_decode::mul1_half2(reader[u].template state<6>(), reader[u].template state<7>()));
                }
                #pragma unroll
                for (int r = 0; r < Rows; ++r) {
                    Pair x;
                    __builtin_memcpy(&x, shared.activation + r*CacheH*128 + local + channel*8, 4);
                    sum[r][0] = __builtin_amdgcn_fdot2(low, x, sum[r][0], false);
                    sum[r][1] = __builtin_amdgcn_fdot2(high, x, sum[r][1], false);
                }
                }
            }
            if constexpr(Rows > 4) {
                asm volatile("" ::: "memory");
                __builtin_amdgcn_sched_barrier(0);
            }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < Rows; ++r) {
        #pragma unroll
        for (int half = 0; half < (N8 ? 1 : 2); ++half) {
            float v = sum[r][half];
            v += __shfl_down(v, 2, 4);
            v += __shfl_down(v, 1, 4);
            if (!(lane & 3) && (Rows==8 || first + r < d.rows))
                a.partial[(size_t(split)*total_rows+first+r)*w.output+tile_n*16+lane/4+(N8 ? output_half : half)*8] = v;
        }
    }
    // Every writing wave releases its VMEM before the CTA's single arrival.
    // The atomic RMW chain carries publication between CTAs at agent scope.
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();
    const int ci = row_block * (w.output / 128) + ob;
    if (!threadIdx.x) {
        const auto old = __hip_atomic_fetch_add(a.counters+ci, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        shared.last = old == unsigned(Splits * (128 / Columns) - 1);
    }
    __syncthreads();
    if (!shared.last) return;
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    finish<Rows, Waves, Splits>(a, ob, first);
    __syncthreads();
    if (!threadIdx.x) __hip_atomic_store(a.counters+ci, 0u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

template<int Input, int Splits, int Bits, int Rows, int Columns, int U, bool HalfWave, bool Pipe = false>
__global__ __launch_bounds__((HalfWave || Columns == 8) ? Columns * 4 : Columns * 2) void projection(DenseR1Launch a) {
    __shared__ Storage<Input, Splits, Rows> shared;
    int split = int(blockIdx.y),nt = int(blockIdx.x);
    if (a.schedule_rotation) {
        const unsigned count=gridDim.x*gridDim.y;
        const unsigned physical=(blockIdx.y*gridDim.x+blockIdx.x+a.schedule_rotation)%count;
        split=int(physical/gridDim.x);nt=int(physical%gridDim.x);
    }
    projection_body<Input,Splits,Bits,Rows,Columns,U,HalfWave,Pipe>(a, shared, split, nt, Rows==8?0:int(blockIdx.z));
}

// Independent projections of one geometry in one dispatch: blockIdx.z selects
// the member, each with its own weights, strided input/output views, partial
// planes and counters. Every member runs the single-projection arithmetic for
// all of its rows (one row block), so its bits equal its own launch.
template<int Input, int Splits, int Bits, int Rows, int Columns, int U, bool HalfWave, bool Pipe>
__global__ __launch_bounds__((HalfWave || Columns == 8) ? Columns * 4 : Columns * 2) void grouped(DenseR1Group g) {
    __shared__ Storage<Input, Splits, Rows> shared;
    int split = int(blockIdx.y),nt = int(blockIdx.x);
    const DenseR1Launch& a = g.member[blockIdx.z];
    if (a.schedule_rotation) {
        const unsigned count=gridDim.x*gridDim.y;
        const unsigned physical=(blockIdx.y*gridDim.x+blockIdx.x+a.schedule_rotation)%count;
        split=int(physical/gridDim.x);nt=int(physical%gridDim.x);
    }
    projection_body<Input,Splits,Bits,Rows,Columns,U,HalfWave,Pipe>(a, shared, split, nt, 0);
}

template<int Input, int Splits, int Rows, int Columns, int U, bool HalfWave, bool Grouped>
const void* bits(int b, bool pipe) {
    if constexpr (Grouped) {
        if (pipe) {
            if (b == 4) return reinterpret_cast<const void*>(grouped<Input,Splits,4,Rows,Columns,U,HalfWave,true>);
            if (b == 5) return reinterpret_cast<const void*>(grouped<Input,Splits,5,Rows,Columns,U,HalfWave,true>);
            return nullptr;
        }
        if (b == 4) return reinterpret_cast<const void*>(grouped<Input,Splits,4,Rows,Columns,U,HalfWave,false>);
        if (b == 5) return reinterpret_cast<const void*>(grouped<Input,Splits,5,Rows,Columns,U,HalfWave,false>);
    } else {
        if (pipe) {  // pipelined weight batches for the single plans: the 64/128-column 16-output bodies only
            if constexpr (Columns >= 64 && !HalfWave) {
                if (b == 4) return reinterpret_cast<const void*>(projection<Input,Splits,4,Rows,Columns,U,HalfWave,true>);
                if (b == 5) return reinterpret_cast<const void*>(projection<Input,Splits,5,Rows,Columns,U,HalfWave,true>);
            }
            return nullptr;
        }
        if (b == 4) return reinterpret_cast<const void*>(projection<Input,Splits,4,Rows,Columns,U,HalfWave>);
        if (b == 5) return reinterpret_cast<const void*>(projection<Input,Splits,5,Rows,Columns,U,HalfWave>);
    }
    return nullptr;
}
template<int Input, int Splits, int Rows, int U, bool Grouped>
const void* columns(int c, int b, int outputs, bool pipe) {
    if (c == 8) return bits<Input,Splits,Rows,8,U,true,Grouped>(b, pipe);
    #define COL(C) if(c==C) return outputs==8 ? bits<Input,Splits,Rows,C,U,true,Grouped>(b, pipe) : bits<Input,Splits,Rows,C,U,false,Grouped>(b, pipe)
    COL(16); COL(32); COL(64); COL(128);
    #undef COL
    return nullptr;
}
template<int Input, int Splits, int Rows, bool Grouped>
const void* prefetch(int u, int c, int b, int outputs, bool pipe) {
    if(outputs!=8 && outputs!=16)return nullptr;
    if (u == 1) return columns<Input,Splits,Rows,1,Grouped>(c,b,outputs,pipe);
    if (u == 2) return columns<Input,Splits,Rows,2,Grouped>(c,b,outputs,pipe);
    if (u == 4) return columns<Input,Splits,Rows,4,Grouped>(c,b,outputs,pipe);
    return nullptr;
}
template<int Input, int Splits, bool Grouped = false>
const void* rows(const DenseR1Launch& a) {
    #define ROW(R) case R: return prefetch<Input,Splits,R,Grouped>(a.prefetch,a.columns_per_block,a.projection.weights.bits,a.outputs_per_wave,a.pipelined)
    switch (a.projection.rows_per_block) { ROW(1); ROW(2); ROW(3); ROW(4); ROW(5); ROW(6); ROW(7); ROW(8); default: return nullptr; }
    #undef ROW
}
} // namespace strix_trellis::dense_r1
