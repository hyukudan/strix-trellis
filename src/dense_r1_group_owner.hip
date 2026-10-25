#include "dense_r1_device.hpp"
#include <cstdlib>
// Dense-r1 projections with the partials folded by the CTA that owns each output H128 (no global partial planes, no
// arrival counters, no second pass). One CTA of 16 waves per (member, output H128). The logical splits are processed
// in passes of Pass splits: the pass's input H128 span is transformed into LDS exactly as the dense-r1 body stages it;
// for every (split, 16-output tile) job a wave runs the body's chain (K16 ascending, low/high accumulators, pairs in
// the same order, lane reduction by shuffles 2 then 1) and writes the split partial to LDS; the waves owning the rows
// add the pass's partials to their register fold in ascending split order from +0. The association is the S-split
// plan's finish(), and so is the epilogue (H128, svh, bias, alpha, beta): bits equal to the dense-r1 plan.
// Instantiated for wo_a 4096 -> 1024 S32 (Pass 8), q_b 1280 -> 16384 S4 (Pass 4) and the o_b K-half 4096 -> 5120 S4
// (Pass 2).
namespace strix_trellis::dense_r1 {
namespace {
constexpr int OwnerWaves = 16;

template<int Input, int Splits, int Pass>
constexpr int owner_span_h128() {  // largest number of input H128 a pass touches
    int most = 0;
    for (int p = 0; p < Splits / Pass; ++p) {
        const int begin = (Input / 16) * (p * Pass) / Splits, end = (Input / 16) * (p * Pass + Pass) / Splits;
        const int n = (end + 7) / 8 - begin / 8;
        if (n > most) most = n;
    }
    return most;
}

template<int Input, int Splits, int Pass, int Bits, int Rows, bool Prefetch>
__global__ __launch_bounds__(OwnerWaves * 32) void owner_fold(DenseR1Group g) {
    constexpr int Tiles = Input / 16 / Splits;
    constexpr int Span = owner_span_h128<Input, Splits, Pass>();
    static_assert(Tiles * Splits == Input / 16 && Splits % Pass == 0);
    using Reader = packed_decode::SaddrLane<Bits>;
    struct Shared {
        _Float16 activation[Rows][Span][128];
        float partial[Pass][Rows][128];
    };
    __shared__ Shared sh;
    const DenseR1Launch& a = g.member[blockIdx.y];
    const auto d = a.projection;
    const auto w = d.weights;
    const int ob = int(blockIdx.x);
    const int lane = threadIdx.x & 31;
    const int wave = __builtin_amdgcn_readfirstlane(int(threadIdx.x) / 32);
    const size_t stride = size_t(w.output / 16) * Bits * 8;
    float v[4] = {};  // fold of row `wave` (waves 0..Rows-1), columns ob*128 + lane + 32 j
    constexpr int Jobs = Pass * 8, Passes = Splits / Pass;
    // the words of four K16 tiles of job (pass, job) from tile `base` on, as loaded
    using Raw = typename Reader::Raw;
    auto batch = [&](int pass, int job, int base, Raw (&rd)[4]) {
        const int begin = (pass * Pass + job / 8) * Tiles;
        const uint32_t* p = w.trellis + size_t(begin) * stride + size_t(ob * 8 + job % 8) * Bits * 8;
        #pragma unroll
        for (int t = 0; t < 4; ++t) rd[t] = Reader::issue(p + size_t(base + t < Tiles ? base + t : base) * stride);
    };
    // Prefetch: the wave requests its next batch (in consumption order, across jobs, the staging and the passes) before
    // it consumes the current one; the arithmetic is unchanged
    Raw next[4];
    int next_pass = 0, next_job = wave, next_base = 0;
    auto request = [&] {
        batch(next_pass, next_job, next_base, next);
        next_base += 4;
        if (next_base >= Tiles) {
            next_base = 0;
            next_job += OwnerWaves;
            if (next_job >= Jobs) { next_job = wave; ++next_pass; }
        }
    };
    if (Prefetch && wave < Jobs) request();
    #pragma unroll 1
    for (int pass = 0; pass < Passes; ++pass) {
        const int first_tile = pass * Pass * Tiles;
        const int first_h = first_tile / 8;
        const int last_h = (first_tile + Pass * Tiles + 7) / 8;  // exclusive
        for (int job = wave; job < Rows * Span; job += OwnerWaves) {
            const int r = job / Span, h = job % Span;
            if (first_h + h >= last_h) continue;
            float x4[4], x[4], s[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = (first_h + h) * 128 + lane + 32 * j;
                x[j] = (Rows == 8 || r < d.rows) ? d.x[size_t(r) * d.x_row_stride + col] : 0;
                s[j] = __half2float(__ushort_as_half(w.suh[col]));
            }
            // all loads in flight before the first RN32 boundary (an asm volatile that orders them)
            #pragma unroll
            for (int j = 0; j < 4; ++j) x4[j] = float(_Float16(materialize_fp32(x[j] * s[j])));
            hadamard128(x4);
            #pragma unroll
            for (int j = 0; j < 4; ++j) sh.activation[r][h][lane + 32 * j] = _Float16(x4[j]);
        }
        __syncthreads();
        #pragma unroll 1
        for (int job = wave; job < Jobs; job += OwnerWaves) {
            const int h = job / 8, tn = job % 8;
            const int begin = (pass * Pass + h) * Tiles;
            float sum[Rows][2] = {};
            #pragma unroll 1
            for (int base = 0; base < Tiles; base += 4) {
                Raw raw[4];
                if constexpr (Prefetch) {
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) raw[t] = next[t];
                    if (next_pass < Passes) request();
                } else batch(pass, job, base, raw);
                Reader rd[4];
                #pragma unroll
                for (int t = 0; t < 4; ++t) rd[t] = Reader(raw[t]);
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    if (base + t >= Tiles) break;
                    const int local = (begin + base + t - first_h * 8) * 16 + 2 * (lane & 3);
                    Pair input[Rows][2];
                    #pragma unroll
                    for (int r = 0; r < Rows; ++r) {
                        __builtin_memcpy(&input[r][0], &sh.activation[r][0][0] + local, 4);
                        __builtin_memcpy(&input[r][1], &sh.activation[r][0][0] + local + 8, 4);
                    }
                    #pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        uint32_t s0, s1;
                        if (pair == 0) { s0 = rd[t].template state<0>(); s1 = rd[t].template state<1>(); }
                        else if (pair == 1) { s0 = rd[t].template state<2>(); s1 = rd[t].template state<3>(); }
                        else if (pair == 2) { s0 = rd[t].template state<4>(); s1 = rd[t].template state<5>(); }
                        else { s0 = rd[t].template state<6>(); s1 = rd[t].template state<7>(); }
                        const Pair weight = __builtin_bit_cast(Pair, packed_decode::mul1_half2(s0, s1));
                        #pragma unroll
                        for (int r = 0; r < Rows; ++r)
                            sum[r][pair / 2] = __builtin_amdgcn_fdot2(weight, input[r][pair & 1], sum[r][pair / 2], false);
                    }
                }
            }
            #pragma unroll
            for (int r = 0; r < Rows; ++r) {
                #pragma unroll
                for (int half = 0; half < 2; ++half) {
                    float x = sum[r][half];
                    x += __shfl_down(x, 2, 4);
                    x += __shfl_down(x, 1, 4);
                    if (!(lane & 3)) sh.partial[h][r][tn * 16 + lane / 4 + half * 8] = x;
                }
            }
        }
        __syncthreads();
        if (wave < Rows && (Rows == 8 || wave < d.rows)) {
            #pragma unroll
            for (int h = 0; h < Pass; ++h) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) v[j] += sh.partial[h][wave][lane + 32 * j];
            }
        }
        __syncthreads();
    }
    if (wave < Rows && (Rows == 8 || wave < d.rows)) {
        float scale[4], bias[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = ob * 128 + lane + 32 * j;
            scale[j] = __half2float(__ushort_as_half(w.svh[col]));
            bias[j] = w.bias ? w.bias[col] : 0;
        }
        hadamard128(v);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = ob * 128 + lane + 32 * j;
            const size_t at = size_t(wave) * d.y_row_stride + col;
            float value = v[j] * scale[j];
            if (w.bias) value += bias[j];
            value *= d.alpha;
            if (d.beta != 0) value += d.beta * d.y[at];
            d.y[at] = value;
        }
    }
}

template<int Input, int Splits, int Pass, int Bits, bool Prefetch>
const void* owner_rows(int rows) {
    switch (rows) {
        case 1: return reinterpret_cast<const void*>(owner_fold<Input, Splits, Pass, Bits, 1, Prefetch>);
        case 2: return reinterpret_cast<const void*>(owner_fold<Input, Splits, Pass, Bits, 2, Prefetch>);
        case 3: return reinterpret_cast<const void*>(owner_fold<Input, Splits, Pass, Bits, 3, Prefetch>);
        case 4: return reinterpret_cast<const void*>(owner_fold<Input, Splits, Pass, Bits, 4, Prefetch>);
        case 5: return reinterpret_cast<const void*>(owner_fold<Input, Splits, Pass, Bits, 5, Prefetch>);
        case 6: return reinterpret_cast<const void*>(owner_fold<Input, Splits, Pass, Bits, 6, Prefetch>);
        case 7: return reinterpret_cast<const void*>(owner_fold<Input, Splits, Pass, Bits, 7, Prefetch>);
        case 8: return reinterpret_cast<const void*>(owner_fold<Input, Splits, Pass, Bits, 8, Prefetch>);
    }
    return nullptr;
}
template<int Input, int Splits, int Pass>
const void* owner_bits(int bits, int rows) {
    static const bool prefetch = [] { const char* e = std::getenv("STRIX_TRELLIS_OWNER_PREFETCH"); return !e || *e != '0'; }();
    if (bits == 4) return prefetch ? owner_rows<Input, Splits, Pass, 4, true>(rows) : owner_rows<Input, Splits, Pass, 4, false>(rows);
    if (bits == 5) return prefetch ? owner_rows<Input, Splits, Pass, 5, true>(rows) : owner_rows<Input, Splits, Pass, 5, false>(rows);
    return nullptr;
}
}  // namespace

const void* owner_function(const DenseR1Launch& a) {
    const auto& w = a.projection.weights;
    if (a.projection.rows != a.projection.rows_per_block || w.output % 128) return nullptr;
    if (w.input == 4096 && w.output == 1024 && a.splits == 32) return owner_bits<4096, 32, 8>(w.bits, a.projection.rows);
    if (w.input == 1280 && w.output == 16384 && a.splits == 4) return owner_bits<1280, 4, 4>(w.bits, a.projection.rows);
    if (w.input == 4096 && w.output == 5120 && a.splits == 4) return owner_bits<4096, 4, 2>(w.bits, a.projection.rows);
    return nullptr;
}
}  // namespace strix_trellis::dense_r1

namespace strix_trellis {
hipError_t launch_dense_r1_group_owner_trusted(const DenseR1Group& g, int count, hipStream_t stream) {
    if (count < 1 || count > dense_r1_group_max) return hipErrorInvalidValue;
    const void* f = dense_r1::owner_function(g.member[0]);
    if (!f) return hipErrorInvalidValue;
    void* args[] = {const_cast<DenseR1Group*>(&g)};
    return hipLaunchKernel(f, dim3(g.member[0].projection.weights.output / 128, count), dim3(16 * 32), args, 0, stream);
}
hipError_t launch_dense_r1_owner_trusted(const DenseR1Launch& a, hipStream_t stream) {
    DenseR1Group g{};
    g.member[0] = a;
    return launch_dense_r1_group_owner_trusted(g, 1, stream);
}
hipError_t query_dense_r1_group_owner_resources(const DenseR1Launch& a, size_t shared, bool* supported) {
    if (!supported) return hipErrorInvalidValue;
    *supported = false;
    const void* f = dense_r1::owner_function(a);
    if (!f) return hipErrorInvalidValue;
    hipFuncAttributes at{};
    const auto e = hipFuncGetAttributes(&at, f);
    if (e != hipSuccess) return e;
    std::printf("DENSE_R1_OWNER_RESOURCE input=%d output=%d rows=%d bits=%d vgpr=%d lds=%zu scratch=%zu\n", a.projection.weights.input,
                a.projection.weights.output, a.projection.rows, a.projection.weights.bits, at.numRegs, at.sharedSizeBytes, at.localSizeBytes);
    *supported = at.maxThreadsPerBlock >= 512 && at.sharedSizeBytes <= shared && !at.localSizeBytes;
    return hipSuccess;
}
}  // namespace strix_trellis
