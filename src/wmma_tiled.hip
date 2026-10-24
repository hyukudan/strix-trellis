// Register-blocked, double-buffered WMMA prefill core for gfx1151.
// Numerics are those of wmma_core (gemm.hip): every output accumulates the
// split's K16 tiles in ascending order with v_wmma_f32_16x16x16_f16 from +0,
// on the same decoded FP16 Q and the same FP16 input. Only the decomposition
// changes: a block owns BM x BN outputs, decodes each Q tile once per block
// into LDS ([n][k], two K16 tiles per stage) and every wave reuses A and B
// fragments across its MT x NT accumulators.
#include "wmma_tiled.hpp"
#include "trellis_device.hpp"
#include "packed_decoder.hpp"
#include "hadamard.hpp"
#include <cstdlib>
#include <cstring>

namespace strix_trellis {
namespace wmma_tiled {
namespace {
using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Half8 = _Float16 __attribute__((ext_vector_type(8)));
// A fragment is loaded as 8 dwords (2 x ds_load_b128) and bit-cast to Half16 only at
// the WMMA: loading <16 x half> directly makes gfx11 true16 codegen split and re-pack
// every dword (v_lshrrev_b32 + v_mov_b16 per dword). Bits are identical either way.
typedef uint32_t Frag __attribute__((ext_vector_type(8), aligned(16)));
using Float8 = float __attribute__((ext_vector_type(8)));
using U4 = uint32_t __attribute__((ext_vector_type(4)));

constexpr int KT = 2;            // K16 tiles per stage
constexpr int BK = 16 * KT;
constexpr int STRIDE = BK + 8;   // halves per LDS row: 80 B keeps b128 alignment

// The three ring words of one lane, loaded early and combined at decode time,
// so the loads stay in flight across a stage of WMMA. Same bit addressing as
// packed_decode::SaddrLane / optimized::AlignedLane<Bits,true>.
template<int Bits> struct RawLane {
    uint32_t a = 0, b = 0, c = 0;
    __device__ __forceinline__ void load(const uint32_t* data) {
        constexpr int count = Bits * 8;
        const int start = ((threadIdx.x & 31) * 8 + 1) * Bits + 256 * Bits - 16;
        const int index = start / 32 - count;
        a = data[index < 0 ? index + count : index];
        b = data[index + 1 >= count ? index + 1 - count : index + 1];
        c = data[index + 2 >= count ? index + 2 - count : index + 2];
    }
};
template<int Bits> struct Stream {
    uint32_t high, low, tail;
    __device__ __forceinline__ explicit Stream(const RawLane<Bits>& r) {
        const int shift = ((((threadIdx.x & 31) * 8 + 1) * Bits + 256 * Bits - 16) & 31);
        high = (r.a << shift) | (r.b >> (32 - shift));
        low = (r.b << shift) | (r.c >> (32 - shift));
        tail = r.c << shift;
    }
    template<int J> __device__ __forceinline__ uint32_t state() const {
        constexpr int offset = J * Bits;
        if constexpr (offset <= 16) return (high >> (16 - offset)) & 65535u;
        else if constexpr (offset < 32) return ((high << (offset - 16)) | (low >> (48 - offset))) & 65535u;
        else if constexpr (offset <= 48) return (low >> (48 - offset)) & 65535u;
        else return ((low << (offset - 48)) | (tail >> (80 - offset))) & 65535u;
    }
};
// Two consecutive K values of one column; low half = first state.
template<int CB> __device__ __forceinline__ uint32_t decode_pair(uint32_t first, uint32_t second) {
    if constexpr (CB == 2) {
        return __builtin_bit_cast(uint32_t, packed_decode::mul1_half2(first, second));
    } else {
        _Float16 x = _Float16(codebook_value(first, CB)), y = _Float16(codebook_value(second, CB));
        return uint32_t(__builtin_bit_cast(uint16_t, x)) | (uint32_t(__builtin_bit_cast(uint16_t, y)) << 16);
    }
}

template<int BM, int BN, int WM, int WN> struct Geometry {
    static constexpr int waves_m = BM / WM, waves_n = BN / WN, waves = waves_m * waves_n;
    static constexpr int threads = waves * 32;
    static constexpr int MT = WM / 16, NT = WN / 16;
    static constexpr int a_chunks = BM * (BK / 8);            // 16-byte chunks per stage
    static constexpr int a_per_thread = (a_chunks + threads - 1) / threads;
    static constexpr int q_tiles = (BN / 16) * KT;            // packed tiles per stage
    static constexpr int q_per_wave = (q_tiles + waves - 1) / waves;
    static constexpr int stage_halves = (BM + BN) * STRIDE;
    static_assert(BM % WM == 0 && BN % WN == 0 && WM % 16 == 0 && WN % 16 == 0);
};

// Probe (benchmark builds only, timing decomposition, wrong outputs):
// 1 = no Q decode/LDS store, 2 = no WMMA, 3 = WMMA + fragment reads only.
template<int Bits, int CB, int BM, int BN, int WM, int WN, bool Fused, int Probe = 0>
__global__ __launch_bounds__((Geometry<BM, BN, WM, WN>::threads)) void tiled_core(
    const uint32_t* __restrict__ packed, const _Float16* __restrict__ input,
    float* __restrict__ partial, int inputs, int outputs, int rows, int splits,
    const uint16_t* __restrict__ scales, const float* __restrict__ bias,
    float* __restrict__ output, float alpha, float beta, int raster_m) {
    using G = Geometry<BM, BN, WM, WN>;
    constexpr int MT = G::MT, NT = G::NT;
    __shared__ __attribute__((aligned(16))) _Float16 lds[2 * G::stage_halves];
    const int thread = threadIdx.x, lane = thread & 31;
    const int wave = __builtin_amdgcn_readfirstlane(thread / 32);
    const int wave_m = wave / G::waves_n, wave_n = wave % G::waves_n;
    // Grouped rasterisation: consecutive blocks cover raster_m row blocks of one column
    // block, then the next column block, so resident blocks share A rows and Q slabs.
    // Only the block -> tile mapping changes; every tile is computed exactly as before.
    const int blocks_m = (rows + BM - 1) / BM;
    const int pid = int(blockIdx.x), per_group = raster_m * (outputs / BN);
    const int group = pid / per_group, first_m = group * raster_m;
    const int group_m = blocks_m - first_m < raster_m ? blocks_m - first_m : raster_m;
    const int local = pid - group * per_group;
    const int first_row = (first_m + local % group_m) * BM, first_col = (local / group_m) * BN;
    const int tiles_n = outputs / 16, tiles_k = inputs / 16;
    // Same split bounds as wmma_core. Pinned in SGPRs so LLVM cannot rematerialise the
    // 64-bit division inside the stage loop.
    int begin = int(int64_t(tiles_k) * blockIdx.z / splits);
    int end = int(int64_t(tiles_k) * (blockIdx.z + 1) / splits);
    asm volatile("" : "+s"(begin), "+s"(end));
    const int stages = (end - begin + KT - 1) / KT;

    U4 a_reg[G::a_per_thread];
    RawLane<Bits> q_reg[G::q_per_wave];

    uint32_t probe_sink = 0;
    auto fetch = [&](int stage) {
        if constexpr (Probe == 3) return;
        const int k0 = begin + stage * KT;
        #pragma unroll
        for (int i = 0; i < G::a_per_thread; ++i) {
            const int chunk = thread + i * G::threads;
            const int r = chunk / (BK / 8), part = chunk % (BK / 8);
            const int row = first_row + r;
            const bool valid = chunk < G::a_chunks && row < rows && k0 + part / 2 < end;
            a_reg[i] = valid ? *reinterpret_cast<const U4*>(input + size_t(row) * inputs + size_t(k0) * 16 + part * 8) : U4{0, 0, 0, 0};
        }
        #pragma unroll
        for (int i = 0; i < G::q_per_wave; ++i) {
            const int t = wave + i * G::waves;            // wave-uniform
            const int kt = t / (BN / 16), nt = t % (BN / 16);
            if (t < G::q_tiles && k0 + kt < end && first_col / 16 + nt < tiles_n)
                q_reg[i].load(packed + (size_t(k0 + kt) * tiles_n + first_col / 16 + nt) * (Bits * 8));
        }
    };
    auto store = [&](int stage, int buffer) {
        if constexpr (Probe == 3) return;
        const int k0 = begin + stage * KT;
        _Float16* as = lds + buffer * G::stage_halves;
        _Float16* bs = as + BM * STRIDE;
        #pragma unroll
        for (int i = 0; i < G::a_per_thread; ++i) {
            const int chunk = thread + i * G::threads;
            if (chunk < G::a_chunks) {
                const int r = chunk / (BK / 8), part = chunk % (BK / 8);
                *reinterpret_cast<U4*>(as + r * STRIDE + part * 8) = a_reg[i];
            }
        }
        #pragma unroll
        for (int i = 0; i < G::q_per_wave; ++i) {
            const int t = wave + i * G::waves;
            const int kt = t / (BN / 16), nt = t % (BN / 16);
            if constexpr (Probe == 1) { probe_sink ^= q_reg[i].a ^ q_reg[i].b ^ q_reg[i].c; continue; }
            if (t < G::q_tiles && k0 + kt < end && first_col / 16 + nt < tiles_n) {
                Stream<Bits> s(q_reg[i]);
                const int r0 = 2 * (lane % 4), c = lane / 4;
                _Float16* base = bs + (nt * 16 + c) * STRIDE + kt * 16 + r0;
                *reinterpret_cast<uint32_t*>(base) = decode_pair<CB>(s.template state<0>(), s.template state<1>());
                *reinterpret_cast<uint32_t*>(base + 8) = decode_pair<CB>(s.template state<2>(), s.template state<3>());
                *reinterpret_cast<uint32_t*>(base + 8 * STRIDE) = decode_pair<CB>(s.template state<4>(), s.template state<5>());
                *reinterpret_cast<uint32_t*>(base + 8 * STRIDE + 8) = decode_pair<CB>(s.template state<6>(), s.template state<7>());
            }
        }
    };

    if constexpr (Probe != 0) {
        for (int j = thread; j < 2 * G::stage_halves; j += G::threads) lds[j] = _Float16(0.0f);
        __syncthreads();
    }
    Float8 acc[MT][NT] = {};
    if (stages > 0) {
        fetch(0);
        store(0, 0);
        __syncthreads();
    }
    for (int stage = 0; stage < stages; ++stage) {
        const int buffer = stage & 1;
        if (stage + 1 < stages) fetch(stage + 1);
        const _Float16* as = lds + buffer * G::stage_halves;
        const _Float16* bs = as + BM * STRIDE;
        const int k0 = begin + stage * KT;
        #pragma unroll
        for (int kt = 0; kt < KT; ++kt) {
            if (k0 + kt < end) {                          // uniform; never multiply zeros into acc
                Frag right[NT];
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const _Float16* p = bs + (wave_n * WN + nt * 16 + lane % 16) * STRIDE + kt * 16;
                    right[nt] = *reinterpret_cast<const Frag*>(p);
                }
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    const _Float16* p = as + (wave_m * WM + mt * 16 + lane % 16) * STRIDE + kt * 16;
                    const Frag left = *reinterpret_cast<const Frag*>(p);
                    #pragma unroll
                    for (int nt = 0; nt < NT; ++nt) {
                        if constexpr (Probe == 2) {   // consume both whole fragments, no WMMA
                            Half16 x = __builtin_bit_cast(Half16, left) * __builtin_bit_cast(Half16, right[nt]);
                            acc[mt][nt][0] += float(x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7] +
                                                    x[8] + x[9] + x[10] + x[11] + x[12] + x[13] + x[14] + x[15]);
                        }
                        else acc[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                            __builtin_bit_cast(Half16, left), __builtin_bit_cast(Half16, right[nt]), acc[mt][nt]);
                    }
                }
            }
        }
        if (stage + 1 < stages) store(stage + 1, buffer ^ 1);
        __syncthreads();
    }

    if constexpr (Probe == 1) acc[0][0][0] += float(probe_sink & 1u);
    if constexpr (!Fused) {
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int col = first_col + wave_n * WN + nt * 16 + lane % 16;
                if (col < outputs) {
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int row = first_row + wave_m * WM + mt * 16 + 2 * j + lane / 16;
                        if (row < rows) partial[(size_t(blockIdx.z) * rows + row) * outputs + col] = acc[mt][nt][j];
                    }
                }
            }
    } else {
        // splits == 1, BN % 128 == 0 and outputs % BN == 0 (checked by the launcher).
        // Per 16-row slice of every wave row, stage FP32 [waves_m*16][BN] in LDS
        // and run output_transform's per-group arithmetic in its exact order.
        constexpr int SLICE = G::waves_m * 16, FSTRIDE = BN + 4;
        static_assert(SLICE * FSTRIDE * 4 <= 2 * G::stage_halves * 2, "epilogue staging exceeds LDS");
        float* f = reinterpret_cast<float*>(lds);
        #pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    f[(wave_m * 16 + 2 * j + lane / 16) * FSTRIDE + wave_n * WN + nt * 16 + lane % 16] = acc[mt][nt][j];
            __syncthreads();
            constexpr int groups = SLICE * (BN / 128);
            for (int g = wave; g < groups; g += G::waves) {
                const int local = g / (BN / 128), group = g % (BN / 128);
                const int row = first_row + (local / 16) * WM + mt * 16 + local % 16;
                if (row < rows) {
                    const int begin_col = first_col + group * 128;
                    float values[4] = {};
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) values[j] += f[local * FSTRIDE + group * 128 + lane + 32 * j];
                    hadamard128(values);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int channel = begin_col + lane + 32 * j;
                        const size_t index = size_t(row) * outputs + channel;
                        float value = values[j] * __half2float(__ushort_as_half(scales[channel]));
                        if (bias) value += bias[channel];
                        value *= alpha;
                        if (beta != 0) value += beta * output[index];
                        output[index] = value;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// Row blocks per rasterisation group (block -> tile mapping only; bits unchanged).
// Default per tile (measured: 2 is best for the 128- and 256-row tiles at large K);
// STRIX_TRELLIS_WMMA_RASTER overrides it for measurements.
int raster_rows(int tile_default) {
    static const int value = [] {
        const char* v = std::getenv("STRIX_TRELLIS_WMMA_RASTER");
        int r = v ? std::atoi(v) : 0;
        return r >= 1 && r <= 64 ? r : 0;
    }();
    return value ? value : tile_default;
}
struct Config { int bm, bn, wm, wn, raster; };
// Index = tile id accepted by launch_wmma_tiled(tile=...).
constexpr Config configs[] = {
    {128, 128, 64, 32, 2},   // 0: 8 waves
    {64, 128, 64, 32, 8},    // 1: 4 waves
    {128, 64, 64, 32, 8},    // 2: 4 waves
    {64, 64, 32, 32, 8},     // 3: 4 waves
    {256, 128, 64, 64, 2},   // 4: 8 waves, 16 accumulators per wave
    {128, 128, 64, 64, 8},   // 5: 4 waves, 16 accumulators per wave
    {128, 256, 64, 64, 8},   // 6: 8 waves
    {32, 128, 32, 32, 8},    // 7: 4 waves
};
constexpr int config_count = int(sizeof(configs) / sizeof(configs[0]));

template<int Bits, int CB, int BM, int BN, int WM, int WN>
hipError_t launch_one(const DeviceWeights& w, const _Float16* input, float* partial, int rows, int splits,
                      hipStream_t stream, const WmmaFusedOutput* fused, int raster) {
    using G = Geometry<BM, BN, WM, WN>;
    dim3 grid(((rows + BM - 1) / BM) * (w.output / BN), 1, splits);   // N % BN == 0 (launcher)
    if (fused && fused->output) {
        if constexpr (BN % 128 == 0)
            tiled_core<Bits, CB, BM, BN, WM, WN, true><<<grid, G::threads, 0, stream>>>(w.trellis, input, partial,
                w.input, w.output, rows, splits, w.svh, w.bias, fused->output, fused->alpha, fused->beta, raster);
        else return hipErrorNotSupported;
    } else {
        tiled_core<Bits, CB, BM, BN, WM, WN, false><<<grid, G::threads, 0, stream>>>(w.trellis, input, partial,
            w.input, w.output, rows, splits, nullptr, nullptr, nullptr, 1.f, 0.f, raster);
    }
    return hipGetLastError();
}
#ifdef STRIX_TRELLIS_WMMA_PROBE
template<int Bits, int BM, int BN, int WM, int WN>
hipError_t launch_probe_one(const DeviceWeights& w, const _Float16* input, float* partial, int rows, int splits,
                            hipStream_t stream, int probe, int raster) {
    using G = Geometry<BM, BN, WM, WN>;
    dim3 grid(((rows + BM - 1) / BM) * (w.output / BN), 1, splits);
#define PROBE_CASE(P) case P: tiled_core<Bits, 2, BM, BN, WM, WN, false, P><<<grid, G::threads, 0, stream>>>(w.trellis, input,         partial, w.input, w.output, rows, splits, nullptr, nullptr, nullptr, 1.f, 0.f, raster); break
    switch (probe) { PROBE_CASE(1); PROBE_CASE(2); PROBE_CASE(3); default: return hipErrorNotSupported; }
#undef PROBE_CASE
    return hipGetLastError();
}
template<int Bits>
hipError_t launch_probe_bits(const DeviceWeights& w, const _Float16* input, float* partial, int rows, int splits,
                             hipStream_t stream, int tile, int probe) {
    switch (tile) {
#define TILE_CASE(I) case I: return launch_probe_one<Bits, configs[I].bm, configs[I].bn, configs[I].wm, configs[I].wn>(w, input, partial, rows, splits, stream, probe, raster_rows(configs[I].raster))
    TILE_CASE(0); TILE_CASE(1); TILE_CASE(2); TILE_CASE(3); TILE_CASE(4); TILE_CASE(5); TILE_CASE(6); TILE_CASE(7);
#undef TILE_CASE
    }
    return hipErrorNotSupported;
}
#endif
template<int Bits, int CB>
hipError_t launch_bits(const DeviceWeights& w, const _Float16* input, float* partial, int rows, int splits,
                       hipStream_t stream, const WmmaFusedOutput* fused, int tile) {
    switch (tile) {
#define TILE_CASE(I) case I: return launch_one<Bits, CB, configs[I].bm, configs[I].bn, configs[I].wm, configs[I].wn>(w, input, partial, rows, splits, stream, fused, raster_rows(configs[I].raster))
    TILE_CASE(0); TILE_CASE(1); TILE_CASE(2); TILE_CASE(3); TILE_CASE(4); TILE_CASE(5); TILE_CASE(6); TILE_CASE(7);
#undef TILE_CASE
    }
    return hipErrorNotSupported;
}
} // namespace

// Measured policy (7 EAQ tensors x M128-4096, cb2 K5/K6, every tile; README per-rank shapes
// confirmed in the final ABA). Within 1.2 % of the best tile on average, worst 1.19x, and faster
// than wmma_core in every measured cell. -1 = keep wmma_core (fewer than 128 rows: unmeasured).
int policy_tile(const DeviceWeights& w, int rows, int splits) {
    (void)splits;   // the split count is fixed by the numerics; every tile honours it
    if (rows < 128 || w.codebook != 2 || w.bits < 4 || w.bits > 6) return -1;
    if (rows <= 128) return w.output % 256 == 0 && w.output < 16384 ? 6 : 0;
    if (rows <= 1024) return 4;
    return 0;
}
} // namespace wmma_tiled

bool wmma_tiled_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRIX_TRELLIS_WMMA_TILED");
        return !(v && std::strcmp(v, "0") == 0);
    }();
    return enabled;
}
bool wmma_tiled_fused_output_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("STRIX_TRELLIS_WMMA_TILED_FUSED");
        return !(v && std::strcmp(v, "0") == 0);
    }();
    return enabled;
}
static bool tiled_geometry_ok(const DeviceWeights& w, int rows, int splits) {
    return w.trellis && rows > 0 && splits > 0 && w.input > 0 && w.output > 0 &&
        w.input % 128 == 0 && w.output % 128 == 0 && w.bits >= 1 && w.bits <= 8 &&
        w.codebook >= 0 && w.codebook <= 2;
}
bool wmma_tiled_fuses_output(const DeviceWeights& w, int rows, int splits, int tile) {
    if (!tiled_geometry_ok(w, rows, splits) || splits != 1 || !w.svh) return false;
    if (tile < 0) tile = wmma_tiled::policy_tile(w, rows, splits);
    if (tile < 0 || tile >= wmma_tiled::config_count) return false;
    const auto& c = wmma_tiled::configs[tile];
    return c.bn % 128 == 0 && w.output % c.bn == 0;
}
#ifdef STRIX_TRELLIS_WMMA_PROBE
hipError_t launch_wmma_tiled_probe(const DeviceWeights& w, const _Float16* input, float* partial, int rows,
    int splits, hipStream_t stream, int tile, int probe) {
    if (!tiled_geometry_ok(w, rows, splits) || w.codebook != 2 || tile < 0 || tile >= wmma_tiled::config_count ||
        w.output % wmma_tiled::configs[tile].bn) return hipErrorNotSupported;
    if (w.bits == 4) return wmma_tiled::launch_probe_bits<4>(w, input, partial, rows, splits, stream, tile, probe);
    if (w.bits == 5) return wmma_tiled::launch_probe_bits<5>(w, input, partial, rows, splits, stream, tile, probe);
    if (w.bits == 6) return wmma_tiled::launch_probe_bits<6>(w, input, partial, rows, splits, stream, tile, probe);
    return hipErrorNotSupported;
}
#endif
hipError_t launch_wmma_tiled(const DeviceWeights& w, const _Float16* input, float* partial, int rows,
    int splits, hipStream_t stream, const WmmaFusedOutput* fused, int tile) {
    if (!tiled_geometry_ok(w, rows, splits) || !input || (uintptr_t(input) & 15u)) return hipErrorNotSupported;
    if (tile < 0) tile = wmma_tiled::policy_tile(w, rows, splits);
    if (tile < 0 || tile >= wmma_tiled::config_count) return hipErrorNotSupported;
    if (w.output % wmma_tiled::configs[tile].bn) return hipErrorNotSupported;   // whole N blocks only
    if (fused && fused->output) {
        if (!wmma_tiled_fuses_output(w, rows, splits, tile)) return hipErrorNotSupported;
    } else if (!partial) return hipErrorNotSupported;
#define BITS_CASE(B) case B: switch (w.codebook) { \
        case 0: return wmma_tiled::launch_bits<B, 0>(w, input, partial, rows, splits, stream, fused, tile); \
        case 1: return wmma_tiled::launch_bits<B, 1>(w, input, partial, rows, splits, stream, fused, tile); \
        case 2: return wmma_tiled::launch_bits<B, 2>(w, input, partial, rows, splits, stream, fused, tile); } break
    switch (w.bits) {
        BITS_CASE(1); BITS_CASE(2); BITS_CASE(3); BITS_CASE(4);
        BITS_CASE(5); BITS_CASE(6); BITS_CASE(7); BITS_CASE(8);
    }
#undef BITS_CASE
    return hipErrorNotSupported;
}
} // namespace strix_trellis
