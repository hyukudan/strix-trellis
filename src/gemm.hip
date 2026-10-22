// Independent fused EXL3 decode / native wave32 WMMA core for gfx1151.
#include "gemm.hpp"
#include "wmma_packed.hpp"
#include "v14_core_policy.hpp"
#include "launch_policy.hpp"
#include "trellis_device.hpp"
#include <cstdint>

namespace strix_trellis {
namespace {
using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));

template<int Bits, int Codebook, int TileM, bool SwizzleQ = false, int PadA = 0>
__global__ __launch_bounds__(128) void wmma_core(const uint32_t* packed,
    const _Float16* input, float* partial, int inputs, int outputs,
    int rows, int splits) {
    // Four waves own distinct N16 tiles and reuse decoded Q across TileM rows.
    // Padding distributes simultaneous A row loads across LDS banks.
    constexpr int AStride = 16 + PadA;
    __shared__ _Float16 a[TileM * AStride];
    __shared__ _Float16 q[4 * 16 * 16];
    const int thread = threadIdx.x, lane = thread & 31, wave = thread / 32;
    const int first_row = int(blockIdx.y) * TileM;
    const int tile_n = int(blockIdx.x) * 4 + wave;
    const int tiles_n = outputs / 16, tiles_k = inputs / 16;
    const int begin = int(int64_t(tiles_k) * blockIdx.z / splits);
    const int end = int(int64_t(tiles_k) * (blockIdx.z + 1) / splits);
    Float8 acc[TileM / 16] = {};
    for (int tile_k = begin; tile_k < end; ++tile_k) {
        for (int j = thread; j < TileM * 16; j += 128) {
            int row = first_row + j / 16;
            a[(j / 16) * AStride + j % 16] = row < rows ? input[size_t(row) * inputs + tile_k * 16 + j % 16] : _Float16(0);
        }
        // No early exit: all waves must reach both workgroup barriers.
        if (tile_n < tiles_n) {
            WaveTile<Bits> tile(packed + (size_t(tile_k) * tiles_n + tile_n) * (Bits * 8));
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int r = 2 * (lane % 4) + (j % 2) + 8 * ((j / 2) % 2);
                int c = lane / 4 + 8 * (j / 4);
                // XOR only changes LDS coordinates; the decoded Q is unchanged.
                q[wave * 256 + r * 16 + (c ^ (SwizzleQ ? (r & 7) * 2 : 0))] = _Float16(codebook_value(tile.state(8 * lane + j), Codebook));
            }
        }
        __syncthreads();
        if (tile_n < tiles_n) {
            Half16 right;
            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                // Replicate both operands in lanes 16..31 (RDNA3/3.5 wave32).
                right[k] = q[wave * 256 + k * 16 + ((lane % 16) ^ (SwizzleQ ? (k & 7) * 2 : 0))];
            }
            #pragma unroll
            for (int m = 0; m < TileM / 16; ++m) {
                Half16 left;
                #pragma unroll
                for (int k = 0; k < 16; ++k)
                    left[k] = a[(m * 16 + lane % 16) * AStride + k];
                acc[m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(left, right, acc[m]);
            }
        }
        __syncthreads();
    }
    if (tile_n < tiles_n) {
        #pragma unroll
        for (int m = 0; m < TileM / 16; ++m) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int row = first_row + m * 16 + 2 * j + lane / 16;
                if (row < rows)
                    partial[(size_t(blockIdx.z) * rows + row) * outputs + tile_n * 16 + lane % 16] = acc[m][j];
            }
        }
    }
}

template<int Bits, int TileM, bool SwizzleQ = false, int PadA = 0> hipError_t launch_codebook(const DeviceWeights& w,
    const _Float16* input, float* partial, int rows, int splits, hipStream_t stream) {
    dim3 grid((w.output / 16 + 3) / 4, rows / TileM + (rows % TileM != 0), splits);
    #define CB_CASE(C) case C: wmma_core<Bits,C,TileM,SwizzleQ,PadA><<<grid,128,0,stream>>>(w.trellis,input,partial,w.input,w.output,rows,splits); break
    switch (w.codebook) { CB_CASE(0); CB_CASE(1); CB_CASE(2); }
    #undef CB_CASE
    return hipGetLastError();
}
} // namespace

template<int TileM, bool SwizzleQ = false, int PadA = 0> hipError_t launch_tile(const DeviceWeights& w, const _Float16* input,
    float* partial, int rows, int splits, hipStream_t stream) {
    if (!w.trellis || !input || !partial || rows <= 0 || splits <= 0 ||
        w.input <= 0 || w.output <= 0 || w.input % 128 || w.output % 128 ||
        w.bits < 1 || w.bits > 8 || w.codebook < 0 || w.codebook > 2)
        return hipErrorInvalidValue;
    #define BITS_CASE(B) case B: return launch_codebook<B,TileM,SwizzleQ,PadA>(w,input,partial,rows,splits,stream)
    switch (w.bits) {
        BITS_CASE(1); BITS_CASE(2); BITS_CASE(3); BITS_CASE(4);
        BITS_CASE(5); BITS_CASE(6); BITS_CASE(7); BITS_CASE(8);
    }
    #undef BITS_CASE
    return hipErrorInvalidValue;
}
hipError_t launch_wmma_core(const DeviceWeights& w, const _Float16* input,
    float* partial, int rows, int splits, hipStream_t stream) {
    const bool pad18=(w.bits==5 && w.codebook==2) || medium_prefill_candidate(w,rows);
    if(wmma_packed_candidate(w.input,w.output,w.bits,w.codebook,rows,splits,
        wmma_tile_rows(w,rows),pad18,!(uintptr_t(input)&3u),!(uintptr_t(partial)&3u),!(uintptr_t(w.trellis)&3u)))
        return launch_wmma_packed(w,input,partial,rows,splits,stream);
    // Shared with workspace query so the grid and split policy cannot drift.
    if(wmma_tile_rows(w,rows)==64) {
        if((w.bits==5 && w.codebook==2) || medium_prefill_candidate(w,rows))
            return launch_tile<64,true,2>(w,input,partial,rows,splits,stream);
        return launch_tile<64,true>(w,input,partial,rows,splits,stream);
    }
    return launch_tile<16>(w,input,partial,rows,splits,stream);
}
#ifdef TRELLIS_GEMM_REUSE_BENCH
// Benchmark-only selection; the public launch contract is unchanged.
hipError_t launch_wmma_reuse_variant(const DeviceWeights& w, const _Float16* input,
    float* partial, int rows, int splits, hipStream_t stream, int tile_m) {
    if (tile_m == 0) return launch_wmma_core(w,input,partial,rows,splits,stream);
    if (tile_m == -65) return launch_tile<64,true,2>(w,input,partial,rows,splits,stream);
    if (tile_m == -64) return launch_tile<64,true>(w,input,partial,rows,splits,stream);
    if (tile_m == 128) return launch_tile<128>(w,input,partial,rows,splits,stream);
    if (tile_m == 64) return launch_tile<64>(w,input,partial,rows,splits,stream);
    return launch_tile<16>(w,input,partial,rows,splits,stream);
}
#endif
} // namespace strix_trellis
