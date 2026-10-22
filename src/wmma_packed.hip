// Independent minimal Q decoder substitution, copied from our gemm.hip.
#include "wmma_packed.hpp"
#include "launch_policy.hpp"
#include "v14_core_policy.hpp"
#include "packed_decoder.hpp"
#include <type_traits>
#include <limits>
#include <string_view>
namespace strix_trellis::wmma_packed {
namespace {
using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));
using HalfPair = _Float16 __attribute__((ext_vector_type(2)));
static_assert(sizeof(HalfPair)==4 && sizeof(__half2)==4);
template<int Bits, int Mode, int TileM, bool SwizzleQ = false, int PadA = 0>
__global__ __launch_bounds__(128) void decoder_core(const uint32_t* packed,
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
            // The only substituted block. SADDR base is uniform per wave.
            if constexpr (Mode == 0) {
            WaveTile<Bits> tile(packed + (size_t(tile_k) * tiles_n + tile_n) * (Bits * 8));
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int r = 2 * (lane % 4) + (j % 2) + 8 * ((j / 2) % 2);
                int c = lane / 4 + 8 * (j / 4);
                // XOR only changes LDS coordinates; the decoded Q is unchanged.
                q[wave * 256 + r * 16 + (c ^ (SwizzleQ ? (r & 7) * 2 : 0))] = _Float16(codebook_value(tile.state(8 * lane + j), 2));
            }
            } else {
                int copytile_n = tile_n;
                if constexpr (Mode == 2) copytile_n = __builtin_amdgcn_readfirstlane(copytile_n);
                const uint32_t* base = packed + (size_t(tile_k) * tiles_n + copytile_n) * (Bits * 8);
                using Reader = std::conditional_t<Mode == 1, optimized::AlignedLane<Bits,true>, packed_decode::SaddrLane<Bits,true>>;
                Reader tile(base);
                {
                    int r = 2 * (lane % 4) + 0, c = lane / 4 + 0;
                    __half2 decoded;
                    if constexpr (Mode == 1) decoded = optimized::mul1_half2(tile.template state<0>(), tile.template state<1>());
                    else decoded = packed_decode::mul1_half2(tile.template state<0>(), tile.template state<1>());
                    HalfPair pair = __builtin_bit_cast(HalfPair, decoded);
                    q[wave * 256 + r * 16 + (c ^ (SwizzleQ ? (r & 7) * 2 : 0))] = pair[0];
                    q[wave * 256 + (r + 1) * 16 + (c ^ (SwizzleQ ? ((r + 1) & 7) * 2 : 0))] = pair[1];
                }
                {
                    int r = 2 * (lane % 4) + 8, c = lane / 4 + 0;
                    __half2 decoded;
                    if constexpr (Mode == 1) decoded = optimized::mul1_half2(tile.template state<2>(), tile.template state<3>());
                    else decoded = packed_decode::mul1_half2(tile.template state<2>(), tile.template state<3>());
                    HalfPair pair = __builtin_bit_cast(HalfPair, decoded);
                    q[wave * 256 + r * 16 + (c ^ (SwizzleQ ? (r & 7) * 2 : 0))] = pair[0];
                    q[wave * 256 + (r + 1) * 16 + (c ^ (SwizzleQ ? ((r + 1) & 7) * 2 : 0))] = pair[1];
                }
                {
                    int r = 2 * (lane % 4) + 0, c = lane / 4 + 8;
                    __half2 decoded;
                    if constexpr (Mode == 1) decoded = optimized::mul1_half2(tile.template state<4>(), tile.template state<5>());
                    else decoded = packed_decode::mul1_half2(tile.template state<4>(), tile.template state<5>());
                    HalfPair pair = __builtin_bit_cast(HalfPair, decoded);
                    q[wave * 256 + r * 16 + (c ^ (SwizzleQ ? (r & 7) * 2 : 0))] = pair[0];
                    q[wave * 256 + (r + 1) * 16 + (c ^ (SwizzleQ ? ((r + 1) & 7) * 2 : 0))] = pair[1];
                }
                {
                    int r = 2 * (lane % 4) + 8, c = lane / 4 + 8;
                    __half2 decoded;
                    if constexpr (Mode == 1) decoded = optimized::mul1_half2(tile.template state<6>(), tile.template state<7>());
                    else decoded = packed_decode::mul1_half2(tile.template state<6>(), tile.template state<7>());
                    HalfPair pair = __builtin_bit_cast(HalfPair, decoded);
                    q[wave * 256 + r * 16 + (c ^ (SwizzleQ ? (r & 7) * 2 : 0))] = pair[0];
                    q[wave * 256 + (r + 1) * 16 + (c ^ (SwizzleQ ? ((r + 1) & 7) * 2 : 0))] = pair[1];
                }
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

} // namespace
} // namespace strix_trellis::wmma_packed
namespace strix_trellis {
hipError_t launch_wmma_packed(const DeviceWeights& w,const _Float16* a,float* p,
    int rows,int splits,hipStream_t stream) {
    const int tile=wmma_tile_rows(w,rows);
    const bool pad18=(w.bits==5 && w.codebook==2) || medium_prefill_candidate(w,rows);
    if(!w.trellis || !a || !p || !wmma_packed_candidate(w.input,w.output,w.bits,w.codebook,
       rows,splits,tile,pad18,!(uintptr_t(a)&3u),!(uintptr_t(p)&3u),!(uintptr_t(w.trellis)&3u)))
        return hipErrorInvalidValue;
    dim3 grid((w.output/16+3)/4,rows/64,splits);
    if(w.bits==5) wmma_packed::decoder_core<5,2,64,true,2><<<grid,128,0,stream>>>(w.trellis,a,p,w.input,w.output,rows,splits);
    else wmma_packed::decoder_core<6,2,64,true,2><<<grid,128,0,stream>>>(w.trellis,a,p,w.input,w.output,rows,splits);
    return hipGetLastError();
}
}
