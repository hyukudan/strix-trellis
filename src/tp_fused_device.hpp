#pragma once
// Independently implemented complete H128 projection shared by the explicit
// TP dense and expert paths. No global transformed activation or partial exists.
#include "backend.hpp"
#include "hadamard.hpp"
#include "packed_decoder.hpp"
#include "tp_fused_layout.hpp"

namespace strix_trellis::tp_fused {
using Pair = _Float16 __attribute__((ext_vector_type(2)));

// Contract: 512 threads (two physical groups of eight wave32s), one output
// H128 block, complete H128 input/output dimensions and codebook mul1. The
// caller supplies 4*Rows*128 floats plus 2*Rows*128 halves in disjoint LDS.
// Access supplies original-basis input and consumes the complete scaled/bias
// projection. It may map rows directly to model/routing storage. Every thread
// participates in all barriers, including the final barrier, so callers may
// reuse LDS for another projection immediately after return.
template<int Bits, int Rows, class Access>
__device__ __forceinline__ void project(const DeviceWeights& weights,
    const Access& access, int output_h128, float* partial, _Float16* cache) {
    static_assert(Bits >= 2 && Bits <= 5);
    static_assert(Rows >= 1 && Rows <= 8);
    const int lane = threadIdx.x & 31;
    const int group = __builtin_amdgcn_readfirstlane(int(threadIdx.x) / 256);
    const int wave = __builtin_amdgcn_readfirstlane((int(threadIdx.x) & 255) / 32);
    const int output_tile = __builtin_amdgcn_readfirstlane(output_h128 * 8 + wave);
    const int maximum_h128 = tp_fused_maximum_h128(weights.input);
    _Float16* activation = cache + group * Rows * 128;
    #pragma unroll
    for (int phase = 0; phase < 2; ++phase) {
        const int split = group + 2 * phase;
        const auto partition = tp_fused_partition(weights.input, split);
        const int begin = partition.begin_tile;
        const int end = partition.end_tile;
        const int first_h128 = partition.first_h128;
        const int end_h128 = partition.end_h128;
        float sum[Rows][2] = {};
        // Uniform iteration count is essential: groups can cover different
        // numbers of H128 blocks despite having the same number of K16 tiles.
        for (int step = 0; step < maximum_h128; ++step) {
            const int h128 = first_h128 + step;
            if (h128 < end_h128 && wave < Rows) {
                float value[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int column = h128 * 128 + lane + 32 * j;
                    const float scaled = materialize_fp32(access.load_input(wave, column) *
                        __half2float(__ushort_as_half(weights.suh[column])));
                    value[j] = float(_Float16(scaled));
                }
                hadamard128(value);
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    activation[wave * 128 + lane + 32 * j] = _Float16(value[j]);
            }
            __syncthreads();
            if (h128 < end_h128) {
                // Transform the complete H128 first, then crop the logical
                // K16 partition. Splitting the H transform itself is invalid
                // for q_b's 320-column partitions (and other boundary cases).
                const int first_tile = begin > h128 * 8 ? begin : h128 * 8;
                const int last_tile = end < (h128 + 1) * 8 ? end : (h128 + 1) * 8;
                for (int k = first_tile; k < last_tile; ++k) {
                    packed_decode::SaddrLane<Bits> reader(weights.trellis +
                        (size_t(k) * (weights.output / 16) + output_tile) * (Bits * 8));
                    Pair input[Rows][2];
                    const int local_column = (k - h128 * 8) * 16 + 2 * (lane & 3);
                    #pragma unroll
                    for (int row = 0; row < Rows; ++row) {
                        __builtin_memcpy(&input[row][0], activation + row * 128 + local_column, 4);
                        __builtin_memcpy(&input[row][1], activation + row * 128 + local_column + 8, 4);
                    }
                    #pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        uint32_t first, second;
                        if (pair == 0) { first = reader.template state<0>(); second = reader.template state<1>(); }
                        else if (pair == 1) { first = reader.template state<2>(); second = reader.template state<3>(); }
                        else if (pair == 2) { first = reader.template state<4>(); second = reader.template state<5>(); }
                        else { first = reader.template state<6>(); second = reader.template state<7>(); }
                        const Pair weight = __builtin_bit_cast(Pair, packed_decode::mul1_half2(first, second));
                        #pragma unroll
                        for (int row = 0; row < Rows; ++row)
                            sum[row][pair / 2] = __builtin_amdgcn_fdot2(
                                weight, input[row][pair & 1], sum[row][pair / 2], false);
                    }
                }
            }
            __syncthreads(); // No group may overwrite activations still in use.
        }
        #pragma unroll
        for (int row = 0; row < Rows; ++row) {
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                float value = sum[row][half];
                value += __shfl_down(value, 2, 4);
                value += __shfl_down(value, 1, 4);
                if (!(lane & 3)) {
                    const int column = wave * 16 + lane / 4 + half * 8;
                    partial[(split * Rows + row) * 128 + column] = value;
                }
            }
        }
        __syncthreads();
    }
    const int row = int(threadIdx.x) / 32;
    if (row < Rows) {
        float value[4] = {};
        // Preserve the declared S4 association, including its +0 origin.
        #pragma unroll
        for (int split = 0; split < 4; ++split) {
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                value[j] += partial[(split * Rows + row) * 128 + lane + 32 * j];
        }
        hadamard128(value);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int column = output_h128 * 128 + lane + 32 * j;
            float result = value[j] * __half2float(__ushort_as_half(weights.svh[column]));
            if (weights.bias) result += weights.bias[column];
            access.store_output(row, column, result);
        }
    }
    __syncthreads(); // Access output and LDS can be consumed by the next call.
}
} // namespace strix_trellis::tp_fused
