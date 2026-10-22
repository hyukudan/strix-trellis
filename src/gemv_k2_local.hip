// Experimental intra-block split-K and software prefetch, independently
// implemented on the project's existing K2 arithmetic. Never selected by V16.
#include "gemv_k2_local.hpp"
#include "k2_local_decode.hpp"
#include <hip/hip_fp16.h>
#include <cmath>
// Keep experiment-only transform symbols separate from the qualified backend.
#define input_transform k2_local_input_transform
#define output_transform k2_local_output_transform
#include "transforms.hpp"
#undef input_transform
#undef output_transform

namespace strix_trellis {
namespace k2_local_kernel {

template<int Unroll, bool Pipeline>
__global__ __launch_bounds__(128) void kernel(const uint32_t* __restrict__ packed,
    const _Float16* __restrict__ activation, float* __restrict__ output,
    int input, int columns) {
    static_assert(Unroll == 1 || Unroll == 2 || Unroll == 4);
    const int wave = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int tile_n = __builtin_amdgcn_readfirstlane(blockIdx.x);
    const int tiles = input / 16;
    const int begin = tiles * wave / 4, end = tiles * (wave + 1) / 4;
    float lo = 0.0f, hi = 0.0f;
    int k = begin;
    if constexpr (Pipeline) {
        // Ordinary C++ loads remain visible to LLVM's hazard/wait machinery.
        // This exposes independent next-tile loads; ISA inspection must verify
        // overlap and VGPR use. It does not promise a hardware async copy.
        if (k + Unroll <= end) {
            Raw current[Unroll];
            #pragma unroll
            for (int j = 0; j < Unroll; ++j)
                current[j] = read_raw(packed, activation, k + j, columns, tile_n);
            for (; k + 2 * Unroll <= end; k += Unroll) {
                Raw next[Unroll];
                #pragma unroll
                for (int j = 0; j < Unroll; ++j)
                    next[j] = read_raw(packed, activation, k + Unroll + j, columns, tile_n);
                #pragma unroll
                for (int j = 0; j < Unroll; ++j) consume(current[j], lo, hi);
                #pragma unroll
                for (int j = 0; j < Unroll; ++j) current[j] = next[j];
            }
            #pragma unroll
            for (int j = 0; j < Unroll; ++j) consume(current[j], lo, hi);
            k += Unroll;
        }
    } else {
        for (; k + Unroll <= end; k += Unroll) {
            Raw raw[Unroll];
            #pragma unroll
            for (int j = 0; j < Unroll; ++j)
                raw[j] = read_raw(packed, activation, k + j, columns, tile_n);
            #pragma unroll
            for (int j = 0; j < Unroll; ++j) consume(raw[j], lo, hi);
        }
    }
    for (; k < end; ++k) consume(read_raw(packed, activation, k, columns, tile_n), lo, hi);
    lo += __shfl_down(lo, 2, 4); lo += __shfl_down(lo, 1, 4);
    hi += __shfl_down(hi, 2, 4); hi += __shfl_down(hi, 1, 4);
    __shared__ float partial[4][16]; // 256 bytes, instead of four global planes.
    if (!(lane & 3)) {
        partial[wave][lane / 4] = lo;
        partial[wave][lane / 4 + 8] = hi;
    }
    // Every wave participates, including an empty K partition. No early return.
    __syncthreads();
    if (wave == 0 && lane < 16) {
        // Exactly the qualified output_transform's split order and +0 seed.
        float value = 0.0f;
        #pragma unroll
        for (int split = 0; split < 4; ++split) value += partial[split][lane];
        output[size_t(tile_n) * 16 + lane] = value;
    }
}
} // namespace k2_local_kernel

hipError_t launch_gemv_k2_local_core(const DeviceWeights& w, const _Float16* a,
    float* p, K2LocalSchedule schedule, hipStream_t stream) {
    K2LocalShape shape{};
    if (!k2_local_shape(w.input, w.output, w.bits, w.codebook, false, shape) ||
        !k2_local_schedule_name(schedule) ||
        !k2_local_span(w.trellis, shape.packed_bytes, 4) ||
        !k2_local_span(a, shape.activation_bytes, 4) ||
        !k2_local_span(p, shape.output_bytes, 4) ||
        !k2_local_disjoint(w.trellis, shape.packed_bytes, a, shape.activation_bytes) ||
        !k2_local_disjoint(w.trellis, shape.packed_bytes, p, shape.output_bytes) ||
        !k2_local_disjoint(a, shape.activation_bytes, p, shape.output_bytes))
        return hipErrorInvalidValue;
    const dim3 grid(w.output / 16), block(128);
    #define RUN(U, P) k2_local_kernel::kernel<U, P><<<grid, block, 0, stream>>>(w.trellis, a, p, w.input, w.output)
    switch (schedule) {
    case K2LocalSchedule::U1: RUN(1, false); break;
    case K2LocalSchedule::U2: RUN(2, false); break;
    case K2LocalSchedule::U4: RUN(4, false); break;
    case K2LocalSchedule::Pipeline2: RUN(2, true); break;
    case K2LocalSchedule::Pipeline4: RUN(4, true); break;
    }
    #undef RUN
    return hipGetLastError();
}
size_t gemv_k2_local_workspace_bytes(const DeviceWeights& w) {
    K2LocalShape shape{};
    return k2_local_shape(w.input, w.output, w.bits, w.codebook, true, shape)
        ? shape.workspace_bytes : SIZE_MAX;
}
hipError_t launch_gemv_k2_local_linear(const DeviceWeights& w, const float* input,
    float* output, void* workspace, size_t workspace_bytes, K2LocalSchedule schedule,
    hipStream_t stream, float alpha, float beta) {
    K2LocalShape shape{};
    if (!k2_local_shape(w.input, w.output, w.bits, w.codebook, true, shape) ||
        !k2_local_schedule_name(schedule) || !std::isfinite(alpha) || !std::isfinite(beta) ||
        workspace_bytes < shape.workspace_bytes ||
        !k2_local_span(workspace, workspace_bytes, 256) ||
        !k2_local_span(input, size_t(w.input) * 4, 4) ||
        !k2_local_span(output, shape.output_bytes, 4) ||
        !k2_local_span(w.trellis, shape.packed_bytes, 4) ||
        !k2_local_span(w.suh, shape.activation_bytes, 2) ||
        !k2_local_span(w.svh, size_t(w.output) * 2, 2) ||
        (w.bias && !k2_local_span(w.bias, shape.output_bytes, 4))) return hipErrorInvalidValue;
    const void* pointers[] = {input, output, workspace, w.trellis, w.suh, w.svh, w.bias};
    const size_t bytes[] = {size_t(w.input) * 4, shape.output_bytes, workspace_bytes,
        shape.packed_bytes, shape.activation_bytes, size_t(w.output) * 2,
        w.bias ? shape.output_bytes : 0};
    // Immutable weights may share read-only storage; every writable span and
    // the original activation must be disjoint from all other logical spans.
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 7; ++j)
            if (!k2_local_disjoint(pointers[i], bytes[i], pointers[j], bytes[j]))
                return hipErrorInvalidValue;
    auto* a = static_cast<_Float16*>(workspace);
    auto* p = reinterpret_cast<float*>(static_cast<unsigned char*>(workspace) + shape.partial_offset);
    k2_local_input_transform<<<(w.input / 128 + 3) / 4, 128, 0, stream>>>(input, w.suh, a, w.input, 1);
    auto status = hipGetLastError();
    if (status != hipSuccess) return status;
    status = launch_gemv_k2_local_core(w, a, p, schedule, stream);
    if (status != hipSuccess) return status;
    // The four internal partitions have already been summed. There is one plane.
    k2_local_output_transform<<<(w.output / 128 + 3) / 4, 128, 0, stream>>>(p, w.svh,
        w.bias, output, w.output, 1, 1, alpha, beta);
    return hipGetLastError();
}
} // namespace strix_trellis
