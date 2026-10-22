#include "tp_dense.hpp"
#include "tp_fused_device.hpp"
#include <cmath>

namespace strix_trellis::tp_dense {
struct Access {
    TpDenseArguments arguments;
    int first_row;
    __device__ __forceinline__ float load_input(int row, int column) const {
        row += first_row;
        if (row >= arguments.rows) return 0;
        return arguments.x[size_t(row) * arguments.x_row_stride + column];
    }
    __device__ __forceinline__ void store_output(int row, int column, float value) const {
        row += first_row;
        if (row >= arguments.rows) return;
        const size_t index = size_t(row) * arguments.y_row_stride + column;
        value *= arguments.alpha;
        if (arguments.beta != 0) value += arguments.beta * arguments.y[index];
        arguments.y[index] = value;
    }
};

template<int Bits, int Rows>
__global__ __launch_bounds__(512) void projection(TpDenseArguments arguments) {
    extern __shared__ unsigned char storage[] __attribute__((aligned(16)));
    float* partial = reinterpret_cast<float*>(storage);
    _Float16* cache = reinterpret_cast<_Float16*>(partial + 4 * Rows * 128);
    tp_fused::project<Bits, Rows>(arguments.weights, Access{arguments, int(blockIdx.y) * Rows}, int(blockIdx.x),
                                 partial, cache);
}

const void* function(int bits, int rows) {
    #define ENTRY(B, R) reinterpret_cast<const void*>(projection<B, R>)
    static const void* const table[2][8] = {
        {ENTRY(4,1), ENTRY(4,2), ENTRY(4,3), ENTRY(4,4), ENTRY(4,5), ENTRY(4,6), ENTRY(4,7), ENTRY(4,8)},
        {ENTRY(5,1), ENTRY(5,2), ENTRY(5,3), ENTRY(5,4), ENTRY(5,5), ENTRY(5,6), ENTRY(5,7), ENTRY(5,8)}
    };
    #undef ENTRY
    return (bits == 4 || bits == 5) && rows >= 1 && rows <= 8 ? table[bits - 4][rows - 1] : nullptr;
}
} // namespace strix_trellis::tp_dense

namespace strix_trellis {
hipError_t query_tp_dense_resources(int bits, int rows_per_block, size_t available_shared,
                                   bool* supported) {
    if (!supported) return hipErrorInvalidValue;
    *supported = false;
    const void* function = tp_dense::function(bits, rows_per_block);
    if (!function) return hipErrorInvalidValue;
    hipFuncAttributes attributes{};
    const hipError_t status = hipFuncGetAttributes(&attributes, function);
    if (status != hipSuccess) return status;
    const size_t required = size_t(rows_per_block) * 128 * (4 * sizeof(float) + 2 * sizeof(uint16_t));
    *supported = tp_dense_image_limits(attributes.maxThreadsPerBlock, attributes.sharedSizeBytes,
        attributes.localSizeBytes, attributes.maxDynamicSharedSizeBytes, available_shared, required);
    return hipSuccess;
}

hipError_t launch_tp_dense_trusted(const TpDenseArguments& arguments, hipStream_t stream) {
    TpDenseShape shape{};
    const DeviceWeights& weights = arguments.weights;
    if (!tp_dense_shape(weights.input, weights.output, weights.bits, weights.codebook,
            arguments.rows, arguments.x_row_stride, arguments.y_row_stride, shape, arguments.rows_per_block) ||
        !weights.trellis || !weights.suh || !weights.svh || !arguments.x || !arguments.y ||
        (reinterpret_cast<uintptr_t>(weights.trellis) & 3u) ||
        (reinterpret_cast<uintptr_t>(weights.suh) & 1u) ||
        (reinterpret_cast<uintptr_t>(weights.svh) & 1u) ||
        (weights.bias && (reinterpret_cast<uintptr_t>(weights.bias) & 3u)) ||
        (reinterpret_cast<uintptr_t>(arguments.x) & 3u) ||
        (reinterpret_cast<uintptr_t>(arguments.y) & 3u) ||
        !std::isfinite(arguments.alpha) || !std::isfinite(arguments.beta))
        return hipErrorInvalidValue;
    void* kernel_arguments[] = {const_cast<TpDenseArguments*>(&arguments)};
    return hipLaunchKernel(tp_dense::function(weights.bits, int(shape.rows_per_block)),
        dim3(shape.blocks, shape.row_blocks), dim3(shape.threads), kernel_arguments, shape.shared_bytes, stream);
}
} // namespace strix_trellis
