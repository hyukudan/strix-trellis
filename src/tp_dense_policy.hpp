#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>

namespace strix_trellis {
struct TpDenseShape {
    uint64_t packed_bytes = 0;
    uint64_t input_span_bytes = 0;
    uint64_t output_span_bytes = 0;
    size_t shared_bytes = 0;
    unsigned blocks = 0;
    unsigned row_blocks = 0;
    unsigned rows_per_block = 0;
    unsigned threads = 0;
};

// Strides are measured in F32 elements, not bytes. The span includes row gaps
// so the C API can validate allocation bounds and conservatively reject aliases.
inline bool tp_dense_span(uint64_t rows, uint64_t columns, uint64_t stride,
                          uint64_t& bytes) {
    bytes = 0;
    if (!rows || !columns || stride < columns) return false;
    constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();
    if (rows > 1 && stride > (maximum - columns) / (rows - 1)) return false;
    const uint64_t elements = (rows - 1) * stride + columns;
    if (elements > maximum / sizeof(float)) return false;
    bytes = elements * sizeof(float);
    return true;
}

inline bool tp_dense_geometry(int input, int output) {
    return (input == 1280 && output == 16384) || // q_b half output
           (input == 4096 && output == 5120) || // o_b half K
           (input == 5120 && output == 1152) || // shared gate/up half output
           (input == 1152 && output == 5120) || // shared down half K
           (input == 5120 && output == 1280) || // q_a
           (input == 5120 && output == 512);    // kv
}

inline bool tp_dense_shape(int input, int output, int bits, int codebook,
    int rows, uint64_t x_stride, uint64_t y_stride, TpDenseShape& result,
    int rows_per_block = 0) {
    result = {};
    if (!tp_dense_geometry(input, output) || (bits != 4 && bits != 5) ||
        codebook != 2 || rows < 1 || rows > 8) return false;
    if (rows_per_block == 0) rows_per_block = rows;
    if (rows_per_block < 1 || rows_per_block > rows) return false;
    uint64_t input_bytes = 0, output_bytes = 0;
    if (!tp_dense_span(uint64_t(rows), uint64_t(input), x_stride, input_bytes) ||
        !tp_dense_span(uint64_t(rows), uint64_t(output), y_stride, output_bytes)) return false;
    result.packed_bytes = uint64_t(input) * uint64_t(output) * uint64_t(bits) / 8;
    result.input_span_bytes = input_bytes;
    result.output_span_bytes = output_bytes;
    result.shared_bytes = size_t(rows_per_block) * 128 * (4 * sizeof(float) + 2 * sizeof(uint16_t));
    result.blocks = unsigned(output / 128);
    result.row_blocks = unsigned((rows + rows_per_block - 1) / rows_per_block);
    result.rows_per_block = unsigned(rows_per_block);
    result.threads = 512;
    return true;
}

inline bool tp_dense_image_limits(int maximum_threads, size_t static_shared,
    size_t local_bytes, int maximum_dynamic, size_t available_shared,
    size_t required_dynamic) {
    if (maximum_threads < 512 || local_bytes || maximum_dynamic < 0 ||
        size_t(maximum_dynamic) < required_dynamic || static_shared > available_shared)
        return false;
    return required_dynamic <= available_shared - static_shared;
}
} // namespace strix_trellis
