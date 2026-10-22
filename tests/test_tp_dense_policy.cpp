#include "../src/tp_dense_policy.hpp"
#include "../src/tp_fused_layout.hpp"
#include <cassert>
#include <climits>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <vector>

using namespace strix_trellis;

int main() {
    const int geometries[][2] = {
        {1280, 16384}, {4096, 5120}, {5120, 1152},
        {1152, 5120}, {5120, 1280}, {5120, 512}
    };
    unsigned positive_cases = 0;
    for (const auto& geometry : geometries) {
        const int input = geometry[0], output = geometry[1];
        for (int bits : {4, 5}) for (int rows = 1; rows <= 8; ++rows) {
            TpDenseShape shape{};
            assert(tp_dense_shape(input, output, bits, 2, rows,
                uint64_t(input) + 37, uint64_t(output) + 19, shape));
            assert(shape.input_span_bytes == uint64_t((rows - 1) * (input + 37) + input) * 4);
            assert(shape.output_span_bytes == uint64_t((rows - 1) * (output + 19) + output) * 4);
            assert(shape.packed_bytes == uint64_t(input) * output * bits / 8);
            assert(shape.shared_bytes == size_t(rows) * 2560);
            assert(shape.shared_bytes <= 20480);
            assert(shape.blocks == unsigned(output / 128) && shape.threads == 512);
            assert(shape.row_blocks == 1 && shape.rows_per_block == unsigned(rows));
            assert(tp_dense_image_limits(512, 0, 0, 65536, 65536, shape.shared_bytes));
            for (int tile = 1; tile <= rows; ++tile) {
                TpDenseShape tiled{};
                assert(tp_dense_shape(input, output, bits, 2, rows,
                    uint64_t(input) + 37, uint64_t(output) + 19, tiled, tile));
                assert(tiled.shared_bytes == size_t(tile) * 2560);
                assert(tiled.row_blocks == unsigned((rows + tile - 1) / tile));
                assert(tiled.rows_per_block == unsigned(tile));
                assert(tiled.input_span_bytes == shape.input_span_bytes &&
                       tiled.output_span_bytes == shape.output_span_bytes);
                std::vector<int> row_coverage(size_t(rows), 0);
                for (unsigned block = 0; block < tiled.row_blocks; ++block)
                    for (int local = 0; local < tile; ++local) {
                        const unsigned row = block * unsigned(tile) + unsigned(local);
                        if (row < unsigned(rows)) ++row_coverage[row];
                    }
                for (int count : row_coverage) assert(count == 1);
            }
            ++positive_cases;
        }
    }
    assert(positive_cases == 96);

    TpDenseShape shape{};
    for (int input : {INT_MIN, -1, 0, 128, 1281, 8192, INT_MAX})
        assert(!tp_dense_shape(input, 16384, 5, 2, 1, 8192, 16384, shape));
    for (int output : {INT_MIN, -1, 0, 127, 16383, 32768, INT_MAX})
        assert(!tp_dense_shape(1280, output, 5, 2, 1, 1280, 32768, shape));
    for (int bits = -1; bits <= 9; ++bits)
        assert(tp_dense_shape(1280, 16384, bits, 2, 1, 1280, 16384, shape) == (bits == 4 || bits == 5));
    for (int codebook : {INT_MIN, -1, 0, 1, 3, INT_MAX})
        assert(!tp_dense_shape(1280, 16384, 5, codebook, 1, 1280, 16384, shape));
    for (int rows : {INT_MIN, -1, 0, 9, 256, INT_MAX})
        assert(!tp_dense_shape(1280, 16384, 5, 2, rows, 1280, 16384, shape));
    assert(!tp_dense_shape(1280, 16384, 5, 2, 8, 1279, 16384, shape));
    assert(!tp_dense_shape(1280, 16384, 5, 2, 8, 1280, 16383, shape));
    constexpr uint64_t maximum = std::numeric_limits<uint64_t>::max();
    assert(!tp_dense_shape(1280, 16384, 5, 2, 8, maximum, 16384, shape));
    assert(!tp_dense_shape(1280, 16384, 5, 2, 8, 1280, maximum, shape));
    for (int tile : {INT_MIN, -1, 4, 8, INT_MAX})
        assert(!tp_dense_shape(1280, 16384, 5, 2, 3, 1280, 16384, shape, tile));
    assert(shape.packed_bytes == 0 && shape.shared_bytes == 0 && shape.blocks == 0);
    uint64_t span = 99;
    assert(!tp_dense_span(0, 1, 1, span) && span == 0);
    assert(!tp_dense_span(1, 0, 1, span));
    assert(!tp_dense_span(2, 128, maximum / 4, span));
    assert(tp_dense_span(1, 128, maximum, span) && span == 512);
    assert(tp_dense_span(2, 1, maximum / 4 - 1, span) && span == (maximum / 4) * 4);
    assert(!tp_dense_image_limits(511, 0, 0, 65536, 65536, 20480));
    assert(!tp_dense_image_limits(512, 0, 1, 65536, 65536, 20480));
    assert(!tp_dense_image_limits(512, 0, 0, -1, 65536, 20480));
    assert(!tp_dense_image_limits(512, 0, 0, 20479, 65536, 20480));
    assert(!tp_dense_image_limits(512, 32, 0, 20480, 20511, 20480));
    assert(tp_dense_image_limits(512, 32, 0, 20480, 20512, 20480));
    assert(!tp_dense_image_limits(512, maximum, 0, 20480, 65536, 20480));

    // Prove actual device-layout coverage for every H128 input up to 5120,
    // including unequal physical-group H counts and non-H-aligned S4 cuts.
    for (int input = 128; input <= 5120; input += 128) {
        std::vector<int> coverage(size_t(input / 16), 0);
        const int iterations = tp_fused_maximum_h128(input);
        int group_iterations[2] = {};
        for (int group = 0; group < 2; ++group) for (int phase = 0; phase < 2; ++phase) {
            const int split = group + phase * 2;
            const auto partition = tp_fused_partition(input, split);
            assert(partition.begin_tile == (input / 16) * split / 4);
            assert(partition.end_tile == (input / 16) * (split + 1) / 4);
            for (int step = 0; step < iterations; ++step) {
                ++group_iterations[group];
                const int h = partition.first_h128 + step;
                if (h >= partition.end_h128) continue;
                assert(h >= 0 && h < input / 128);
                const int begin = partition.begin_tile > h * 8 ? partition.begin_tile : h * 8;
                const int end = partition.end_tile < (h + 1) * 8 ? partition.end_tile : (h + 1) * 8;
                for (int k = begin; k < end; ++k) {
                    assert(k >= 0 && k < input / 16);
                    assert(k * 16 >= h * 128 && k * 16 + 15 < (h + 1) * 128);
                    ++coverage[size_t(k)];
                }
            }
        }
        assert(group_iterations[0] == group_iterations[1]);
        for (int count : coverage) assert(count == 1);
    }
    const auto qb0 = tp_fused_partition(1280, 0);
    const auto qb1 = tp_fused_partition(1280, 1);
    assert(qb0.end_tile == 20 && qb0.end_h128 == 3);
    assert(qb1.begin_tile == 20 && qb1.first_h128 == 2);
    assert(qb1.begin_tile * 16 == 320 && qb1.first_h128 * 128 == 256);
    const auto uneven0 = tp_fused_partition(1408, 0);
    const auto uneven1 = tp_fused_partition(1408, 1);
    assert(uneven0.end_h128 - uneven0.first_h128 == 3);
    assert(uneven1.end_h128 - uneven1.first_h128 == 4);
    assert(tp_fused_maximum_h128(1408) == 4);
}
