#pragma once
#include "tp_fused_layout.hpp"
#include "gemv_policy.hpp"
#include <cstddef>
#include <cstdint>

#ifdef __HIPCC__
#define STRIX_SPLIT_HD __host__ __device__
#else
#define STRIX_SPLIT_HD
#endif
namespace strix_trellis {
// Legacy split count is part of the numerical identity, independent of tiling.
inline int tp_split_count(int input, int output) {
    return measured_vector_splits(input, output, 4, 2, 1);
}
STRIX_SPLIT_HD constexpr TpFusedPartition tp_split_partition(int input, int split, int count) {
    const int begin = (input / 16) * split / count;
    const int end = (input / 16) * (split + 1) / count;
    return {begin, end, begin / 8, (end + 7) / 8};
}
inline bool tp_split_largek(int input, int output) {
    return input == 5120 && (output == 1280 || output == 512 || output == 1152);
}
inline bool tp_split_geometry(int input, int output) {
    return tp_split_largek(input,output) || (input==1280 && output==16384) ||
        (input==4096 && output==5120);
}
STRIX_SPLIT_HD constexpr int tp_split_cache_h128(int input) { return input==4096 ? 8 : 3; }
STRIX_SPLIT_HD constexpr int tp_split_device_count(int input,int output) {
    return input==5120 ? (output==512 ? 32 : 16) : 4;
}
struct TpSplitLayout {
    size_t partial_bytes = 0, counter_offset = 0, counter_bytes = 0, bytes = 0;
    unsigned counters = 0;
};
// Bounded shapes only: reject before multiplying. Counters are per row tile,
// not per row; P always uses global rows and is independent of row tiling.
inline bool tp_split_layout(int input, int output, int rows, int tile, TpSplitLayout& l) {
    l = {};
    if (!tp_split_geometry(input, output) || rows < 1 || rows > 8 || tile < 0 || tile > rows) return false;
    if (!tile) tile = rows;
    l.partial_bytes = size_t(tp_split_count(input, output)) * rows * output * sizeof(float);
    l.counter_offset = (l.partial_bytes + 255) & ~size_t(255);
    l.counters = unsigned((rows + tile - 1) / tile * (output / 128));
    l.counter_bytes = (size_t(l.counters) * sizeof(uint32_t) + 255) & ~size_t(255);
    l.bytes = l.counter_offset + l.counter_bytes;
    return true;
}
} // namespace strix_trellis
#undef STRIX_SPLIT_HD
