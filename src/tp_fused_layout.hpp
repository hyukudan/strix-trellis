#pragma once

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_TP_HOST_DEVICE __host__ __device__
#else
#define STRIX_TP_HOST_DEVICE
#endif

namespace strix_trellis {
struct TpFusedPartition {
    int begin_tile;
    int end_tile;
    int first_h128;
    int end_h128;
};

// Supported callers bound input to H128 multiples <=5120 and split to [0,4).
// These helpers are shared by the actual GPU schedule and CPU coverage proof.
STRIX_TP_HOST_DEVICE constexpr TpFusedPartition tp_fused_partition(int input, int split) {
    const int tiles = input / 16;
    const int begin = tiles * split / 4;
    const int end = tiles * (split + 1) / 4;
    return {begin, end, begin / 8, (end + 7) / 8};
}

STRIX_TP_HOST_DEVICE constexpr int tp_fused_maximum_h128(int input) {
    int maximum = 0;
    for (int split = 0; split < 4; ++split) {
        const auto partition = tp_fused_partition(input, split);
        const int count = partition.end_h128 - partition.first_h128;
        if (count > maximum) maximum = count;
    }
    return maximum;
}
} // namespace strix_trellis

#undef STRIX_TP_HOST_DEVICE
