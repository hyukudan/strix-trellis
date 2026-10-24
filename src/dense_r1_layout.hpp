#pragma once
#include "tp_dense_policy.hpp"
#include "tp_split_layout.hpp"

#ifdef __HIPCC__
#define STRIX_DENSE_R1_HD __host__ __device__
#else
#define STRIX_DENSE_R1_HD
#endif
namespace strix_trellis {
inline bool dense_r1_geometry(int input, int output) {
    return tp_dense_geometry(input, output) ||
        (input == 5120 && (output == 64640 || output == 129280));
}
// Grouped geometry (wo_a groups). Its K partition is the generic GEMV's
// single-row policy for the member's bits (K4: 32, K5: 16), so every row of a
// grouped call has the bits of that row submitted alone through strix_gemv.
inline bool dense_r1_group_geometry(int input, int output) { return input == 4096 && output == 1024; }
inline int dense_r1_group_splits(int input, int output, int bits) {
    return measured_vector_splits(input, output, bits, 2, 1);
}
inline int dense_r1_splits(int input, int output, bool legacy) {
    return legacy ? measured_vector_splits(input, output, 4, 2, 1) : 4;
}
inline size_t dense_r1_workspace(int input, int output, int rows, int tile, bool legacy) {
    if (!dense_r1_geometry(input, output) || rows < 1 || rows > 8 || tile < 0 || tile > rows) return 0;
    if (!tile) tile = rows;
    const size_t partial = size_t(dense_r1_splits(input, output, legacy)) * rows * output * 4;
    return ((partial + 255) & ~size_t(255)) +
        ((size_t((rows + tile - 1) / tile) * (output / 128) * 4 + 255) & ~size_t(255));
}
template<int Input, int Splits>
STRIX_DENSE_R1_HD constexpr int dense_r1_cache_h128() {
    int maximum=0;
    for(int s=0;s<Splits;++s) {
        const int begin=(Input/16)*s/Splits,end=(Input/16)*(s+1)/Splits;
        const int count=(end+7)/8-begin/8;
        if(count>maximum)maximum=count;
    }
    return maximum;
}
template<int Bits>
STRIX_DENSE_R1_HD constexpr uint32_t dense_r1_half_window(uint32_t high,uint32_t low,int half) {
    static_assert(Bits==4 || Bits==5);
    return half ? (high << (4*Bits)) | (low >> (32-4*Bits)) : high;
}
template<int Input,int Splits>
STRIX_DENSE_R1_HD constexpr bool dense_r1_uniform_h128() {
    for(int s=0;s<Splits;++s) {
        const int begin=(Input/16)*s/Splits,end=(Input/16)*(s+1)/Splits;
        if((end+7)/8-begin/8 != dense_r1_cache_h128<Input,Splits>())return false;
    }
    return true;
}
} // namespace strix_trellis
#undef STRIX_DENSE_R1_HD
