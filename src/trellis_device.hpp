// Independent mathematical EXL3 codebook and ring-bitstream implementation.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

namespace strix_trellis {
__device__ __forceinline__ float codebook_value(uint32_t state, int cb) {
    if (cb == 2) {
        uint32_t product = state * 0x83dcd12du;
        unsigned sum = __builtin_amdgcn_udot4(product, 0x01010101u, 0, false);
        // The 16-bit representation of 1024 + sum is 0x6400 + sum.
        return __half2float(__float2half_rn(fmaf(float(1024 + sum),
            __half2float(__ushort_as_half(0x1eee)),
            __half2float(__ushort_as_half(0xc931)))));
    }
    uint32_t mixed = cb == 1 ? state * 0xcbac1fedu : state * 89226354u + 64248484u;
    uint32_t halves = (mixed & 0x8fff8fffu) ^ 0x3b603b60u;
    return __half2float(__float2half_rn(
        __half2float(__ushort_as_half(halves & 65535u)) +
        __half2float(__ushort_as_half(halves >> 16))));
}

// All lanes participate, including lanes without a source word.
// Bits in each little-endian uint32 are traversed most-significant first.
template<int Bits> struct WaveTile {
    uint32_t lo, hi;
    static constexpr int words = Bits * 8;
    __device__ explicit WaveTile(const uint32_t* data) {
        int lane = threadIdx.x & 31;
        lo = lane < words ? data[lane] : 0;
        if constexpr (words > 32) hi = lane + 32 < words ? data[lane + 32] : 0;
        else hi = 0;
    }
    __device__ __forceinline__ uint32_t word(int index) const {
        uint32_t first = __shfl(lo, index & 31, 32);
        if constexpr (words > 32) {
            uint32_t second = __shfl(hi, index & 31, 32);
            return index < 32 ? first : second;
        }
        return first;
    }
    template<int Group>
    __device__ __forceinline__ uint64_t group(int first_position) const {
        constexpr int width = 16 + (Group - 1) * Bits;
        int end = (first_position + Group) * Bits + 256 * Bits;
        int start = end - width;
        uint64_t pair = (uint64_t(word((start / 32) % words)) << 32) |
                        word(((end - 1) / 32) % words);
        return pair >> ((32 - (end & 31)) & 31);
    }
    __device__ __forceinline__ uint32_t state(int position) const {
        int end = (position + 1) * Bits + 256 * Bits;
        int start = end - 16;
        uint32_t earlier = word((start / 32) % words);
        uint32_t later = word(((end - 1) / 32) % words);
        uint64_t pair = (uint64_t(earlier) << 32) | later;
        return uint32_t(pair >> ((32 - (end & 31)) & 31)) & 65535u;
    }
};
}
