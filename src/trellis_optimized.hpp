// Independent mul1 microkernel experiments. No production dispatch contract.
#pragma once
#include "trellis_device.hpp"

namespace strix_trellis::optimized {
__device__ __forceinline__ uint32_t mul1_sum(uint32_t state) {
    return __builtin_amdgcn_udot4(state * 0x83dcd12du, 0x01010101u, 0, false);
}
__device__ __forceinline__ __half mul1_half(uint32_t state) {
    return __hfma(__ushort_as_half(0x6400u + mul1_sum(state)),
                  __ushort_as_half(0x1eee), __ushort_as_half(0xc931));
}
// Low half is the first state. Suitable for a subsequent packed FP16 dot.
__device__ __forceinline__ __half2 mul1_half2(uint32_t first, uint32_t second) {
    __half2 input = __halves2half2(__ushort_as_half(0x6400u + mul1_sum(first)),
                                  __ushort_as_half(0x6400u + mul1_sum(second)));
    return __hfma2(input, __halves2half2(__ushort_as_half(0x1eee), __ushort_as_half(0x1eee)),
                         __halves2half2(__ushort_as_half(0xc931), __ushort_as_half(0xc931)));
}
// A lane owns eight consecutive states. The at most three source words are
// loaded directly, rather than obtained through two shuffle banks. The extra
// requests are deliberately exposed as an alternative, not assumed free.
template<int Bits> struct DirectLane {
    uint32_t words[3];
    int start;
    __device__ explicit DirectLane(const uint32_t* data) {
        constexpr int ring_words = Bits * 8;
        start = ((threadIdx.x & 31) * 8 + 1) * Bits + 256 * Bits - 16;
        int index = start / 32;
        #pragma unroll
        for (int j = 0; j < 3; ++j) words[j] = data[(index + j) % ring_words];
    }
    template<int J> __device__ __forceinline__ uint32_t state() const {
        int offset = (start & 31) + J * Bits;
        int index = offset / 32;
        int shift = offset & 31;
        // A zero shift is well-defined; the extra word is unused in that case.
        uint64_t window = (uint64_t(words[index]) << 32) | words[index + (shift > 16)];
        return uint32_t(window >> (48 - shift)) & 65535u;
    }
};
template<int Bits, int Base, int Group>
__device__ __forceinline__ void group_states(const WaveTile<Bits>& tile, uint32_t (&out)[Group]) {
    static_assert(16 + (Group - 1) * Bits <= 64);
    uint64_t window = tile.template group<Group>((threadIdx.x & 31) * 8 + Base);
    #pragma unroll
    for (int j = 0; j < Group; ++j) out[j] = uint32_t(window >> ((Group - 1 - j) * Bits)) & 65535u;
}
// Align the lane's bit stream once. All state shifts below are compile-time
// constants, avoiding the per-state word selection of DirectLane.
template<int Bits, bool FastWrap = false> struct AlignedLane {
    static_assert(Bits >= 1 && Bits <= 8);
    uint32_t high, low, tail;
    __device__ explicit AlignedLane(const uint32_t* data) {
        constexpr int count = Bits * 8;
        int start = ((threadIdx.x & 31) * 8 + 1) * Bits + 256 * Bits - 16;
        int index = start / 32;
        int shift = start & 31; // Nonzero for integer Bits in [1,8].
        uint32_t a, b, c;
        if constexpr (FastWrap) {
            // Only the first source word can precede the ring; the last two
            // may wrap once. Bounded conditionals avoid constant division.
            index -= count;
            a = data[index < 0 ? index + count : index];
            b = data[index + 1 >= count ? index + 1 - count : index + 1];
            c = data[index + 2 >= count ? index + 2 - count : index + 2];
        } else {
            a = data[index % count]; b = data[(index + 1) % count];
            c = data[(index + 2) % count];
        }
        high = (a << shift) | (b >> (32 - shift));
        low = (b << shift) | (c >> (32 - shift));
        tail = c << shift;
    }
    template<int J> __device__ __forceinline__ uint32_t state() const {
        constexpr int offset = J * Bits;
        if constexpr (offset <= 16) return (high >> (16 - offset)) & 65535u;
        else if constexpr (offset < 32) return ((high << (offset - 16)) | (low >> (48 - offset))) & 65535u;
        else if constexpr (offset <= 48) return (low >> (48 - offset)) & 65535u;
        else return ((low << (offset - 48)) | (tail >> (80 - offset))) & 65535u;
    }
};
}
