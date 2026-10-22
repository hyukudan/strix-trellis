#pragma once
// Experimental K2/R1 schedule. No production dispatch policy uses this header.
#include <cstddef>
#include <cstdint>
#include <limits>

namespace strix_trellis {
enum class K2LocalSchedule { U1, U2, U4, Pipeline2, Pipeline4 };
inline const char* k2_local_schedule_name(K2LocalSchedule schedule) {
    switch (schedule) {
    case K2LocalSchedule::U1: return "u1";
    case K2LocalSchedule::U2: return "u2";
    case K2LocalSchedule::U4: return "u4";
    case K2LocalSchedule::Pipeline2: return "pipeline2";
    case K2LocalSchedule::Pipeline4: return "pipeline4";
    }
    return nullptr;
}
struct K2LocalShape {
    size_t packed_bytes = 0, activation_bytes = 0, output_bytes = 0;
    size_t partial_offset = 0, workspace_bytes = 0;
};
inline bool k2_local_shape(int input, int output, int bits, int codebook,
                           bool full_operator, K2LocalShape& shape) {
    shape = {};
    if (input < 16 || input > 8192 || output < 16 || output > 131072 ||
        input % 16 || output % 16 || bits != 2 || codebook != 2 ||
        (full_operator && (input % 128 || output % 128))) return false;
    // The domain above proves every product fits even in a 32-bit size_t.
    shape.packed_bytes = size_t(input) * size_t(output) / 4;
    shape.activation_bytes = size_t(input) * 2;
    shape.output_bytes = size_t(output) * 4;
    shape.partial_offset = (shape.activation_bytes + 255) & ~size_t(255);
    shape.workspace_bytes = shape.partial_offset + shape.output_bytes;
    return true;
}
inline bool k2_local_span(const void* pointer, size_t bytes, size_t alignment) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(pointer);
    return pointer && alignment && !(base % alignment) && bytes <= UINTPTR_MAX - base;
}
inline bool k2_local_disjoint(const void* first, size_t first_bytes,
                              const void* second, size_t second_bytes) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(first);
    const uintptr_t b = reinterpret_cast<uintptr_t>(second);
    if (first_bytes > UINTPTR_MAX - a || second_bytes > UINTPTR_MAX - b) return false;
    return !first_bytes || !second_bytes || a + first_bytes <= b || b + second_bytes <= a;
}
} // namespace strix_trellis
