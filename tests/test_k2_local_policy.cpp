#include "../src/k2_local_policy.hpp"
#include <cassert>
#include <climits>
#include <initializer_list>
#include <cstring>
#include <limits>
using namespace strix_trellis;
int main() {
    K2LocalShape shape{};
    assert(k2_local_shape(8192, 5120, 2, 2, true, shape));
    assert(shape.packed_bytes == 10485760);
    assert(shape.partial_offset == 16384);
    assert(shape.workspace_bytes == 16384 + 5120 * 4);
    // A diagnostic N16 is valid for the core, but never for an H128 operator.
    assert(k2_local_shape(16, 16, 2, 2, false, shape));
    assert(shape.workspace_bytes == 320);
    assert(!k2_local_shape(16, 16, 2, 2, true, shape));
    assert(!shape.workspace_bytes);
    const int invalid[] = {INT_MIN, -1, 0, 1, 15, 17, 8193, INT_MAX};
    for (int input : invalid) assert(!k2_local_shape(input, 128, 2, 2, false, shape));
    for (int output : {INT_MIN, -1, 0, 1, 15, 17, 131073, INT_MAX})
        assert(!k2_local_shape(128, output, 2, 2, false, shape));
    for (int bits = 0; bits <= 9; ++bits) for (int cb = -1; cb <= 3; ++cb)
        assert(k2_local_shape(128, 128, bits, cb, true, shape) == (bits == 2 && cb == 2));
    assert(k2_local_shape(8192, 131072, 2, 2, true, shape));
    assert(shape.packed_bytes == size_t(268435456));
    assert(k2_local_schedule_name(K2LocalSchedule::U1));
    assert(k2_local_schedule_name(K2LocalSchedule::U2));
    assert(k2_local_schedule_name(K2LocalSchedule::U4));
    assert(k2_local_schedule_name(K2LocalSchedule::Pipeline2));
    assert(k2_local_schedule_name(K2LocalSchedule::Pipeline4));
    assert(!k2_local_schedule_name(static_cast<K2LocalSchedule>(-1)));
    assert(!k2_local_schedule_name(static_cast<K2LocalSchedule>(5)));
    // Pointer wrap and adjacent/subrange aliases are launch-safety properties.
    auto p = [](uintptr_t value) { return reinterpret_cast<const void*>(value); };
    assert(k2_local_span(p(256), 1024, 256));
    assert(!k2_local_span(p(258), 1024, 4));
    assert(!k2_local_span(nullptr, 1024, 4));
    assert(!k2_local_span(p(UINTPTR_MAX - 255), 256, 256));
    assert(k2_local_disjoint(p(256), 256, p(512), 256));
    assert(!k2_local_disjoint(p(256), 257, p(512), 256));
    assert(!k2_local_disjoint(p(512), 256, p(256), 257));
    assert(!k2_local_disjoint(p(UINTPTR_MAX - 15), 16, p(256), 1));
    assert(k2_local_disjoint(p(256), 256, nullptr, 0));
}
