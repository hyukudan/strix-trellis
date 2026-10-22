#pragma once
#include "backend.hpp"
#include "tp_dense_policy.hpp"

namespace strix_trellis {
struct TpDenseArguments {
    DeviceWeights weights;
    const float* x;
    float* y;
    uint64_t x_row_stride;
    uint64_t y_row_stride;
    int rows;
    float alpha;
    float beta;
    // Zero selects all rows. Smaller tiles increase independent CTAs while
    // trading away weight reuse; this is an explicit measured-choice input.
    int rows_per_block = 0;
};

// Creation-time function-image admission. The C API also verifies gfx1151,
// wave32, allocation extents, aliases, stream ownership and object lifetimes.
// A false supported result is not a measured speed decision.
hipError_t query_tp_dense_resources(int bits, int rows_per_block, size_t available_shared,
                                    bool* supported);

// Explicit opt-in complete original-basis projection: exactly one kernel and
// zero allocations/copies/synchronizations. F32 row strides are in elements.
// The caller has validated the declared buffer spans and retains all resources.
// No external workspace is needed; default V16 dispatch is unaffected.
hipError_t launch_tp_dense_trusted(const TpDenseArguments&, hipStream_t);
} // namespace strix_trellis
