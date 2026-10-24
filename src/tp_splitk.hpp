#pragma once
#include "tp_dense.hpp"
#include "tp_split_layout.hpp"
namespace strix_trellis {
// Explicit split plans leave the original zero-workspace 0.3 API unchanged.
struct TpSplitLaunch {
    TpDenseArguments member[2]{};
    float* partial[2]{};
    uint32_t* counters = nullptr;
    int members = 1;
    bool swiglu = false;
    bool two_kernel = false;
    float* diagnostic[2]{}; // Optional contiguous [R,N] pre-activation outputs.
    uint16_t* activation = nullptr; // Probe only: [S,R,cache_h128,128], first member.
    unsigned schedule_rotation = 0; // Probe only: permute physical CTA arrival order.
};
hipError_t query_tp_split_resources(const TpSplitLaunch&, size_t available_shared, bool* supported);
hipError_t launch_tp_split_trusted(const TpSplitLaunch&, hipStream_t);
} // namespace strix_trellis
