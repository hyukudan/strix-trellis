#pragma once
#include "backend.hpp"
#include "k2_local_policy.hpp"

namespace strix_trellis {
// Requires a gfx1151 device with wave32; qualification must check that platform.
// Opt-in experimental R1 K2/mul1 core. Four K partitions reduce in LDS and emit
// one N-element F32 plane, not [4,N]. The caller provides immutable packed
// weights and aligned, disjoint device storage of the declared logical sizes.
// No allocation, copy, synchronization or pointer metadata query is performed.
hipError_t launch_gemv_k2_local_core(const DeviceWeights&, const _Float16*, float*,
                                    K2LocalSchedule, hipStream_t);
// Includes both H128 transforms, scale vectors, optional bias and alpha/beta.
// The workspace query returns SIZE_MAX for unsupported H128 geometry.
size_t gemv_k2_local_workspace_bytes(const DeviceWeights&);
hipError_t launch_gemv_k2_local_linear(const DeviceWeights&, const float*, float*,
    void* workspace, size_t workspace_bytes, K2LocalSchedule, hipStream_t,
    float alpha = 1.0f, float beta = 0.0f);
} // namespace strix_trellis
