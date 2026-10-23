// Independently implemented, explicitly selected expert tensor-parallel path.
#pragma once
#include "backend.hpp"
#include <cstdint>
#include <type_traits>

namespace strix_trellis {
constexpr unsigned tp_moe_top_k = 6;
constexpr unsigned tp_moe_max_rows = 8;
constexpr unsigned tp_moe_max_experts = tp_moe_top_k * tp_moe_max_rows;
constexpr unsigned tp_moe_input = 5120;
constexpr unsigned tp_moe_hidden = 1152;
constexpr unsigned tp_moe_output = 5120;

// Built and copied to device memory once when an immutable plan is prepared.
// Each expert occurs once in the union; all routes selecting it share weights.
struct TpMoeExpert {
    DeviceWeights gate{}, up{}, down{};
    uint32_t row_count = 0;
    uint32_t token_rows[tp_moe_max_rows]{};
    uint32_t route_slots[tp_moe_max_rows]{};
};
struct TpMoeTable {
    TpMoeExpert experts[tp_moe_max_experts]{};
    uint32_t expert_indices[tp_moe_max_rows][tp_moe_top_k]{};
    float route_scores[tp_moe_max_rows][tp_moe_top_k]{};
};
static_assert(std::is_trivially_copyable_v<TpMoeTable>);

// Strides are in float elements. Hidden is the native row-major
// [rows, top_k, 1152] intermediate, consumed directly by the down kernel.
struct TpMoeLaunch {
    const TpMoeTable* table = nullptr;
    const float* x = nullptr;
    float* y = nullptr;
    float* hidden = nullptr;
    uint64_t x_stride = 0;
    uint64_t y_stride = 0;
    unsigned rows = 0;
    unsigned expert_count = 0;
    float alpha = 1.0f;
    float beta = 0.0f;
    // Optional persistent-bank mode: route IDs are device bank indices and
    // may change in place between properly ordered submissions/replays.
    const TpMoeExpert* bank = nullptr;
    unsigned bank_count = 0;
    const uint32_t* device_route_ids = nullptr;
    const float* device_route_scores = nullptr;
    // Additive split3 variant: H remains at hidden; P follows it.
    float* partials = nullptr;
    unsigned persistent_grid = 0;
    // Immutable plans classify at creation; device-bank plans must reclassify
    // their mutable routes on the GPU for every submission.
    unsigned gate_row_classes = 0, down_row_classes = 0;
    // Plan-owned, zero-initialized arrival counters of the folded R1 path.
    unsigned* fold_counters = nullptr;
};

// Creation-time check for the exact rows image and down image. It propagates
// HIP errors and rejects images unable to launch their 512-thread blocks.
hipError_t query_tp_moe_resources(unsigned rows, size_t device_shared_bytes,
    bool* supported);
hipError_t query_tp_moe_split_resources(unsigned rows, size_t device_shared_bytes,
    bool* supported);
hipError_t query_tp_moe_persistent_grid(unsigned rows, unsigned compute_units, unsigned* grid);
hipError_t launch_tp_moe_split_full(const TpMoeLaunch&, hipStream_t);
hipError_t launch_tp_moe_split_gate_up(const TpMoeLaunch&, hipStream_t);
hipError_t launch_tp_moe_split_core(const TpMoeLaunch&, hipStream_t);
hipError_t launch_tp_moe_split_finish(const TpMoeLaunch&, hipStream_t);
// Qualification only: original two-phase S4 tree exported before finishing.
hipError_t launch_tp_moe_reference_partials(const TpMoeLaunch&, hipStream_t);
const void* tp_moe_split_function(unsigned stage, unsigned rows);
// Distinct opt-in identity; the original and split S4 creators never use it.
hipError_t query_tp_moe_a8r_resources(unsigned rows, size_t device_shared_bytes, bool* supported);
const void* tp_moe_a8r_function(unsigned stage, unsigned rows);
hipError_t launch_tp_moe_a8r_gate_up(const TpMoeLaunch&, hipStream_t);
hipError_t launch_tp_moe_a8r_core(const TpMoeLaunch&, hipStream_t);
hipError_t launch_tp_moe_a8r_finish(const TpMoeLaunch&, hipStream_t);
hipError_t launch_tp_moe_a8r_full(const TpMoeLaunch&, hipStream_t);
// Diagnostic proxy for redundant device-owner resolution at the B grid.
hipError_t launch_tp_moe_route_probe(unsigned rows, unsigned bank_count,
    const uint32_t* ids, const float* scores, uint32_t* output, hipStream_t);
// No allocation, table upload, host route loop, synchronization or adapter copy.
hipError_t launch_tp_moe_gate_up(const TpMoeLaunch&, hipStream_t);
hipError_t launch_tp_moe_down(const TpMoeLaunch&, hipStream_t);
}
