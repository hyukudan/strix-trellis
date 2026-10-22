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
};

// Creation-time check for the exact rows image and down image. It propagates
// HIP errors and rejects images unable to launch their 512-thread blocks.
hipError_t query_tp_moe_resources(unsigned rows, size_t device_shared_bytes,
    bool* supported);
// No allocation, table upload, host route loop, synchronization or adapter copy.
hipError_t launch_tp_moe_gate_up(const TpMoeLaunch&, hipStream_t);
hipError_t launch_tp_moe_down(const TpMoeLaunch&, hipStream_t);
}
