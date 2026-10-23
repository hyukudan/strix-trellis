// Independent expert-TP SwiGLU: grouped gate/up and deterministic top-6 down.
#include "tp_moe.hpp"
#include "tp_moe_activation.hpp"
#include "tp_moe_policy.hpp"
#include "tp_moe_routes.hpp"
#include "tp_moe_split_layout.hpp"
#include "tp_fused_device.hpp"
#include <cmath>

namespace strix_trellis::tp_moe_detail {
struct RowMap {
    unsigned row_count;
    const uint32_t* token_rows;
    const uint32_t* route_slots;
};
struct GateAccess {
    RowMap mapping;
    const TpMoeLaunch& launch;
    float* gate;
    __device__ __forceinline__ float load_input(int row, int column) const {
        return unsigned(row) < mapping.row_count
            ? launch.x[uint64_t(mapping.token_rows[row]) * launch.x_stride + column]
            : 0.0f;
    }
    __device__ __forceinline__ void store_output(int row, int column, float value) const {
        if (unsigned(row) < mapping.row_count) gate[row * 128 + (column & 127)] = value;
    }
};
struct UpAccess {
    RowMap mapping;
    const TpMoeLaunch& launch;
    const float* gate;
    __device__ __forceinline__ float load_input(int row, int column) const {
        return unsigned(row) < mapping.row_count
            ? launch.x[uint64_t(mapping.token_rows[row]) * launch.x_stride + column]
            : 0.0f;
    }
    __device__ __forceinline__ void store_output(int row, int column, float value) const {
        if (unsigned(row) >= mapping.row_count) return;
        const float g = gate[row * 128 + (column & 127)];
        const uint64_t route = uint64_t(mapping.token_rows[row]) * tp_moe_top_k +
            mapping.route_slots[row];
        launch.hidden[route * tp_moe_hidden + column] = tp_moe_swiglu(g, value);
    }
};

template<int Rows>
__device__ __forceinline__ void gate_up_rows(const TpMoeExpert& e,
    RowMap mapping, const TpMoeLaunch& launch, float* gate, float* partial, _Float16* cache) {
    GateAccess g{mapping, launch, gate};
    if (e.gate.bits == 2) tp_fused::project<2, Rows>(e.gate, g, blockIdx.x, partial, cache);
    else tp_fused::project<3, Rows>(e.gate, g, blockIdx.x, partial, cache);
    // project ends with a CTA barrier, publishing every gate value before up.
    UpAccess u{mapping, launch, gate};
    if (e.up.bits == 2) tp_fused::project<2, Rows>(e.up, u, blockIdx.x, partial, cache);
    else tp_fused::project<3, Rows>(e.up, u, blockIdx.x, partial, cache);
}

template<int Capacity>
__global__ __launch_bounds__(512) void gate_up_kernel(TpMoeLaunch launch) {
    __shared__ float gate[Capacity * 128];
    __shared__ float partial[4 * Capacity * 128];
    __shared__ _Float16 cache[2 * Capacity * 128];
    __shared__ TpMoeDeviceRows routed;
    __shared__ bool owner;
    const TpMoeExpert* expert;
    RowMap mapping;
    if (launch.bank) {
        // Only one thread scans the bounded 48-route window and builds the
        // shared row map. The bank itself is persistent device memory.
        if (threadIdx.x == 0)
            owner = tp_moe_resolve_owner(launch.device_route_ids, launch.rows,
                launch.bank_count, blockIdx.y, routed);
        __syncthreads();
        if (!owner) return; // Uniform CTA exit, before any projection barrier.
        expert = launch.bank + routed.bank_index;
        mapping = {routed.row_count, routed.token_rows, routed.route_slots};
    } else {
        expert = launch.table->experts + blockIdx.y;
        mapping = {expert->row_count, expert->token_rows, expert->route_slots};
    }
    // Most window unions contain experts selected by only one token. Select
    // capacity per expert in the kernel, retaining a single grouped dispatch.
    if (mapping.row_count == 1) gate_up_rows<1>(*expert, mapping, launch, gate, partial, cache);
    else if constexpr (Capacity >= 2) {
        if (mapping.row_count <= 2) gate_up_rows<2>(*expert, mapping, launch, gate, partial, cache);
        else if constexpr (Capacity >= 4) {
            if (mapping.row_count <= 4) gate_up_rows<4>(*expert, mapping, launch, gate, partial, cache);
            else if constexpr (Capacity >= 8)
                gate_up_rows<8>(*expert, mapping, launch, gate, partial, cache);
        }
    }
}

struct DownAccess {
    const TpMoeLaunch& launch;
    const float* input;
    float* combined;
    float score;
    __device__ __forceinline__ float load_input(int, int column) const {
        return input[column];
    }
    __device__ __forceinline__ void store_output(int, int column, float value) const {
        const unsigned local = unsigned(column) & 127;
        // The route loop is ascending slot order, starting from +0. No atomic
        // order or inter-block communication enters the numerical contract.
        const float weighted = materialize_fp32(score * value);
        combined[local] = materialize_fp32(combined[local] + weighted);
    }
};

__global__ __launch_bounds__(512) void down_kernel(TpMoeLaunch launch) {
    __shared__ float partial[4 * 128];
    __shared__ _Float16 cache[2 * 128];
    __shared__ float combined[128];
    __shared__ uint32_t selected[tp_moe_top_k];
    __shared__ float scores[tp_moe_top_k];
    __shared__ bool valid;
    const unsigned token = blockIdx.y;
    if (threadIdx.x == 0) {
        for (unsigned slot = 0; slot < tp_moe_top_k; ++slot) {
            selected[slot] = launch.bank ? launch.device_route_ids[token * tp_moe_top_k + slot]
                : launch.table->expert_indices[token][slot];
            scores[slot] = launch.bank ? launch.device_route_scores[token * tp_moe_top_k + slot]
                : launch.table->route_scores[token][slot];
        }
        valid = !launch.bank || tp_moe_device_row_valid(selected, scores, launch.bank_count);
    }
    if (threadIdx.x < 128) combined[threadIdx.x] = 0.0f;
    __syncthreads();
    if (!valid) {
        if (threadIdx.x < 128) {
            const uint64_t output = uint64_t(token) * launch.y_stride +
                uint64_t(blockIdx.x) * 128 + threadIdx.x;
            launch.y[output] = __builtin_bit_cast(float, uint32_t(0x7fc00000u));
        }
        return; // All threads exit; no invalid ID reaches the expert bank.
    }
    for (unsigned slot = 0; slot < tp_moe_top_k; ++slot) {
        const float score = scores[slot];
        // A zero-score route contributes +0 without reading its intermediate
        // or weights. The sampled score is uniform across this entire CTA.
        if (score == 0.0f) continue;
        const unsigned expert_index = selected[slot];
        const auto& weights = launch.bank ? launch.bank[expert_index].down
            : launch.table->experts[expert_index].down;
        const float* input = launch.hidden +
            (uint64_t(token) * tp_moe_top_k + slot) * tp_moe_hidden;
        DownAccess access{launch, input, combined, score};
        if (weights.bits == 2)
            tp_fused::project<2, 1>(weights, access, blockIdx.x, partial, cache);
        else tp_fused::project<3, 1>(weights, access, blockIdx.x, partial, cache);
    }
    // Every project ends with a barrier. Each active thread writes one unique
    // output element, including the optional old-Y term exactly once.
    if (threadIdx.x < 128) {
        const uint64_t output = uint64_t(token) * launch.y_stride +
            uint64_t(blockIdx.x) * 128 + threadIdx.x;
        float value = materialize_fp32(launch.alpha * combined[threadIdx.x]);
        if (launch.beta != 0.0f)
            value = materialize_fp32(value + materialize_fp32(launch.beta * launch.y[output]));
        launch.y[output] = value;
    }
}

struct ReferenceAccess {
    const float* input;
    __device__ __forceinline__ float load_input(int, int column) const { return input[column]; }
    __device__ __forceinline__ void store_output(int, int, float) const {}
};
__global__ __launch_bounds__(512) void reference_partials_kernel(TpMoeLaunch l) {
    __shared__ float p[4 * 128];
    __shared__ _Float16 cache[2 * 128];
    const unsigned row = blockIdx.y / 6, slot = blockIdx.y % 6;
    const auto* ids = l.bank ? l.device_route_ids + row * 6 : l.table->expert_indices[row];
    const auto* scores = l.bank ? l.device_route_scores + row * 6 : l.table->route_scores[row];
    if (l.bank && !tp_moe_device_row_valid(ids, scores, l.bank_count)) return;
    if (scores[slot] == 0.0f) return;
    const auto& w = l.bank ? l.bank[ids[slot]].down : l.table->experts[ids[slot]].down;
    ReferenceAccess a{l.hidden + (uint64_t(row) * 6 + slot) * 1152};
    if (w.bits == 2) tp_fused::project<2,1>(w, a, blockIdx.x, p, cache);
    else tp_fused::project<3,1>(w, a, blockIdx.x, p, cache);
    const unsigned split = threadIdx.x / 128, column = threadIdx.x % 128;
    l.partials[tp_moe_split_index(row,slot,split,blockIdx.x*128+column)] = p[split*128+column];
}

const void* gate_function(unsigned rows) {
    if (rows == 1) return reinterpret_cast<const void*>(gate_up_kernel<1>);
    if (rows == 2) return reinterpret_cast<const void*>(gate_up_kernel<2>);
    if (rows <= 4) return reinterpret_cast<const void*>(gate_up_kernel<4>);
    return reinterpret_cast<const void*>(gate_up_kernel<8>);
}
bool launch_valid(const TpMoeLaunch& l) {
    const bool routing = l.bank
        ? !l.table && l.bank_count >= 6 && l.bank_count <= 256 &&
            l.device_route_ids && l.device_route_scores
        : l.table && !l.bank_count && !l.device_route_ids && !l.device_route_scores &&
            l.expert_count >= 6 && l.expert_count <= 6 * l.rows;
    return routing && l.x && l.y && l.hidden && l.rows >= 1 && l.rows <= 8 &&
        l.x_stride >= tp_moe_input && l.y_stride >= tp_moe_output &&
        std::isfinite(l.alpha) && std::isfinite(l.beta);
}
}

namespace strix_trellis {
hipError_t launch_tp_moe_reference_partials(const TpMoeLaunch& l, hipStream_t stream) {
    if (!l.partials || !tp_moe_detail::launch_valid(l)) return hipErrorInvalidValue;
    tp_moe_detail::reference_partials_kernel<<<dim3(40,l.rows*6),512,0,stream>>>(l);
    return hipGetLastError();
}
hipError_t query_tp_moe_resources(unsigned rows, size_t available, bool* supported) {
    if (!supported || rows < 1 || rows > 8) return hipErrorInvalidValue;
    *supported = false;
    const void* functions[]{tp_moe_detail::gate_function(rows),
        reinterpret_cast<const void*>(tp_moe_detail::down_kernel)};
    for (const void* function : functions) {
        hipFuncAttributes a{};
        const hipError_t status = hipFuncGetAttributes(&a, function);
        if (status != hipSuccess) return status;
        if (a.maxThreadsPerBlock < 512 || a.sharedSizeBytes > available) return hipSuccess;
    }
    // Scratch spills are a measurement concern, not a correctness admission
    // condition. ISA/resource inspection remains part of native qualification.
    *supported = true;
    return hipSuccess;
}
hipError_t launch_tp_moe_gate_up(const TpMoeLaunch& l, hipStream_t stream) {
    if (!tp_moe_detail::launch_valid(l)) return hipErrorInvalidValue;
    const dim3 grid(tp_moe_hidden / 128, l.bank ? l.rows * tp_moe_top_k : l.expert_count);
    if (l.rows == 1) tp_moe_detail::gate_up_kernel<1><<<grid, 512, 0, stream>>>(l);
    else if (l.rows == 2) tp_moe_detail::gate_up_kernel<2><<<grid, 512, 0, stream>>>(l);
    else if (l.rows <= 4) tp_moe_detail::gate_up_kernel<4><<<grid, 512, 0, stream>>>(l);
    else tp_moe_detail::gate_up_kernel<8><<<grid, 512, 0, stream>>>(l);
    return hipGetLastError();
}
hipError_t launch_tp_moe_down(const TpMoeLaunch& l, hipStream_t stream) {
    if (!tp_moe_detail::launch_valid(l)) return hipErrorInvalidValue;
    tp_moe_detail::down_kernel<<<dim3(tp_moe_output / 128, l.rows), 512, 0, stream>>>(l);
    return hipGetLastError();
}
}
