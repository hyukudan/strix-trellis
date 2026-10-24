#pragma once
#include <cstdint>
#include "tp_fused_layout.hpp"
#include "tp_moe_routes.hpp"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_MOE_SPLIT_HD __host__ __device__
#else
#define STRIX_MOE_SPLIT_HD
#endif
#ifndef STRIX_TRELLIS_TP_MOE_CORE_WAVES
#define STRIX_TRELLIS_TP_MOE_CORE_WAVES 4
#endif
#ifndef STRIX_TRELLIS_TP_MOE_GATE_TILES
#define STRIX_TRELLIS_TP_MOE_GATE_TILES 1
#endif
#ifndef STRIX_TRELLIS_TP_MOE_GROUPED_THREADS
#define STRIX_TRELLIS_TP_MOE_GROUPED_THREADS 192
#endif
namespace strix_trellis {
constexpr unsigned tp_moe_pipeline_core_waves = STRIX_TRELLIS_TP_MOE_CORE_WAVES;
constexpr unsigned tp_moe_pipeline_core_threads = 32 * tp_moe_pipeline_core_waves;
constexpr unsigned tp_moe_gate_finish_tiles = STRIX_TRELLIS_TP_MOE_GATE_TILES;
constexpr unsigned tp_moe_gate_finish_threads = 64 * tp_moe_gate_finish_tiles;
constexpr unsigned tp_moe_grouped_finish_threads = STRIX_TRELLIS_TP_MOE_GROUPED_THREADS;
static_assert(tp_moe_grouped_finish_threads == 192 || tp_moe_grouped_finish_threads == 256);
// All regions are multiples of 256 bytes. No arrival counters or clearing are
// needed: each active route/split/channel has exactly one B writer.
STRIX_MOE_SPLIT_HD constexpr uint64_t tp_moe_split_hidden_bytes(unsigned rows) {
    return rows >= 1 && rows <= 8 ? uint64_t(rows) * 27648 : 0;
}
STRIX_MOE_SPLIT_HD constexpr uint64_t tp_moe_split_partial_bytes(unsigned rows) {
    return rows >= 1 && rows <= 8 ? uint64_t(rows) * 491520 : 0;
}
STRIX_MOE_SPLIT_HD constexpr uint64_t tp_moe_split_activation_bytes(unsigned rows) {
#ifdef STRIX_TRELLIS_TP_MOE_PIPELINED_K2
    // Two independent scaled/transformed 5120-half inputs per route. Down
    // reuses the same region after gate/up has finished consuming it.
    return rows >= 1 && rows <= 8 ? uint64_t(rows) * 122880 : 0;
#else
    (void)rows;
    return 0;
#endif
}
#if defined(STRIX_TRELLIS_TP_MOE_PREPARE_ROUTES) || defined(STRIX_TRELLIS_TP_MOE_DEVICE_ROUTE_MAPS)
#define STRIX_TRELLIS_TP_MOE_ROUTE_MAPS 1
#endif
STRIX_MOE_SPLIT_HD constexpr uint64_t tp_moe_split_route_bytes(unsigned rows) {
#ifdef STRIX_TRELLIS_TP_MOE_ROUTE_MAPS
    // Two maps for each of at most 48 owner candidates; each map is 72 B.
    return rows >= 1 && rows <= 8 ? 8192 : 0;
#else
    (void)rows;
    return 0;
#endif
}
#ifdef STRIX_TRELLIS_TP_MOE_PREPARE_ROUTES
constexpr unsigned tp_moe_route_nodes = 1;
#else
constexpr unsigned tp_moe_route_nodes = 0;
#endif
// Extra node of device-routed plans: their owner maps are prepared once per
// replay ahead of the gate stage (and ahead of a staged down core).
#if defined(STRIX_TRELLIS_TP_MOE_PIPELINED_K2) && defined(STRIX_TRELLIS_TP_MOE_DEVICE_ROUTE_MAPS)
constexpr unsigned tp_moe_device_route_nodes = 1;
#else
constexpr unsigned tp_moe_device_route_nodes = 0;
#endif
#ifdef STRIX_TRELLIS_TP_MOE_PIPELINED_K2
constexpr unsigned tp_moe_split_gate_nodes = 3 + tp_moe_route_nodes;
constexpr unsigned tp_moe_split_core_nodes = 2 + tp_moe_route_nodes;
#else
constexpr unsigned tp_moe_split_gate_nodes = 1, tp_moe_split_core_nodes = 1;
#endif
#ifdef STRIX_TRELLIS_TP_MOE_PERSISTENT
constexpr unsigned tp_moe_split_nodes = 1;
#elif defined(STRIX_TRELLIS_TP_MOE_FUSE_DOWN_PREPARE)
constexpr unsigned tp_moe_split_nodes = 5 + tp_moe_route_nodes;
#else
constexpr unsigned tp_moe_split_nodes = tp_moe_split_gate_nodes + tp_moe_split_core_nodes + 1;
#endif
STRIX_MOE_SPLIT_HD constexpr unsigned tp_moe_adaptive_nodes(unsigned rows) {
#ifdef STRIX_TRELLIS_TP_MOE_ADAPTIVE_ROWS
    return rows > 2 && rows <= 8 ? 1 : 0;
#else
    (void)rows;
    return 0;
#endif
}
#ifdef STRIX_TRELLIS_TP_MOE_PARALLEL_FINISH
constexpr unsigned tp_moe_finish_extra_nodes = 1;
#else
constexpr unsigned tp_moe_finish_extra_nodes = 0;
#endif
// Maximum for mutable device routes. Immutable plans can omit empty classes.
STRIX_MOE_SPLIT_HD constexpr unsigned tp_moe_full_nodes(unsigned rows) {
#if defined(STRIX_TRELLIS_TP_MOE_PIPELINED_K2) && defined(STRIX_TRELLIS_TP_MOE_DEVICE_MERGED_CLASSES)
    // Device-routed plans run both row classes in one core per projection.
    (void)rows;
    return tp_moe_split_nodes + tp_moe_device_route_nodes + tp_moe_finish_extra_nodes;
#else
    return tp_moe_split_nodes + 2 * tp_moe_adaptive_nodes(rows) + tp_moe_device_route_nodes +
        tp_moe_finish_extra_nodes;
#endif
}
STRIX_MOE_SPLIT_HD constexpr uint64_t tp_moe_split_workspace_bytes(unsigned rows) {
    return tp_moe_split_hidden_bytes(rows) + tp_moe_split_partial_bytes(rows) +
        tp_moe_split_activation_bytes(rows) + tp_moe_split_route_bytes(rows);
}
STRIX_MOE_SPLIT_HD constexpr uint64_t tp_moe_pipeline_gate_index(unsigned rows,
    unsigned projection, unsigned row, unsigned slot, unsigned split, unsigned channel) {
    return (((uint64_t(projection) * rows * 6 + row * 6 + slot) * 4 + split) * 1152) + channel;
}
STRIX_MOE_SPLIT_HD constexpr uint64_t tp_moe_pipeline_activation_index(unsigned row,
    unsigned slot, bool down, unsigned projection = 0) {
    const uint64_t route = uint64_t(row) * 6 + slot;
    return down ? route * 1152 : (2 * route + projection) * 5120;
}
STRIX_MOE_SPLIT_HD constexpr uint64_t tp_moe_split_index(unsigned row,
    unsigned slot, unsigned split, unsigned channel) {
    return (((uint64_t(row) * 6 + slot) * 4 + split) * 5120) + channel;
}
// Compact in original token order, preserving each original top-6 slot. Called
// only by the owner; malformed tokens and signed-zero scores have no P reader.
STRIX_MOE_SPLIT_HD inline void tp_moe_split_active_rows(TpMoeDeviceRows& map,
    const uint32_t* ids, const float* scores, unsigned bank_count) {
    unsigned active = 0;
    for (unsigned i = 0; i < map.row_count; ++i) {
        const unsigned row = map.token_rows[i], slot = map.route_slots[i];
        if (!tp_moe_device_row_valid(ids + row * 6, scores + row * 6, bank_count) ||
            scores[row * 6 + slot] == 0.0f) continue;
        map.token_rows[active] = row;
        map.route_slots[active++] = slot;
    }
    map.row_count = active;
}
} // namespace strix_trellis
#undef STRIX_MOE_SPLIT_HD
