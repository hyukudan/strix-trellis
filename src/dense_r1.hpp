#pragma once
#include "tp_dense.hpp"
#include "dense_r1_layout.hpp"

namespace strix_trellis {
// One logical accumulator per S partition, regardless of physical CTA width.
// Scratch/counters belong to the caller and must be used serially on a stream.
struct DenseR1Launch {
    TpDenseArguments projection{};
    float* partial = nullptr; // [splits, rows, output]
    uint32_t* counters = nullptr; // [ceil(rows/tile), output/128], initially zero
    int splits = 4;
    int columns_per_block = 64;
    int prefetch = 2;
    int outputs_per_wave = 16; // 8 exposes both N8 halves as independent waves.
    unsigned schedule_rotation = 0; // Correctness perturbation, zero in timings.
    uint16_t* activation = nullptr; // Optional [rows,input] diagnostic, ob=0.
    bool pipelined = false; // weight batches requested ahead (same arithmetic); single plans: 64/128 cols, 16 outputs
    bool late_stage = false; // previous activation staging order (same values); single plans, see dense_r1_late_staging
};
hipError_t launch_dense_r1_trusted(const DenseR1Launch&, hipStream_t);
hipError_t query_dense_r1_resources(const DenseR1Launch&, size_t available_shared, bool* supported);
// Up to eight independent projections of one geometry/schedule in one dispatch
// (grid z = member). Each member owns its partial planes and counters and has
// a single row block (rows_per_block == rows).
inline constexpr int dense_r1_group_max = 8;
struct DenseR1Group {
    DenseR1Launch member[dense_r1_group_max]{};
};
hipError_t launch_dense_r1_group_trusted(const DenseR1Group&, int count, hipStream_t);
// Partials folded by the CTA owning each output H128 (no partial planes, no counters): wo_a S32, q_b S4, o_b S4.
hipError_t launch_dense_r1_group_owner_trusted(const DenseR1Group&, int count, hipStream_t);
hipError_t launch_dense_r1_owner_trusted(const DenseR1Launch&, hipStream_t);  // single projection (q_b, o_b K-half)
hipError_t query_dense_r1_group_owner_resources(const DenseR1Launch& first, size_t available_shared, bool* supported);
// q_a (5120 -> 1280, S16) + kv (5120 -> 512, S32) LEGACY members of one input in one dispatch.
struct DenseR1Pair {
    DenseR1Launch member[2]{};
};
hipError_t launch_dense_r1_pair_trusted(const DenseR1Pair&, hipStream_t);
hipError_t query_dense_r1_pair_resources(const DenseR1Pair&, size_t available_shared, bool* supported);
hipError_t query_dense_r1_group_resources(const DenseR1Launch& first, size_t available_shared, bool* supported);
} // namespace strix_trellis
