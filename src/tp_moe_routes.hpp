#pragma once
// Bounded device routing helpers, also executed directly by CPU policy tests.
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_MOE_HOST_DEVICE __host__ __device__
#else
#define STRIX_MOE_HOST_DEVICE
#endif

namespace strix_trellis {
// Trivial shared routing metadata; weights remain in the persistent bank.
// No constructor or per-thread array initialization is needed on the GPU.
struct TpMoeDeviceRows {
    uint32_t bank_index;
    uint32_t row_count;
    uint32_t token_rows[8];
    uint32_t route_slots[8];
};

// Each candidate is one flattened (token, top-6 slot). Exactly its first
// occurrence owns an expert's grouped projection. Invalid IDs own no work.
// On malformed duplicate selection within a token, map only the first slot;
// the down validator marks that token invalid before consuming hidden values.
STRIX_MOE_HOST_DEVICE inline bool tp_moe_resolve_owner(const uint32_t* ids,
    unsigned rows, unsigned bank_count, unsigned candidate, TpMoeDeviceRows& out) {
    out = {};
    if (!ids || rows < 1 || rows > 8 || bank_count < 6 || bank_count > 256 ||
        candidate >= rows * 6) return false;
    const uint32_t expert = ids[candidate];
    if (expert >= bank_count) return false;
    for (unsigned previous = 0; previous < candidate; ++previous)
        if (ids[previous] == expert) return false;
    out.bank_index = expert;
    for (unsigned row = 0; row < rows; ++row) {
        for (unsigned slot = 0; slot < 6; ++slot) {
            if (ids[row * 6 + slot] != expert) continue;
            const unsigned local = out.row_count++;
            out.token_rows[local] = row;
            out.route_slots[local] = slot;
            break;
        }
    }
    return true;
}

STRIX_MOE_HOST_DEVICE constexpr bool tp_moe_device_score_valid(float score) {
    // Both comparisons reject NaN; the upper bound rejects positive infinity.
    // Negative zero is a valid zero score. No fast-math assumptions are used.
    return score >= 0.0f && score <= 0x1.fffffep+127f;
}
STRIX_MOE_HOST_DEVICE inline bool tp_moe_device_row_valid(const uint32_t* ids,
    const float* scores, unsigned bank_count) {
    if (!ids || !scores || bank_count < 6 || bank_count > 256) return false;
    for (unsigned slot = 0; slot < 6; ++slot) {
        if (ids[slot] >= bank_count || !tp_moe_device_score_valid(scores[slot])) return false;
        for (unsigned previous = 0; previous < slot; ++previous)
            if (ids[slot] == ids[previous]) return false;
    }
    return true;
}
} // namespace strix_trellis
#undef STRIX_MOE_HOST_DEVICE
