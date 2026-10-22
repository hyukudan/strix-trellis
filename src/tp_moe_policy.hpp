// Host-only contract for an immutable top-6 expert union window.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace strix_trellis {
struct TpMoeRouting {
    uint32_t row_count[48]{};
    uint32_t token_rows[48][8]{};
    uint32_t route_slots[48][8]{};
    uint32_t expert_indices[8][6]{};
    float route_scores[8][6]{};
};

// Every bank entry must occur in the window, IDs are unique in the bank and
// within each token's top-6 selection. Different tokens may share any experts.
// Scores are caller-normalized nonnegative finite values; no renormalization
// is silently introduced. Failure leaves output untouched.
inline bool tp_moe_build_routing(const uint32_t* expert_ids, unsigned count,
    const uint32_t* route_ids, const float* route_scores, unsigned rows,
    TpMoeRouting& output) {
    if (!expert_ids || !route_ids || !route_scores || rows < 1 || rows > 8 ||
        count < 6 || count > 48 || count > rows * 6) return false;
    for (unsigned e = 0; e < count; ++e)
        for (unsigned p = 0; p < e; ++p)
            if (expert_ids[e] == expert_ids[p]) return false;
    TpMoeRouting candidate{};
    for (unsigned row = 0; row < rows; ++row) {
        for (unsigned slot = 0; slot < 6; ++slot) {
            const unsigned route = row * 6 + slot;
            const uint32_t id = route_ids[route];
            const float score = route_scores[route];
            if (!std::isfinite(score) || score < 0.0f) return false;
            for (unsigned previous = 0; previous < slot; ++previous)
                if (route_ids[row * 6 + previous] == id) return false;
            unsigned expert = 0;
            while (expert < count && expert_ids[expert] != id) ++expert;
            if (expert == count) return false;
            const unsigned local_row = candidate.row_count[expert]++;
            if (local_row >= 8) return false;
            candidate.token_rows[expert][local_row] = row;
            candidate.route_slots[expert][local_row] = slot;
            candidate.expert_indices[row][slot] = expert;
            candidate.route_scores[row][slot] = score;
        }
    }
    for (unsigned expert = 0; expert < count; ++expert)
        if (candidate.row_count[expert] == 0) return false;
    output = candidate;
    return true;
}

inline bool tp_moe_weight_shape(int input, int output, int bits,
    int codebook, bool down) {
    return (bits == 2 || bits == 3) && codebook == 2 &&
        input == (down ? 1152 : 5120) && output == (down ? 5120 : 1152);
}
inline uint64_t tp_moe_hidden_bytes(unsigned rows) {
    return rows >= 1 && rows <= 8 ? uint64_t(rows) * 6 * 1152 * sizeof(float) : 0;
}
}
