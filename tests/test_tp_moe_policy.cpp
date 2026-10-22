#include "../src/tp_moe_policy.hpp"
#include "../src/tp_moe_routes.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>
using namespace strix_trellis;

static void exercise(unsigned rows, unsigned mode) {
    std::vector<uint32_t> routes(rows * 6), ids;
    std::vector<float> scores(rows * 6);
    for (unsigned r = 0; r < rows; ++r) for (unsigned s = 0; s < 6; ++s) {
        // Shared six, overlapping windows, and disjoint top-six selections.
        const unsigned number = mode == 0 ? (s + r) % 6 : mode == 1 ? r * 3 + s : r * 6 + s;
        const uint32_t id = 10007 + number * 97;
        routes[r * 6 + s] = id;
        scores[r * 6 + s] = float(s + 1) / 32.0f;
        if (std::find(ids.begin(), ids.end(), id) == ids.end()) ids.push_back(id);
    }
    // Bank order intentionally differs from ID order and selection order.
    std::reverse(ids.begin(), ids.end());
    TpMoeRouting routing{};
    assert(tp_moe_build_routing(ids.data(), unsigned(ids.size()), routes.data(),
        scores.data(), rows, routing));
    std::array<unsigned, 48> writes{};
    std::array<double, 48> routed{};
    // Simulate an expert-grouped producer writing directly to route slots.
    for (unsigned e = 0; e < ids.size(); ++e) {
        assert(routing.row_count[e] >= 1 && routing.row_count[e] <= rows);
        for (unsigned local = 0; local < routing.row_count[e]; ++local) {
            const unsigned token = routing.token_rows[e][local];
            const unsigned slot = routing.route_slots[e][local];
            const unsigned route = token * 6 + slot;
            assert(token < rows && slot < 6 && routes[route] == ids[e]);
            assert(routing.expert_indices[token][slot] == e);
            assert(routing.route_scores[token][slot] == scores[route]);
            assert(++writes[route] == 1);
            // This mock projection depends on BOTH token and expert identity.
            routed[route] = double(ids[e]) * (token + 3) + slot;
        }
    }
    // A token-ordered consumer must reproduce an independent route lookup.
    for (unsigned token = 0; token < rows; ++token) {
        double grouped = 0, direct = 0;
        for (unsigned slot = 0; slot < 6; ++slot) {
            const unsigned route = token * 6 + slot;
            assert(writes[route] == 1);
            const auto e = routing.expert_indices[token][slot];
            assert(ids[e] == routes[route]);
            grouped += routed[route] * routing.route_scores[token][slot];
            direct += (double(routes[route]) * (token + 3) + slot) * scores[route];
        }
        assert(grouped == direct);
    }
    if (mode == 2) assert(ids.size() == rows * 6);
    if (rows == 8 && mode == 2) assert(ids.size() == 48);
    if (mode == 0) for (unsigned e = 0; e < 6; ++e) assert(routing.row_count[e] == rows);
}

static void device_windows() {
    // The SAME owner election and row validation functions run in the HIP
    // kernels. Check they write each valid route once, even with sparse indices
    // into the largest persistent bank and changing unions between windows.
    for (unsigned rows = 1; rows <= 8; ++rows) for (unsigned mode = 0; mode < 3; ++mode) {
        uint32_t routes[48]{};
        float scores[48]{};
        for (unsigned row = 0; row < rows; ++row) for (unsigned slot = 0; slot < 6; ++slot) {
            const unsigned id = mode == 0 ? (row + slot) % 6 : mode == 1 ? row * 3 + slot : row * 6 + slot;
            routes[row * 6 + slot] = 255 - id * 5;
            scores[row * 6 + slot] = float(slot) / 32;
        }
        unsigned writes[48]{};
        unsigned owners = 0;
        for (unsigned candidate = 0; candidate < rows * 6; ++candidate) {
            TpMoeDeviceRows group{};
            if (!tp_moe_resolve_owner(routes, rows, 256, candidate, group)) continue;
            ++owners;
            assert(group.bank_index == routes[candidate]);
            assert(group.row_count >= 1 && group.row_count <= rows);
            for (unsigned earlier = 0; earlier < candidate; ++earlier)
                assert(routes[earlier] != group.bank_index);
            for (unsigned local = 0; local < group.row_count; ++local) {
                const unsigned row = group.token_rows[local];
                const unsigned slot = group.route_slots[local];
                assert(row < rows && slot < 6);
                assert(routes[row * 6 + slot] == group.bank_index);
                assert(++writes[row * 6 + slot] == 1);
            }
        }
        assert(owners == (mode == 0 ? 6 : mode == 1 ? rows * 3 + 3 : rows * 6));
        for (unsigned row = 0; row < rows; ++row) {
            assert(tp_moe_device_row_valid(routes + row * 6, scores + row * 6, 256));
            for (unsigned slot = 0; slot < 6; ++slot) assert(writes[row * 6 + slot] == 1);
        }
    }
    uint32_t ids[48]{};
    float scores[6]{};
    for (unsigned row = 0; row < 8; ++row) for (unsigned slot = 0; slot < 6; ++slot)
        ids[row * 6 + slot] = slot;
    for (unsigned slot = 0; slot < 6; ++slot) scores[slot] = float(slot) / 8;
    TpMoeDeviceRows map{};
    assert(tp_moe_resolve_owner(ids, 8, 6, 0, map) && map.row_count == 8);
    ids[1] = ids[0]; // A duplicated row must not overflow the shared map.
    assert(tp_moe_resolve_owner(ids, 8, 6, 0, map) && map.row_count == 8);
    assert(!tp_moe_device_row_valid(ids, scores, 6));
    assert(tp_moe_device_row_valid(ids + 6, scores, 6));
    ids[1] = 256;
    assert(!tp_moe_resolve_owner(ids, 8, 256, 1, map));
    assert(!tp_moe_device_row_valid(ids, scores, 256));
    ids[1] = UINT32_MAX;
    assert(!tp_moe_resolve_owner(ids, 8, 256, 1, map));
    assert(!tp_moe_device_row_valid(ids, scores, 256));
    ids[1] = 1;
    for (float bad : {-1.0f, std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        scores[0] = bad;
        assert(!tp_moe_device_row_valid(ids, scores, 6));
    }
    scores[0] = -0.0f;
    assert(tp_moe_device_row_valid(ids, scores, 6));
    for (unsigned bad : {0u, 5u, 257u, UINT32_MAX}) {
        assert(!tp_moe_resolve_owner(ids, 8, bad, 0, map));
        assert(!tp_moe_device_row_valid(ids, scores, bad));
    }
    assert(!tp_moe_resolve_owner(ids, 0, 6, 0, map));
    assert(!tp_moe_resolve_owner(ids, 9, 6, 0, map));
    assert(!tp_moe_resolve_owner(ids, 8, 6, 48, map));
    assert(!tp_moe_resolve_owner(nullptr, 8, 6, 0, map));
    assert(!tp_moe_device_row_valid(nullptr, scores, 6));
    assert(!tp_moe_device_row_valid(ids, nullptr, 6));
}

int main() {
    for (unsigned rows = 1; rows <= 8; ++rows)
        for (unsigned mode = 0; mode < 3; ++mode) exercise(rows, mode);
    device_windows();
    uint32_t ids[7] = {7, 100, 9001, 4, 60000, 0, 43};
    uint32_t routes[12] = {7, 100, 9001, 4, 60000, 0, 7, 100, 9001, 4, 60000, 0};
    float scores[12] = {0, .1f, .2f, .3f, .4f, .5f, 0, .1f, .2f, .3f, .4f, .5f};
    TpMoeRouting saved{};
    assert(tp_moe_build_routing(ids, 6, routes, scores, 2, saved));
    auto rejected = [&](const uint32_t* bank, unsigned count, const uint32_t* route,
                        const float* score, unsigned rows) {
        TpMoeRouting output = saved;
        assert(!tp_moe_build_routing(bank, count, route, score, rows, output));
        assert(std::memcmp(&saved, &output, sizeof(saved)) == 0);
    };
    rejected(nullptr, 6, routes, scores, 2);
    rejected(ids, 6, nullptr, scores, 2);
    rejected(ids, 6, routes, nullptr, 2);
    for (unsigned rows : {0u, 9u, std::numeric_limits<unsigned>::max()})
        rejected(ids, 6, routes, scores, rows);
    for (unsigned count : {0u, 5u, 49u, std::numeric_limits<unsigned>::max()})
        rejected(ids, count, routes, scores, 2);
    rejected(ids, 7, routes, scores, 1); // More entries than possible routes.
    rejected(ids, 7, routes, scores, 2); // Inactive bank entry violates exact union.
    const uint32_t old_id = ids[5]; ids[5] = ids[0];
    rejected(ids, 6, routes, scores, 2); ids[5] = old_id;
    const uint32_t old_route = routes[5]; routes[5] = routes[0];
    rejected(ids, 6, routes, scores, 2); routes[5] = 12345;
    rejected(ids, 6, routes, scores, 2); routes[5] = old_route;
    for (float bad : {-1.0f, std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        const float old = scores[7]; scores[7] = bad;
        rejected(ids, 6, routes, scores, 2); scores[7] = old;
    }
    for (int bits = -1; bits <= 8; ++bits) for (int codebook = -1; codebook <= 3; ++codebook) {
        const bool supported = (bits == 2 || bits == 3) && codebook == 2;
        assert(tp_moe_weight_shape(5120, 1152, bits, codebook, false) == supported);
        assert(tp_moe_weight_shape(1152, 5120, bits, codebook, true) == supported);
        assert(!tp_moe_weight_shape(5120, 2304, bits, codebook, false));
        assert(!tp_moe_weight_shape(2304, 5120, bits, codebook, true));
        assert(!tp_moe_weight_shape(1152, 5120, bits, codebook, false));
    }
    for (unsigned rows = 0; rows <= 9; ++rows)
        assert(tp_moe_hidden_bytes(rows) ==
            (rows >= 1 && rows <= 8 ? uint64_t(rows) * 6 * 1152 * 4 : 0));
    assert(!tp_moe_hidden_bytes(std::numeric_limits<unsigned>::max()));
    std::cout << "PASS tp_moe_policy 24 immutable and 24 device windows, union48/bank256, exact routing, negatives and shapes\n";
}
