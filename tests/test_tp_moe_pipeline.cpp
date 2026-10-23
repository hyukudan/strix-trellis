#define STRIX_TRELLIS_TP_MOE_PIPELINED_K2 1
#include "../src/tp_moe_split_layout.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
using namespace strix_trellis;

int main() {
    static_assert(tp_moe_split_gate_nodes == 3 + tp_moe_route_nodes);
    static_assert(tp_moe_split_core_nodes == 2 + tp_moe_route_nodes);
    static_assert(tp_moe_split_nodes == 6 + 2 * tp_moe_route_nodes);
    static_assert(sizeof(TpMoeDeviceRows) == 72);
    for (unsigned rows = 1; rows <= 8; ++rows) {
        // Route epilogues can reuse all transformed inputs only after B ends.
        // Their F32 vectors fit exactly without touching exported H or P.
        assert(uint64_t(rows) * 6 * 5120 * sizeof(float) ==
               tp_moe_split_activation_bytes(rows));
        assert(tp_moe_full_nodes(rows) == tp_moe_split_nodes + 2 * tp_moe_adaptive_nodes(rows) + tp_moe_finish_extra_nodes);
        assert(tp_moe_split_workspace_bytes(rows) == uint64_t(rows) * 642048 + tp_moe_split_route_bytes(rows));
#ifdef STRIX_TRELLIS_TP_MOE_PREPARE_ROUTES
        // Both full 48-owner maps must fit after all live activations, including R1.
        assert(tp_moe_split_route_bytes(rows) == 8192);
        assert(2 * 48 * sizeof(TpMoeDeviceRows) <= tp_moe_split_route_bytes(rows));
        const auto maps_begin = tp_moe_split_hidden_bytes(rows) +
            tp_moe_split_partial_bytes(rows) + tp_moe_split_activation_bytes(rows);
        assert(maps_begin % 256 == 0);
        assert(maps_begin + 2 * 48 * sizeof(TpMoeDeviceRows) <= tp_moe_split_workspace_bytes(rows));
#endif
        const size_t partials = tp_moe_split_partial_bytes(rows) / sizeof(float);
        const size_t halves = tp_moe_split_activation_bytes(rows) / 2;
        std::vector<unsigned char> gp(partials), gx(halves), dx(halves);
        for (unsigned row = 0; row < rows; ++row) for (unsigned slot = 0; slot < 6; ++slot) {
            for (unsigned projection = 0; projection < 2; ++projection) {
                const auto start = tp_moe_pipeline_activation_index(row, slot, false, projection);
                for (unsigned i = 0; i < 5120; ++i) assert(start+i < halves && ++gx[start+i] == 1);
                for (unsigned split = 0; split < 4; ++split) for (unsigned column = 0; column < 1152; ++column) {
                    const auto index = tp_moe_pipeline_gate_index(rows, projection, row, slot, split, column);
                    assert(index < partials && ++gp[index] == 1);
                }
            }
            const auto start = tp_moe_pipeline_activation_index(row, slot, true);
            for (unsigned i = 0; i < 1152; ++i) assert(start+i < halves && ++dx[start+i] == 1);
        }
        assert(std::all_of(gx.begin(),gx.end(),[](auto v){return v == 1;}));
        const size_t gate_end = size_t(rows)*2*6*4*1152;
        assert(std::all_of(gp.begin(),gp.begin()+gate_end,[](auto v){return v == 1;}));
        assert(std::all_of(gp.begin()+gate_end,gp.end(),[](auto v){return v == 0;}));
        const size_t down_end = size_t(rows)*6*1152;
        assert(std::all_of(dx.begin(),dx.begin()+down_end,[](auto v){return v == 1;}));
        assert(std::all_of(dx.begin()+down_end,dx.end(),[](auto v){return v == 0;}));
    }
    for (unsigned rows : {0u,9u,~0u}) {
        assert(tp_moe_split_workspace_bytes(rows) == 0);
        assert(tp_moe_split_activation_bytes(rows) == 0);
        assert(tp_moe_split_route_bytes(rows) == 0);
    }
    for (unsigned width : {1152u,5120u}) {
        const unsigned blocks = width/128, grid = (blocks+3)/4;
        for (unsigned rows = 1; rows <= 8; ++rows) {
            std::vector<unsigned> units(rows*blocks);
            for (unsigned block = 0; block < grid; ++block)
                for (unsigned wave = 0; wave < 4; ++wave)
                    for (unsigned unit = block*4+wave; unit < rows*blocks; unit += grid*4)
                        assert(++units[unit] == 1);
            assert(std::all_of(units.begin(),units.end(),[](auto n){return n == 1;}));
        }
        for (unsigned waves : {2u,4u,8u}) {
            std::vector<unsigned> writes(width);
            for (unsigned block = 0; block < width/(waves*16); ++block)
                for (unsigned wave = 0; wave < waves; ++wave)
                    for (unsigned lane = 0; lane < 32; lane += 4)
                        for (unsigned half = 0; half < 2; ++half) {
                            const unsigned column = (block*waves+wave)*16+lane/4+half*8;
                            assert(column < width && ++writes[column] == 1);
                        }
            assert(std::all_of(writes.begin(),writes.end(),[](auto n){return n == 1;}));
        }
    }
    std::cout << "PASS tp_moe_pipeline workspace, activation lifetimes, both gate planes, physical grid ownership\n";
}
