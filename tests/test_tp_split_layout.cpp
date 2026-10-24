#include "../src/tp_split_layout.hpp"
#include <algorithm>
#include <cassert>
#include <climits>
#include <numeric>
#include <random>
#include <vector>
using namespace strix_trellis;
int main() {
    std::mt19937 rng(20261021);
    for (const auto& shape : {std::pair<int,int>{5120,1280}, {5120,512}, {5120,1152},
                            {1280,16384}, {4096,5120}, {1152,5120}}) {
        const int input = shape.first, output = shape.second;
        const int s = tp_split_count(input, output);
        assert(s == measured_vector_splits(input, output, 4, 2, 1));
        for (int bits : {4,5}) for (int r = 1; r <= 8; ++r)
            assert(s == measured_vector_splits(input, output, bits, 2, r));
        std::vector<int> tiles(size_t(input / 16));
        for (int p = 0; p < s; ++p) {
            const auto part = tp_split_partition(input,p,s);
            assert(part.first_h128 >= 0 && part.end_h128 <= input / 128);
            assert(part.end_h128-part.first_h128 == ((input/16)/s+7)/8);
            for (int k = part.begin_tile; k < part.end_tile; ++k) {
                ++tiles[k];
                assert(k >= part.first_h128 * 8 && k < part.end_h128 * 8);
            }
            if (input == 5120) assert(part.end_h128 - part.first_h128 <= 3);
        }
        for (int count : tiles) assert(count == 1);
        if (!tp_split_geometry(input,output)) continue;
        for (int rows = 1; rows <= 8; ++rows) for (int tile = 1; tile <= rows; ++tile) {
            TpSplitLayout l{};
            assert(tp_split_layout(input,output,rows,tile,l));
            assert(l.partial_bytes == size_t(s) * rows * output * 4);
            assert(!(l.bytes % 256) && !(l.counter_offset % 256));
            assert(l.counters == unsigned((rows + tile - 1) / tile * (output / 128)));
            std::vector<unsigned char> writers(l.partial_bytes / 4);
            std::vector<int> counters(l.counters);
            std::vector<int> arrivals(static_cast<size_t>(s)); std::iota(arrivals.begin(),arrivals.end(),0);
            for (int repeat = 0; repeat < 20; ++repeat) {
                std::shuffle(arrivals.begin(),arrivals.end(),rng);
                int completed = 0;
                for (unsigned rt = 0; rt < unsigned((rows+tile-1)/tile); ++rt)
                    for (int ob = 0; ob < output / 128; ++ob) {
                        const auto ci = rt * (output / 128) + ob;
                        for (int split : arrivals) {
                            for (int row = int(rt)*tile; row < std::min(rows,int(rt+1)*tile); ++row)
                                for (int col = ob*128; col < (ob+1)*128; ++col) {
                                    const auto index = (size_t(split)*rows+row)*output+col;
                                    assert(index < writers.size());
                                    if (!repeat) ++writers[index];
                                }
                            if (++counters[ci] == s) { ++completed; counters[ci] = 0; }
                        }
                    }
                assert(completed == int(l.counters));
                for (int c : counters) assert(c == 0);
            }
            for (int c : writers) assert(c == 1);
        }
    }
    TpSplitLayout l{};
    for (int r : {INT_MIN,-1,0,9,INT_MAX}) assert(!tp_split_layout(5120,1280,r,0,l));
    for (int t : {INT_MIN,-1,9,INT_MAX}) assert(!tp_split_layout(5120,1280,8,t,l));
    assert(tp_split_layout(1280,16384,1,0,l) && l.partial_bytes==262144);
    assert(tp_split_layout(4096,5120,8,0,l) && l.partial_bytes==655360);
    assert(!tp_split_layout(1152,5120,1,0,l));
    assert(tp_split_layout(5120,1280,8,8,l) && l.bytes == 655616);
    assert(tp_split_layout(5120,512,8,8,l) && l.bytes == 524544);
    assert(tp_split_layout(5120,1152,8,8,l) && l.bytes == 590080);
}
