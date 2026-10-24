#include "../src/dense_r1_layout.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
using namespace strix_trellis;

template<int Bits> void windows() {
    uint32_t seed=0x359bd417u;
    for(int trial=0;trial<65536;++trial) {
        seed=seed*1664525u+1013904223u;const uint32_t high=seed;
        seed=seed*1664525u+1013904223u;const uint32_t low=seed;
        // Treat the reader's two words as one big-endian bit window. Its
        // original eight states are independent of the optimized half reader.
        const uint64_t stream=(uint64_t(high)<<32)|low;
        for(int half=0;half<2;++half)for(int j=0;j<4;++j) {
            const auto actual=(dense_r1_half_window<Bits>(high,low,half)>>(16-j*Bits))&65535u;
            const auto expected=uint32_t(stream>>(48-(half*4+j)*Bits))&65535u;
            assert(actual==expected);
        }
    }
}
int main() {
    windows<4>();windows<5>();
    static_assert(dense_r1_cache_h128<5120,4>()==10);
    static_assert(dense_r1_cache_h128<5120,16>()==3);
    static_assert(dense_r1_cache_h128<5120,32>()==2);
    static_assert(dense_r1_cache_h128<1280,4>()==3);
    static_assert(dense_r1_cache_h128<4096,4>()==8);
    static_assert(dense_r1_cache_h128<1152,4>()==3);
    static_assert(dense_r1_uniform_h128<5120,4>() && dense_r1_uniform_h128<5120,16>() && dense_r1_uniform_h128<5120,32>());
    static_assert(dense_r1_uniform_h128<1280,4>() && dense_r1_uniform_h128<4096,4>() && dense_r1_uniform_h128<1152,4>());
    const int outputs[]={512,1152,1280,5120,16384,64640,129280};
    for(int output:outputs)for(int columns:{8,16,32,64,128})for(int per_wave:{8,16})for(int splits:{4,16,32}) {
        if(columns==8 && per_wave==16)continue;
        std::vector<unsigned char> written(size_t(splits)*output,0);
        std::vector<unsigned> arrivals(output/128,0);
        for(int physical=0;physical<output/columns*splits;++physical) {
            const int nt=physical%(output/columns),s=physical/(output/columns);
            ++arrivals[nt/(128/columns)];
            const int waves=columns/per_wave;
            for(int wave=0;wave<waves;++wave)for(int half=0;half<(per_wave==8?1:2);++half)for(int lane=0;lane<32;lane+=4) {
                const int tile=per_wave==8?(nt*waves+wave)/2:nt*waves+wave;
                const int col=tile*16+lane/4+(per_wave==8?(nt*waves+wave)&1:half)*8;
                assert(col>=0 && col<output && !written[size_t(s)*output+col]++);
            }
        }
        for(auto count:written)assert(count==1);
        for(auto count:arrivals)assert(count==unsigned(splits*128/columns));
    }
    for(int rows=1;rows<=8;++rows)for(int tile=0;tile<=rows;++tile)for(bool legacy:{false,true}) {
        const auto bytes=dense_r1_workspace(5120,512,rows,tile,legacy);
        assert(bytes>size_t((legacy?32:4)*rows*512*4) && bytes%256==0);
    }
    assert(!dense_r1_workspace(5120,512,9,0,false));
    assert(!dense_r1_workspace(5120,512,6,7,false));
    assert(!dense_r1_workspace(5120,513,6,0,false));
    std::cout<<"PASS dense_r1_layout: two 65536-window sweeps, physical ownership and arrival counts\n";
}
