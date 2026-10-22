// Compile-time native ABI plus CPU counterexamples for both real geometries.
#include "../../grouped_k3_math/abi.hpp"
#include "../contract.hpp"
#include <cstdio>
#include <cstring>
using namespace grouped_k3_perf;
int main() {
    for (int i : {5120,2304}) {
        const int n=i==5120?2304:5120, s=i==5120?16:4;
        Layout l{};
        require(layout(i,n,s,1,l),"geometry layout");
        require(l.a==std::size_t(i)*2 && l.p==std::size_t(s)*n*4 &&
                l.stride==std::size_t(i==5120?157696:86528),"actual A/P SUM offsets");
        const int tiles=i/16;
        int end=0;
        for (int split=0;split<s;++split) {
            const int lo=(tiles*split/s)*16, hi=(tiles*(split+1)/s)*16;
            require(lo==end && hi>lo && lo%16==0 && hi%16==0,"exact split coverage");
            end=hi;
        }
        require(end==i,"split tail");
        for (unsigned g:selections) {
            bool seen[64]{};
            for (unsigned c=0;c<clones(g);++c) {
                grouped_k3_review::Batch b{};
                b.input=i;b.output=n;b.splits=s;b.count=g;
                for (unsigned j=0;j<g;++j) {
                    const unsigned id=start(g,c)+j;
                    require(id<64 && !seen[id],"clone full unique coverage");seen[id]=true;
                    b.groups[j].alpha=1;b.groups[j].beta=0;
                }
                auto native=grouped_k3_math::bridge(b);
                require(native.count==int(g) && native.input==i && native.splits==s,"bridge shape");
                for (unsigned j=g;j<32;++j) {
                    grouped_k3_math::NativeGroup zero{};
                    require(!std::memcmp(&native.groups[j],&zero,sizeof(zero)),"inactive bytes zero");
                }
            }
            for (bool b:seen) require(b,"missing clone");
        }
        std::printf("PASS CPU ABI2320 input=%d output=%d splits=%d stride=%zu coverage64\n",i,n,s,l.stride);
    }
    require(instances*packed_bytes==283115520 && instances*packed_bytes>=256ull*1024*1024,"resident packed math");
}
