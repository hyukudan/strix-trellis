// CPU executable compiled with HIP's host frontend, linked without HIP runtime.
#include "abi.hpp"
#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept>
using namespace grouped_k3_review;
static void check(bool b) { if(!b) throw std::runtime_error("CPU guard/layout failure"); }
int main() {
    Environment e{41,41,0,0,true};
    for(auto shape : {std::array<int,3>{5120,2304,16},{2304,5120,4}}) {
        std::vector<Descriptor> d(32);
        std::vector<std::unique_ptr<unsigned char[]>> storage;
        for(auto& v:d) {
            v.input=shape[0];v.output=shape[1];v.splits=shape[2];
            uint64_t sizes[8]={uint64_t(v.input)*v.output*3/8,uint64_t(v.input)*4,
                uint64_t(v.input)*2,uint64_t(v.output)*2,uint64_t(v.output)*4,
                uint64_t(v.output)*4,uint64_t(v.input)*2,uint64_t(v.output)*v.splits*4};
            for(unsigned f=0;f<8;++f) {
                // Real host allocations, never dereferenced through descriptor types.
                auto mem=std::make_unique<unsigned char[]>(sizes[f]+512);
                uintptr_t base=reinterpret_cast<uintptr_t>(mem.get());
                uint64_t offset=(256-base%256)%256;
                v.regions[f]={base,sizes[f]+512,offset,sizes[f],41,0,true};
                storage.push_back(std::move(mem));
            }
        }
        Plan p;
        for(unsigned n:{1,2,4,8,16,31,32}) {
            check(prepare(d.data(),n,e,p)); auto b=pilot::bridge(p.batch());
            for(unsigned g=0;g<32;++g) {
                check(reinterpret_cast<uintptr_t>(b.groups[g].a)==reinterpret_cast<uintptr_t>(p.batch().groups[g].a));
                check(std::memcmp(&b.groups[g],&p.batch().groups[g],72)==0);
            }
        }
        auto bad=d; bad[31].regions[P].capacity--; check(!prepare(bad.data(),32,e,p));
        bad=d; bad[31].regions[A]=bad[0].regions[A];check(!prepare(bad.data(),32,e,p));
        bad=d; bad[31].alpha=INFINITY;check(!prepare(bad.data(),32,e,p));
        check(!prepare(d.data(),33,e,p));
        for(unsigned fault=0;fault<6;++fault) {
            auto ne=e;
            if(fault==0)ne.device=-1;if(fault==1)ne.stream_device=1;
            if(fault==2)ne.stream_live=false;if(fault==3)ne.owner=0;
            if(fault==4)ne.stream_owner=99;
            bad=d;if(fault==5)bad[31].regions[P].live=false;
            check(!prepare(bad.data(),32,ne,p));
        }
        std::cout<<"CPU real host allocation guard geometry="<<shape[0]<<","<<shape[1]<<","<<shape[2]<<" counts=1,2,4,8,16,31,32 negative_environment=6 PASS\n";
    }
    std::cout<<"HIP HOST layout Group="<<sizeof(pilot::NativeGroup)<<" Batch="<<sizeof(pilot::NativeBatch)
        <<" Trace="<<sizeof(pilot::Trace)<<" all offsets asserted; bridge all32 values PASS; no HIP runtime linked/called\n";
}
