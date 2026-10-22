// Runtime-free host test: no HIP library, GPU pointer, or device discovery.
#include "abi.hpp"
#include "first_error.hpp"
#include <cstdio>
#include <cstdlib>
using namespace grouped_k3_math;
static void check(bool b) {if(!b)std::abort();}
int main() {
    for(int fault=-1;fault<3;++fault) {
        unsigned attempts=0,mask=0;
        int status=enqueue_three([&](unsigned stage) {
            check(stage==attempts++);mask|=1u<<stage;
            return int(stage)==fault?79:0;
        },0);
        check(status==(fault<0?0:79));
        check(attempts==unsigned(fault<0?3:fault+1));
        check(mask==unsigned(fault<0?7:(1<<(fault+1))-1));
    }
    grouped_k3_review::Batch original{};
    original.input=5120;original.output=2304;original.splits=16;original.count=1;
    original.groups[0].alpha=-.75f;original.groups[0].beta=-0.f;
    auto copy=bridge(original);original.groups[0].alpha=97.f;
    check(copy.groups[0].alpha==-.75f && std::signbit(copy.groups[0].beta));
    NativeGroup zero{};
    for(unsigned g=1;g<32;++g)check(!std::memcmp(&zero,&copy.groups[g],sizeof(zero)));
    grouped_k3_review::Layout layout{};
    check(grouped_k3_review::layout(5120,2304,16,32,layout) && layout.stride==157696 && layout.total==5046272);
    check(grouped_k3_review::layout(2304,5120,4,32,layout) && layout.stride==86528 && layout.total==2768896);
    std::puts("PASS CPU only: HIP-host ABI72/2320, typed field bridge, frozen copy/inactive zero, actual shared first-error template, SUM");
    std::puts("No native allocation-failure injection, HIP enqueue failure, fdot2 or GPU evidence");
}
