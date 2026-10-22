// Executes the very same forwarding hook as native QA, with no HIP/HSA library.
#include <cassert>
#include <cstdio>
#include <cstring>
struct hipFuncAttributes {int maxThreadsPerBlock;size_t sharedSizeBytes,localSizeBytes;int numRegs;};
using hipError_t=int;
constexpr int hipSuccess=0;
#include "attribute_hook.hpp"
using namespace grouped_k3_math;
static int result=0,forwarded=0;
static const void* passed_function=nullptr;
static hipFuncAttributes* passed_output=nullptr;
extern "C" int __real_hipFuncGetAttributes(hipFuncAttributes* a,const void* f) {
    ++forwarded;passed_function=f;passed_output=a;
    if(!result)*a={128,0,0,19};
    return result;
}
static int identities[8];
static const void* math[3]={&identities[4],&identities[5],&identities[6]};
static void query(int k) {
    hipFuncAttributes a;std::memset(&a,0xa5,sizeof(a));auto before=a;
    int n=forwarded;
    assert(__wrap_hipFuncGetAttributes(&a,&identities[k])==result);
    assert(forwarded==n+1 && passed_output==&a && passed_function==&identities[k]);
    if(result)assert(!std::memcmp(&a,&before,sizeof(a)));
}
static uint64_t begin(int old=4) {
    attribute_trace={};result=0;
    auto id=attribute_trace.reserve();
    for(int j=0;j<old;++j)query(j);
    attribute_trace.associate(id,&identities[7]);attribute_trace.phase=AttrPhase::math;
    return id;
}
static void finish(){for(int j=4;j<7;++j)query(j);}
int main() {
    auto id=begin();finish();assert(attribute_trace.admit(id,math) && attribute_trace.complete());
    for(size_t j=0;j<7;++j) {
        auto& e=attribute_trace.events[j];assert(e.creation==id && e.context==&identities[7] &&
            e.function==&identities[j] && e.valid && e.attributes.numRegs==19);
    }
    id=begin();attribute_trace.events[1].function=attribute_trace.events[0].function;finish();assert(!attribute_trace.admit(id,math));
    id=begin(3);finish();assert(!attribute_trace.admit(id,math) && attribute_trace.calls==6);
    id=begin(5);finish();assert(!attribute_trace.admit(id,math) && attribute_trace.calls==8);
    id=begin();query(4);query(4);query(6);assert(!attribute_trace.admit(id,math));
    id=begin();finish();const void* duplicate[3]={math[0],math[0],math[2]};assert(!attribute_trace.admit(id,duplicate));
    for(auto phase:{AttrPhase::prepare,AttrPhase::submit,AttrPhase::capture,AttrPhase::replay,AttrPhase::reuse,AttrPhase::cleanup}) {
        id=begin();finish();assert(attribute_trace.admit(id,math));
        {AttrRegion region(attribute_trace,phase,id);query(0);}
        assert(!attribute_trace.complete() && attribute_trace.calls==8);
        id=begin();finish();assert(attribute_trace.admit(id,math));
        {AttrRegion region(attribute_trace,phase,id);}
        assert(attribute_trace.complete());
    }
    for(int failure=0;failure<7;++failure) {
        attribute_trace={};id=attribute_trace.reserve();
        for(int j=0;j<7;++j) {
            if(j==4){attribute_trace.associate(id,&identities[7]);attribute_trace.phase=AttrPhase::math;}
            result=j==failure?713:0;query(j);
        }
        assert(!attribute_trace.admit(id,math) && attribute_trace.first_error==713);
        assert(!attribute_trace.events[failure].valid && attribute_trace.events[failure].attributes.numRegs==0);
        result=719;query(0);assert(attribute_trace.first_error==713);
    }
    id=begin();for(int j=0;j<130;++j)query(0);
    assert(attribute_trace.overflow && attribute_trace.stored==128 && attribute_trace.calls==134 && !attribute_trace.complete());
    attribute_trace={};for(int j=0;j<9;++j)attribute_trace.reserve();assert(attribute_trace.overflow);
    attribute_trace={};result=0;query(0);assert(attribute_trace.failed);
    // Two admitted contexts: cardinality is derived from actual creations.
    attribute_trace={};
    for(int c=0;c<2;++c) {
        id=attribute_trace.reserve();for(int j=0;j<4;++j)query(j);
        attribute_trace.associate(id,&identities[c]);attribute_trace.phase=AttrPhase::math;finish();
        assert(attribute_trace.admit(id,math));
    }
    assert(attribute_trace.complete() && attribute_trace.calls==14 && attribute_trace.creations==2);
    std::printf("PASS same forwarding hook: seven/omitted/extra/duplicate/late/error-first/capacity/two-context; trace_bytes=%zu\n",sizeof(attribute_trace));
}
