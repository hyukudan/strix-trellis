// Native CPU executable: validates the real private factories without libamdhip64.
#include "gemv_k2_rows.hpp"
#include <cassert>
#include <climits>
#include <cstdio>
#include <initializer_list>
using namespace strix_trellis;
using namespace strix_trellis::k2_rows_private;
int main() {
    size_t passed=0,rejected=0;
    constexpr uintptr_t b=uintptr_t(1)<<32,a=uintptr_t(1)<<48,p=uintptr_t(1)<<57;
    for(int I:{16,80,128,176,272,112,240,1280,8192})for(int N:{16,32,48,80,5120,32768})
    for(int R:{2,4,8})for(int S=1;S<=32;++S)for(int T:{R,R+3,INT_MAX})for(int O:{0,1}) {
        OwnedRange B{(void*)b,size_t(I)*size_t(N)/4},A{(void*)a,size_t(T)*size_t(I)*2},P{(void*)p,size_t(S)*size_t(T)*size_t(N)*4};
        DeviceWeights w{(uint32_t*)b,nullptr,nullptr,nullptr,I,N,2,2};
        const _Float16* av=(const _Float16*)(a+size_t(O)*size_t(I)*2);float* pv=(float*)(p+size_t(O)*size_t(N)*4);
        Plan plan;bool valid=O<=T-R;
        assert(make_diagnostic(plan,w,av,pv,R,T,O,S,B,A,P)==valid);++passed;
        bool perf=valid && S==4 && ((I==8192&&N==5120)||(I==1280&&N==32768));
        assert(make_performance(plan,w,av,pv,R,T,O,S,B,A,P)==perf);++passed;
        assert(!make_diagnostic(plan,w,av,pv,R,R,1,S,B,A,P));
        assert(!make_diagnostic(plan,w,av,pv,R,T,-1,S,B,A,P));
        assert(!make_diagnostic(plan,w,av,pv,3,T,O,S,B,A,P));
        assert(!make_diagnostic(plan,w,av+1,pv,R,T,O,S,B,A,P));
        assert(!make_diagnostic(plan,w,av,pv,R,T,O,S,{(void*)b,B.bytes-1},A,P));
        assert(!make_diagnostic(plan,w,av,pv,R,T,O,S,B,A,A));
        assert(!make_diagnostic(plan,w,av,pv,R,T,O,S,B,A,{(void*)p,1}));
        rejected+=7;
    }
    assert(!contained({(void*)(UINTPTR_MAX-3),8},(void*)(UINTPTR_MAX-3),4));
    assert(!disjoint({(void*)(UINTPTR_MAX-3),8},{(void*)16,4}));
    printf("PASS actual_factory accepted_or_guard_checked=%zu invalid_rejected=%zu overflow_rejected=2\n",passed,rejected);
}
