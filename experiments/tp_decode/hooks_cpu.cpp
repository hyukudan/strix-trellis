// Compile the actual ordinary wrappers against CPU stand-ins; never HIP-linked.
#include <cstdlib>
#include <cstdio>
#include <new>
#include "contract.hpp"
using hipError_t=int;using hipMemcpyKind=int;using hipStream_t=void*;using hipEvent_t=void*;
struct hipFuncAttributes{};struct hipDeviceProp_t{};struct hipPointerAttribute_t{};
constexpr int hipErrorInvalidValue=1;
#include "submit_hooks.hpp"
static unsigned forwards=0;
#define STUB(N,P) extern "C" int __real_##N P {++forwards;return 713;}
STUB(hipMalloc,(void**,size_t))
STUB(hipFree,(void*))
STUB(hipMemcpy,(void*,const void*,size_t,hipMemcpyKind))
STUB(hipMemcpyAsync,(void*,const void*,size_t,hipMemcpyKind,hipStream_t))
STUB(hipMemset,(void*,int,size_t))
STUB(hipMemsetAsync,(void*,int,size_t,hipStream_t))
STUB(hipStreamSynchronize,(hipStream_t))
STUB(hipDeviceSynchronize,(void))
STUB(hipEventSynchronize,(hipEvent_t))
STUB(hipFuncGetAttributes,(hipFuncAttributes*,const void*))
STUB(hipGetDeviceProperties,(hipDeviceProp_t*,int))
STUB(hipGetDevice,(int*))
STUB(hipPointerGetAttributes,(hipPointerAttribute_t*,const void*))
STUB(hipMemGetAddressRange,(void**,size_t*,void*))
STUB(hipEventRecord,(hipEvent_t,hipStream_t))
STUB(hipEventElapsedTime,(float*,hipEvent_t,hipEvent_t))
#undef STUB
extern "C" void* tp_dso_allocate(size_t);
extern "C" void tp_dso_release(void*);
static void check(bool value){if(!value)std::abort();}
int main(){
#define TEST(N,A) check(__wrap_##N A==713);{tp_decode::SubmitScope guard;check(__wrap_##N A==hipErrorInvalidValue);}
    TEST(hipMalloc,(nullptr,0)) TEST(hipFree,(nullptr)) TEST(hipMemcpy,(nullptr,nullptr,0,0))
    TEST(hipMemcpyAsync,(nullptr,nullptr,0,0,nullptr)) TEST(hipMemset,(nullptr,0,0)) TEST(hipMemsetAsync,(nullptr,0,0,nullptr))
    TEST(hipStreamSynchronize,(nullptr)) TEST(hipDeviceSynchronize,()) TEST(hipEventSynchronize,(nullptr))
    TEST(hipFuncGetAttributes,(nullptr,nullptr)) TEST(hipGetDeviceProperties,(nullptr,0)) TEST(hipGetDevice,(nullptr))
    TEST(hipPointerGetAttributes,(nullptr,nullptr)) TEST(hipMemGetAddressRange,(nullptr,nullptr,nullptr))
    TEST(hipEventRecord,(nullptr,nullptr)) TEST(hipEventElapsedTime,(nullptr,nullptr,nullptr))
#undef TEST
    check(forwards==16);
    void* p=__wrap_malloc(16);check(p!=nullptr);__wrap_free(p);
    p=__wrap_calloc(2,16);check(p!=nullptr);p=__wrap_realloc(p,64);check(p!=nullptr);__wrap_free(p);
    p=__wrap__Znwm(32);check(p!=nullptr);__wrap__ZdlPv(p);
    p=__wrap__Znam(32);check(p!=nullptr);__wrap__ZdaPv(p);
    p=__wrap__Znwm(32);__wrap__ZdlPvm(p,32);p=__wrap__Znam(32);__wrap__ZdaPvm(p,32);
    {tp_decode::SubmitScope guard;check(!__wrap_malloc(4)&&!__wrap_calloc(1,4)&&!__wrap_realloc(nullptr,4));}
    bool caught=false;try{tp_decode::SubmitScope guard;(void)__wrap__Znwm(4);}catch(const std::bad_alloc&){caught=true;}check(caught);
    caught=false;try{tp_decode::SubmitScope guard;(void)__wrap__Znam(4);}catch(const std::bad_alloc&){caught=true;}check(caught);
    p=__wrap__Znwm(32);{tp_decode::SubmitScope guard;__wrap__ZdlPv(p);}__wrap__ZdlPv(p);
    p=__wrap__Znam(32);{tp_decode::SubmitScope guard;__wrap__ZdaPv(p);}__wrap__ZdaPv(p);
    p=__wrap__Znwm(32);{tp_decode::SubmitScope guard;__wrap__ZdlPvm(p,32);}__wrap__ZdlPvm(p,32);
    p=__wrap__Znam(32);{tp_decode::SubmitScope guard;__wrap__ZdaPvm(p,32);}__wrap__ZdaPvm(p,32);
    p=__wrap_malloc(32);{tp_decode::SubmitScope guard;__wrap_free(p);}__wrap_free(p);
    size_t count=tp_decode::forbidden_calls[5];
    // Counterexample: DSO-internal operations are outside ordinary --wrap scope.
    // They must still allocate/free correctly, without global symbol replacement.
    {tp_decode::SubmitScope guard;p=tp_dso_allocate(77);check(p!=nullptr);tp_dso_release(p);}
    check(tp_decode::forbidden_calls[5]==count);
    std::puts("PASS 16 actual HIP link wrappers forward 713/block submit; ordinary C/C++ heap wrappers; DSO boundary counterexample remains outside scope");
}
