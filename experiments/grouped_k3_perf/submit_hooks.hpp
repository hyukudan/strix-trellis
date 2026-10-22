// Link-boundary user/static-library guard, not a global driver-query claim.
#pragma once
#include "submit_guard.hpp"
#define PERF_WRAP(NAME,TYPE,PARAMS,ARGS,KIND) \
extern "C" hipError_t __real_##NAME PARAMS; \
extern "C" hipError_t __wrap_##NAME PARAMS { \
    if(grouped_k3_perf::forbidden(grouped_k3_perf::SideEffect::KIND))return hipErrorInvalidValue; \
    return __real_##NAME ARGS; }
PERF_WRAP(hipMalloc,hipError_t,(void** p,size_t n),(p,n),allocation)
PERF_WRAP(hipFree,hipError_t,(void* p),(p),allocation)
PERF_WRAP(hipMemcpy,hipError_t,(void* d,const void* s,size_t n,hipMemcpyKind k),(d,s,n,k),copy)
PERF_WRAP(hipMemcpyAsync,hipError_t,(void* d,const void* s,size_t n,hipMemcpyKind k,hipStream_t st),(d,s,n,k,st),copy)
PERF_WRAP(hipMemsetAsync,hipError_t,(void* d,int v,size_t n,hipStream_t st),(d,v,n,st),copy)
PERF_WRAP(hipStreamSynchronize,hipError_t,(hipStream_t st),(st),sync)
PERF_WRAP(hipPointerGetAttributes,hipError_t,(hipPointerAttribute_t* a,const void* p),(a,p),query)
PERF_WRAP(hipMemGetAddressRange,hipError_t,(void** p,size_t* n,void* d),(p,n,d),query)
PERF_WRAP(hipStreamGetDevice,hipError_t,(hipStream_t st,hipDevice_t* d),(st,d),query)
PERF_WRAP(hipStreamGetFlags,hipError_t,(hipStream_t st,unsigned* f),(st,f),query)
PERF_WRAP(hipGetDevice,hipError_t,(int* d),(d),query)
PERF_WRAP(hipEventRecord,hipError_t,(hipEvent_t e,hipStream_t st),(e,st),event)
PERF_WRAP(hipEventElapsedTime,hipError_t,(float* ms,hipEvent_t a,hipEvent_t b),(ms,a,b),event)
#undef PERF_WRAP
// malloc/calloc/realloc calls from linked user/static code, including native operator new below.
extern "C" void* __real_malloc(size_t);
extern "C" void* __wrap_malloc(size_t n){if(grouped_k3_perf::forbidden(grouped_k3_perf::SideEffect::heap))return nullptr;return __real_malloc(n);}
extern "C" void* __real_calloc(size_t,size_t);
extern "C" void* __wrap_calloc(size_t n,size_t s){if(grouped_k3_perf::forbidden(grouped_k3_perf::SideEffect::heap))return nullptr;return __real_calloc(n,s);}
extern "C" void* __real_realloc(void*,size_t);
extern "C" void* __wrap_realloc(void* p,size_t n){if(grouped_k3_perf::forbidden(grouped_k3_perf::SideEffect::heap))return nullptr;return __real_realloc(p,n);}
