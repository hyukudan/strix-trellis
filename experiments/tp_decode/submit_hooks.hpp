// Ordinary linker wrappers only. No global new/delete definitions or DSO interposition.
#pragma once
#include "contract.hpp"
#define TP_WRAP(NAME,PARAMS,ARGS,KIND) \
extern "C" hipError_t __real_##NAME PARAMS; \
extern "C" hipError_t __wrap_##NAME PARAMS { \
    if(tp_decode::forbidden(KIND))return hipErrorInvalidValue;return __real_##NAME ARGS; }
TP_WRAP(hipMalloc,(void** p,size_t n),(p,n),0)
TP_WRAP(hipFree,(void* p),(p),0)
TP_WRAP(hipMemcpy,(void* d,const void* s,size_t n,hipMemcpyKind k),(d,s,n,k),1)
TP_WRAP(hipMemcpyAsync,(void* d,const void* s,size_t n,hipMemcpyKind k,hipStream_t st),(d,s,n,k,st),1)
TP_WRAP(hipMemset,(void* d,int v,size_t n),(d,v,n),1)
TP_WRAP(hipMemsetAsync,(void* d,int v,size_t n,hipStream_t st),(d,v,n,st),1)
TP_WRAP(hipStreamSynchronize,(hipStream_t st),(st),2)
TP_WRAP(hipDeviceSynchronize,(void),(),2)
TP_WRAP(hipEventSynchronize,(hipEvent_t e),(e),2)
TP_WRAP(hipFuncGetAttributes,(hipFuncAttributes* a,const void* f),(a,f),3)
TP_WRAP(hipGetDeviceProperties,(hipDeviceProp_t* p,int d),(p,d),3)
TP_WRAP(hipGetDevice,(int* d),(d),3)
TP_WRAP(hipPointerGetAttributes,(hipPointerAttribute_t* a,const void* p),(a,p),3)
TP_WRAP(hipMemGetAddressRange,(void** p,size_t* n,void* d),(p,n,d),3)
TP_WRAP(hipEventRecord,(hipEvent_t e,hipStream_t st),(e,st),4)
TP_WRAP(hipEventElapsedTime,(float* ms,hipEvent_t a,hipEvent_t b),(ms,a,b),4)
#undef TP_WRAP
extern "C" void* __real_malloc(size_t);
extern "C" void* __wrap_malloc(size_t n){if(tp_decode::forbidden(5))return nullptr;return __real_malloc(n);}
extern "C" void* __real_calloc(size_t,size_t);
extern "C" void* __wrap_calloc(size_t n,size_t s){if(tp_decode::forbidden(5))return nullptr;return __real_calloc(n,s);}
extern "C" void* __real_realloc(void*,size_t);
extern "C" void* __wrap_realloc(void* p,size_t n){if(tp_decode::forbidden(5))return nullptr;return __real_realloc(p,n);}
extern "C" void __real_free(void*);
extern "C" void __wrap_free(void* p){if(!tp_decode::forbidden(5))__real_free(p);}
// --wrap redirects only static link references. Calls inside libamdhip64 keep its ABI.
extern "C" void* __real__Znwm(size_t);
extern "C" void* __wrap__Znwm(size_t n){if(tp_decode::forbidden(5))throw std::bad_alloc();return __real__Znwm(n);}
extern "C" void* __real__Znam(size_t);
extern "C" void* __wrap__Znam(size_t n){if(tp_decode::forbidden(5))throw std::bad_alloc();return __real__Znam(n);}
extern "C" void __real__ZdlPv(void*);
extern "C" void __wrap__ZdlPv(void* p){if(!tp_decode::forbidden(5))__real__ZdlPv(p);}
extern "C" void __real__ZdaPv(void*);
extern "C" void __wrap__ZdaPv(void* p){if(!tp_decode::forbidden(5))__real__ZdaPv(p);}
extern "C" void __real__ZdlPvm(void*,size_t);
extern "C" void __wrap__ZdlPvm(void* p,size_t n){if(!tp_decode::forbidden(5))__real__ZdlPvm(p,n);}
extern "C" void __real__ZdaPvm(void*,size_t);
extern "C" void __wrap__ZdaPvm(void* p,size_t n){if(!tp_decode::forbidden(5))__real__ZdaPvm(p,n);}
