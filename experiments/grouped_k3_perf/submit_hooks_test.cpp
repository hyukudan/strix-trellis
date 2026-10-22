// Compile the ACTUAL native wrappers with CPU ABI stand-ins, never HIP/HSA.
#include <cstdlib>
#include <cstdio>
#include "contract.hpp"
using hipError_t=int;using hipMemcpyKind=int;using hipStream_t=void*;using hipEvent_t=void*;using hipDevice_t=int;
struct hipPointerAttribute_t {int unused;};constexpr int hipErrorInvalidValue=1;
#include "submit_hooks.hpp"
using namespace grouped_k3_perf;
static unsigned forwards=0;
extern "C" int __real_hipMalloc(void** p,size_t n){(void)p;(void)n;++forwards;return 713;}
extern "C" int __real_hipFree(void* p){(void)p;++forwards;return 713;}
extern "C" int __real_hipMemcpy(void* d,const void* s,size_t n,hipMemcpyKind k){(void)d;(void)s;(void)n;(void)k;++forwards;return 713;}
extern "C" int __real_hipMemcpyAsync(void* d,const void* s,size_t n,hipMemcpyKind k,hipStream_t st){(void)d;(void)s;(void)n;(void)k;(void)st;++forwards;return 713;}
extern "C" int __real_hipMemsetAsync(void* d,int v,size_t n,hipStream_t st){(void)d;(void)v;(void)n;(void)st;++forwards;return 713;}
extern "C" int __real_hipStreamSynchronize(hipStream_t st){(void)st;++forwards;return 713;}
extern "C" int __real_hipPointerGetAttributes(hipPointerAttribute_t* a,const void* p){(void)a;(void)p;++forwards;return 713;}
extern "C" int __real_hipMemGetAddressRange(void** p,size_t* n,void* d){(void)p;(void)n;(void)d;++forwards;return 713;}
extern "C" int __real_hipStreamGetDevice(hipStream_t st,hipDevice_t* d){(void)st;(void)d;++forwards;return 713;}
extern "C" int __real_hipStreamGetFlags(hipStream_t st,unsigned* f){(void)st;(void)f;++forwards;return 713;}
extern "C" int __real_hipGetDevice(int* d){(void)d;++forwards;return 713;}
extern "C" int __real_hipEventRecord(hipEvent_t e,hipStream_t st){(void)e;(void)st;++forwards;return 713;}
extern "C" int __real_hipEventElapsedTime(float* ms,hipEvent_t a,hipEvent_t b){(void)ms;(void)a;(void)b;++forwards;return 713;}
extern "C" void* __real_malloc(size_t n){return std::malloc(n);}
extern "C" void* __real_calloc(size_t n,size_t s){return std::calloc(n,s);}
extern "C" void* __real_realloc(void* p,size_t n){return std::realloc(p,n);}
int main(){
require(__wrap_hipMalloc(nullptr,0)==713,"actual HIP error forwarding");
{SubmitScope scope;require(__wrap_hipMalloc(nullptr,0)==hipErrorInvalidValue,"side effect must be blocked");}
require(__wrap_hipFree(nullptr)==713,"actual HIP error forwarding");
{SubmitScope scope;require(__wrap_hipFree(nullptr)==hipErrorInvalidValue,"side effect must be blocked");}
require(__wrap_hipMemcpy(nullptr,nullptr,0,0)==713,"actual HIP error forwarding");
{SubmitScope scope;require(__wrap_hipMemcpy(nullptr,nullptr,0,0)==hipErrorInvalidValue,"side effect must be blocked");}
require(__wrap_hipMemcpyAsync(nullptr,nullptr,0,0,nullptr)==713,"actual HIP error forwarding");
{SubmitScope scope;require(__wrap_hipMemcpyAsync(nullptr,nullptr,0,0,nullptr)==hipErrorInvalidValue,"side effect must be blocked");}
require(__wrap_hipMemsetAsync(nullptr,0,0,nullptr)==713,"actual HIP error forwarding");
{SubmitScope scope;require(__wrap_hipMemsetAsync(nullptr,0,0,nullptr)==hipErrorInvalidValue,"side effect must be blocked");}
require(__wrap_hipStreamSynchronize(nullptr)==713,"actual HIP error forwarding");
{SubmitScope scope;require(__wrap_hipStreamSynchronize(nullptr)==hipErrorInvalidValue,"side effect must be blocked");}
require(__wrap_hipPointerGetAttributes(nullptr,nullptr)==713,"actual HIP error forwarding");
{SubmitScope scope;require(__wrap_hipPointerGetAttributes(nullptr,nullptr)==hipErrorInvalidValue,"side effect must be blocked");}
require(__wrap_hipMemGetAddressRange(nullptr,nullptr,nullptr)==713,"actual HIP error forwarding");
{SubmitScope scope;require(__wrap_hipMemGetAddressRange(nullptr,nullptr,nullptr)==hipErrorInvalidValue,"side effect must be blocked");}
require(__wrap_hipStreamGetDevice(nullptr,nullptr)==713,"actual HIP error forwarding");
{SubmitScope scope;require(__wrap_hipStreamGetDevice(nullptr,nullptr)==hipErrorInvalidValue,"side effect must be blocked");}
require(__wrap_hipStreamGetFlags(nullptr,nullptr)==713,"actual HIP error forwarding");
{SubmitScope scope;require(__wrap_hipStreamGetFlags(nullptr,nullptr)==hipErrorInvalidValue,"side effect must be blocked");}
require(__wrap_hipGetDevice(nullptr)==713,"actual HIP error forwarding");
{SubmitScope scope;require(__wrap_hipGetDevice(nullptr)==hipErrorInvalidValue,"side effect must be blocked");}
require(__wrap_hipEventRecord(nullptr,nullptr)==713,"actual HIP error forwarding");
{SubmitScope scope;require(__wrap_hipEventRecord(nullptr,nullptr)==hipErrorInvalidValue,"side effect must be blocked");}
require(__wrap_hipEventElapsedTime(nullptr,nullptr,nullptr)==713,"actual HIP error forwarding");
{SubmitScope scope;require(__wrap_hipEventElapsedTime(nullptr,nullptr,nullptr)==hipErrorInvalidValue,"side effect must be blocked");}
require(forwards==13,"forbidden side effect forwarded");
{SubmitScope scope;require(__wrap_malloc(8)==nullptr && __wrap_calloc(1,8)==nullptr && __wrap_realloc(nullptr,8)==nullptr,"heap side effect missed");}
require(!clean_submit(),"missed violations");
std::puts("PASS actual link wrappers forward original errors and block alloc/copy/sync/query/event/heap in submit; CPU only");
}
