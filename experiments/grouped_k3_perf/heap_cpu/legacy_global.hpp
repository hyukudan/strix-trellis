// Native and CPU mock share the actual global C++ heap prohibition.
#pragma once
#include "../submit_guard.hpp"
#include <cstdlib>
#include <new>
inline void* perf_allocate(size_t n){
    if(grouped_k3_perf::forbidden(grouped_k3_perf::SideEffect::heap))throw std::bad_alloc();
    if(void* p=std::malloc(n?n:1))return p;
    throw std::bad_alloc();
}
__attribute__((noinline)) void* operator new(size_t n){return perf_allocate(n);}
__attribute__((noinline)) void* operator new[](size_t n){return perf_allocate(n);}
void operator delete(void* p) noexcept {if(!grouped_k3_perf::forbidden(grouped_k3_perf::SideEffect::heap))std::free(p);}
void operator delete[](void* p) noexcept {if(!grouped_k3_perf::forbidden(grouped_k3_perf::SideEffect::heap))std::free(p);}
void operator delete(void* p,size_t) noexcept {if(!grouped_k3_perf::forbidden(grouped_k3_perf::SideEffect::heap))std::free(p);}
void operator delete[](void* p,size_t) noexcept {if(!grouped_k3_perf::forbidden(grouped_k3_perf::SideEffect::heap))std::free(p);}
