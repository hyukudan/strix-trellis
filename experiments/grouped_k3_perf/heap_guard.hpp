// Ordinary operator references at the final static link boundary only.
// Independent implementation; remedy motivated by HIGH review heap_wrap.cpp.
#pragma once
#include "submit_guard.hpp"
#include <new>
#include <cstddef>
#define PERF_NEW(symbol) \
extern "C" void* __real_##symbol(std::size_t); \
extern "C" __attribute__((noinline)) void* __wrap_##symbol(std::size_t size) { \
    if (grouped_k3_perf::forbidden(grouped_k3_perf::SideEffect::heap)) throw std::bad_alloc(); \
    return __real_##symbol(size); }
#define PERF_DELETE(symbol, signature, arguments) \
extern "C" void __real_##symbol signature noexcept; \
extern "C" __attribute__((noinline)) void __wrap_##symbol signature noexcept { \
    if (!grouped_k3_perf::forbidden(grouped_k3_perf::SideEffect::heap)) __real_##symbol arguments; }
PERF_NEW(_Znwm)
PERF_NEW(_Znam)
PERF_DELETE(_ZdlPv, (void* p), (p))
PERF_DELETE(_ZdaPv, (void* p), (p))
PERF_DELETE(_ZdlPvm, (void* p, std::size_t size), (p,size))
PERF_DELETE(_ZdaPvm, (void* p, std::size_t size), (p,size))
#undef PERF_NEW
#undef PERF_DELETE
