// Review-only minimal remedy prototype. Does not modify the frozen harness.
#include "../../experiments/grouped_k3_perf/submit_guard.hpp"
#include <new>
#include <cstddef>
using namespace grouped_k3_perf;
#define ALLOC(SYM) \
extern "C" void* __real_##SYM(std::size_t); \
extern "C" void* __wrap_##SYM(std::size_t n) { \
    if (forbidden(SideEffect::heap)) throw std::bad_alloc(); \
    return __real_##SYM(n); }
#define FREE(SYM, PARAMS, ARGS) \
extern "C" void __real_##SYM PARAMS noexcept; \
extern "C" void __wrap_##SYM PARAMS noexcept { \
    if (!forbidden(SideEffect::heap)) __real_##SYM ARGS; }
ALLOC(_Znwm)
ALLOC(_Znam)
FREE(_ZdlPv, (void* p), (p))
FREE(_ZdaPv, (void* p), (p))
FREE(_ZdlPvm, (void* p, std::size_t n), (p,n))
FREE(_ZdaPvm, (void* p, std::size_t n), (p,n))
