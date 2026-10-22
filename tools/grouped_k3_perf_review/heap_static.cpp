// Separate static-library call site exercises the proposed link-boundary guard.
#include <new>
extern "C" void* cpu_static_allocate() { return ::operator new(24); }
