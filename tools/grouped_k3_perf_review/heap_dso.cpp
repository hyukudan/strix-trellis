// CPU-only surrogate for a legitimate shared-library allocation, not HIP code.
#include <new>
#include <cstddef>
extern "C" int cpu_dso_roundtrip(unsigned kind) {
    try {
        void* p = (kind == 1 || kind == 3) ? ::operator new[](24) : ::operator new(24);
        if (kind == 0) ::operator delete(p);
        if (kind == 1) ::operator delete[](p);
        if (kind == 2) ::operator delete(p, std::size_t(24));
        if (kind == 3) ::operator delete[](p, std::size_t(24));
        return 0;
    } catch (const std::bad_alloc&) { return 1; }
}
extern "C" void* cpu_dso_allocate(unsigned kind) {
    return (kind == 1 || kind == 3) ? ::operator new[](24) : ::operator new(24);
}
extern "C" void cpu_dso_release(void* p, unsigned kind) {
    if (kind == 0) ::operator delete(p);
    if (kind == 1) ::operator delete[](p);
    if (kind == 2) ::operator delete(p, std::size_t(24));
    if (kind == 3) ::operator delete[](p, std::size_t(24));
}
