// Only this review fixture is executed; no native HIP executable is loaded.
#include "../../experiments/grouped_k3_perf/submit_guard.hpp"
#ifndef REVIEW_LINK_WRAP
#include "../../experiments/grouped_k3_perf/heap_guard.hpp"
#endif
#include <new>
#include <cstdio>
extern "C" int cpu_dso_roundtrip(unsigned);
extern "C" void* cpu_static_allocate();
extern "C" void* cpu_dso_allocate(unsigned);
extern "C" void cpu_dso_release(void*, unsigned);
using namespace grouped_k3_perf;
int main() {
    unsigned dso_failures=0, user_rejections=0;
    for (unsigned k=0;k<4;++k) if (cpu_dso_roundtrip(k)) return 2;
    {
        SubmitScope scope;
        for (unsigned k=0;k<4;++k) dso_failures+=cpu_dso_roundtrip(k);
    }
    auto dso_forbidden=forbidden_calls[unsigned(SideEffect::heap)];
    forbidden_calls[unsigned(SideEffect::heap)]=0;
    void* pointers[4];
    for (unsigned k=0;k<4;++k) pointers[k]=cpu_dso_allocate(k);
    { SubmitScope scope; for (unsigned k=0;k<4;++k) cpu_dso_release(pointers[k],k); }
    auto dso_delete_forbidden=forbidden_calls[unsigned(SideEffect::heap)];
#ifndef REVIEW_LINK_WRAP
    for (unsigned k=0;k<4;++k) cpu_dso_release(pointers[k],k);
#endif
    forbidden_calls[unsigned(SideEffect::heap)]=0;
    {
        SubmitScope scope;
        try { void* p=::operator new(24); ::operator delete(p); }
        catch (const std::bad_alloc&) { ++user_rejections; }
        try { void* p=cpu_static_allocate(); ::operator delete(p); }
        catch (const std::bad_alloc&) { ++user_rejections; }
    }
    auto user_forbidden=forbidden_calls[unsigned(SideEffect::heap)];
    for (unsigned k=0;k<4;++k) pointers[k]=cpu_dso_allocate(k);
    { SubmitScope scope;
      ::operator delete(pointers[0]); ::operator delete[](pointers[1]);
      ::operator delete(pointers[2], std::size_t(24));
      ::operator delete[](pointers[3], std::size_t(24)); }
    auto user_delete_forbidden=forbidden_calls[unsigned(SideEffect::heap)]-user_forbidden;
    for (unsigned k=0;k<4;++k) cpu_dso_release(pointers[k],k);
#ifdef REVIEW_LINK_WRAP
    bool expected=dso_failures==0 && dso_forbidden==0 && dso_delete_forbidden==0;
    const char* mode="link_wrap";
#else
    bool expected=dso_failures==4 && dso_forbidden==4 && dso_delete_forbidden==4;
    const char* mode="frozen_global_replacement";
#endif
    std::printf("mode=%s dso_failures=%u dso_forbidden=%zu dso_delete_forbidden=%zu user_rejections=%u user_forbidden=%zu user_delete_forbidden=%zu\n",
                mode,dso_failures,dso_forbidden,dso_delete_forbidden,user_rejections,user_forbidden,user_delete_forbidden);
    return expected && user_rejections==2 && user_forbidden==2 && user_delete_forbidden==4 ? 0 : 3;
}
