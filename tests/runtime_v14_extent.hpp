// Host-only interval decisions shared by native QA and the CPU reproduction.
#pragma once
#include <cstdint>
#include <cstddef>
#include <limits>
namespace v14_extent {
inline bool overlap(uintptr_t a,size_t an,uintptr_t b,size_t bn) {
    return an && bn && a < b+bn && b < a+an;
}
inline bool tail(uintptr_t allocation,size_t requested,uintptr_t base,size_t extent,
                 size_t guard,size_t required,uintptr_t& pointer,size_t& remaining) {
    if(requested<guard+4 || allocation!=base || extent<requested ||
       extent>std::numeric_limits<uintptr_t>::max()-base) return false;
    pointer=allocation+requested-guard-4;
    remaining=extent-(pointer-base);
    return pointer%4==0 && remaining>=guard+4 && remaining<required &&
           required<=std::numeric_limits<uintptr_t>::max()-pointer;
}
}
