// Private trace ABI. No production entry points or math operations.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include "../../reports/grouped_k3_private_guard.hpp"
namespace pilot {
struct NativeGroup {
    const uint32_t* packed;
    const float* x;
    const uint16_t* suh;
    const uint16_t* svh;
    const float* bias;
    float* y;
    _Float16* a;
    float* partial;
    float alpha, beta;
};
struct NativeBatch { int input, output, splits, count; NativeGroup groups[32]; };
static_assert(sizeof(_Float16)==2 && sizeof(void*)==8 && sizeof(int)==4);
static_assert(std::is_standard_layout_v<NativeBatch> && std::is_trivially_copyable_v<NativeBatch>);
static_assert(sizeof(NativeGroup)==72 && alignof(NativeGroup)==8);
static_assert(sizeof(NativeBatch)==2320 && alignof(NativeBatch)==8);
#define PILOT_OFFSET(T, F, O) static_assert(offsetof(T,F)==O)
PILOT_OFFSET(NativeGroup,packed,0); PILOT_OFFSET(NativeGroup,x,8);
PILOT_OFFSET(NativeGroup,suh,16); PILOT_OFFSET(NativeGroup,svh,24);
PILOT_OFFSET(NativeGroup,bias,32); PILOT_OFFSET(NativeGroup,y,40);
PILOT_OFFSET(NativeGroup,a,48); PILOT_OFFSET(NativeGroup,partial,56);
PILOT_OFFSET(NativeGroup,alpha,64); PILOT_OFFSET(NativeGroup,beta,68);
PILOT_OFFSET(NativeBatch,input,0); PILOT_OFFSET(NativeBatch,output,4);
PILOT_OFFSET(NativeBatch,splits,8); PILOT_OFFSET(NativeBatch,count,12);
PILOT_OFFSET(NativeBatch,groups,16);
#undef PILOT_OFFSET
static_assert(sizeof(NativeBatch)==sizeof(grouped_k3_review::Batch));
static_assert(sizeof(NativeGroup)==sizeof(grouped_k3_review::Group));
static_assert(offsetof(NativeGroup,a)==offsetof(grouped_k3_review::Group,a));
// Field value conversion only. Never access a half object through uint16_t.
inline NativeBatch bridge(const grouped_k3_review::Batch& b) {
    NativeBatch n{};
    n.input=b.input; n.output=b.output; n.splits=b.splits; n.count=b.count;
    for(unsigned i=0;i<32;++i) {
        const auto& s=b.groups[i]; auto& d=n.groups[i];
        d.packed=s.packed; d.x=s.x; d.suh=s.suh; d.svh=s.svh;
        d.bias=s.bias; d.y=s.y; d.a=static_cast<_Float16*>(static_cast<void*>(s.a));
        d.partial=s.partial; d.alpha=s.alpha; d.beta=s.beta;
    }
    return n;
}
struct Bits { uint64_t pointers[8]; uint32_t alpha, beta; };
struct Slot { Bits group; int metadata[4]; uint32_t x,y,z,predecessor,tag; };
struct Header { Bits groups[32]; int metadata[4]; };
struct Trace {
    uint64_t before[32];
    Header headers[3];
    Slot input[320], core[18432], output[320];
    uint64_t after[32];
};
inline Bits bits(const NativeGroup& g) {
    Bits b{{reinterpret_cast<uintptr_t>(g.packed),reinterpret_cast<uintptr_t>(g.x),
        reinterpret_cast<uintptr_t>(g.suh),reinterpret_cast<uintptr_t>(g.svh),
        reinterpret_cast<uintptr_t>(g.bias),reinterpret_cast<uintptr_t>(g.y),
        reinterpret_cast<uintptr_t>(g.a),reinterpret_cast<uintptr_t>(g.partial)},0,0};
    std::memcpy(&b.alpha,&g.alpha,4); std::memcpy(&b.beta,&g.beta,4); return b;
}
constexpr uint32_t tag(unsigned stage,unsigned g) { return 0x71300000u+stage*256+g; }
} // namespace pilot
