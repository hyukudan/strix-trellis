// Private mathematical ABI. No Trace argument or device descriptor array.
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "../../reports/grouped_k3_private_guard.hpp"
namespace grouped_k3_math {
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
#define MATH_OFFSET(T,F,O) static_assert(offsetof(T,F)==O)
MATH_OFFSET(NativeGroup,packed,0); MATH_OFFSET(NativeGroup,x,8);
MATH_OFFSET(NativeGroup,suh,16); MATH_OFFSET(NativeGroup,svh,24);
MATH_OFFSET(NativeGroup,bias,32); MATH_OFFSET(NativeGroup,y,40);
MATH_OFFSET(NativeGroup,a,48); MATH_OFFSET(NativeGroup,partial,56);
MATH_OFFSET(NativeGroup,alpha,64); MATH_OFFSET(NativeGroup,beta,68);
MATH_OFFSET(NativeBatch,input,0); MATH_OFFSET(NativeBatch,output,4);
MATH_OFFSET(NativeBatch,splits,8); MATH_OFFSET(NativeBatch,count,12);
MATH_OFFSET(NativeBatch,groups,16);
#undef MATH_OFFSET
// Fieldwise pointer conversion through void*, never a struct/half type pun.
inline NativeBatch bridge(const grouped_k3_review::Batch& b) {
    NativeBatch n{};
    n.input=b.input; n.output=b.output; n.splits=b.splits; n.count=b.count;
    for (unsigned i=0;i<32;++i) {
        const auto& s=b.groups[i]; auto& d=n.groups[i];
        d.packed=s.packed; d.x=s.x; d.suh=s.suh; d.svh=s.svh;
        d.bias=s.bias; d.y=s.y;
        d.a=static_cast<_Float16*>(static_cast<void*>(s.a));
        d.partial=s.partial; d.alpha=s.alpha; d.beta=s.beta;
    }
    return n;
}
} // namespace grouped_k3_math
