#pragma once
// Private experiment. Never passed to the V11 archive or production API.
#include "backend.hpp"
#include <limits>
namespace strix_trellis::k2_rows_private {
enum class Mode { Global, LDS8 };
struct OwnedRange { const void* base; size_t bytes; };
class Plan {
    const uint32_t* b_{};
    const _Float16* a_{};
    float* p_{};
    int input_{}, output_{}, rows_{}, total_{}, splits_{};
    bool valid_{};
    friend bool make_diagnostic(Plan&,const DeviceWeights&,const _Float16*,float*,
        int,int,int,int,OwnedRange,OwnedRange,OwnedRange);
    friend void submit(const Plan&,Mode,hipStream_t);
};
inline bool contained(OwnedRange owner,const void* pointer,size_t bytes) {
    uintptr_t b=reinterpret_cast<uintptr_t>(owner.base), p=reinterpret_cast<uintptr_t>(pointer);
    return b && p>=b && owner.bytes<=UINTPTR_MAX-b && p<=b+owner.bytes && bytes<=b+owner.bytes-p;
}
inline bool disjoint(OwnedRange a,OwnedRange b) {
    uintptr_t x=reinterpret_cast<uintptr_t>(a.base),y=reinterpret_cast<uintptr_t>(b.base);
    return x && y && a.bytes<=UINTPTR_MAX-x && b.bytes<=UINTPTR_MAX-y &&
        (x+a.bytes<=y || y+b.bytes<=x);
}
// Factories run before timing. Pointers are already shifted by row_offset.
inline bool make_diagnostic(Plan& plan,const DeviceWeights& w,const _Float16* a,float* p,
    int rows,int total,int offset,int splits,OwnedRange B,OwnedRange A,OwnedRange P) {
    static_assert(sizeof(size_t)==8);
    plan=Plan{};
    if((rows!=2&&rows!=4&&rows!=8)||total<rows||offset<0||offset>total-rows||
       splits<1||splits>32||w.input<16||w.input>8192||w.input%16||
       w.output<16||w.output>32768||w.output%16||w.bits!=2||w.codebook!=2||
       !w.trellis||!a||!p||(reinterpret_cast<uintptr_t>(w.trellis)&3)||
       (reinterpret_cast<uintptr_t>(a)&3)||(reinterpret_cast<uintptr_t>(p)&3))return false;
    const size_t I=size_t(w.input), N=size_t(w.output), R=size_t(rows), T=size_t(total), S=size_t(splits);
    // Domain bounds above prove these products fit size_t64, including INT_MAX total.
    const size_t bbytes=I*N/4, abytes=R*I*2, pbytes=((S-1)*T+R)*N*4;
    // Bind the shifted views to the declared parent's row origin as well as its extent.
    if(reinterpret_cast<uintptr_t>(a)!=reinterpret_cast<uintptr_t>(A.base)+size_t(offset)*I*2 ||
       reinterpret_cast<uintptr_t>(p)!=reinterpret_cast<uintptr_t>(P.base)+size_t(offset)*N*4 ||
       !contained(B,w.trellis,bbytes)||!contained(A,a,abytes)||!contained(P,p,pbytes)||
       !disjoint(B,A)||!disjoint(B,P)||!disjoint(A,P))return false;
    plan.b_=w.trellis;plan.a_=a;plan.p_=p;plan.input_=w.input;plan.output_=w.output;
    plan.rows_=rows;plan.total_=total;plan.splits_=splits;plan.valid_=true;return true;
}
inline bool make_performance(Plan& plan,const DeviceWeights& w,const _Float16* a,float* p,
    int rows,int total,int offset,int splits,OwnedRange B,OwnedRange A,OwnedRange P) {
    plan=Plan{};
    if(splits!=4 || !((w.input==8192&&w.output==5120)||(w.input==1280&&w.output==32768)))return false;
    return make_diagnostic(plan,w,a,p,rows,total,offset,splits,B,A,P);
}
// No allocation, transfer, synchronization, storage query or last-error query.
void submit(const Plan&,Mode,hipStream_t);
}
