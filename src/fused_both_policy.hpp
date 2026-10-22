#pragma once
#include <cstddef>
#include <cstdint>
namespace strix_trellis {
struct FusedBothLayout {
    size_t blocks=0,dynamic_bytes=0,static_bytes=0;
};
inline bool fused_both_device_limits(int threads,int dimension_x,size_t shared) {
    return threads>=1024 && dimension_x>=1024 && shared>=18432;
}
inline bool fused_both_image_limits(int threads,size_t shared,size_t local,
    int dynamic,size_t available) {
    return threads>=1024 && shared==2048 && local==0 && dynamic>=16384 && available>=18432;
}
template<class Error,class Attributes,class GetAttributes>
Error probe_fused_both_image(size_t available,bool* supported,Error success,
    Error invalid,const void* function,GetAttributes get_attributes) {
    if(!supported) return invalid;
    *supported=false;
    Attributes a{};
    Error e=get_attributes(&a,function);
    if(e!=success) return e;
    *supported=fused_both_image_limits(a.maxThreadsPerBlock,a.sharedSizeBytes,
        a.localSizeBytes,a.maxDynamicSharedSizeBytes,available);
    return success;
}
inline bool fused_both_aligned(const void* p,uintptr_t alignment) {
    return p && !(reinterpret_cast<uintptr_t>(p)&(alignment-1));
}
// Whole-operation R1 only. Bounded before all size_t layout arithmetic.
// Allocation extents, aliasing, finite scalars and workspace remain C API duties.
inline bool fused_both_layout(int input,int output,int bits,int codebook,int rows,
    bool prefer_wmma,bool capable,const void* packed,const void* x,
    const void* suh,const void* svh,const void* bias,const void* y,FusedBothLayout& out) {
    out={};
    if(!capable || prefer_wmma || rows!=1 || codebook!=2 || bits!=5 || output!=5120 ||
       (input!=8192 && input!=2304) || !fused_both_aligned(packed,4) ||
       !fused_both_aligned(x,4) || !fused_both_aligned(y,4) ||
       !fused_both_aligned(suh,2) || !fused_both_aligned(svh,2) ||
       (bias && !fused_both_aligned(bias,4))) return false;
    out.blocks=size_t(output)/128;
    out.dynamic_bytes=size_t(4)*((size_t(input)/128+3)/4)*128*2;
    out.static_bytes=2048;
    return true;
}
}
