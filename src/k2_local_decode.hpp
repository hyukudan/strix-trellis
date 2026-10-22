#pragma once
// Project-owned V16 K2 arithmetic, reused by the local-split experiment and QA.
// This is not imported or translated from an external implementation.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

namespace strix_trellis::k2_local_kernel {
__device__ __forceinline__ uint32_t word_load(const unsigned char* base,int index) {
    uint32_t offset=uint32_t(index)*4u;
    asm("" : "+v"(offset));
    return *reinterpret_cast<const uint32_t*>(base+offset);
}
using Pair = _Float16 __attribute__((ext_vector_type(2)));
using namespace strix_trellis;
struct Raw { uint32_t a,b,x,y; };
__device__ __forceinline__ Raw read_raw(const uint32_t* p,const _Float16* x,int k,int n,int tn) {
    int lane=threadIdx.x&31;
    int index=((lane*8+1)*2+512-16)/32-16;
    int ia=index<0?index+16:index;
    int ib=index+1>=16?index+1-16:index+1;
    auto* base=reinterpret_cast<const unsigned char*>(p+(size_t(k)*(n/16)+tn)*16);
    Raw r;
    r.a=word_load(base,ia);
    r.b=word_load(base,ib);
    // memcpy preserves the activation object's alias/provenance contract.
    __builtin_memcpy(&r.x,x+k*16+2*(lane&3),4);
    __builtin_memcpy(&r.y,x+k*16+2*(lane&3)+8,4);
    return r;
}
__device__ __forceinline__ void states(Raw r,uint32_t (&s)[8]) {
    int shift=(((threadIdx.x&31)*8+1)*2+512-16)&31;
    uint32_t w=shift==2?r.a:((r.a<<16)|(r.b>>16));
    #pragma unroll
    for(int j=0;j<8;++j)s[j]=w>>(14-2*j);
    // Upper bits survive intentionally: staged consumes only unsigned16.
}
__device__ __forceinline__ Pair sad_pair(uint32_t a,uint32_t b) {
    uint32_t low=__builtin_amdgcn_sad_u8(a,0,0x64006400u);
    uint32_t both=__builtin_amdgcn_sad_hi_u8(b,0,low);
    return __builtin_bit_cast(Pair,__hfma2(__builtin_bit_cast(__half2,both),
        __builtin_bit_cast(__half2,uint32_t(0x1eee1eee)),
        __builtin_bit_cast(__half2,uint32_t(0xc931c931))));
}
__device__ __forceinline__ void staged(const uint32_t (&s)[8],uint32_t (&h)[8]) {
    uint32_t t[8];
    // All temporaries are earlyclobber: none may alias any live state input.
    // Pure arithmetic only; memory remains visible to LLVM.
    asm(
        "v_pk_mul_lo_u16 %0, %16, %24 op_sel:[0,0] op_sel_hi:[0,1]\n\t"
        "v_pk_mul_lo_u16 %1, %17, %24 op_sel:[0,0] op_sel_hi:[0,1]\n\t"
        "v_pk_mul_lo_u16 %2, %18, %24 op_sel:[0,0] op_sel_hi:[0,1]\n\t"
        "v_pk_mul_lo_u16 %3, %19, %24 op_sel:[0,0] op_sel_hi:[0,1]\n\t"
        "v_pk_mul_lo_u16 %4, %20, %24 op_sel:[0,0] op_sel_hi:[0,1]\n\t"
        "v_pk_mul_lo_u16 %5, %21, %24 op_sel:[0,0] op_sel_hi:[0,1]\n\t"
        "v_pk_mul_lo_u16 %6, %22, %24 op_sel:[0,0] op_sel_hi:[0,1]\n\t"
        "v_pk_mul_lo_u16 %7, %23, %24 op_sel:[0,0] op_sel_hi:[0,1]\n\t"
        "v_mad_u32_u16 %8, %16, %25, %0\n\t"
        "v_mad_u32_u16 %9, %17, %25, %1\n\t"
        "v_mad_u32_u16 %10, %18, %25, %2\n\t"
        "v_mad_u32_u16 %11, %19, %25, %3\n\t"
        "v_mad_u32_u16 %12, %20, %25, %4\n\t"
        "v_mad_u32_u16 %13, %21, %25, %5\n\t"
        "v_mad_u32_u16 %14, %22, %25, %6\n\t"
        "v_mad_u32_u16 %15, %23, %25, %7"
        : "=&v"(t[0]),"=&v"(t[1]),"=&v"(t[2]),"=&v"(t[3]),
          "=&v"(t[4]),"=&v"(t[5]),"=&v"(t[6]),"=&v"(t[7]),
          "=&v"(h[0]),"=&v"(h[1]),"=&v"(h[2]),"=&v"(h[3]),
          "=&v"(h[4]),"=&v"(h[5]),"=&v"(h[6]),"=&v"(h[7])
        : "v"(s[0]),"v"(s[1]),"v"(s[2]),"v"(s[3]),
          "v"(s[4]),"v"(s[5]),"v"(s[6]),"v"(s[7]),
          "s"(0x83dc0000u),"s"(0xd12du));
}
__device__ __forceinline__ void consume(Raw r,float& lo,float& hi) {
    uint32_t s[8],h[8];states(r,s);
    staged(s,h);
    Pair a=__builtin_bit_cast(Pair,r.x),b=__builtin_bit_cast(Pair,r.y);
    #pragma unroll
    for(int pair=0;pair<4;++pair) {
        uint32_t x=h[2*pair],y=h[2*pair+1];
        Pair w=sad_pair(x,y);
        // Preserve pair order 0,1,2,3 and the original two FP32 chains.
        if(pair<2)lo=__builtin_amdgcn_fdot2(w,(pair&1)?b:a,lo,false);
        else hi=__builtin_amdgcn_fdot2(w,(pair&1)?b:a,hi,false);
    }
}

} // namespace strix_trellis::k2_local_kernel
