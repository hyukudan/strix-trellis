#pragma once
// Independently derived modulo-32 hash and packed codebook experiment.
// The input state is unsigned16; upper bits of its container are ignored.
#include "trellis_optimized.hpp"
namespace strix_trellis::combined_decode {
template<int Bits,bool FastWrap=true> struct SaddrLane {
    static_assert(Bits>=1 && Bits<=8 && FastWrap);
    uint32_t high,low,tail;
    __device__ __forceinline__ static uint32_t load(const unsigned char* base,int index) {
        uint32_t offset=uint32_t(index)*4u;
        // Keep an unsigned32 VMEM offset opaque. The C++ load remains visible to
        // LLVM's dependency/waitcnt machinery; no asynchronous load is hidden.
        asm("" : "+v"(offset));
        return *reinterpret_cast<const uint32_t*>(base+offset);
    }
    __device__ explicit SaddrLane(const uint32_t* data) {
        constexpr int count=Bits*8;
        // This reader requires a pointer uniform within each participating wave.
        // The call site scalarizes tile_n. Preserve pointer provenance instead
        // of rebuilding a generic pointer from integer halves.
        auto* base=reinterpret_cast<const unsigned char*>(data);
        int start=((threadIdx.x&31)*8+1)*Bits+256*Bits-16;
        int index=start/32-count,shift=start&31;
        int ia=index<0?index+count:index;
        int ib=index+1>=count?index+1-count:index+1;
        int ic=index+2>=count?index+2-count:index+2;
        uint32_t a=load(base,ia),b=load(base,ib),c=load(base,ic);
        high=(a<<shift)|(b>>(32-shift));
        low=(b<<shift)|(c>>(32-shift));tail=c<<shift;
    }
    template<int J> __device__ __forceinline__ uint32_t state() const {
        constexpr int offset=J*Bits;
        if constexpr(offset<=16)return(high>>(16-offset))&65535u;
        else if constexpr(offset<32)return((high<<(offset-16))|(low>>(48-offset)))&65535u;
        else if constexpr(offset<=48)return(low>>(48-offset))&65535u;
        else return((low<<(offset-48))|(tail>>(80-offset)))&65535u;
    }
};
template<uint32_t Multiplier,uint32_t Offset=0>
__device__ __forceinline__ uint32_t hash16_isa(uint32_t state) {
    uint32_t high, result;
    // Low result = Offset.low; high = state*Multiplier.high + Offset.high.
    // The following full-width MAD supplies the low product and its carry.
    if constexpr(Offset==0) {
        // Preserve s in low16 while forming the high contribution for M-1.
        // Then s*(M-1).low + packed(s, s*(M-1).high) equals s*M mod2^32.
        constexpr uint32_t adjusted=Multiplier-1u;
        constexpr uint32_t upper=(adjusted&0xffff0000u)|1u;
        constexpr uint32_t lower=adjusted&65535u;
        asm("v_pk_mul_lo_u16 %0, %1, %2 op_sel:[0,0] op_sel_hi:[0,1]"
            : "=v"(high) : "v"(state), "s"(upper));
        asm("v_mad_u32_u16 %0, %1, %2, %1"
            : "=v"(result) : "v"(high), "s"(lower));
    } else {
        constexpr uint32_t upper=Multiplier&0xffff0000u;
        constexpr uint32_t lower=Multiplier&65535u;
        asm("v_pk_mad_u16 %0, %1, %2, %3 op_sel:[0,0,0] op_sel_hi:[0,1,1]"
            : "=v"(high) : "v"(state), "s"(upper), "v"(Offset));
        asm("v_mad_u32_u16 %0, %1, %2, %3"
            : "=v"(result) : "v"(state), "s"(lower), "v"(high));
    }
    return result;
}
__device__ __forceinline__ __half2 mul1_half2_isa(uint32_t first,uint32_t second) {
    uint32_t a=hash16_isa<0x83dcd12du>(first),b=hash16_isa<0x83dcd12du>(second);
    uint32_t low,both;
    constexpr uint32_t seed=0x64006400u;
    asm("v_sad_u8 %0, %1, 0, %2" : "=v"(low) : "v"(a), "s"(seed));
    asm("v_sad_hi_u8 %0, %1, 0, %2" : "=v"(both) : "v"(b), "v"(low));
    return __hfma2(__builtin_bit_cast(__half2,both),
                  __builtin_bit_cast(__half2,uint32_t(0x1eee1eeeu)),
                  __builtin_bit_cast(__half2,uint32_t(0xc931c931u)));
}
template<int Codebook> __device__ __forceinline__ __half2 codebook_half2_isa(uint32_t a,uint32_t b) {
    static_assert(Codebook>=0 && Codebook<=2);
    if constexpr(Codebook==2) return mul1_half2_isa(a,b);
    else {
        constexpr uint32_t multiplier=Codebook==0?89226354u:0xcbac1fedu;
        constexpr uint32_t offset=Codebook==0?64248484u:0u;
        uint32_t x=(hash16_isa<multiplier,offset>(a)&0x8fff8fffu)^0x3b603b60u;
        uint32_t y=(hash16_isa<multiplier,offset>(b)&0x8fff8fffu)^0x3b603b60u;
        uint32_t lows=(x&65535u)|(y<<16),highs=(x>>16)|(y&0xffff0000u);
        return __hadd2(__builtin_bit_cast(__half2,lows),__builtin_bit_cast(__half2,highs));
    }
}
} // namespace strix_trellis::optimized
