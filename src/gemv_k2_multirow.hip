// Private multirow experiment; arithmetic extracted from our unchanged R1 V1.
#include "gemv_k2_multirow.hpp"
#include "v14_core_policy.hpp"
#include <hip/hip_fp16.h>
namespace strix_trellis::k2_multirow {
__device__ __forceinline__ uint32_t word_load(const unsigned char* base,int index) {
    uint32_t offset=uint32_t(index)*4u;
    asm("" : "+v"(offset));
    return *reinterpret_cast<const uint32_t*>(base+offset);
}
using Pair = _Float16 __attribute__((ext_vector_type(2)));
using namespace strix_trellis;
struct RawPacked { uint32_t a,b; };
__device__ __forceinline__ RawPacked read_raw(const uint32_t* p,int k,int n,int tn) {
    int lane=threadIdx.x&31;
    int index=((lane*8+1)*2+512-16)/32-16;
    int ia=index<0?index+16:index;
    int ib=index+1>=16?index+1-16:index+1;
    auto* base=reinterpret_cast<const unsigned char*>(p+(size_t(k)*size_t(n/16)+size_t(tn))*16);
    return {word_load(base,ia),word_load(base,ib)};
}
__device__ __forceinline__ void states(RawPacked r,uint32_t (&s)[8]) {
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
// Exhaustive arithmetic probes include only the preserved V1 helpers.
template<int Rows,bool LDS>
__device__ __forceinline__ void consume(RawPacked raw,const _Float16* a,const uint32_t* tile,
    int input,int k,int kk,float (&sums)[Rows][2]) {
    // Empty arithmetic dependencies constrain compiler scheduling across K16s.
    // They do not hide memory or alter the preserved V1 arithmetic block.
    #pragma unroll
    for(int r=0;r<Rows;++r)
        asm("" : "+v"(raw.a),"+v"(raw.b) : "v"(sums[r][0]),"v"(sums[r][1]));
    uint32_t s[8],h[8];states(raw,s);staged(s,h);
    uint32_t slot=threadIdx.x&3;
    asm("" : "+v"(slot) : "v"(h[0]),"v"(h[1]),"v"(h[2]),"v"(h[3]),
        "v"(h[4]),"v"(h[5]),"v"(h[6]),"v"(h[7]));
    // Hash first: the 24 asm values never overlap an array of U4 activations.
    Pair activations[Rows][2];
    #pragma unroll
    for(int r=0;r<Rows;++r) {
        if constexpr(LDS) {
            activations[r][0]=__builtin_bit_cast(Pair,tile[r*64+kk*8+slot]);
            activations[r][1]=__builtin_bit_cast(Pair,tile[r*64+kk*8+slot+4]);
        } else {
            size_t channel=size_t(r)*size_t(input)+size_t(k)*16+2*size_t(slot);
            __builtin_memcpy(&activations[r][0],a+channel,4);
            __builtin_memcpy(&activations[r][1],a+channel+8,4);
        }
    }
    #pragma unroll
    for(int pair=0;pair<4;++pair) {
        Pair w=sad_pair(h[2*pair],h[2*pair+1]);
        #pragma unroll
        for(int r=0;r<Rows;++r)
            sums[r][pair/2]=__builtin_amdgcn_fdot2(w,activations[r][pair&1],sums[r][pair/2],false);
    }
}
template<int Rows,bool LDS>
__global__ __launch_bounds__(128) void kernel(const uint32_t* __restrict__ b,
    const _Float16* __restrict__ a,float* __restrict__ p,int input,int n,int total,int splits) {
    static_assert(Rows==2||Rows==4||Rows==8);
    int tn=__builtin_amdgcn_readfirstlane(blockIdx.x*4+threadIdx.x/32);
    bool active=tn<n/16;
    if constexpr(!LDS) {if(!active)return;}
    __shared__ uint32_t tile[LDS?Rows*8*8:1];
    const int begin=int(int64_t(input/16)*int64_t(blockIdx.y)/splits);
    const int end=int(int64_t(input/16)*(int64_t(blockIdx.y)+1)/splits);
    float sums[Rows][2]={};
    for(int base=begin;base<end;base+=8) {
        int chunk_end=base+8<end?base+8:end;
        if constexpr(LDS) {
            for(int q=threadIdx.x;q<Rows*8*8;q+=128) {
                int r=q/64,pair=q%64;
                if(base+pair/8<end) {
                    uint32_t v;
                    __builtin_memcpy(&v,a+size_t(r)*size_t(input)+size_t(base)*16+size_t(pair)*2,4);
                    tile[q]=v;
                }
            }
            __syncthreads();
        }
        if(active) {
            int k=base;
            for(;k+4<=end && k+4<=chunk_end;k+=4) {
                RawPacked raw[4]; // Exactly eight B u32 per lane, independent of Rows.
                #pragma unroll
                for(int j=0;j<4;++j)raw[j]=read_raw(b,k+j,n,tn);
                // Keep all four B tiles resident before starting the first hash.
                asm("" : "+v"(raw[0].a),"+v"(raw[0].b),"+v"(raw[1].a),"+v"(raw[1].b),
                    "+v"(raw[2].a),"+v"(raw[2].b),"+v"(raw[3].a),"+v"(raw[3].b));
                #pragma unroll
                for(int j=0;j<4;++j)consume<Rows,LDS>(raw[j],a,tile,input,k+j,k+j-base,sums);
            }
            for(;k<chunk_end;++k)consume<Rows,LDS>(read_raw(b,k,n,tn),a,tile,input,k,k-base,sums);
        }
        if constexpr(LDS) __syncthreads();
    }
    if(active) {
        int lane=threadIdx.x&31;
        #pragma unroll
        for(int r=0;r<Rows;++r) {
            #pragma unroll
            for(int h=0;h<2;++h) {
                float v=sums[r][h];v+=__shfl_down(v,2,4);v+=__shfl_down(v,1,4);
                if(!(lane&3))p[(size_t(blockIdx.y)*size_t(total)+size_t(r))*size_t(n)+size_t(tn)*16+lane/4+h*8]=v;
            }
        }
    }
}
} // namespace strix_trellis::k2_multirow
namespace strix_trellis {
hipError_t launch_gemv_k2_multirow(const DeviceWeights& w,const _Float16* a,float* p,
    int rows,int total_rows,int splits,hipStream_t stream) {
    if(!w.trellis || !a || !p || !k2_multirow_candidate(w.input,w.output,w.bits,w.codebook,
       rows,total_rows,splits,!(uintptr_t(a)&3u),!(uintptr_t(p)&3u),!(uintptr_t(w.trellis)&3u)))
        return hipErrorInvalidValue;
    dim3 grid((w.output/16+3)/4,splits),block(128);
    #define RUN(R) case R: k2_multirow::kernel<R,true><<<grid,block,0,stream>>>(w.trellis,a,p,w.input,w.output,total_rows,splits); break
    switch(rows) {RUN(2);RUN(4);RUN(8);}
    #undef RUN
    return hipGetLastError();
}
}
