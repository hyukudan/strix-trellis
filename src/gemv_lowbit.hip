// Independent private K3/K4 experiment, derived from our reviewed K2 D4/V1.
#include "gemv_lowbit.hpp"
#include "gemv_lds.hpp"
#include <hip/hip_fp16.h>
namespace strix_trellis_lowbit {
using Pair = _Float16 __attribute__((ext_vector_type(2)));
static_assert(sizeof(size_t)==8);
struct Raw { uint32_t a,b,x,y; };
template<int K,bool Coop> __device__ __forceinline__ Raw read_raw(
    const uint32_t* p,const _Float16* x,int k,int n,int tn) {
    const int lane=threadIdx.x&31;
    const uint32_t* base=p+(size_t(k)*size_t(n/16)+size_t(tn))*size_t(8*K);
    Raw r{};
    if constexpr(Coop) {
        // The address expression itself is dominated by the valid-lane branch.
        if constexpr(K==3) { if(lane<24) r.b=base[lane]; }
        else r.b=base[lane];
    } else {
        int index=K==3?((24*lane+755)/32-24):lane-1;
        int ia=index<0?index+8*K:index;
        r.a=base[ia]; r.b=base[index+1];
    }
    const auto* aligned=static_cast<const _Float16*>(__builtin_assume_aligned(x,4));
    __builtin_memcpy(&r.x,aligned+size_t(k)*16+2*(lane&3),4);
    __builtin_memcpy(&r.y,aligned+size_t(k)*16+2*(lane&3)+8,4);
    return r;
}
template<int K,bool Coop> __device__ __forceinline__ void states(Raw r,uint32_t (&s)[8]) {
    const int lane=threadIdx.x&31;
    if constexpr(Coop) {
        // All 32 lanes have rejoined; each sender belongs to this Raw slot.
        if constexpr(K==3) {
            int index=(24*lane+755)/32-24;
            r.a=__shfl(r.b,index<0?index+24:index,32);
            r.b=__shfl(r.b,index+1,32);
        } else r.a=__shfl(r.b,(lane+31)&31,32);
    }
    if constexpr(K==3) {
        int shift=(24*lane+755)&31;
        uint32_t high=(r.a<<shift)|(r.b>>(32-shift)),low=r.b<<shift;
        #pragma unroll
        for(int j=0;j<6;++j)s[j]=high>>(16-3*j);
        s[6]=(high<<2)|(low>>30); s[7]=(high<<5)|(low>>27);
    } else {
        s[0]=(r.a<<4)|(r.b>>28); s[1]=(r.a<<8)|(r.b>>24);
        s[2]=(r.a<<12)|(r.b>>20); s[3]=r.b>>16;
        s[4]=r.b>>12;s[5]=r.b>>8;s[6]=r.b>>4;s[7]=r.b;
    }
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

template<int K,bool Coop> __device__ __forceinline__ void consume(Raw r,float& lo,float& hi) {
    uint32_t s[8],h[8];states<K,Coop>(r,s);staged(s,h);
    Pair a=__builtin_bit_cast(Pair,r.x),b=__builtin_bit_cast(Pair,r.y);
    #pragma unroll
    for(int pair=0;pair<4;++pair) {
        Pair w=sad_pair(h[2*pair],h[2*pair+1]);
        if(pair<2)lo=__builtin_amdgcn_fdot2(w,(pair&1)?b:a,lo,false);
        else hi=__builtin_amdgcn_fdot2(w,(pair&1)?b:a,hi,false);
    }
}
template<int K,int U,bool Coop> __global__ __launch_bounds__(128) void kernel(
    const uint32_t* __restrict__ p,const _Float16* __restrict__ x,
    float* __restrict__ out,int input,int n,int total_rows,int splits) {
    int tn=__builtin_amdgcn_readfirstlane(blockIdx.x*4+threadIdx.x/32);
    if(tn>=n/16)return;
    int begin=int(int64_t(input/16)*blockIdx.y/splits);
    int end=int(int64_t(input/16)*(blockIdx.y+1)/splits);
    int k=begin;float lo=0,hi=0;
    if constexpr(U==4) {
        for(;k+4<=end;k+=4) {
            Raw raw[4];
            #pragma unroll
            for(int j=0;j<4;++j)raw[j]=read_raw<K,Coop>(p,x,k+j,n,tn);
            // Register-only compiler fence: all raw slots precede any decode,
            // including cooperative shuffles. No instruction, memory or wait.
            if constexpr(Coop) {
                asm volatile("" : "+v"(raw[0].b),"+v"(raw[0].x),"+v"(raw[0].y),
                    "+v"(raw[1].b),"+v"(raw[1].x),"+v"(raw[1].y),
                    "+v"(raw[2].b),"+v"(raw[2].x),"+v"(raw[2].y),
                    "+v"(raw[3].b),"+v"(raw[3].x),"+v"(raw[3].y));
            } else {
                asm volatile("" : "+v"(raw[0].a),"+v"(raw[0].b),"+v"(raw[0].x),"+v"(raw[0].y),
                    "+v"(raw[1].a),"+v"(raw[1].b),"+v"(raw[1].x),"+v"(raw[1].y),
                    "+v"(raw[2].a),"+v"(raw[2].b),"+v"(raw[2].x),"+v"(raw[2].y),
                    "+v"(raw[3].a),"+v"(raw[3].b),"+v"(raw[3].x),"+v"(raw[3].y));
            }
            #pragma unroll
            for(int j=0;j<4;++j)consume<K,Coop>(raw[j],lo,hi);
        }
    }
    for(;k<end;++k)consume<K,Coop>(read_raw<K,Coop>(p,x,k,n,tn),lo,hi);
    int lane=threadIdx.x&31;
    lo+=__shfl_down(lo,2,4);lo+=__shfl_down(lo,1,4);
    hi+=__shfl_down(hi,2,4);hi+=__shfl_down(hi,1,4);
    if(!(lane&3)) {
        size_t offset=size_t(blockIdx.y)*size_t(total_rows)*size_t(n)+size_t(tn)*16+lane/4;
        out[offset]=lo;out[offset+8]=hi;
    }
}
hipError_t launch(const strix_trellis::DeviceWeights& w,const _Float16* x,float* y,
    int rows,int total_rows,int splits,int candidate,hipStream_t stream) {
    // Alignment fallback precedes every alignment assumption in read_raw.
    if((reinterpret_cast<uintptr_t>(x)&3u)||!w.trellis||!x||!y||rows!=1||total_rows<1||
       w.codebook!=2||w.input<=0||w.output<=0||w.input>5120||w.output>5120||
       w.input%16||w.output%16||splits<1||splits>16||
       !((w.bits==3&&candidate>=0&&candidate<=2)||(w.bits==4&&candidate>=3&&candidate<=4)))
        return strix_trellis::launch_gemv_lds_core(w,x,y,rows,total_rows,splits,stream);
    dim3 grid((w.output/16+3)/4,splits),block(128);
    #define RUN(K,U,C) kernel<K,U,C><<<grid,block,0,stream>>>(w.trellis,x,y,w.input,w.output,total_rows,splits)
    switch(candidate) {
    case 0:RUN(3,1,false);break;case 1:RUN(3,4,false);break;case 2:RUN(3,4,true);break;
    case 3:RUN(4,4,false);break;case 4:RUN(4,4,true);break;
    }
    #undef RUN
    return hipGetLastError();
}
template<int K,bool Coop> __global__ void reader_probe(const uint32_t* rings,uint32_t* out,int count) {
    int tile=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x&31;
    if(tile>=count)return;
    const uint32_t* base=rings+size_t(tile)*8*K;Raw r{};
    if constexpr(Coop) {
        if constexpr(K==3) {if(lane<24)r.b=base[lane];} else r.b=base[lane];
    } else {
        int index=K==3?(24*lane+755)/32-24:lane-1;
        r.a=base[index<0?index+8*K:index];r.b=base[index+1];
    }
    uint32_t s[8];states<K,Coop>(r,s);
    for(int j=0;j<8;++j)out[size_t(tile)*256+lane*8+j]=s[j]&65535u;
}
hipError_t probe(int bits,bool coop,const uint32_t* p,uint32_t* out,int count,hipStream_t stream) {
    dim3 grid((count+3)/4),block(128);
    #define RUN(K,C) reader_probe<K,C><<<grid,block,0,stream>>>(p,out,count)
    if(bits==3) {if(coop){RUN(3,true);}else{RUN(3,false);}}
    else {if(coop){RUN(4,true);}else{RUN(4,false);}}
    #undef RUN
    return hipGetLastError();
}
__global__ void hash_probe(uint32_t* hashes,uint32_t* pairs) {
    uint32_t index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=65536*8)return;
    constexpr uint32_t high[8]={0,0xffff0000,0x80000000,0x12340000,0xabcd0000,0x55550000,0xaaaa0000,0x00010000};
    uint32_t s[8],h[8];
    #pragma unroll
    for(int j=0;j<8;++j)s[j]=((index+j*8191)&65535)|high[index/65536];
    staged(s,h);
    #pragma unroll
    for(int j=0;j<8;++j)hashes[size_t(index)*8+j]=h[j];
    #pragma unroll
    for(int j=0;j<4;++j)pairs[size_t(index)*4+j]=__builtin_bit_cast(uint32_t,sad_pair(h[2*j],h[2*j+1]));
}
hipError_t probe_hash(uint32_t* hashes,uint32_t* pairs,hipStream_t stream) {
    hash_probe<<<65536*8/128,128,0,stream>>>(hashes,pairs);return hipGetLastError();
}
}
