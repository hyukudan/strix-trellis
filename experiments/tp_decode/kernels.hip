// New schedule only. No third-party implementation code is used.
#include "kernels.hpp"
#include "../../src/hadamard.hpp"
#include "../../src/packed_decoder.hpp"
#include <algorithm>
namespace tp_decode {
using Pair=_Float16 __attribute__((ext_vector_type(2)));
using strix_trellis::hadamard128;
using strix_trellis::materialize_fp32;
namespace decode=strix_trellis::packed_decode;

__device__ __forceinline__ void input_h(const Arguments& a,int r,int h,float (&v)[4]) {
    int lane=threadIdx.x&31;
    #pragma unroll
    for(int j=0;j<4;++j){int c=h*128+lane+32*j;
        float scaled=materialize_fp32(a.x[size_t(r)*a.xs+c]*__half2float(__ushort_as_half(a.w.suh[c])));
        v[j]=float(_Float16(scaled));}
    hadamard128(v);
}
__device__ __forceinline__ void output_h(const Arguments& a,int r,int h,float (&v)[4]) {
    hadamard128(v);int lane=threadIdx.x&31;
    #pragma unroll
    for(int j=0;j<4;++j){int c=h*128+lane+32*j;size_t y=size_t(r)*a.ys+c;
        float value=v[j]*__half2float(__ushort_as_half(a.w.svh[c]));
        if(a.w.bias)value+=a.w.bias[c];
        value*=a.alpha;
        if(a.beta!=0)value+=a.beta*a.y[y];
        a.y[y]=value;}
}
__global__ __launch_bounds__(128) void tp_input(Arguments a) {
    int h=blockIdx.x*4+threadIdx.x/32,nh=a.w.input/128;
    if(h>=a.rows*nh)return;int r=h/nh;h%=nh;
    float v[4];input_h(a,r,h,v);
    #pragma unroll
    for(int j=0;j<4;++j)a.a[size_t(r)*a.w.input+h*128+(threadIdx.x&31)+32*j]=_Float16(v[j]);
}
__global__ __launch_bounds__(128) void tp_output(Arguments a) {
    int h=blockIdx.x*4+threadIdx.x/32,nh=a.w.output/128;
    if(h>=a.rows*nh)return;int r=h/nh;h%=nh;float v[4]={};
    for(int s=0;s<4;++s){
        #pragma unroll
        for(int j=0;j<4;++j)v[j]+=a.p[(size_t(s)*a.rows+r)*a.w.output+h*128+(threadIdx.x&31)+32*j];}
    output_h(a,r,h,v);
}
// Only RAW words are prefetched. All VMEM remains visible to LLVM wait tracking.
struct Raw {
    uint32_t a,b,c;
    __device__ __forceinline__ static Raw load(const uint32_t* base) {
        int start=((threadIdx.x&31)*8+1)*5+1280-16,index=start/32-40;
        return {decode::SaddrLane<5>::load(reinterpret_cast<const unsigned char*>(base),index<0?index+40:index),
                decode::SaddrLane<5>::load(reinterpret_cast<const unsigned char*>(base),index+1>=40?index+1-40:index+1),
                decode::SaddrLane<5>::load(reinterpret_cast<const unsigned char*>(base),index+2>=40?index+2-40:index+2)};
    }
    template<int J> __device__ __forceinline__ uint32_t state() const {
        int shift=(((threadIdx.x&31)*8+1)*5+1280-16)&31;
        uint32_t high=(a<<shift)|(b>>(32-shift));
        uint32_t low=(b<<shift)|(c>>(32-shift));
        constexpr int offset=J*5;
        if constexpr(offset<=16)return(high>>(16-offset))&65535u;
        else if constexpr(offset<32)return((high<<(offset-16))|(low>>(48-offset)))&65535u;
        else return(low>>(48-offset))&65535u;
    }
};
template<int R,int P> __device__ __forceinline__ void pair_dot(const Raw& raw,const Pair (&x)[R][2],float (&sum)[R][2]) {
    Pair w=__builtin_bit_cast(Pair,decode::mul1_half2(raw.template state<2*P>(),raw.template state<2*P+1>()));
    #pragma unroll
    for(int r=0;r<R;++r)sum[r][P/2]=__builtin_amdgcn_fdot2(w,x[r][P&1],sum[r][P/2],false);
}
template<int R> __device__ __forceinline__ void dot(Raw raw,const _Float16* a,int stride,int k,float (&sum)[R][2]) {
    // This boundary consumes CURRENT only. LLVM still owns every VMEM dependency.
    asm volatile("" : "+v"(raw.a), "+v"(raw.b), "+v"(raw.c) : "v"(sum[0][0]), "v"(sum[0][1]));
    Pair x[R][2];int c=16*k+2*(threadIdx.x&3);
    #pragma unroll
    for(int r=0;r<R;++r){__builtin_memcpy(&x[r][0],a+size_t(r)*stride+c,4);__builtin_memcpy(&x[r][1],a+size_t(r)*stride+c+8,4);}
    pair_dot<R,0>(raw,x,sum);pair_dot<R,1>(raw,x,sum);pair_dot<R,2>(raw,x,sum);pair_dot<R,3>(raw,x,sum);
}
template<int R> __device__ __forceinline__ void publish(float (&sum)[R][2],float* p,int stride) {
    int lane=threadIdx.x&31;
    #pragma unroll
    for(int r=0;r<R;++r){
        #pragma unroll
        for(int h=0;h<2;++h){float v=sum[r][h];v+=__shfl_down(v,2,4);v+=__shfl_down(v,1,4);
            if(!(lane&3))p[size_t(r)*stride+lane/4+h*8]=v;}}
}
// A/B have exact rows and chunk8 only for R5..8. C caches whole H then crops K16.
template<int R,Variant V> __global__ __launch_bounds__(128) void tp_core(Arguments a) {
    extern __shared__ _Float16 cache[];
    int s=blockIdx.y,b=(a.w.input/16)*s/4,e=(a.w.input/16)*(s+1)/4;
    int tn=__builtin_amdgcn_readfirstlane(int(blockIdx.x)*4+int(threadIdx.x)/32);
    float sum[R][2]={};
    int hb=b/8,he=(e+7)/8;
    if constexpr(V==Variant::C) {
        int nh=he-hb;
        for(int z=threadIdx.x/32;z<R*nh;z+=4){int r=z/nh,h=hb+z%nh;float v[4];input_h(a,r,h,v);
            #pragma unroll
            for(int j=0;j<4;++j){int c=h*128+(threadIdx.x&31)+j*32;_Float16 value=_Float16(v[j]);
                cache[r*nh*128+(h-hb)*128+(threadIdx.x&31)+j*32]=value;
                if(blockIdx.x==0&&c>=b*16&&c<e*16)a.a[size_t(r)*a.w.input+c]=value;}}
        __syncthreads();
        for(int k=b;k<e;++k)dot<R>(Raw::load(a.w.trellis+(size_t(k)*(a.w.output/16)+tn)*40),cache,(he-hb)*128,k-hb*8,sum);
    } else {
        constexpr int chunk=R>=5?8:1024;
        for(int cb=b;cb<e;cb+=chunk){int ce=min(cb+chunk,e);
            const _Float16* data=a.a;int stride=a.w.input,origin=0;
            if constexpr(R>=5){
                for(int z=threadIdx.x;z<R*128;z+=128){int r=z/128,c=z%128;cache[z]=c<(ce-cb)*16?a.a[size_t(r)*a.w.input+cb*16+c]:_Float16(0);}
                __syncthreads();data=cache;stride=128;origin=cb;
            }
            if constexpr(V==Variant::B){
                int k=cb;
                Raw current=Raw::load(a.w.trellis+(size_t(k)*(a.w.output/16)+tn)*40);
                // Two-step steady state: after CURRENT is consumed, its RAW
                // registers can hold FOLLOWING while NEXT is decoded. Each
                // non-final tile has its successor issued before its own dot.
                // Uniform loop bounds avoid a conditional next/current merge.
                for(;k+2<ce;k+=2){
                    Raw next=Raw::load(a.w.trellis+(size_t(k+1)*(a.w.output/16)+tn)*40);
                    asm volatile("" ::: "memory");
                    dot<R>(current,data,stride,k-origin,sum);
                    Raw following=Raw::load(a.w.trellis+(size_t(k+2)*(a.w.output/16)+tn)*40);
                    asm volatile("" ::: "memory");
                    dot<R>(next,data,stride,k+1-origin,sum);
                    current=following;
                }
                // Guard before forming the final successor pointer; never end+1.
                if(k+1<ce){Raw next=Raw::load(a.w.trellis+(size_t(k+1)*(a.w.output/16)+tn)*40);
                    asm volatile("" ::: "memory");
                    dot<R>(current,data,stride,k-origin,sum);dot<R>(next,data,stride,k+1-origin,sum);
                } else dot<R>(current,data,stride,k-origin,sum);
            } else for(int k=cb;k<ce;++k)dot<R>(Raw::load(a.w.trellis+(size_t(k)*(a.w.output/16)+tn)*40),data,stride,k-origin,sum);
            if constexpr(R>=5)__syncthreads();
        }
    }
    publish<R>(sum,a.p+size_t(s)*R*a.w.output+tn*16,a.w.output);
}
// Eight waves per physical group; all threads reach every barrier. P is logical S.
template<int R,bool Diagnostic> __global__ __launch_bounds__(R==1?1024:512) void tp_both(Arguments a) {
    constexpr int G=R==1?4:2;
    extern __shared__ unsigned char storage[];
    float* partial=reinterpret_cast<float*>(storage);
    _Float16* cache=reinterpret_cast<_Float16*>(partial+4*R*128);
    int group=threadIdx.x/256,wave=(threadIdx.x%256)/32,lane=threadIdx.x&31;
    int maxh=0;for(int s=0;s<4;++s){int b=(a.w.input/16)*s/4,e=(a.w.input/16)*(s+1)/4;maxh=max(maxh,(e+7)/8-b/8);}
    _Float16* local=cache+group*R*maxh*128;
    for(int phase=0;phase<4/G;++phase){int s=group+G*phase,b=(a.w.input/16)*s/4,e=(a.w.input/16)*(s+1)/4;
        int hb=b/8,nh=(e+7)/8-hb;
        for(int z=wave;z<R*nh;z+=8){int r=z/nh,h=hb+z%nh;float v[4];input_h(a,r,h,v);
            #pragma unroll
            for(int j=0;j<4;++j){int c=h*128+lane+32*j;_Float16 value=_Float16(v[j]);
                local[r*nh*128+(h-hb)*128+lane+32*j]=value;
                if constexpr(Diagnostic)if(blockIdx.x==0&&c>=b*16&&c<e*16)a.a[size_t(r)*a.w.input+c]=value;}}
        __syncthreads();
        int tn=__builtin_amdgcn_readfirstlane(int(blockIdx.x)*8+wave);
        float sum[R][2]={};
        for(int k=b;k<e;++k)dot<R>(Raw::load(a.w.trellis+(size_t(k)*(a.w.output/16)+tn)*40),local,nh*128,k-hb*8,sum);
        publish<R>(sum,partial+s*R*128+wave*16,128);
        if constexpr(Diagnostic)publish<R>(sum,a.p+size_t(s)*R*a.w.output+tn*16,a.w.output);
        __syncthreads();
    }
    int row=threadIdx.x/32;
    if(row<R){float v[4]={};for(int s=0;s<4;++s){
        #pragma unroll
        for(int j=0;j<4;++j)v[j]+=partial[(s*R+row)*128+lane+j*32];}
        output_h(a,row,blockIdx.x,v);}
}
#define ROWS(M,V) M(1,V) M(2,V) M(3,V) M(4,V) M(5,V) M(6,V) M(7,V) M(8,V)
#define ENTRY(R,V) {reinterpret_cast<const void*>(tp_core<R,Variant::V>),#V "/R" #R,R,128,Variant::V},
const Function* functions(size_t& count) {
    static const Function table[]={ROWS(ENTRY,A) ROWS(ENTRY,B) ROWS(ENTRY,C)
        {reinterpret_cast<const void*>(tp_both<1,false>),"D/R1",1,1024,Variant::D},
        {reinterpret_cast<const void*>(tp_both<6,false>),"D/R6",6,512,Variant::D},
        {reinterpret_cast<const void*>(tp_both<1,true>),"DiagnosticD/R1",1,1024,Variant::DiagnosticD},
        {reinterpret_cast<const void*>(tp_both<6,true>),"DiagnosticD/R6",6,512,Variant::DiagnosticD},
        {reinterpret_cast<const void*>(tp_input),"input",0,128,Variant::Canonical},
        {reinterpret_cast<const void*>(tp_output),"output",0,128,Variant::Canonical}};
    count=sizeof(table)/sizeof(*table);return table;
}
#undef ENTRY
#undef ROWS
size_t shared_bytes(const Function& f,int i) {
    int maxh=0;for(int s=0;s<4;++s)maxh=std::max(maxh,((i/16)*(s+1)/4+7)/8-((i/16)*s/4)/8);
    if(f.variant==Variant::C)return 2*f.rows*128*maxh;
    if(f.variant==Variant::D||f.variant==Variant::DiagnosticD)return 4*4*f.rows*128+(f.rows==1?4:2)*2*f.rows*128*maxh;
    if(f.variant==Variant::A||f.variant==Variant::B)return f.rows>=5?256*f.rows:0;
    return 0;
}
hipError_t enqueue(const Arguments& a,Variant v,hipStream_t stream) {
    if(!eligible(v,a.w.input,a.w.output,a.rows))return hipErrorInvalidValue;
    if(v==Variant::Old)return strix_trellis::launch_linear(a.w,a.x,a.y,a.rows,a.a,stream,a.alpha,a.beta,false);
    if(v!=Variant::C&&v!=Variant::D&&v!=Variant::DiagnosticD){
        tp_input<<<(a.rows*(a.w.input/128)+3)/4,128,0,stream>>>(a);
        auto err=hipGetLastError();if(err!=hipSuccess)return err;
    }
    if(v==Variant::Canonical){auto err=strix_trellis::baseline_core(a.w,a.a,a.p,a.rows,a.rows,4,stream);if(err!=hipSuccess)return err;}
    else {size_t n;auto table=functions(n);const Function* selected=nullptr;
        for(size_t j=0;j<n;++j)if(table[j].variant==v&&table[j].rows==a.rows)selected=table+j;
        if(!selected)return hipErrorInvalidValue;
        void* args[]={const_cast<Arguments*>(&a)};
        auto err=hipLaunchKernel(selected->pointer,dim3(a.w.output/(v==Variant::D||v==Variant::DiagnosticD?128:64),v==Variant::D||v==Variant::DiagnosticD?1:4),dim3(selected->threads),args,shared_bytes(*selected,a.w.input),stream);
        if(err!=hipSuccess)return err;
    }
    if(v!=Variant::D&&v!=Variant::DiagnosticD){tp_output<<<(a.rows*(a.w.output/128)+3)/4,128,0,stream>>>(a);return hipGetLastError();}
    return hipSuccess;
}
} // namespace tp_decode
