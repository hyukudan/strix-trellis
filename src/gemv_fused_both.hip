// Literal independent prototype b79f058; see reports/runtime-v15-candidate.md.
#include "gemv_fused_both.hpp"
#include "fused_both_policy.hpp"
#include "packed_decoder.hpp"
#include "hadamard.hpp"
namespace strix_trellis::fused_both {
using Pair = _Float16 __attribute__((ext_vector_type(2)));
static_assert(sizeof(Pair)==4);
template<int Bits,Mode M>
__global__ __launch_bounds__(1024) void both(const uint32_t* __restrict__ packed,
    const float* __restrict__ original,const uint16_t* __restrict__ suh,
    const uint16_t* __restrict__ svh,const float* __restrict__ bias,
    float* __restrict__ y,float* __restrict__ dump,_Float16* debug_a,
    float* debug_h,int input,int output,float alpha,float beta) {
    extern __shared__ _Float16 cache[] __attribute__((aligned(16)));
    __shared__ float partial[4][128];
    const int lane=threadIdx.x&31;
    const int wave=__builtin_amdgcn_readfirstlane(int(threadIdx.x)/32);
    const int g=__builtin_amdgcn_readfirstlane(wave/8);
    const int nw=__builtin_amdgcn_readfirstlane(wave%8);
    const int tile=__builtin_amdgcn_readfirstlane(int(blockIdx.x)*8+nw);
    const int H=input/128, stride=((H+3)/4)*128;
    const int Hbegin=H*g/4, Hend=H*(g+1)/4;
    const int kbegin=8*Hbegin, kend=8*Hend;
    for(int h=Hbegin+nw;h<Hend;h+=8) {
        float v[4];
        #pragma unroll
        for(int j=0;j<4;++j) {
            int c=h*128+lane+32*j;
            float scaled=materialize_fp32(original[c]*__half2float(__ushort_as_half(suh[c])));
            v[j]=float(_Float16(scaled));
        }
        hadamard128(v);
        #pragma unroll
        for(int j=0;j<4;++j) {
            int local=(h-Hbegin)*128+lane+32*j;
            cache[g*stride+local]=_Float16(v[j]);
            if constexpr(M==Mode::Diagnostic) if(blockIdx.x==0) {
                if(debug_a)debug_a[h*128+lane+32*j]=_Float16(v[j]);
                if(debug_h)debug_h[h*128+lane+32*j]=v[j];
            }
        }
    }
    __syncthreads(); // B1: all groups, including empty partitions.
    float acc[2]={};
    for(int k=kbegin;k<kend;++k) {
        packed_decode::SaddrLane<Bits> reader(packed+
            (size_t(k)*(output/16)+tile)*Bits*8);
        Pair av[2];
        int local=(k-kbegin)*16+2*(lane&3);
        __builtin_memcpy(&av[0],cache+g*stride+local,4);
        __builtin_memcpy(&av[1],cache+g*stride+local+8,4);
        #pragma unroll
        for(int p=0;p<4;++p) {
            uint32_t a,b;
            if(p==0){a=reader.template state<0>();b=reader.template state<1>();}
            else if(p==1){a=reader.template state<2>();b=reader.template state<3>();}
            else if(p==2){a=reader.template state<4>();b=reader.template state<5>();}
            else {a=reader.template state<6>();b=reader.template state<7>();}
            Pair weight=__builtin_bit_cast(Pair,packed_decode::mul1_half2(a,b));
            acc[p/2]=__builtin_amdgcn_fdot2(weight,av[p&1],acc[p/2],false);
        }
    }
    #pragma unroll
    for(int h=0;h<2;++h) {
        float v=acc[h];v+=__shfl_down(v,2,4);v+=__shfl_down(v,1,4);
        if((lane&3)==0) {
            int c=16*nw+lane/4+8*h;
            partial[g][c]=v;
            if constexpr(M!=Mode::Direct)
                dump[size_t(g)*output+size_t(blockIdx.x)*128+c]=v;
        }
    }
    __syncthreads(); // B2: publish P before any wave returns.
    if constexpr(M!=Mode::GlobalPartial) {
        if(wave!=0)return;
        float v[4]={};
        for(int split=0;split<4;++split) {
            #pragma unroll
            for(int j=0;j<4;++j)v[j]+=partial[split][lane+32*j];
        }
        hadamard128(v);
        #pragma unroll
        for(int j=0;j<4;++j) {
            size_t c=size_t(blockIdx.x)*128+lane+32*j;
            float value=v[j]*__half2float(__ushort_as_half(svh[c]));
            if(bias)value+=bias[c];
            value*=alpha;
            if(beta!=0)value+=beta*y[c];
            y[c]=value;
        }
    }
}

const void* direct_function() { return reinterpret_cast<const void*>(both<5,Mode::Direct>); }
}
namespace strix_trellis {
hipError_t query_fused_both_resources(size_t shared,bool* supported) {
    return probe_fused_both_image<hipError_t,hipFuncAttributes>(shared,supported,
        hipSuccess,hipErrorInvalidValue,fused_both::direct_function(),hipFuncGetAttributes);
}
hipError_t launch_fused_both_trusted(const DeviceWeights& w,const float* x,float* y,
    int rows,bool prefer_wmma,hipStream_t stream,float alpha,float beta) {
    FusedBothLayout layout{};
    if(!fused_both_layout(w.input,w.output,w.bits,w.codebook,rows,prefer_wmma,
        w.fused_both_capable,w.trellis,x,w.suh,w.svh,w.bias,y,layout))
        return hipErrorInvalidValue;
    fused_both::both<5,fused_both::Mode::Direct><<<layout.blocks,1024,layout.dynamic_bytes,stream>>>(
        w.trellis,x,w.suh,w.svh,w.bias,y,nullptr,nullptr,nullptr,w.input,w.output,alpha,beta);
    return hipGetLastError();
}
}
