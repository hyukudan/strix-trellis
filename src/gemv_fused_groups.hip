// Independently implemented from the reviewed G4 ownership and operation maps.
#include "gemv_fused_groups.hpp"
#include "packed_decoder.hpp"
#include "hadamard.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace strix_trellis::fused_groups {
using Pair = _Float16 __attribute__((ext_vector_type(2)));
template<int Bits,int Groups,Mode M>
__global__ __launch_bounds__(256*Groups) void grouped(const uint32_t* __restrict__ packed,
    const _Float16* __restrict__ a,const uint16_t* __restrict__ svh,
    const float* __restrict__ bias,float* __restrict__ y,float* __restrict__ dump,
    int input,int output,float alpha,float beta) {
    __shared__ uint32_t activation[Groups][512];
    __shared__ float partial[Groups][8][128];
    const int lane=threadIdx.x&31, local=threadIdx.x&255;
    const int wave=__builtin_amdgcn_readfirstlane(int(threadIdx.x)/32);
    const int g=__builtin_amdgcn_readfirstlane(wave/8);
    const int nw=__builtin_amdgcn_readfirstlane(wave%8);
    const int tile=__builtin_amdgcn_readfirstlane(int(blockIdx.x)*8+nw);
    const int length=(input/16)/Groups, end=(g+1)*length;
    float acc[8][2]={};
    for(int base=g*length;base<end;base+=8) {
        #pragma unroll
        for(int pass=0;pass<2;++pass) {
            int q=local+256*pass, r=q/64, p=q%64;
            if(base+p/8<end) {
                uint32_t word;
                __builtin_memcpy(&word,a+size_t(r)*input+2*(base*8+p),4);
                activation[g][q]=word;
            }
        }
        __syncthreads();
        for(int kk=0;kk<8 && base+kk<end;++kk) {
            packed_decode::SaddrLane<Bits> reader(packed+
                (size_t(base+kk)*(output/16)+tile)*Bits*8);
            Pair av[8][2];
            #pragma unroll
            for(int r=0;r<8;++r) {
                av[r][0]=__builtin_bit_cast(Pair,activation[g][r*64+kk*8+(lane&3)]);
                av[r][1]=__builtin_bit_cast(Pair,activation[g][r*64+kk*8+(lane&3)+4]);
            }
            #pragma unroll
            for(int p=0;p<4;++p) {
                uint32_t first,second;
                if(p==0){first=reader.template state<0>();second=reader.template state<1>();}
                else if(p==1){first=reader.template state<2>();second=reader.template state<3>();}
                else if(p==2){first=reader.template state<4>();second=reader.template state<5>();}
                else {first=reader.template state<6>();second=reader.template state<7>();}
                Pair weight=__builtin_bit_cast(Pair,packed_decode::mul1_half2(first,second));
                #pragma unroll
                for(int r=0;r<8;++r)
                    acc[r][p/2]=__builtin_amdgcn_fdot2(weight,av[r][p&1],acc[r][p/2],false);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<8;++r) {
        #pragma unroll
        for(int h=0;h<2;++h) {
            float v=acc[r][h];
            v+=__shfl_down(v,2,4);v+=__shfl_down(v,1,4);
            if((lane&3)==0) {
                int c=nw*16+lane/4+h*8;
                if constexpr(M!=Mode::GlobalPartial) partial[g][r][c]=v;
                if constexpr(M!=Mode::Direct)
                    dump[(size_t(g)*8+r)*output+size_t(blockIdx.x)*128+c]=v;
            }
        }
    }
    // Every thread publishes before any wave may leave the CTA.
    __syncthreads();
    if constexpr(M!=Mode::GlobalPartial) {
        if(wave>=8)return;
        float values[4]={};
        // Deliberately preserve the +0 origin and ascending split association.
        for(int split=0;split<Groups;++split) {
            #pragma unroll
            for(int j=0;j<4;++j) values[j]+=partial[split][wave][lane+32*j];
        }
        hadamard128(values);
        #pragma unroll
        for(int j=0;j<4;++j) {
            size_t c=size_t(blockIdx.x)*128+lane+32*j;
            size_t index=size_t(wave)*output+c;
            float value=values[j]*__half2float(__ushort_as_half(svh[c]));
            if(bias)value+=bias[c];
            value*=alpha;
            if(beta!=0)value+=beta*y[index];
            y[index]=value;
        }
    }
}

hipError_t guard(const Request& q) {
    const auto& w=q.weights;
    if(q.rows!=8||(q.groups!=1&&q.groups!=2&&q.groups!=4)||w.codebook!=2||(w.bits!=5&&w.bits!=6)||
       w.input<=0||w.input>8192||w.output<=0||w.output>131072||w.input%128||w.output%128||
       !std::isfinite(q.alpha)||!std::isfinite(q.beta)||
       (q.mode!=Mode::GlobalPartial&&q.mode!=Mode::Direct&&q.mode!=Mode::Diagnostic))
        return hipErrorInvalidValue;
    // Bounded products: packed <=805306368, P <=16777216 bytes.
    const void* ptrs[]={w.trellis,q.a,w.svh,w.bias,q.y,q.partial};
    size_t sizes[]={size_t(w.input)*w.output*w.bits/8,size_t(8)*w.input*2,
        size_t(w.output)*2,w.bias?size_t(w.output)*4:0,size_t(8)*w.output*4,
        q.mode==Mode::Direct?0:size_t(q.groups)*8*w.output*4};
    size_t align[]={4,4,2,4,4,4};
    for(int i=0;i<6;++i) {
        if(!sizes[i]) {if(ptrs[i]||q.regions[i].data||q.regions[i].bytes)return hipErrorInvalidValue;continue;}
        uintptr_t start=reinterpret_cast<uintptr_t>(ptrs[i]);
        if(!start||start%align[i]||q.regions[i].data!=ptrs[i]||q.regions[i].bytes!=sizes[i]||
           sizes[i]>std::numeric_limits<uintptr_t>::max()-start)return hipErrorInvalidValue;
        for(int j=0;j<i;++j) if(sizes[j]) {
            uintptr_t other=reinterpret_cast<uintptr_t>(ptrs[j]);
            if(start<other+sizes[j]&&other<start+sizes[i])return hipErrorInvalidValue;
        }
    }
    return hipSuccess;
}
template<int B,int G,Mode M> static hipError_t resource_one(const hipDeviceProp_t& p) {
    hipFuncAttributes a{};
    hipError_t e=hipFuncGetAttributes(&a,reinterpret_cast<const void*>(grouped<B,G,M>));
    if(e!=hipSuccess)return e;
    std::printf("RESOURCE bits=%d groups=%d mode=%d maxThreads=%d shared=%zu regs=%d local=%zu\n",
        B,G,int(M),a.maxThreadsPerBlock,a.sharedSizeBytes,a.numRegs,a.localSizeBytes);
    if(a.maxThreadsPerBlock<256*G||a.sharedSizeBytes>p.sharedMemPerBlock||a.localSizeBytes)
        return hipErrorInvalidConfiguration;
    return hipSuccess;
}
hipError_t resources() {
    int device;hipError_t e=hipGetDevice(&device);if(e!=hipSuccess)return e;
    hipDeviceProp_t p{};e=hipGetDeviceProperties(&p,device);if(e!=hipSuccess)return e;
    std::printf("DEVICE name=%s arch=%s wave=%d maxThreads=%d maxDim=%d shared=%zu\n",
        p.name,p.gcnArchName,p.warpSize,p.maxThreadsPerBlock,p.maxThreadsDim[0],p.sharedMemPerBlock);
    if(p.warpSize!=32||std::strncmp(p.gcnArchName,"gfx1151",7)||p.maxThreadsPerBlock<1024||
       p.maxThreadsDim[0]<1024||p.sharedMemPerBlock<24576)return hipErrorInvalidConfiguration;
    #define CHECK(B,G,M) e=resource_one<B,G,Mode::M>(p);if(e!=hipSuccess)return e
    #define GROUP(G) CHECK(5,G,GlobalPartial);CHECK(5,G,Direct);CHECK(5,G,Diagnostic);CHECK(6,G,GlobalPartial);CHECK(6,G,Direct);CHECK(6,G,Diagnostic)
    GROUP(1);GROUP(2);GROUP(4);
    #undef GROUP
    #undef CHECK
    return hipSuccess;
}
hipError_t prepare(const Request& q,Token& out) {
    out=Token{};
    hipError_t e=guard(q);if(e!=hipSuccess)return e;
    int device;e=hipGetDevice(&device);if(e!=hipSuccess)return e;
    unsigned flags;e=hipStreamGetFlags(q.stream,&flags);if(e!=hipSuccess)return e;
    for(auto region:q.regions) if(region.bytes) {
        hipPointerAttribute_t a{};e=hipPointerGetAttributes(&a,region.data);if(e!=hipSuccess)return e;
        if(a.type!=hipMemoryTypeDevice||a.isManaged||a.device!=device)return hipErrorInvalidValue;
        void* base=nullptr;size_t bytes=0;
        e=hipMemGetAddressRange(&base,&bytes,region.data);if(e!=hipSuccess)return e;
        uintptr_t start=reinterpret_cast<uintptr_t>(region.data),begin=reinterpret_cast<uintptr_t>(base);
        if(start<begin||start-begin>bytes||region.bytes>bytes-(start-begin))return hipErrorInvalidValue;
    }
    out.request_=q;out.ready_=true;return hipSuccess;
}
template<int B,int G> static void dispatch(const Request& q) {
    const auto& w=q.weights;
    #define RUN(M) grouped<B,G,Mode::M><<<w.output/128,256*G,0,q.stream>>>(w.trellis,q.a,w.svh,w.bias,q.y,q.partial,w.input,w.output,q.alpha,q.beta)
    if(q.mode==Mode::GlobalPartial){RUN(GlobalPartial);}
    else if(q.mode==Mode::Direct){RUN(Direct);}else{RUN(Diagnostic);}
    #undef RUN
}
hipError_t submit(const Token& token) {
    if(!token.ready_)return hipErrorInvalidValue;
    const auto& q=token.request_;
    #define GROUP(G) if(q.weights.bits==5)dispatch<5,G>(q);else dispatch<6,G>(q)
    if(q.groups==1){GROUP(1);}else if(q.groups==2){GROUP(2);}else{GROUP(4);}
    #undef GROUP
    return hipGetLastError();
}
}
