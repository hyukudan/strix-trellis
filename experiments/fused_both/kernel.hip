// Private independent composition of the reviewed R1 H4 ownership map.
#include "kernel.hpp"
#include "packed_decoder.hpp"
#include "hadamard.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

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

static size_t dynamic_bytes(int input){return size_t(4)*((input/128+3)/4)*128*2;}
hipError_t guard(const Request& q) {
    const auto& w=q.weights;
    if(q.rows!=1||q.groups!=4||w.codebook!=2||(w.bits!=5&&w.bits!=6)||
       w.input<=0||w.input>8192||w.output<=0||w.output>131072||w.input%128||w.output%128||
       !std::isfinite(q.alpha)||!std::isfinite(q.beta)||
       (q.mode!=Mode::GlobalPartial&&q.mode!=Mode::Direct&&q.mode!=Mode::Diagnostic))
        return hipErrorInvalidValue;
    const void* ptrs[]={w.trellis,q.x,w.suh,w.svh,w.bias,q.y,q.partial,q.debug_a,q.debug_h};
    size_t sizes[]={size_t(w.input)*w.output*w.bits/8,size_t(w.input)*4,size_t(w.input)*2,
        size_t(w.output)*2,w.bias?size_t(w.output)*4:0,size_t(w.output)*4,
        q.mode==Mode::Direct?0:size_t(4)*w.output*4,
        q.mode==Mode::Diagnostic&&q.debug_a?size_t(w.input)*2:0,
        q.mode==Mode::Diagnostic&&q.debug_h?size_t(w.input)*4:0};
    size_t align[]={4,4,2,2,4,4,4,2,4};
    for(int i=0;i<9;++i) {
        if(!sizes[i]) {
            if(ptrs[i]||q.regions[i].data||q.regions[i].bytes)return hipErrorInvalidValue;
            continue;
        }
        uintptr_t start=reinterpret_cast<uintptr_t>(ptrs[i]);
        if(!start||start%align[i]||q.regions[i].data!=ptrs[i]||q.regions[i].bytes<sizes[i]||
           q.regions[i].bytes>std::numeric_limits<uintptr_t>::max()-start)return hipErrorInvalidValue;
        for(int j=0;j<i;++j)if(sizes[j]) {
            uintptr_t other=reinterpret_cast<uintptr_t>(ptrs[j]);
            if(start<other+q.regions[j].bytes&&other<start+q.regions[i].bytes)return hipErrorInvalidValue;
        }
    }
    return hipSuccess;
}
template<int B,Mode M> static hipError_t resource_one(const hipDeviceProp_t& p,size_t dynamic) {
    hipFuncAttributes a{};
    auto e=hipFuncGetAttributes(&a,reinterpret_cast<const void*>(both<B,M>));
    if(e!=hipSuccess)return e;
    std::printf("RESOURCE bits=%d mode=%d maxThreads=%d shared=%zu dynamic_limit=%d requested=%zu regs=%d local=%zu\n",
        B,int(M),a.maxThreadsPerBlock,a.sharedSizeBytes,a.maxDynamicSharedSizeBytes,dynamic,a.numRegs,a.localSizeBytes);
    if(a.maxThreadsPerBlock<1024||a.localSizeBytes||a.sharedSizeBytes>p.sharedMemPerBlock||
       dynamic>p.sharedMemPerBlock-a.sharedSizeBytes||a.maxDynamicSharedSizeBytes<0||
       dynamic>size_t(a.maxDynamicSharedSizeBytes))return hipErrorInvalidConfiguration;
    return hipSuccess;
}
static hipError_t device(int& id,hipDeviceProp_t& p) {
    auto e=hipGetDevice(&id);if(e!=hipSuccess)return e;
    e=hipGetDeviceProperties(&p,id);if(e!=hipSuccess)return e;
    if(p.warpSize!=32||std::strncmp(p.gcnArchName,"gfx1151",7)||
       (p.gcnArchName[7]!=0&&p.gcnArchName[7]!=':')||p.maxThreadsPerBlock<1024||
       p.maxThreadsDim[0]<1024)return hipErrorInvalidConfiguration;
    return hipSuccess;
}
hipError_t resources() {
    int id;hipDeviceProp_t p{};auto e=device(id,p);if(e!=hipSuccess)return e;
    std::printf("DEVICE name=%s arch=%s wave=%d shared=%zu\n",p.name,p.gcnArchName,p.warpSize,p.sharedMemPerBlock);
    #define CHECK(B,M) e=resource_one<B,Mode::M>(p,16384);if(e!=hipSuccess)return e
    CHECK(5,GlobalPartial);CHECK(5,Direct);CHECK(5,Diagnostic);
    CHECK(6,GlobalPartial);CHECK(6,Direct);CHECK(6,Diagnostic);
    #undef CHECK
    return hipSuccess;
}
template<int B> static hipError_t selected(const Request& q,const hipDeviceProp_t& p,size_t d) {
    if(q.mode==Mode::GlobalPartial)return resource_one<B,Mode::GlobalPartial>(p,d);
    if(q.mode==Mode::Direct)return resource_one<B,Mode::Direct>(p,d);
    return resource_one<B,Mode::Diagnostic>(p,d);
}
hipError_t prepare(const Request& q,Token& out) {
    out=Token{};
    auto e=guard(q);if(e!=hipSuccess)return e;
    int id;hipDeviceProp_t p{};e=device(id,p);if(e!=hipSuccess)return e;
    if(q.weights.output/128>p.maxGridSize[0]||p.maxThreadsDim[1]<1||p.maxThreadsDim[2]<1)
        return hipErrorInvalidConfiguration;
    if(q.stream) {
        int stream_device=-1;unsigned flags=0;
        e=hipStreamGetDevice(q.stream,&stream_device);if(e!=hipSuccess)return e;
        e=hipStreamGetFlags(q.stream,&flags);if(e!=hipSuccess)return e;
        if(stream_device!=id)return hipErrorInvalidValue;
    }
    for(auto r:q.regions)if(r.bytes) {
        hipPointerAttribute_t a{};e=hipPointerGetAttributes(&a,r.data);if(e!=hipSuccess)return e;
        if(a.type!=hipMemoryTypeDevice||a.isManaged||a.device!=id)return hipErrorInvalidValue;
        void* base=nullptr;size_t bytes=0;
        e=hipMemGetAddressRange(&base,&bytes,const_cast<void*>(r.data));if(e!=hipSuccess)return e;
        auto start=reinterpret_cast<uintptr_t>(r.data),begin=reinterpret_cast<uintptr_t>(base);
        if(start<begin||start-begin>bytes||r.bytes>bytes-(start-begin))return hipErrorInvalidValue;
    }
    size_t d=dynamic_bytes(q.weights.input);
    e=q.weights.bits==5?selected<5>(q,p,d):selected<6>(q,p,d);if(e!=hipSuccess)return e;
    out.request_=q;out.dynamic_=d;out.ready_=true;return hipSuccess;
}
template<int B> static void dispatch(const Request& q,size_t d) {
    const auto& w=q.weights;
    #define RUN(M) both<B,Mode::M><<<w.output/128,1024,d,q.stream>>>(w.trellis,q.x,w.suh,w.svh,w.bias,q.y,q.partial,q.debug_a,q.debug_h,w.input,w.output,q.alpha,q.beta)
    if(q.mode==Mode::GlobalPartial){RUN(GlobalPartial);}
    else if(q.mode==Mode::Direct){RUN(Direct);}else{RUN(Diagnostic);}
    #undef RUN
}
hipError_t submit(const Token& token) {
    if(!token.ready_)return hipErrorInvalidValue;
    if(token.request_.weights.bits==5)dispatch<5>(token.request_,token.dynamic_);
    else dispatch<6>(token.request_,token.dynamic_);
    return hipGetLastError();
}
}
