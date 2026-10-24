// Independent optional original-F32 input fusion; no backend policy changes.
#include "gemv_fused.hpp"
#include "hadamard.hpp"
#include "trellis_optimized.hpp"
namespace strix_trellis {
using FusedPair = _Float16 __attribute__((ext_vector_type(2)));
// Optional completion (R1, Complete=true): the last of the 2*splits CTAs that write one 128-column group folds
// that group's partials and applies the output transform, with the same code and order as output_transform
// (ascending splits from +0, H128, svh scale, bias, alpha, beta). counters: one per group, zero between launches.
// Completion counters are kFusedCounterStride words apart (one 128-byte line each).
constexpr int kFusedCounterStride = 32;
struct FusedCompletion {
    const uint16_t* svh=nullptr;const float* bias=nullptr;float* output=nullptr;uint32_t* counters=nullptr;
    float alpha=1,beta=0;
};
template<int Bits, bool Mul1, bool Debug, bool Complete=false>
__global__ void fused_input_kernel(const uint32_t* __restrict__ packed,
    const float* __restrict__ original,const uint16_t* __restrict__ suh,
    float* __restrict__ partial,int input,int output,int total_rows,int splits,
    int codebook,_Float16* debug_a,float* debug_h,FusedCompletion done={}) {
    extern __shared__ _Float16 cache[];
    const int wave=threadIdx.x/32,lane=threadIdx.x&31;
    const int beginH=(input/128)*blockIdx.y/splits;
    const int endH=(input/128)*(blockIdx.y+1)/splits;
    for(int h=beginH+wave;h<endH;h+=4) {
        float values[4];
        #pragma unroll
        for(int j=0;j<4;++j) {
            int channel=h*128+lane+32*j;
            float scaled=materialize_fp32(original[channel]*__half2float(__ushort_as_half(suh[channel])));
            values[j]=float(_Float16(scaled));
        }
        hadamard128(values);
        #pragma unroll
        for(int j=0;j<4;++j) {
            int local=(h-beginH)*128+lane+32*j;
            cache[local]=_Float16(values[j]);
            if constexpr(Debug) if(blockIdx.x==0) {
                if(debug_a) debug_a[beginH*128+local]=_Float16(values[j]);
                if(debug_h) debug_h[beginH*128+local]=values[j];
            }
        }
    }
    __syncthreads(); // All four waves, including empty H chunks.
    const int tile_n=blockIdx.x*4+wave;
    const int begin=beginH*8,end=endH*8,rows=1;
    float sums[1][2]={};
    for(int k=begin;k<end;++k) {
        optimized::AlignedLane<Bits,true> tile(packed+(size_t(k)*(output/16)+tile_n)*Bits*8);
        // Each activation pair is shared by the low and high output columns.
        FusedPair activations[1][2];
        #pragma unroll
        for(int r=0;r<1;++r) if(r<rows) {
            int channel=k*16+2*(lane&3);
            __builtin_memcpy(&activations[r][0],cache+channel-begin*16,sizeof(FusedPair));
            __builtin_memcpy(&activations[r][1],cache+channel-begin*16+8,sizeof(FusedPair));
        }
        #pragma unroll
        for(int pair=0;pair<4;++pair) {
            uint32_t a,b;
            // Explicit instantiations keep bit offsets compile-time constants.
            if(pair==0) {a=tile.template state<0>();b=tile.template state<1>();}
            else if(pair==1) {a=tile.template state<2>();b=tile.template state<3>();}
            else if(pair==2) {a=tile.template state<4>();b=tile.template state<5>();}
            else {a=tile.template state<6>();b=tile.template state<7>();}
            FusedPair w;
            if constexpr(Mul1) {
                __half2 decoded=optimized::mul1_half2(a,b);
                w=__builtin_bit_cast(FusedPair,decoded);
            } else {w[0]=_Float16(codebook_value(a,codebook));w[1]=_Float16(codebook_value(b,codebook));}
            #pragma unroll
            for(int r=0;r<1;++r) if(r<rows)
                sums[r][pair/2]=__builtin_amdgcn_fdot2(w,activations[r][pair&1],sums[r][pair/2],false);
        }
    }
    #pragma unroll
    for(int h=0;h<2;++h) {
        float v=sums[0][h];v+=__shfl_down(v,2,4);v+=__shfl_down(v,1,4);
        if(!(lane&3)) partial[size_t(blockIdx.y)*total_rows*output+tile_n*16+lane/4+h*8]=v;
    }
    if constexpr(Complete) {
        __shared__ bool last;
        // Every writing wave releases its partials before the CTA's single arrival.
        __builtin_amdgcn_fence(__ATOMIC_RELEASE,"agent");
        __syncthreads();
        const int group=blockIdx.x/2;
        uint32_t* counter=done.counters+size_t(group)*kFusedCounterStride;
        if(!threadIdx.x) {
            const auto old=__hip_atomic_fetch_add(counter,1u,__ATOMIC_ACQ_REL,__HIP_MEMORY_SCOPE_AGENT);
            last=old==unsigned(2*splits-1);
            if(last) __hip_atomic_store(counter,0u,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
        }
        __syncthreads();
        if(!last) return;
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent");
        // output_transform for row 0 of this group. Each of the 128 threads folds one column: the same sum as
        // output_transform's (from +0, ascending splits), with the loads issued eight splits at a time.
        __shared__ float folded[128];
        const int begin=group*128;
        {
            const float* column=partial+begin+threadIdx.x;
            const size_t stride=size_t(total_rows)*output;
            float sum=0;
            for(int s0=0;s0<splits;s0+=8) {
                float batch[8];
                #pragma unroll
                for(int k=0;k<8;++k) batch[k]=s0+k<splits ? column[size_t(s0+k)*stride] : 0.f;
                #pragma unroll
                for(int k=0;k<8;++k) if(s0+k<splits) sum+=batch[k];
            }
            folded[threadIdx.x]=sum;
        }
        __syncthreads();
        if(wave) return;
        float values[4];
        #pragma unroll
        for(int j=0;j<4;++j) values[j]=folded[lane+32*j];
        hadamard128(values);
        #pragma unroll
        for(int j=0;j<4;++j) {
            const int channel=begin+lane+32*j;
            float value=values[j]*__half2float(__ushort_as_half(done.svh[channel]));
            if(done.bias) value+=done.bias[channel];
            value*=done.alpha;
            if(done.beta!=0) value+=done.beta*done.output[channel];
            done.output[channel]=value;
        }
    }
}
template<int Bits,bool Mul1,bool Debug>
static hipError_t run(const DeviceWeights& w,const float* x,float* y,int total,int splits,
    hipStream_t stream,_Float16* a,float* h) {
    // ceil(H/splits) blocks, exactly 16 KiB at I8192/split1.
    size_t lds=size_t((w.input/128+splits-1)/splits)*128*2;
    fused_input_kernel<Bits,Mul1,Debug><<<dim3(w.output/64,splits),128,lds,stream>>>(
        w.trellis,x,w.suh,y,w.input,w.output,total,splits,w.codebook,a,h);
    return hipGetLastError();
}
template<bool Debug>
static hipError_t dispatch(const DeviceWeights& w,const float* x,float* y,int rows,
    int total,int splits,hipStream_t stream,_Float16* a,float* h) {
    if(!w.trellis||!w.suh||!x||!y||rows!=1||total<rows||splits<1||splits>65535||
        w.input<=0||w.input>8192||w.output<=0||w.input%128||w.output%128||
        w.codebook<0||w.codebook>2) return hipErrorInvalidValue;
    #define CASE(B) case B:return w.codebook==2?run<B,true,Debug>(w,x,y,total,splits,stream,a,h):run<B,false,Debug>(w,x,y,total,splits,stream,a,h)
    switch(w.bits) {CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);default:return hipErrorInvalidValue;}
    #undef CASE
}
hipError_t launch_gemv_fused_input(const DeviceWeights& w,const float* x,float* y,
    int rows,int total,int splits,hipStream_t stream) {
    return dispatch<false>(w,x,y,rows,total,splits,stream,nullptr,nullptr);
}
hipError_t launch_gemv_fused_input_debug(const DeviceWeights& w,const float* x,float* y,
    int rows,int total,int splits,hipStream_t stream,_Float16* a,float* h) {
    return dispatch<true>(w,x,y,rows,total,splits,stream,a,h);
}
#ifdef STRIX_TRELLIS_PLAN_FUSED_COMPLETION
template<int Bits,bool Mul1>
static hipError_t run_complete(const DeviceWeights& w,const float* x,float* partial,int splits,
    hipStream_t stream,const FusedCompletion& done) {
    size_t lds=size_t((w.input/128+splits-1)/splits)*128*2;
    fused_input_kernel<Bits,Mul1,false,true><<<dim3(w.output/64,splits),128,lds,stream>>>(
        w.trellis,x,w.suh,partial,w.input,w.output,1,splits,w.codebook,nullptr,nullptr,done);
    return hipGetLastError();
}
hipError_t launch_gemv_fused_complete(const DeviceWeights& w,const float* x,float* partial,float* y,
    uint32_t* counters,int splits,hipStream_t stream,float alpha,float beta) {
    if(!w.trellis||!w.suh||!w.svh||!x||!partial||!y||!counters||splits<1||splits>65535||
        w.input<=0||w.input>8192||w.output<=0||w.input%128||w.output%128||
        w.codebook<0||w.codebook>2) return hipErrorInvalidValue;
    FusedCompletion done;
    done.svh=w.svh;done.bias=w.bias;done.output=y;done.counters=counters;done.alpha=alpha;done.beta=beta;
    #define CASE(B) case B:return w.codebook==2?run_complete<B,true>(w,x,partial,splits,stream,done):run_complete<B,false>(w,x,partial,splits,stream,done)
    switch(w.bits) {CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);default:return hipErrorInvalidValue;}
    #undef CASE
}
#endif
}
