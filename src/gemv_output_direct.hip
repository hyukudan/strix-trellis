// Literal extraction of the independently implemented and reviewed G4 kernel.
#include "gemv_output_direct.hpp"
#include "fused_output_policy.hpp"
#include "fused_output_resources.hpp"
#include "packed_decoder.hpp"
#include "hadamard.hpp"
#include <cmath>

namespace strix_trellis::output_direct {
enum class Mode { GlobalPartial, Direct, Diagnostic };
using Pair = _Float16 __attribute__((ext_vector_type(2)));
template<int Bits,Mode M>
__global__ __launch_bounds__(1024) void g4(const uint32_t* __restrict__ packed,
    const _Float16* __restrict__ a,const uint16_t* __restrict__ svh,
    const float* __restrict__ bias,float* __restrict__ y,float* __restrict__ dump,
    int input,int output,float alpha,float beta) {
    __shared__ uint32_t activation[4][512];
    __shared__ float partial[4][8][128];
    const int lane=threadIdx.x&31, local=threadIdx.x&255;
    const int wave=__builtin_amdgcn_readfirstlane(int(threadIdx.x)/32);
    const int g=__builtin_amdgcn_readfirstlane(wave/8);
    const int nw=__builtin_amdgcn_readfirstlane(wave%8);
    const int tile=__builtin_amdgcn_readfirstlane(int(blockIdx.x)*8+nw);
    const int length=(input/16)/4, end=(g+1)*length;
    float acc[8][2]={};
#ifdef STRIX_TRELLIS_DIRECT_PREFETCH
    // Prefetch: the packed words of the next K16 row are requested before the current row is
    // consumed (the first one before the first activation staging); the window is formed at
    // use. Same addresses, same order of every dot product; only the load issue point moves.
    auto next=packed_decode::SaddrLane<Bits>::issue(packed+(size_t(g*length)*(output/16)+tile)*Bits*8);
#endif
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
        // Same tile addresses as (base+kk)*(output/16)+tile, advanced by one
        // K16 row per step so the compiler does not recompute the product.
        const size_t row_words=size_t(output/16)*Bits*8;
        const uint32_t* row=packed+(size_t(base)*(output/16)+tile)*Bits*8;
        for(int kk=0;kk<8 && base+kk<end;++kk,row+=row_words) {
#ifdef STRIX_TRELLIS_DIRECT_PREFETCH
            const packed_decode::SaddrLane<Bits> reader(next);
            if(base+kk+1<end) next=packed_decode::SaddrLane<Bits>::issue(row+row_words);
            // Keep the next row's loads at the top of this step.
            __builtin_amdgcn_sched_barrier(0);
#else
            packed_decode::SaddrLane<Bits> reader(row);
#endif
            Pair av[8][2];
            #pragma unroll
            for(int r=0;r<8;++r) {
                av[r][0]=__builtin_bit_cast(Pair,activation[g][r*64+kk*8+(lane&3)]);
                av[r][1]=__builtin_bit_cast(Pair,activation[g][r*64+kk*8+(lane&3)+4]);
            }
#ifdef STRIX_TRELLIS_DIRECT_DUAL_ISSUE
            // Decode all four pairs, then issue the two independent chains of every row
            // alternately (acc[r][0]: pairs 0 then 1, acc[r][1]: pairs 2 then 3, as before).
            // The activation is the first fdot2 operand: native gfx11 DOT2 is symmetric in its
            // two multiplicands (alignment, truncation and special values included), so the
            // result is unchanged. This order lets the compiler pair the dot products into
            // dual-issue instructions.
            const Pair weight[4]={
                __builtin_bit_cast(Pair,packed_decode::mul1_half2(reader.template state<0>(),reader.template state<1>())),
                __builtin_bit_cast(Pair,packed_decode::mul1_half2(reader.template state<2>(),reader.template state<3>())),
                __builtin_bit_cast(Pair,packed_decode::mul1_half2(reader.template state<4>(),reader.template state<5>())),
                __builtin_bit_cast(Pair,packed_decode::mul1_half2(reader.template state<6>(),reader.template state<7>()))};
            #pragma unroll
            for(int p:{0,2,1,3}) {
                #pragma unroll
                for(int r=0;r<8;++r)
                    acc[r][p/2]=__builtin_amdgcn_fdot2(av[r][p&1],weight[p],acc[r][p/2],false);
            }
#else
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
#endif
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
        for(int split=0;split<4;++split) {
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


} // namespace strix_trellis::output_direct
namespace strix_trellis {
hipError_t query_fused_output_resources(int expected_device,size_t shared_bytes,bool* supported) {
    const void* functions[4]={
        reinterpret_cast<const void*>(output_direct::g4<5,output_direct::Mode::Direct>),
        reinterpret_cast<const void*>(output_direct::g4<6,output_direct::Mode::Direct>),nullptr,nullptr};
    grouped_direct_functions(&functions[2],&functions[3]);
    return probe_fused_output_images<hipError_t,hipFuncAttributes>(expected_device,shared_bytes,
        supported,hipSuccess,hipErrorInvalidValue,functions,hipGetDevice,hipFuncGetAttributes);
}
hipError_t launch_fused_output_trusted(const DeviceWeights& w,const _Float16* a,
    float* y,hipStream_t stream,float alpha,float beta) {
    if(!w.trellis || !w.svh || !a || !y || !std::isfinite(alpha) || !std::isfinite(beta) ||
       !fused_output_candidate(w.input,w.output,w.bits,w.codebook,8,4,false,
                               w.fused_output_capable,!(reinterpret_cast<uintptr_t>(a)&3u)))
        return hipErrorInvalidValue;
    const int groups=fused_output_groups(w.input,w.output,w.bits,w.codebook,8,4,false,
        w.fused_output_capable,true);
    if(groups!=4) return launch_grouped_direct_trusted(w,a,y,stream,alpha,beta,groups);
    if(w.bits==5)
        output_direct::g4<5,output_direct::Mode::Direct><<<w.output/128,1024,0,stream>>>(
            w.trellis,a,w.svh,w.bias,y,nullptr,w.input,w.output,alpha,beta);
    else
        output_direct::g4<6,output_direct::Mode::Direct><<<w.output/128,1024,0,stream>>>(
            w.trellis,a,w.svh,w.bias,y,nullptr,w.input,w.output,alpha,beta);
    return hipGetLastError();
}
} // namespace strix_trellis
