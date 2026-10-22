// Independent EXL3 GEMV: exact FP16 decode, packed FP16 dot / FP32 accumulation.
#include "gemv_lds.hpp"
#include "trellis_optimized.hpp"
#include "packed_decoder.hpp"
#include "gemv_policy.hpp"
#include "gemv_k2.hpp"
#include "gemv_k2_multirow.hpp"
#include "v14_core_policy.hpp"
#include <type_traits>
namespace strix_trellis {
using BaselinePair = _Float16 __attribute__((ext_vector_type(2)));

template<int Bits, bool Mul1, int Rows, bool Fast=false>
__global__ void gemv_baseline_kernel(const uint32_t* __restrict__ packed,
    const _Float16* __restrict__ x,float* __restrict__ partial,
    int input,int output,int rows,int total_rows,int splits,int codebook) {
    int tile_n=blockIdx.x*4+threadIdx.x/32;
    if constexpr(Fast) tile_n=__builtin_amdgcn_readfirstlane(tile_n);
    if(tile_n>=output/16) return;
    int lane=threadIdx.x&31;
    int begin=(input/16)*blockIdx.y/splits,end=(input/16)*(blockIdx.y+1)/splits;
    float sums[Rows][2]={};
    for(int k=begin;k<end;++k) {
        using Reader=std::conditional_t<Fast,packed_decode::SaddrLane<Bits>,optimized::AlignedLane<Bits,true>>;
        Reader tile(packed+(size_t(k)*(output/16)+tile_n)*Bits*8);
        // Each activation pair is shared by the low and high output columns.
        BaselinePair activations[Rows][2];
        #pragma unroll
        for(int r=0;r<Rows;++r) if(r<rows) {
            int channel=k*16+2*(lane&3);
            __builtin_memcpy(&activations[r][0],x+size_t(r)*input+channel,sizeof(BaselinePair));
            __builtin_memcpy(&activations[r][1],x+size_t(r)*input+channel+8,sizeof(BaselinePair));
        }
        #pragma unroll
        for(int pair=0;pair<4;++pair) {
            uint32_t a,b;
            // Explicit instantiations keep bit offsets compile-time constants.
            if(pair==0) {a=tile.template state<0>();b=tile.template state<1>();}
            else if(pair==1) {a=tile.template state<2>();b=tile.template state<3>();}
            else if(pair==2) {a=tile.template state<4>();b=tile.template state<5>();}
            else {a=tile.template state<6>();b=tile.template state<7>();}
            BaselinePair w;
            if constexpr(Mul1) {
                __half2 decoded;
                if constexpr(Fast) decoded=packed_decode::mul1_half2(a,b);
                else decoded=optimized::mul1_half2(a,b);
                w=__builtin_bit_cast(BaselinePair,decoded);
            } else {w[0]=_Float16(codebook_value(a,codebook));w[1]=_Float16(codebook_value(b,codebook));}
            #pragma unroll
            for(int r=0;r<Rows;++r) if(r<rows)
                sums[r][pair/2]=__builtin_amdgcn_fdot2(w,activations[r][pair&1],sums[r][pair/2],false);
        }
    }
    #pragma unroll
    for(int r=0;r<Rows;++r) if(r<rows) {
        #pragma unroll
        for(int h=0;h<2;++h) {
            float v=sums[r][h];v+=__shfl_down(v,2,4);v+=__shfl_down(v,1,4);
            if(!(lane&3)) partial[(size_t(blockIdx.y)*total_rows+r)*output+tile_n*16+lane/4+h*8]=v;
        }
    }
}
template<int Bits,bool Mul1,bool Fast=false> static hipError_t gemv_baseline_rows(const DeviceWeights& w,
    const _Float16* x,float* y,int rows,int total_rows,int splits,hipStream_t stream) {
    dim3 grid((w.output/16+3)/4,splits),block(128);
    #define RUN(R) gemv_baseline_kernel<Bits,Mul1,R,Fast><<<grid,block,0,stream>>>(w.trellis,x,y,w.input,w.output,rows,total_rows,splits,w.codebook)
    if(rows==1) {RUN(1);} else if(rows==2) {RUN(2);} else if(rows<=4) {RUN(4);} else {RUN(8);}
    #undef RUN
    return hipGetLastError();
}
hipError_t baseline_core(const DeviceWeights& w,const _Float16* x,float* y,
    int rows,int total_rows,int splits,hipStream_t stream) {
    if(!w.trellis||!x||!y||rows<1||rows>8||total_rows<rows||splits<1||
       w.input<=0||w.output<=0||w.input%16||w.output%16||w.codebook<0||w.codebook>2)
        return hipErrorInvalidValue;
    #define CASE(B) case B:return w.codebook==2?gemv_baseline_rows<B,true>(w,x,y,rows,total_rows,splits,stream):gemv_baseline_rows<B,false>(w,x,y,rows,total_rows,splits,stream)
    switch(w.bits) {CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);default:return hipErrorInvalidValue;}
    #undef CASE
}

using GemvPair = _Float16 __attribute__((ext_vector_type(2)));

// memcpy preserves object aliasing rules; the proven alignment makes this a
// packed 32-bit load. This helper is never reached for a two-byte offset.
__device__ inline uint32_t load_activation_pair(const _Float16* x,size_t pair) {
    const char* aligned=static_cast<const char*>(__builtin_assume_aligned(x,4));
    uint32_t value;
    __builtin_memcpy(&value,aligned+pair*4,sizeof(value));
    return value;
}

template<int Bits, bool Mul1, int Rows, int Load, bool StaticRows=false, bool Fast=false>
__global__ void gemv_lds_kernel(const uint32_t* __restrict__ packed,
    const _Float16* __restrict__ x,float* __restrict__ partial,
    int input,int output,int rows,int total_rows,int splits,int codebook) {
    int tile_n=blockIdx.x*4+threadIdx.x/32;
    if constexpr(Fast) tile_n=__builtin_amdgcn_readfirstlane(tile_n);
    // Inactive output waves still participate in every LDS barrier.
    bool active=tile_n<output/16;
    if constexpr(Load==1) {if(!active) return;}
    constexpr int Chunk=Load==2?8:1;
    __shared__ uint32_t activation_tile[Load==2 ? Rows*8*Chunk : 1];
    int lane=threadIdx.x&31;
    int begin=(input/16)*blockIdx.y/splits,end=(input/16)*(blockIdx.y+1)/splits;
    float sums[Rows][2]={};
    for(int base=begin;base<end;base+=Chunk) {
        if constexpr(Load==2) {
            // Stage consecutive channels of eight tiles per row. Cooperative
            // loads are coalesced and both barriers are amortized over Chunk.
            for(int q=threadIdx.x;q<Rows*8*Chunk;q+=128) {
                int r=q/(8*Chunk),pair=q%(8*Chunk);
                if((StaticRows || r<rows) && base+pair/8<end)
                    activation_tile[q]=load_activation_pair(x,size_t(r)*(input/2)+base*8+pair);
            }
            __syncthreads();
        }
        if(active) for(int kk=0;kk<Chunk && base+kk<end;++kk) {
        int k=base+kk;
        using Reader=std::conditional_t<Fast,packed_decode::SaddrLane<Bits>,optimized::AlignedLane<Bits,true>>;
        Reader tile(packed+(size_t(k)*(output/16)+tile_n)*Bits*8);
        // Each activation pair is shared by the low and high output columns.
        GemvPair activations[Rows][2];
        #pragma unroll
        for(int r=0;r<Rows;++r) if(StaticRows || r<rows) {
            int channel=k*16+2*(lane&3);
            if constexpr(Load==0) {
                __builtin_memcpy(&activations[r][0],x+size_t(r)*input+channel,sizeof(GemvPair));
                __builtin_memcpy(&activations[r][1],x+size_t(r)*input+channel+8,sizeof(GemvPair));
            } else if constexpr(Load==1) {
                activations[r][0]=__builtin_bit_cast(GemvPair,load_activation_pair(x,size_t(r)*(input/2)+channel/2));
                activations[r][1]=__builtin_bit_cast(GemvPair,load_activation_pair(x,size_t(r)*(input/2)+(channel+8)/2));
            } else {
                activations[r][0]=__builtin_bit_cast(GemvPair,activation_tile[r*8*Chunk+kk*8+(lane&3)]);
                activations[r][1]=__builtin_bit_cast(GemvPair,activation_tile[r*8*Chunk+kk*8+(lane&3)+4]);
            }
        }
        #pragma unroll
        for(int pair=0;pair<4;++pair) {
            uint32_t a,b;
            // Explicit instantiations keep bit offsets compile-time constants.
            if(pair==0) {a=tile.template state<0>();b=tile.template state<1>();}
            else if(pair==1) {a=tile.template state<2>();b=tile.template state<3>();}
            else if(pair==2) {a=tile.template state<4>();b=tile.template state<5>();}
            else {a=tile.template state<6>();b=tile.template state<7>();}
            GemvPair w;
            if constexpr(Mul1) {
                __half2 decoded;
                if constexpr(Fast) decoded=packed_decode::mul1_half2(a,b);
                else decoded=optimized::mul1_half2(a,b);
                w=__builtin_bit_cast(GemvPair,decoded);
            } else {w[0]=_Float16(codebook_value(a,codebook));w[1]=_Float16(codebook_value(b,codebook));}
            #pragma unroll
            for(int r=0;r<Rows;++r) if(StaticRows || r<rows)
                sums[r][pair/2]=__builtin_amdgcn_fdot2(w,activations[r][pair&1],sums[r][pair/2],false);
        }
        } // active wave
        if constexpr(Load==2) __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<Rows;++r) if(active && (StaticRows || r<rows)) {
        #pragma unroll
        for(int h=0;h<2;++h) {
            float v=sums[r][h];v+=__shfl_down(v,2,4);v+=__shfl_down(v,1,4);
            if(!(lane&3)) partial[(size_t(blockIdx.y)*total_rows+r)*output+tile_n*16+lane/4+h*8]=v;
        }
    }
}
template<int Bits,bool Mul1,int Load> static hipError_t gemv_lds_rows(const DeviceWeights& w,
    const _Float16* x,float* y,int rows,int total_rows,int splits,hipStream_t stream) {
    dim3 grid((w.output/16+3)/4,splits),block(128);
    #define RUN(R) gemv_lds_kernel<Bits,Mul1,R,Load><<<grid,block,0,stream>>>(w.trellis,x,y,w.input,w.output,rows,total_rows,splits,w.codebook)
    if(rows==1) {RUN(1);} else if(rows==2) {RUN(2);} else if(rows<=4) {RUN(4);} else {RUN(8);}
    #undef RUN
    return hipGetLastError();
}
template<int Load> static hipError_t dispatch(const DeviceWeights& w,const _Float16* x,float* y,
    int rows,int total_rows,int splits,hipStream_t stream) {
    if(!w.trellis||!x||!y||rows<1||rows>8||total_rows<rows||splits<1||
       w.input<=0||w.output<=0||w.input%16||w.output%16||w.codebook<0||w.codebook>2)
        return hipErrorInvalidValue;
    #define CASE(B) case B:return w.codebook==2?gemv_lds_rows<B,true,Load>(w,x,y,rows,total_rows,splits,stream):gemv_lds_rows<B,false,Load>(w,x,y,rows,total_rows,splits,stream)
    switch(w.bits) {CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);default:return hipErrorInvalidValue;}
    #undef CASE
}
hipError_t launch_gemv_lds_variant(const DeviceWeights& w,const _Float16* x,float* y,
    int rows,int total_rows,int splits,hipStream_t stream,int mode) {
    if(mode<0||mode>2) return hipErrorInvalidValue;
    // Never form a uint32 pointer on the two-byte aligned fallback path.
    if((reinterpret_cast<uintptr_t>(x)&3u)||mode==0)
        return baseline_core(w,x,y,rows,total_rows,splits,stream);
    if(mode==1) return dispatch<1>(w,x,y,rows,total_rows,splits,stream);
    return dispatch<2>(w,x,y,rows,total_rows,splits,stream);
}
// Identical kernel body to the independently qualified exact-row candidates.
// The policy proves rows==R before removing row predicates.
template<int Bits,int Load,bool Fast> static hipError_t exact_rows(const DeviceWeights& w,
    const _Float16* x,float* y,int rows,int total_rows,int splits,hipStream_t stream) {
    dim3 grid((w.output/16+3)/4,splits),block(128);
    #define RUN(R) case R: gemv_lds_kernel<Bits,true,R,Load,true,Fast><<<grid,block,0,stream>>>(w.trellis,x,y,w.input,w.output,rows,total_rows,splits,w.codebook);break
    switch(rows) {RUN(1);RUN(2);RUN(3);RUN(4);RUN(5);RUN(6);RUN(7);RUN(8);default:return hipErrorInvalidValue;}
    #undef RUN
    return hipGetLastError();
}
hipError_t launch_gemv_lds_core(const DeviceWeights& w,const _Float16* x,float* y,
    int rows,int total_rows,int splits,hipStream_t stream) {
    if(k2_multirow_candidate(w.input,w.output,w.bits,w.codebook,rows,total_rows,splits,
        !(uintptr_t(x)&3u),!(uintptr_t(y)&3u),!(uintptr_t(w.trellis)&3u)))
        return launch_gemv_k2_multirow(w,x,y,rows,total_rows,splits,stream);
    if(pipeline_k2_candidate(w.input,w.output,w.bits,w.codebook,rows,splits,
                            !(reinterpret_cast<uintptr_t>(x)&3u)))
        return launch_gemv_k2_core(w,x,y,total_rows,splits,stream);
    const int exact_mode=exact_gemv_mode(w.input,w.output,w.bits,w.codebook,
        rows,splits,!(reinterpret_cast<uintptr_t>(x)&3u));
    if(exact_mode) {
        if(!w.trellis || !x || !y || total_rows<rows) return hipErrorInvalidValue;
        if(exact_mode==1) return w.bits==5?exact_rows<5,1,false>(w,x,y,rows,total_rows,splits,stream):exact_rows<6,1,false>(w,x,y,rows,total_rows,splits,stream);
        if(exact_mode==2) return w.bits==5?exact_rows<5,1,true>(w,x,y,rows,total_rows,splits,stream):exact_rows<6,1,true>(w,x,y,rows,total_rows,splits,stream);
        if(exact_mode==3) return w.bits==5?exact_rows<5,2,true>(w,x,y,rows,total_rows,splits,stream):exact_rows<6,2,true>(w,x,y,rows,total_rows,splits,stream);
        // Mode 4 is deliberately the measured v9 body, not exact-global plus
        // a split change. The R4 predicates and R8 static body stay intact.
        dim3 grid((w.output/16+3)/4,splits),block(128);
        if(rows==4) gemv_lds_kernel<5,true,4,2,false,true><<<grid,block,0,stream>>>(w.trellis,x,y,w.input,w.output,rows,total_rows,splits,w.codebook);
        else gemv_lds_kernel<5,true,8,2,true,true><<<grid,block,0,stream>>>(w.trellis,x,y,w.input,w.output,rows,total_rows,splits,w.codebook);
        return hipGetLastError();
    }
    // Two frozen, alternating campaigns support only these rows, shapes and
    // splits. Other formats/geometries and two-byte X retain the v8 path.
    const bool measured_rows=rows==1 || rows==2 || rows==4 || rows==8;
    const bool measured_shape=
        (w.bits==5 && splits==4 && ((w.input==1280 && w.output==32768) ||
         (w.input==8192 && w.output==5120) || (w.input==2304 && w.output==5120))) ||
        (w.bits==5 && splits==16 && w.input==5120 && w.output==2304) ||
        (w.bits==5 && splits==32 && w.input==4096 && w.output==1024) ||
        (w.bits==6 && splits==4 && ((w.input==8192 && w.output==5120) ||
         (w.input==5120 && w.output==129280))) ||
        (w.bits==6 && splits==16 && w.input==5120 && w.output==1280) ||
        (w.bits==6 && splits==32 && w.input==5120 && w.output==512);
    if(w.codebook==2 && measured_rows && measured_shape &&
       !(reinterpret_cast<uintptr_t>(x)&3u)) {
        if(!w.trellis || !x || !y || total_rows<rows) return hipErrorInvalidValue;
        dim3 grid((w.output/16+3)/4,splits),block(128);
        #define FAST(B,R,L,STATIC) gemv_lds_kernel<B,true,R,L,STATIC,true><<<grid,block,0,stream>>>(w.trellis,x,y,w.input,w.output,rows,total_rows,splits,w.codebook)
        if(rows<=2) return w.bits==5?gemv_baseline_rows<5,true,true>(w,x,y,rows,total_rows,splits,stream):gemv_baseline_rows<6,true,true>(w,x,y,rows,total_rows,splits,stream);
        if(rows==4) {if(w.bits==5) {FAST(5,4,2,false);} else {FAST(6,4,2,false);}}
        else {if(w.bits==5) {FAST(5,8,2,true);} else {FAST(6,8,2,true);}}
        #undef FAST
        return hipGetLastError();
    }
    // Two independent cold-address turns favored removing the row predicates
    // only when the host proves all eight rows are active. Retain the original
    // pair/dot order, inactive-wave barriers, chunk size and aligned fallback.
    if(rows==8 && w.codebook==2 && (w.bits==5 || w.bits==6) &&
       !(reinterpret_cast<uintptr_t>(x)&3u)) {
        if(!w.trellis || !x || !y || total_rows<rows || splits<1 ||
           w.input<=0 || w.output<=0 || w.input%16 || w.output%16)
            return hipErrorInvalidValue;
        dim3 grid((w.output/16+3)/4,splits),block(128);
        #define STATIC_R8(B) gemv_lds_kernel<B,true,8,2,true><<<grid,block,0,stream>>>(w.trellis,x,y,w.input,w.output,rows,total_rows,splits,w.codebook)
        if(w.bits==5) {STATIC_R8(5);} else {STATIC_R8(6);}
        #undef STATIC_R8
        return hipGetLastError();
    }
    // Same-turn measurements favor batched LDS for multirow mul1 K5/K6.
    // Small rows and unmeasured formats keep the independently copied core.
    int mode=(w.codebook==2 && (w.bits==5 || w.bits==6) && rows>=3)?2:0;
    return launch_gemv_lds_variant(w,x,y,rows,total_rows,splits,stream,mode);
}
}
