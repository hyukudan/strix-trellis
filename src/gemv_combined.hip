#include "gemv_combined.hpp"
#include "trellis_optimized.hpp"
#include "combined_decoder.hpp"
// Independently owned v8 bodies; no production objects or archive symbols.
namespace combined_mode0 {
using namespace ::strix_trellis;
using DeviceWeights = ::strix_trellis::DeviceWeights;
using BaselinePair = _Float16 __attribute__((ext_vector_type(2)));

template<int Bits, bool Mul1, int Rows>
__global__ void gemv_baseline_kernel(const uint32_t* __restrict__ packed,
    const _Float16* __restrict__ x,float* __restrict__ partial,
    int input,int output,int rows,int total_rows,int splits,int codebook) {
    int tile_n=blockIdx.x*4+threadIdx.x/32;
    if(tile_n>=output/16) return;
    int lane=threadIdx.x&31;
    int begin=(input/16)*blockIdx.y/splits,end=(input/16)*(blockIdx.y+1)/splits;
    float sums[Rows][2]={};
    for(int k=begin;k<end;++k) {
        optimized::AlignedLane<Bits,true> tile(packed+(size_t(k)*(output/16)+tile_n)*Bits*8);
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
                __half2 decoded=optimized::mul1_half2(a,b);
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
template<int Bits,bool Mul1> static hipError_t gemv_baseline_rows(const DeviceWeights& w,
    const _Float16* x,float* y,int rows,int total_rows,int splits,hipStream_t stream) {
    dim3 grid((w.output/16+3)/4,splits),block(128);
    #define RUN(R) gemv_baseline_kernel<Bits,Mul1,R><<<grid,block,0,stream>>>(w.trellis,x,y,w.input,w.output,rows,total_rows,splits,w.codebook)
    if(rows==1) {RUN(1);} else if(rows==2) {RUN(2);} else if(rows<=4) {RUN(4);} else {RUN(8);}
    #undef RUN
    return hipGetLastError();
}
hipError_t baseline_core(const DeviceWeights& w,const _Float16* x,float* y,
    int rows,int total_rows,int splits,hipStream_t stream) {
    if(!w.trellis||!x||!y||rows<1||rows>8||total_rows<rows||splits<1||
       w.input<=0||w.output<=0||w.input%16||w.output%16||w.codebook<0||w.codebook>2)
        return hipErrorInvalidValue;
    #define CASE(B) case B:return gemv_baseline_rows<B,true>(w,x,y,rows,total_rows,splits,stream)
    switch(w.bits) {CASE(5);CASE(6);default:return hipErrorInvalidValue;}
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

template<int Bits, bool Mul1, int Rows, int Load, bool StaticRows=false>
__global__ void gemv_lds_kernel(const uint32_t* __restrict__ packed,
    const _Float16* __restrict__ x,float* __restrict__ partial,
    int input,int output,int rows,int total_rows,int splits,int codebook) {
    int tile_n=blockIdx.x*4+threadIdx.x/32;
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
        optimized::AlignedLane<Bits,true> tile(packed+(size_t(k)*(output/16)+tile_n)*Bits*8);
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
                __half2 decoded=optimized::mul1_half2(a,b);
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
    #define CASE(B) case B:return gemv_lds_rows<B,true,Load>(w,x,y,rows,total_rows,splits,stream)
    switch(w.bits) {CASE(5);CASE(6);default:return hipErrorInvalidValue;}
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
hipError_t launch_gemv_lds_core(const DeviceWeights& w,const _Float16* x,float* y,
    int rows,int total_rows,int splits,hipStream_t stream) {
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

namespace combined_mode1 {
using namespace ::strix_trellis;
using DeviceWeights = ::strix_trellis::DeviceWeights;
using BaselinePair = _Float16 __attribute__((ext_vector_type(2)));

template<int Bits, bool Mul1, int Rows>
__global__ void gemv_baseline_kernel(const uint32_t* __restrict__ packed,
    const _Float16* __restrict__ x,float* __restrict__ partial,
    int input,int output,int rows,int total_rows,int splits,int codebook) {
    int tile_n=blockIdx.x*4+threadIdx.x/32;
    if(tile_n>=output/16) return;
    int lane=threadIdx.x&31;
    int begin=(input/16)*blockIdx.y/splits,end=(input/16)*(blockIdx.y+1)/splits;
    float sums[Rows][2]={};
    for(int k=begin;k<end;++k) {
        optimized::AlignedLane<Bits,true> tile(packed+(size_t(k)*(output/16)+tile_n)*Bits*8);
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
                __half2 decoded=combined_decode::mul1_half2_isa(a,b);
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
template<int Bits,bool Mul1> static hipError_t gemv_baseline_rows(const DeviceWeights& w,
    const _Float16* x,float* y,int rows,int total_rows,int splits,hipStream_t stream) {
    dim3 grid((w.output/16+3)/4,splits),block(128);
    #define RUN(R) gemv_baseline_kernel<Bits,Mul1,R><<<grid,block,0,stream>>>(w.trellis,x,y,w.input,w.output,rows,total_rows,splits,w.codebook)
    if(rows==1) {RUN(1);} else if(rows==2) {RUN(2);} else if(rows<=4) {RUN(4);} else {RUN(8);}
    #undef RUN
    return hipGetLastError();
}
hipError_t baseline_core(const DeviceWeights& w,const _Float16* x,float* y,
    int rows,int total_rows,int splits,hipStream_t stream) {
    if(!w.trellis||!x||!y||rows<1||rows>8||total_rows<rows||splits<1||
       w.input<=0||w.output<=0||w.input%16||w.output%16||w.codebook<0||w.codebook>2)
        return hipErrorInvalidValue;
    #define CASE(B) case B:return gemv_baseline_rows<B,true>(w,x,y,rows,total_rows,splits,stream)
    switch(w.bits) {CASE(5);CASE(6);default:return hipErrorInvalidValue;}
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

template<int Bits, bool Mul1, int Rows, int Load, bool StaticRows=false>
__global__ void gemv_lds_kernel(const uint32_t* __restrict__ packed,
    const _Float16* __restrict__ x,float* __restrict__ partial,
    int input,int output,int rows,int total_rows,int splits,int codebook) {
    int tile_n=blockIdx.x*4+threadIdx.x/32;
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
        optimized::AlignedLane<Bits,true> tile(packed+(size_t(k)*(output/16)+tile_n)*Bits*8);
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
                __half2 decoded=combined_decode::mul1_half2_isa(a,b);
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
    #define CASE(B) case B:return gemv_lds_rows<B,true,Load>(w,x,y,rows,total_rows,splits,stream)
    switch(w.bits) {CASE(5);CASE(6);default:return hipErrorInvalidValue;}
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
hipError_t launch_gemv_lds_core(const DeviceWeights& w,const _Float16* x,float* y,
    int rows,int total_rows,int splits,hipStream_t stream) {
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

namespace combined_mode2 {
using namespace ::strix_trellis;
using DeviceWeights = ::strix_trellis::DeviceWeights;
using BaselinePair = _Float16 __attribute__((ext_vector_type(2)));

template<int Bits, bool Mul1, int Rows>
__global__ void gemv_baseline_kernel(const uint32_t* __restrict__ packed,
    const _Float16* __restrict__ x,float* __restrict__ partial,
    int input,int output,int rows,int total_rows,int splits,int codebook) {
    int tile_n=blockIdx.x*4+threadIdx.x/32;
    tile_n=__builtin_amdgcn_readfirstlane(tile_n);
    if(tile_n>=output/16) return;
    int lane=threadIdx.x&31;
    int begin=(input/16)*blockIdx.y/splits,end=(input/16)*(blockIdx.y+1)/splits;
    float sums[Rows][2]={};
    for(int k=begin;k<end;++k) {
        combined_decode::SaddrLane<Bits,true> tile(packed+(size_t(k)*(output/16)+tile_n)*Bits*8);
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
                __half2 decoded=optimized::mul1_half2(a,b);
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
template<int Bits,bool Mul1> static hipError_t gemv_baseline_rows(const DeviceWeights& w,
    const _Float16* x,float* y,int rows,int total_rows,int splits,hipStream_t stream) {
    dim3 grid((w.output/16+3)/4,splits),block(128);
    #define RUN(R) gemv_baseline_kernel<Bits,Mul1,R><<<grid,block,0,stream>>>(w.trellis,x,y,w.input,w.output,rows,total_rows,splits,w.codebook)
    if(rows==1) {RUN(1);} else if(rows==2) {RUN(2);} else if(rows<=4) {RUN(4);} else {RUN(8);}
    #undef RUN
    return hipGetLastError();
}
hipError_t baseline_core(const DeviceWeights& w,const _Float16* x,float* y,
    int rows,int total_rows,int splits,hipStream_t stream) {
    if(!w.trellis||!x||!y||rows<1||rows>8||total_rows<rows||splits<1||
       w.input<=0||w.output<=0||w.input%16||w.output%16||w.codebook<0||w.codebook>2)
        return hipErrorInvalidValue;
    #define CASE(B) case B:return gemv_baseline_rows<B,true>(w,x,y,rows,total_rows,splits,stream)
    switch(w.bits) {CASE(5);CASE(6);default:return hipErrorInvalidValue;}
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

template<int Bits, bool Mul1, int Rows, int Load, bool StaticRows=false>
__global__ void gemv_lds_kernel(const uint32_t* __restrict__ packed,
    const _Float16* __restrict__ x,float* __restrict__ partial,
    int input,int output,int rows,int total_rows,int splits,int codebook) {
    int tile_n=blockIdx.x*4+threadIdx.x/32;
    tile_n=__builtin_amdgcn_readfirstlane(tile_n);
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
        combined_decode::SaddrLane<Bits,true> tile(packed+(size_t(k)*(output/16)+tile_n)*Bits*8);
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
                __half2 decoded=optimized::mul1_half2(a,b);
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
    #define CASE(B) case B:return gemv_lds_rows<B,true,Load>(w,x,y,rows,total_rows,splits,stream)
    switch(w.bits) {CASE(5);CASE(6);default:return hipErrorInvalidValue;}
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
hipError_t launch_gemv_lds_core(const DeviceWeights& w,const _Float16* x,float* y,
    int rows,int total_rows,int splits,hipStream_t stream) {
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

namespace combined_mode3 {
using namespace ::strix_trellis;
using DeviceWeights = ::strix_trellis::DeviceWeights;
using BaselinePair = _Float16 __attribute__((ext_vector_type(2)));

template<int Bits, bool Mul1, int Rows>
__global__ void gemv_baseline_kernel(const uint32_t* __restrict__ packed,
    const _Float16* __restrict__ x,float* __restrict__ partial,
    int input,int output,int rows,int total_rows,int splits,int codebook) {
    int tile_n=blockIdx.x*4+threadIdx.x/32;
    tile_n=__builtin_amdgcn_readfirstlane(tile_n);
    if(tile_n>=output/16) return;
    int lane=threadIdx.x&31;
    int begin=(input/16)*blockIdx.y/splits,end=(input/16)*(blockIdx.y+1)/splits;
    float sums[Rows][2]={};
    for(int k=begin;k<end;++k) {
        combined_decode::SaddrLane<Bits,true> tile(packed+(size_t(k)*(output/16)+tile_n)*Bits*8);
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
                __half2 decoded=combined_decode::mul1_half2_isa(a,b);
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
template<int Bits,bool Mul1> static hipError_t gemv_baseline_rows(const DeviceWeights& w,
    const _Float16* x,float* y,int rows,int total_rows,int splits,hipStream_t stream) {
    dim3 grid((w.output/16+3)/4,splits),block(128);
    #define RUN(R) gemv_baseline_kernel<Bits,Mul1,R><<<grid,block,0,stream>>>(w.trellis,x,y,w.input,w.output,rows,total_rows,splits,w.codebook)
    if(rows==1) {RUN(1);} else if(rows==2) {RUN(2);} else if(rows<=4) {RUN(4);} else {RUN(8);}
    #undef RUN
    return hipGetLastError();
}
hipError_t baseline_core(const DeviceWeights& w,const _Float16* x,float* y,
    int rows,int total_rows,int splits,hipStream_t stream) {
    if(!w.trellis||!x||!y||rows<1||rows>8||total_rows<rows||splits<1||
       w.input<=0||w.output<=0||w.input%16||w.output%16||w.codebook<0||w.codebook>2)
        return hipErrorInvalidValue;
    #define CASE(B) case B:return gemv_baseline_rows<B,true>(w,x,y,rows,total_rows,splits,stream)
    switch(w.bits) {CASE(5);CASE(6);default:return hipErrorInvalidValue;}
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

template<int Bits, bool Mul1, int Rows, int Load, bool StaticRows=false>
__global__ void gemv_lds_kernel(const uint32_t* __restrict__ packed,
    const _Float16* __restrict__ x,float* __restrict__ partial,
    int input,int output,int rows,int total_rows,int splits,int codebook) {
    int tile_n=blockIdx.x*4+threadIdx.x/32;
    tile_n=__builtin_amdgcn_readfirstlane(tile_n);
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
        combined_decode::SaddrLane<Bits,true> tile(packed+(size_t(k)*(output/16)+tile_n)*Bits*8);
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
                __half2 decoded=combined_decode::mul1_half2_isa(a,b);
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
    #define CASE(B) case B:return gemv_lds_rows<B,true,Load>(w,x,y,rows,total_rows,splits,stream)
    switch(w.bits) {CASE(5);CASE(6);default:return hipErrorInvalidValue;}
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
hipError_t launch_gemv_lds_core(const DeviceWeights& w,const _Float16* x,float* y,
    int rows,int total_rows,int splits,hipStream_t stream) {
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

namespace strix_trellis_combined {
hipError_t launch(const DeviceWeights& w,const _Float16* x,float* y,int rows,int total,int splits,hipStream_t stream,int mode) {
 if(mode<0||mode>3||w.codebook!=2||(w.bits!=5&&w.bits!=6))return hipErrorInvalidValue;
 switch(mode) {
 case 0:return combined_mode0::launch_gemv_lds_core(w,x,y,rows,total,splits,stream);
 case 1:return combined_mode1::launch_gemv_lds_core(w,x,y,rows,total,splits,stream);
 case 2:return combined_mode2::launch_gemv_lds_core(w,x,y,rows,total,splits,stream);
 case 3:return combined_mode3::launch_gemv_lds_core(w,x,y,rows,total,splits,stream);
 default:return hipErrorInvalidValue;
 }
}
}
