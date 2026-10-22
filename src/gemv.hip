// Independent EXL3 GEMV: exact FP16 decode, packed FP16 dot / FP32 accumulation.
#include "gemv.hpp"
#include "trellis_optimized.hpp"
namespace strix_trellis {
using GemvPair = _Float16 __attribute__((ext_vector_type(2)));

template<int Bits, bool Mul1, int Rows>
__global__ void gemv_aligned(const uint32_t* __restrict__ packed,
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
        GemvPair activations[Rows][2];
        #pragma unroll
        for(int r=0;r<Rows;++r) if(r<rows) {
            int channel=k*16+2*(lane&3);
            __builtin_memcpy(&activations[r][0],x+size_t(r)*input+channel,sizeof(GemvPair));
            __builtin_memcpy(&activations[r][1],x+size_t(r)*input+channel+8,sizeof(GemvPair));
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
template<int Bits,bool Mul1> static hipError_t gemv_rows(const DeviceWeights& w,
    const _Float16* x,float* y,int rows,int total_rows,int splits,hipStream_t stream) {
    dim3 grid((w.output/16+3)/4,splits),block(128);
    #define RUN(R) gemv_aligned<Bits,Mul1,R><<<grid,block,0,stream>>>(w.trellis,x,y,w.input,w.output,rows,total_rows,splits,w.codebook)
    if(rows==1) {RUN(1);} else if(rows==2) {RUN(2);} else if(rows<=4) {RUN(4);} else {RUN(8);}
    #undef RUN
    return hipGetLastError();
}
hipError_t launch_gemv_core(const DeviceWeights& w,const _Float16* x,float* y,
    int rows,int total_rows,int splits,hipStream_t stream) {
    if(!w.trellis||!x||!y||rows<1||rows>8||total_rows<rows||splits<1||
       w.input<=0||w.output<=0||w.input%16||w.output%16||w.codebook<0||w.codebook>2)
        return hipErrorInvalidValue;
    #define CASE(B) case B:return w.codebook==2?gemv_rows<B,true>(w,x,y,rows,total_rows,splits,stream):gemv_rows<B,false>(w,x,y,rows,total_rows,splits,stream)
    switch(w.bits) {CASE(1);CASE(2);CASE(3);CASE(4);CASE(5);CASE(6);CASE(7);CASE(8);default:return hipErrorInvalidValue;}
    #undef CASE
}
}
