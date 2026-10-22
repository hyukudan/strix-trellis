// Independent blocked-K staging experiment; no production dispatch changes.
#include "wmma_blocked_submit.hpp"
#include "trellis_optimized.hpp"
#include <algorithm>
#include <limits>
#include <string_view>

namespace strix_trellis::blocked {
namespace {
using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));
using UInt8 = uint32_t __attribute__((ext_vector_type(8)));

template<int M, int C> struct Staging {
    _Float16 a[C][M][18];
    _Float16 q[C][4][16][16];
};
template<int P, int Bits, int M, int C>
__device__ __forceinline__ void fast_pair(Staging<M,C>& lds, int sub,
    const optimized::AlignedLane<Bits,true>& decoder) {
    int lane=threadIdx.x&31, wave=threadIdx.x/32;
    int r=2*(lane%4)+8*(P%2), c=lane/4+8*(P/2);
    uint32_t v=__builtin_bit_cast(uint32_t,optimized::mul1_half2(
        decoder.template state<2*P>(),decoder.template state<2*P+1>()));
    // The pair is adjacent LOGICAL ROWS, with two distinct XOR addresses.
    // Extract integer halves explicitly. The tested ROCm compiler duplicated
    // the low half when scalarizing a bitcast to a two-element half vector.
    lds.q[sub][wave][r][c^((r&7)*2)]=__builtin_bit_cast(_Float16,uint16_t(v));
    lds.q[sub][wave][r+1][c^(((r+1)&7)*2)]=__builtin_bit_cast(_Float16,uint16_t(v>>16));
}
template<int Bits, int M, int C, bool Fast>
__device__ __forceinline__ void stage(Staging<M,C>& lds,
    const uint32_t* packed, const _Float16* input, int inputs, int outputs,
    int rows, int first_row, int first_n, int base, int valid) {
    for(int j=threadIdx.x;j<C*M*16;j+=128) {
        int sub=j/(M*16), m=(j/16)%M, k=j%16, row=first_row+m;
        lds.a[sub][m][k]=(sub<valid && row<rows) ?
            input[size_t(row)*inputs+size_t(base+sub)*16+k] : _Float16(0);
    }
    int lane=threadIdx.x&31, wave=threadIdx.x/32, tile_n=first_n+wave;
    // Each decoder dies after its stores. No decoder or Half16 array over C.
    #pragma unroll 1
    for(int sub=0;sub<valid;++sub) if(tile_n<outputs/16) {
        const uint32_t* ptr=packed+(size_t(base+sub)*(outputs/16)+tile_n)*(Bits*8);
        if constexpr(Fast) {
            optimized::AlignedLane<Bits,true> decoder(ptr);
            fast_pair<0>(lds,sub,decoder); fast_pair<1>(lds,sub,decoder);
            fast_pair<2>(lds,sub,decoder); fast_pair<3>(lds,sub,decoder);
        } else {
            WaveTile<Bits> decoder(ptr);
            #pragma unroll
            for(int j=0;j<8;++j) {
                int r=2*(lane%4)+(j%2)+8*((j/2)%2), c=lane/4+8*(j/4);
                lds.q[sub][wave][r][c^((r&7)*2)]=_Float16(codebook_value(decoder.state(8*lane+j),2));
            }
        }
    }
}
template<int M,int C>
__device__ __forceinline__ Half16 left_fragment(const Staging<M,C>& lds,int sub,int m,int lane) {
    Half16 left;
    #pragma unroll
    for(int k=0;k<16;++k) left[k]=lds.a[sub][m*16+lane%16][k];
    return left;
}
template<int M,int C>
__device__ __forceinline__ Half16 right_fragment(const Staging<M,C>& lds,int sub,int wave,int lane) {
    Half16 right;
    #pragma unroll
    for(int k=0;k<16;++k) right[k]=lds.q[sub][wave][k][(lane%16)^((k&7)*2)];
    return right;
}
template<int Bits,int M,int C,bool Fast>
__global__ __launch_bounds__(128) void core(const uint32_t* packed,
    const _Float16* input,float* partial,int inputs,int outputs,int rows,int splits) {
    __shared__ Staging<M,C> lds;
    int lane=threadIdx.x&31,wave=threadIdx.x/32;
    int first_row=int(blockIdx.y)*M,first_n=int(blockIdx.x)*4,tile_n=first_n+wave;
    int begin=int(int64_t(inputs/16)*blockIdx.z/splits);
    int end=int(int64_t(inputs/16)*(int64_t(blockIdx.z)+1)/splits);
    Float8 acc[M/16]={};
    for(int base=begin;base<end;base+=C) {
        int valid=min(C,end-base); // Uniform, begin need not be C-aligned.
        stage<Bits,M,C,Fast>(lds,packed,input,inputs,outputs,rows,first_row,first_n,base,valid);
        __syncthreads();
        if(tile_n<outputs/16) {
            #pragma unroll 1
            for(int sub=0;sub<valid;++sub) {
                Half16 right=right_fragment(lds,sub,wave,lane);
                #pragma unroll
                for(int m=0;m<M/16;++m) {
                    Half16 left=left_fragment(lds,sub,m,lane);
                    acc[m]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(left,right,acc[m]);
                }
            }
        }
        __syncthreads();
    }
    if(tile_n<outputs/16) {
        #pragma unroll
        for(int m=0;m<M/16;++m) {
            #pragma unroll
            for(int j=0;j<8;++j) {
                int row=first_row+m*16+2*j+lane/16;
                if(row<rows) partial[(size_t(blockIdx.z)*rows+row)*outputs+size_t(tile_n)*16+lane%16]=acc[m][j];
            }
        }
    }
}
template<int Bits,int M,int C,bool Fast>
__global__ __launch_bounds__(128) void probe(const uint32_t* packed,
    const _Float16* input,uint16_t* a,uint16_t* q,int inputs,int outputs,int rows,
    int first_row,int first_n,int base,int valid) {
    __shared__ Staging<M,C> lds;
    stage<Bits,M,C,Fast>(lds,packed,input,inputs,outputs,rows,first_row,first_n,base,valid);
    __syncthreads();
    int lane=threadIdx.x&31,wave=threadIdx.x/32;
    for(int sub=0;sub<C;++sub) {
        // Logical checkpoint through the same fragment accessor as core().
        for(int m=0;m<M/16;++m) {
            Half16 left=left_fragment(lds,sub,m,lane);
            UInt8 raw=__builtin_bit_cast(UInt8,left);
            if(wave==0 && lane<16) for(int k=0;k<16;++k)
                a[(size_t(sub)*M+m*16+lane)*16+k]=uint16_t(raw[k/2]>>(16*(k%2)));
        }
        Half16 right={};
        if(sub<valid && first_n+wave<outputs/16) right=right_fragment(lds,sub,wave,lane);
        UInt8 raw=__builtin_bit_cast(UInt8,right);
        if(lane<16) for(int k=0;k<16;++k)
            q[((size_t(sub)*4+wave)*16+k)*16+lane]=uint16_t(raw[k/2]>>(16*(k%2)));
    }
    __syncthreads();
}

bool aligned(const void* p,size_t alignment) { return p && !(reinterpret_cast<uintptr_t>(p)%alignment); }
bool storage(const void* p,size_t bytes,size_t alignment,int device) {
    if(!aligned(p,alignment)) return false;
    hipPointerAttribute_t attr{};
    if(hipPointerGetAttributes(&attr,p)!=hipSuccess) { (void)hipGetLastError(); return false; }
    if(attr.type!=hipMemoryTypeDevice || attr.isManaged || attr.device!=device) return false;
    void* base=nullptr;size_t size=0;
    if(hipMemGetAddressRange(&base,&size,const_cast<void*>(p))!=hipSuccess) { (void)hipGetLastError(); return false; }
    size_t offset=reinterpret_cast<uintptr_t>(p)-reinterpret_cast<uintptr_t>(base);
    return offset<=size && bytes<=size-offset;
}
bool disjoint(const void* a,size_t na,const void* b,size_t nb) {
    auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    return x<y ? na<=y-x : nb<=x-y;
}
hipError_t validate(const DeviceWeights& w,const _Float16* x,int rows,int splits,
    hipStream_t stream,int m,int bk,Decoder decoder,int& device) {
    if((m!=16 && m!=64) || (bk!=16 && bk!=32 && bk!=64) ||
       (decoder!=Decoder::slow && decoder!=Decoder::fast) || rows<1 || rows>1024 ||
       splits<1 || splits>32 || w.input<=0 || w.output<=0 || w.input%128 || w.output%128 ||
       (w.bits!=5 && w.bits!=6) || w.codebook!=2 || !stream ||
       stream==hipStreamLegacy || stream==hipStreamPerThread) return hipErrorInvalidValue;
    hipError_t e=hipGetDevice(&device);if(e!=hipSuccess) return e;
    hipDeviceProp_t prop{};e=hipGetDeviceProperties(&prop,device);if(e!=hipSuccess) return e;
    if(std::string_view(prop.gcnArchName).substr(0,7)!="gfx1151" || prop.warpSize!=32) return hipErrorNotSupported;
    unsigned flags=0;e=hipStreamGetFlags(stream,&flags);if(e!=hipSuccess) return e;
    int stream_device=-1;e=hipStreamGetDevice(stream,&stream_device);if(e!=hipSuccess) return e;
    if(stream_device!=device) return hipErrorInvalidValue;
    size_t packed=size_t(w.input)*w.output;
    if(packed>SIZE_MAX/size_t(w.bits)) return hipErrorInvalidValue;
    packed=packed*w.bits/8;
    if(!storage(w.trellis,packed,4,device) || !storage(x,size_t(rows)*w.input*2,2,device) ||
       !disjoint(w.trellis,packed,x,size_t(rows)*w.input*2)) return hipErrorInvalidValue;
    return hipSuccess;
}
template<int Bits,int M,int C,bool Fast>
hipError_t dispatch(const DeviceWeights& w,const _Float16* x,float* y,int rows,int splits,
    hipStream_t stream,uint16_t* a,uint16_t* q,int first_row,int first_n,int base,int valid) {
    if(a) probe<Bits,M,C,Fast><<<1,128,0,stream>>>(w.trellis,x,a,q,w.input,w.output,rows,first_row,first_n,base,valid);
    else core<Bits,M,C,Fast><<<dim3((size_t(w.output)+63)/64,(rows+M-1)/M,splits),128,0,stream>>>(w.trellis,x,y,w.input,w.output,rows,splits);
    return hipGetLastError();
}
template<int Bits,int M,bool Fast>
hipError_t blocks(const DeviceWeights& w,const _Float16* x,float* y,int rows,int splits,
    hipStream_t stream,int bk,uint16_t* a,uint16_t* q,int fr,int fn,int base,int valid) {
    #define BLOCK(C) dispatch<Bits,M,C,Fast>(w,x,y,rows,splits,stream,a,q,fr,fn,base,valid)
    if(bk==16) return BLOCK(1);if(bk==32) return BLOCK(2);return BLOCK(4);
    #undef BLOCK
}
hipError_t select(const DeviceWeights& w,const _Float16* x,float* y,int rows,int splits,
    hipStream_t stream,int m,int bk,Decoder d,uint16_t* a=nullptr,uint16_t* q=nullptr,
    int fr=0,int fn=0,int base=0,int valid=0) {
    #define SELECT(B,M,F) blocks<B,M,F>(w,x,y,rows,splits,stream,bk,a,q,fr,fn,base,valid)
    #define FORMAT(B,F) if(m==16) return SELECT(B,16,F);else return SELECT(B,64,F)
    if(w.bits==5) {if(d==Decoder::fast) {FORMAT(5,true);}else {FORMAT(5,false);}}
    else {if(d==Decoder::fast) {FORMAT(6,true);}else {FORMAT(6,false);}}
    #undef FORMAT
    #undef SELECT
}
}
hipError_t launch(const DeviceWeights& w,const _Float16* x,float* y,int rows,int splits,
    hipStream_t stream,int m,int bk,Decoder d) {
    int device=0;hipError_t e=validate(w,x,rows,splits,stream,m,bk,d,device);if(e!=hipSuccess) return e;
    size_t bytes=size_t(splits)*rows*w.output*4, xb=size_t(rows)*w.input*2;
    size_t wb=size_t(w.input)*w.output*w.bits/8;
    if(!storage(y,bytes,4,device) || !disjoint(x,xb,y,bytes) || !disjoint(w.trellis,wb,y,bytes)) return hipErrorInvalidValue;
    return select(w,x,y,rows,splits,stream,m,bk,d);
}
hipError_t checkpoint(const DeviceWeights& w,const _Float16* x,uint16_t* a,uint16_t* q,
    int rows,hipStream_t stream,int m,int bk,Decoder d,int fr,int fn,int base,int valid) {
    int device=0;hipError_t e=validate(w,x,rows,1,stream,m,bk,d,device);if(e!=hipSuccess) return e;
    if(fr<0 || fr>=rows || fn<0 || fn>=w.output/16 || base<0 || valid<0 || valid>bk/16 ||
       int64_t(base)+valid>w.input/16) return hipErrorInvalidValue;
    size_t ab=size_t(bk/16)*m*16*2,qb=size_t(bk/16)*4*256*2;
    size_t xb=size_t(rows)*w.input*2,wb=size_t(w.input)*w.output*w.bits/8;
    if(!storage(a,ab,2,device) || !storage(q,qb,2,device) || !disjoint(a,ab,q,qb) ||
       !disjoint(a,ab,x,xb) || !disjoint(q,qb,x,xb) ||
       !disjoint(a,ab,w.trellis,wb) || !disjoint(q,qb,w.trellis,wb)) return hipErrorInvalidValue;
    return select(w,x,nullptr,rows,1,stream,m,bk,d,a,q,fr,fn,base,valid);
}
}

namespace strix_trellis {
hipError_t launch_wmma_reuse_variant(const DeviceWeights&,const _Float16*,float*,int,int,hipStream_t,int);
}
namespace strix_trellis::blocked {
hipError_t prevalidate(const DeviceWeights& w,const _Float16* x,float* y,int rows,
    int splits,hipStream_t stream,int m,int bk,Decoder d,bool baseline,
    std::unique_ptr<const Prevalidated>& out) {
    out.reset();
    if(baseline && (m!=64 || bk!=16 || d!=Decoder::slow)) return hipErrorInvalidValue;
    int device=0;
    hipError_t e=validate(w,x,rows,splits,stream,m,bk,d,device);
    if(e!=hipSuccess) return e;
    size_t bytes=size_t(splits)*rows*w.output*4, xb=size_t(rows)*w.input*2;
    size_t wb=size_t(w.input)*w.output*w.bits/8;
    if(!storage(y,bytes,4,device) || !disjoint(x,xb,y,bytes) ||
       !disjoint(w.trellis,wb,y,bytes)) return hipErrorInvalidValue;
    out.reset(new Prevalidated(w,x,y,rows,splits,stream,m,bk,d,baseline));
    return hipSuccess;
}
hipError_t Prevalidated::submit() const {
    if(baseline_) return launch_wmma_reuse_variant(w_,x_,y_,rows_,splits_,stream_,-65);
    return select(w_,x_,y_,rows_,splits_,stream_,m_,bk_,decoder_);
}
}
