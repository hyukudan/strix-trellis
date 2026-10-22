// Independent minimal Q decoder substitution, copied from our gemm.hip.
#include "wmma_decoder.hpp"
#include "packed_decoder.hpp"
#include <type_traits>
#include <limits>
#include <string_view>
namespace strix_trellis::decoder {
namespace {
using Half16 = _Float16 __attribute__((ext_vector_type(16)));
using Float8 = float __attribute__((ext_vector_type(8)));
using HalfPair = _Float16 __attribute__((ext_vector_type(2)));
static_assert(sizeof(HalfPair)==4 && sizeof(__half2)==4);
template<int Bits, int Mode, int TileM, bool SwizzleQ = false, int PadA = 0>
__global__ __launch_bounds__(128) void decoder_core(const uint32_t* packed,
    const _Float16* input, float* partial, int inputs, int outputs,
    int rows, int splits) {
    // Four waves own distinct N16 tiles and reuse decoded Q across TileM rows.
    // Padding distributes simultaneous A row loads across LDS banks.
    constexpr int AStride = 16 + PadA;
    __shared__ _Float16 a[TileM * AStride];
    __shared__ _Float16 q[4 * 16 * 16];
    const int thread = threadIdx.x, lane = thread & 31, wave = thread / 32;
    const int first_row = int(blockIdx.y) * TileM;
    const int tile_n = int(blockIdx.x) * 4 + wave;
    const int tiles_n = outputs / 16, tiles_k = inputs / 16;
    const int begin = int(int64_t(tiles_k) * blockIdx.z / splits);
    const int end = int(int64_t(tiles_k) * (blockIdx.z + 1) / splits);
    Float8 acc[TileM / 16] = {};
    for (int tile_k = begin; tile_k < end; ++tile_k) {
        for (int j = thread; j < TileM * 16; j += 128) {
            int row = first_row + j / 16;
            a[(j / 16) * AStride + j % 16] = row < rows ? input[size_t(row) * inputs + tile_k * 16 + j % 16] : _Float16(0);
        }
        // No early exit: all waves must reach both workgroup barriers.
        if (tile_n < tiles_n) {
            // The only substituted block. SADDR base is uniform per wave.
            if constexpr (Mode == 0) {
            WaveTile<Bits> tile(packed + (size_t(tile_k) * tiles_n + tile_n) * (Bits * 8));
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int r = 2 * (lane % 4) + (j % 2) + 8 * ((j / 2) % 2);
                int c = lane / 4 + 8 * (j / 4);
                // XOR only changes LDS coordinates; the decoded Q is unchanged.
                q[wave * 256 + r * 16 + (c ^ (SwizzleQ ? (r & 7) * 2 : 0))] = _Float16(codebook_value(tile.state(8 * lane + j), 2));
            }
            } else {
                int copytile_n = tile_n;
                if constexpr (Mode == 2) copytile_n = __builtin_amdgcn_readfirstlane(copytile_n);
                const uint32_t* base = packed + (size_t(tile_k) * tiles_n + copytile_n) * (Bits * 8);
                using Reader = std::conditional_t<Mode == 1, optimized::AlignedLane<Bits,true>, packed_decode::SaddrLane<Bits,true>>;
                Reader tile(base);
                {
                    int r = 2 * (lane % 4) + 0, c = lane / 4 + 0;
                    __half2 decoded;
                    if constexpr (Mode == 1) decoded = optimized::mul1_half2(tile.template state<0>(), tile.template state<1>());
                    else decoded = packed_decode::mul1_half2(tile.template state<0>(), tile.template state<1>());
                    HalfPair pair = __builtin_bit_cast(HalfPair, decoded);
                    q[wave * 256 + r * 16 + (c ^ (SwizzleQ ? (r & 7) * 2 : 0))] = pair[0];
                    q[wave * 256 + (r + 1) * 16 + (c ^ (SwizzleQ ? ((r + 1) & 7) * 2 : 0))] = pair[1];
                }
                {
                    int r = 2 * (lane % 4) + 8, c = lane / 4 + 0;
                    __half2 decoded;
                    if constexpr (Mode == 1) decoded = optimized::mul1_half2(tile.template state<2>(), tile.template state<3>());
                    else decoded = packed_decode::mul1_half2(tile.template state<2>(), tile.template state<3>());
                    HalfPair pair = __builtin_bit_cast(HalfPair, decoded);
                    q[wave * 256 + r * 16 + (c ^ (SwizzleQ ? (r & 7) * 2 : 0))] = pair[0];
                    q[wave * 256 + (r + 1) * 16 + (c ^ (SwizzleQ ? ((r + 1) & 7) * 2 : 0))] = pair[1];
                }
                {
                    int r = 2 * (lane % 4) + 0, c = lane / 4 + 8;
                    __half2 decoded;
                    if constexpr (Mode == 1) decoded = optimized::mul1_half2(tile.template state<4>(), tile.template state<5>());
                    else decoded = packed_decode::mul1_half2(tile.template state<4>(), tile.template state<5>());
                    HalfPair pair = __builtin_bit_cast(HalfPair, decoded);
                    q[wave * 256 + r * 16 + (c ^ (SwizzleQ ? (r & 7) * 2 : 0))] = pair[0];
                    q[wave * 256 + (r + 1) * 16 + (c ^ (SwizzleQ ? ((r + 1) & 7) * 2 : 0))] = pair[1];
                }
                {
                    int r = 2 * (lane % 4) + 8, c = lane / 4 + 8;
                    __half2 decoded;
                    if constexpr (Mode == 1) decoded = optimized::mul1_half2(tile.template state<6>(), tile.template state<7>());
                    else decoded = packed_decode::mul1_half2(tile.template state<6>(), tile.template state<7>());
                    HalfPair pair = __builtin_bit_cast(HalfPair, decoded);
                    q[wave * 256 + r * 16 + (c ^ (SwizzleQ ? (r & 7) * 2 : 0))] = pair[0];
                    q[wave * 256 + (r + 1) * 16 + (c ^ (SwizzleQ ? ((r + 1) & 7) * 2 : 0))] = pair[1];
                }
            }
        }
        __syncthreads();
        if (tile_n < tiles_n) {
            Half16 right;
            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                // Replicate both operands in lanes 16..31 (RDNA3/3.5 wave32).
                right[k] = q[wave * 256 + k * 16 + ((lane % 16) ^ (SwizzleQ ? (k & 7) * 2 : 0))];
            }
            #pragma unroll
            for (int m = 0; m < TileM / 16; ++m) {
                Half16 left;
                #pragma unroll
                for (int k = 0; k < 16; ++k)
                    left[k] = a[(m * 16 + lane % 16) * AStride + k];
                acc[m] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(left, right, acc[m]);
            }
        }
        __syncthreads();
    }
    if (tile_n < tiles_n) {
        #pragma unroll
        for (int m = 0; m < TileM / 16; ++m) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int row = first_row + m * 16 + 2 * j + lane / 16;
                if (row < rows)
                    partial[(size_t(blockIdx.z) * rows + row) * outputs + tile_n * 16 + lane % 16] = acc[m][j];
            }
        }
    }
}

template<int Bits,int Mode,int TileM,bool SwizzleQ,int PadA>
__global__ __launch_bounds__(128) void staging_dump(const uint32_t* packed,
    const _Float16* input,uint16_t* da,uint16_t* dp,uint16_t* dl,
    int inputs,int outputs,int rows,int first_row,int first_n,int tile_k) {
    constexpr int AStride=16+PadA;
    __shared__ _Float16 a[TileM * AStride];
    __shared__ _Float16 q[4 * 16 * 16];
    const int thread=threadIdx.x,lane=thread&31,wave=thread/32;
    const int tile_n=first_n+wave,tiles_n=outputs/16;
    for (int j = thread; j < TileM * 16; j += 128) {
        int row = first_row + j / 16;
        a[(j / 16) * AStride + j % 16] = row < rows ? input[size_t(row) * inputs + tile_k * 16 + j % 16] : _Float16(0);
    }
    for(int j=thread;j<1024;j+=128) q[j]=0;
    __syncthreads();
    if(tile_n<tiles_n) {
            // The only substituted block. SADDR base is uniform per wave.
            if constexpr (Mode == 0) {
            WaveTile<Bits> tile(packed + (size_t(tile_k) * tiles_n + tile_n) * (Bits * 8));
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int r = 2 * (lane % 4) + (j % 2) + 8 * ((j / 2) % 2);
                int c = lane / 4 + 8 * (j / 4);
                // XOR only changes LDS coordinates; the decoded Q is unchanged.
                q[wave * 256 + r * 16 + (c ^ (SwizzleQ ? (r & 7) * 2 : 0))] = _Float16(codebook_value(tile.state(8 * lane + j), 2));
            }
            } else {
                int copytile_n = tile_n;
                if constexpr (Mode == 2) copytile_n = __builtin_amdgcn_readfirstlane(copytile_n);
                const uint32_t* base = packed + (size_t(tile_k) * tiles_n + copytile_n) * (Bits * 8);
                using Reader = std::conditional_t<Mode == 1, optimized::AlignedLane<Bits,true>, packed_decode::SaddrLane<Bits,true>>;
                Reader tile(base);
                {
                    int r = 2 * (lane % 4) + 0, c = lane / 4 + 0;
                    __half2 decoded;
                    if constexpr (Mode == 1) decoded = optimized::mul1_half2(tile.template state<0>(), tile.template state<1>());
                    else decoded = packed_decode::mul1_half2(tile.template state<0>(), tile.template state<1>());
                    HalfPair pair = __builtin_bit_cast(HalfPair, decoded);
                    q[wave * 256 + r * 16 + (c ^ (SwizzleQ ? (r & 7) * 2 : 0))] = pair[0];
                    q[wave * 256 + (r + 1) * 16 + (c ^ (SwizzleQ ? ((r + 1) & 7) * 2 : 0))] = pair[1];
                }
                {
                    int r = 2 * (lane % 4) + 8, c = lane / 4 + 0;
                    __half2 decoded;
                    if constexpr (Mode == 1) decoded = optimized::mul1_half2(tile.template state<2>(), tile.template state<3>());
                    else decoded = packed_decode::mul1_half2(tile.template state<2>(), tile.template state<3>());
                    HalfPair pair = __builtin_bit_cast(HalfPair, decoded);
                    q[wave * 256 + r * 16 + (c ^ (SwizzleQ ? (r & 7) * 2 : 0))] = pair[0];
                    q[wave * 256 + (r + 1) * 16 + (c ^ (SwizzleQ ? ((r + 1) & 7) * 2 : 0))] = pair[1];
                }
                {
                    int r = 2 * (lane % 4) + 0, c = lane / 4 + 8;
                    __half2 decoded;
                    if constexpr (Mode == 1) decoded = optimized::mul1_half2(tile.template state<4>(), tile.template state<5>());
                    else decoded = packed_decode::mul1_half2(tile.template state<4>(), tile.template state<5>());
                    HalfPair pair = __builtin_bit_cast(HalfPair, decoded);
                    q[wave * 256 + r * 16 + (c ^ (SwizzleQ ? (r & 7) * 2 : 0))] = pair[0];
                    q[wave * 256 + (r + 1) * 16 + (c ^ (SwizzleQ ? ((r + 1) & 7) * 2 : 0))] = pair[1];
                }
                {
                    int r = 2 * (lane % 4) + 8, c = lane / 4 + 8;
                    __half2 decoded;
                    if constexpr (Mode == 1) decoded = optimized::mul1_half2(tile.template state<6>(), tile.template state<7>());
                    else decoded = packed_decode::mul1_half2(tile.template state<6>(), tile.template state<7>());
                    HalfPair pair = __builtin_bit_cast(HalfPair, decoded);
                    q[wave * 256 + r * 16 + (c ^ (SwizzleQ ? (r & 7) * 2 : 0))] = pair[0];
                    q[wave * 256 + (r + 1) * 16 + (c ^ (SwizzleQ ? ((r + 1) & 7) * 2 : 0))] = pair[1];
                }
            }
    }
    __syncthreads();
    for(int j=thread;j<TileM*16;j+=128) da[j]=__builtin_bit_cast(uint16_t,a[(j/16)*AStride+j%16]);
    for(int j=thread;j<1024;j+=128) {
        dp[j]=__builtin_bit_cast(uint16_t,q[j]);
        int r=(j%256)/16,c=j%16;
        dl[j]=__builtin_bit_cast(uint16_t,q[(j/256)*256+r*16+(c^(SwizzleQ?(r&7)*2:0))]);
    }
}

template<int B,int D,int M,bool X,int P>
hipError_t dispatch(const Request& r,uint16_t* a=nullptr,uint16_t* p=nullptr,uint16_t* l=nullptr,int fr=0,int fn=0,int k=0) {
    const auto& w=r.weights;
    if(a) staging_dump<B,D,M,X,P><<<1,128,0,r.stream>>>(w.trellis,static_cast<const _Float16*>(r.a.pointer),a,p,l,w.input,w.output,r.rows,fr,fn,k);
    else decoder_core<B,D,M,X,P><<<dim3((w.output/16+3)/4,(r.rows+M-1)/M,r.splits),128,0,r.stream>>>(w.trellis,static_cast<const _Float16*>(r.a.pointer),static_cast<float*>(const_cast<void*>(r.partial.pointer)),w.input,w.output,r.rows,r.splits);
    return hipGetLastError();
}
template<int B,int D>
hipError_t layout(const Request& r,uint16_t* a,uint16_t* p,uint16_t* l,int fr,int fn,int k) {
    if(r.layout==Layout::m16_a16_plain) return dispatch<B,D,16,false,0>(r,a,p,l,fr,fn,k);
    if(r.layout==Layout::m64_a16_xor) return dispatch<B,D,64,true,0>(r,a,p,l,fr,fn,k);
    return dispatch<B,D,64,true,2>(r,a,p,l,fr,fn,k);
}
hipError_t select(const Request& r,uint16_t* a=nullptr,uint16_t* p=nullptr,uint16_t* l=nullptr,int fr=0,int fn=0,int k=0) {
    #define MODE(B) if(r.mode==Mode::baseline) return layout<B,0>(r,a,p,l,fr,fn,k); if(r.mode==Mode::aligned) return layout<B,1>(r,a,p,l,fr,fn,k); return layout<B,2>(r,a,p,l,fr,fn,k)
    if(r.weights.bits==5) { MODE(5); } else { MODE(6); }
    #undef MODE
}
bool storage(Range r,size_t needed,size_t alignment,int device) {
    uintptr_t ptr=reinterpret_cast<uintptr_t>(r.pointer);
    if(!ptr || ptr%alignment || needed>r.capacity || r.capacity>UINTPTR_MAX-ptr) return false;
    hipPointerAttribute_t attr{};
    if(hipPointerGetAttributes(&attr,r.pointer)!=hipSuccess) { (void)hipGetLastError(); return false; }
    if(attr.type!=hipMemoryTypeDevice || attr.isManaged || attr.device!=device) return false;
    void* base=nullptr;size_t capacity=0;
    if(hipMemGetAddressRange(&base,&capacity,const_cast<void*>(r.pointer))!=hipSuccess) { (void)hipGetLastError(); return false; }
    uintptr_t origin=reinterpret_cast<uintptr_t>(base);
    if(ptr<origin) return false;
    size_t offset=ptr-origin;
    return offset<=capacity && r.capacity<=capacity-offset;
}
bool disjoint(Range a,Range b) {
    uintptr_t x=reinterpret_cast<uintptr_t>(a.pointer),y=reinterpret_cast<uintptr_t>(b.pointer);
    return x<y ? a.capacity<=y-x : b.capacity<=x-y;
}
} // namespace
hipError_t prepare(const Request& r,std::unique_ptr<const Token>& out) {
    out.reset();const auto& w=r.weights;
    if(r.rows<1 || r.rows>512 || r.splits<1 || r.splits>32 ||
       w.input<128 || w.input>8192 || w.input%128 || w.output<1 || w.output>32768 ||
       (r.diagnostic ? (w.output!=16 && w.output!=32 && w.output!=48) : w.output%128!=0) ||
       (w.bits!=5 && w.bits!=6) || w.codebook!=2 ||
       (r.mode!=Mode::baseline && r.mode!=Mode::aligned && r.mode!=Mode::saddr) ||
       (r.layout!=Layout::m16_a16_plain && r.layout!=Layout::m64_a16_xor && r.layout!=Layout::m64_a18_xor) ||
       r.a_stride!=w.input || r.partial_stride!=w.output || r.total_rows!=r.rows ||
       r.packed.pointer!=w.trellis || !r.stream || r.stream==hipStreamLegacy || r.stream==hipStreamPerThread) return hipErrorInvalidValue;
    int device=-1;auto e=hipGetDevice(&device);if(e!=hipSuccess) return e;
    hipDeviceProp_t prop{};e=hipGetDeviceProperties(&prop,device);if(e!=hipSuccess) return e;
    if(std::string_view(prop.gcnArchName).substr(0,7)!="gfx1151" || prop.warpSize!=32) return hipErrorNotSupported;
    unsigned flags=0;e=hipStreamGetFlags(r.stream,&flags);if(e!=hipSuccess) return e;
    int stream_device=-1;e=hipStreamGetDevice(r.stream,&stream_device);if(e!=hipSuccess) return e;
    if(stream_device!=device) return hipErrorInvalidValue;
    // All factors are bounded above before these uint64/size_t products.
    static_assert(sizeof(size_t)==8);
    size_t wb=size_t(w.input)*w.output*w.bits/8,ab=size_t(r.rows)*w.input*2,pb=size_t(r.splits)*r.rows*w.output*4;
    if(!storage(r.packed,wb,4,device) || !storage(r.a,ab,2,device) || !storage(r.partial,pb,4,device) ||
       !disjoint(r.packed,r.a) || !disjoint(r.packed,r.partial) || !disjoint(r.a,r.partial)) return hipErrorInvalidValue;
    out.reset(new Token(r));return hipSuccess;
}
hipError_t Token::submit() const { return select(request_); }
hipError_t dump(const Token& t,uint16_t* a,uint16_t* physical,uint16_t* logical,int fr,int fn,int k) {
    const Request& r=t.request_;int device=-1;auto e=hipGetDevice(&device);if(e!=hipSuccess) return e;
    size_t ab=(r.layout==Layout::m16_a16_plain?16:64)*16*2;
    Range ranges[]={{a,ab},{physical,2048},{logical,2048},r.packed,r.a,r.partial};
    if(fr<0 || fr>=r.rows || fn<0 || fn>=r.weights.output/16 || k<0 || k>=r.weights.input/16) return hipErrorInvalidValue;
    for(int i=0;i<3;++i) {
        if(!storage(ranges[i],ranges[i].capacity,2,device)) return hipErrorInvalidValue;
        for(int j=i+1;j<6;++j) if(!disjoint(ranges[i],ranges[j])) return hipErrorInvalidValue;
    }
    return select(r,a,physical,logical,fr,fn,k);
}
} // namespace strix_trellis::decoder
