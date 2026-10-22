// Small design gates: exhaustive codebooks, WMMA layout, split-I GEMV prototype.
#include "trellis_device.hpp"
#include "host_reference.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP(call) do { auto status=(call); if(status!=hipSuccess) { std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(status)); std::exit(1); }} while(0)
using namespace strix_trellis;
using half16 = _Float16 __attribute__((ext_vector_type(16)));
using float8 = float __attribute__((ext_vector_type(8)));

__global__ void exhaustive(uint16_t* output, int cb) {
    int state = blockIdx.x * blockDim.x + threadIdx.x;
    if (state < 65536) output[state] = __half_as_ushort(__float2half_rn(codebook_value(state, cb)));
}

template<int Variant>
__global__ void wmma_layout(const _Float16* a, const _Float16* b, float* output) {
    const int lane = threadIdx.x;
    const int column = lane & 15;
    const int case_offset = blockIdx.x * 256;
    half16 left, right;
    for (int k = 0; k < 16; ++k) {
        left[k] = a[case_offset + ((Variant & 1) ? k * 16 + column : column * 16 + k)];
        right[k] = b[case_offset + ((Variant & 2) ? column * 16 + k : k * 16 + column)];
    }
    float8 accum = {};
    accum = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(left, right, accum);
    for (int element = 0; element < 8; ++element) {
        int row = element * 2 + (lane >> 4);
        output[case_offset + ((Variant & 4) ? column * 16 + row : row * 16 + column)] = accum[element];
    }
}

template<int Bits>
__global__ void core_probe(const uint32_t* packed, const float* x, float* partial, int input, int output, int splits) {
    int tile_n = blockIdx.x * 4 + (threadIdx.x / 32);
    if (tile_n >= output / 16) return;
    int lane = threadIdx.x & 31;
    int tile_count = input / 16;
    int begin = tile_count * blockIdx.y / splits;
    int end = tile_count * (blockIdx.y + 1) / splits;
    float low = 0, high = 0;
    for (int tile_k = begin; tile_k < end; ++tile_k) {
        WaveTile<Bits> tile(packed + (size_t(tile_k) * (output/16) + tile_n) * Bits * 8);
        constexpr int group = Bits <= 4 ? 8 : (Bits == 7 ? 2 : 4);
        #pragma unroll
        for(int base=0;base<8;base+=group) {
            uint64_t window = tile.template group<group>(lane * 8 + base);
            #pragma unroll
            for(int j=0;j<group;++j) {
                int element = base+j;
                int row = ((lane & 3) << 1) + (element & 1) + ((element & 2) << 2);
                unsigned state = uint32_t(window >> ((group-1-j)*Bits)) & 65535u;
                float weight = codebook_value(state,2);
                float activation = x[tile_k * 16 + row];
                if (element < 4) low = fmaf(weight,activation,low);
                else high = fmaf(weight,activation,high);
            }
        }
    }
    low += __shfl_down(low,2,4); high += __shfl_down(high,2,4);
    low += __shfl_down(low,1,4); high += __shfl_down(high,1,4);
    if ((lane & 3) == 0) {
        size_t offset = size_t(blockIdx.y) * output + tile_n * 16 + (lane >> 2);
        partial[offset] = low; partial[offset + 8] = high;
    }
}
__global__ void finish_probe(const float* partial, float* output, int n, int splits) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index>=n) return;
    float sum=0;
    for(int split=0; split<splits; ++split) sum+=partial[split*n+index];
    output[index]=sum;
}

static void codebook_gate() {
    uint16_t* device; HIP(hipMalloc(&device,65536*2)); std::vector<uint16_t> got(65536);
    for(int cb=0;cb<3;++cb) {
        exhaustive<<<256,256>>>(device,cb); HIP(hipGetLastError());
        HIP(hipMemcpy(got.data(),device,65536*2,hipMemcpyDeviceToHost));
        for(int state=0;state<65536;++state) if(got[state]!=host_codebook(state,cb)) {std::fprintf(stderr,"Codebook mismatch %d %d\n",cb,state); std::exit(2);}
        std::printf("{\"gate\":\"exhaustive_codebook\",\"codebook\":%d,\"states\":65536,\"bit_exact\":true}\n",cb);
    }
    HIP(hipFree(device));
}
static void wmma_gate() {
    constexpr int cases=513;
    std::vector<_Float16> a(cases*256),b(a.size()); std::vector<float> expected(a.size()),got(a.size()),magnitude(a.size());
    for(int c=0;c<cases;++c) {
        for(int i=0;i<16;++i) {a[c*256+i*16+i]=1; b[c*256+i*16+i]=1;}
        if(c<256) {std::fill(a.begin()+c*256,a.begin()+(c+1)*256,0); a[c*256+c]=2;}
        else if(c<512) {std::fill(b.begin()+c*256,b.begin()+(c+1)*256,0); b[c*256+c-256]=-3;}
        else for(int i=0;i<256;++i) {a[c*256+i]=_Float16((i*7%13)-6);b[c*256+i]=_Float16((i*11%17)-8);}
        for(int r=0;r<16;++r) for(int n=0;n<16;++n) for(int k=0;k<16;++k) {
            float term=float(a[c*256+r*16+k])*float(b[c*256+k*16+n]);
            expected[c*256+r*16+n]+=term;magnitude[c*256+r*16+n]+=std::abs(term);
        }
    }
    _Float16 *da,*db; float* dc;
    HIP(hipMalloc(&da,a.size()*2)); HIP(hipMalloc(&db,b.size()*2)); HIP(hipMalloc(&dc,got.size()*4));
    HIP(hipMemcpy(da,a.data(),a.size()*2,hipMemcpyHostToDevice)); HIP(hipMemcpy(db,b.data(),b.size()*2,hipMemcpyHostToDevice));
    int winner=-1;
    for(int variant=0;variant<8;++variant) {
        #define TEST_VARIANT(V) if(variant==V) wmma_layout<V><<<cases,32>>>(da,db,dc)
        TEST_VARIANT(0); TEST_VARIANT(1); TEST_VARIANT(2); TEST_VARIANT(3);
        TEST_VARIANT(4); TEST_VARIANT(5); TEST_VARIANT(6); TEST_VARIANT(7);
        #undef TEST_VARIANT
        HIP(hipGetLastError()); HIP(hipMemcpy(got.data(),dc,got.size()*4,hipMemcpyDeviceToHost));
        size_t mismatches=0,impulse_mismatches=0;double max_abs=0,max_normalized=0;
        for(size_t i=0;i<got.size();++i) {
            mismatches+=got[i]!=expected[i];
            if(i<512*256) impulse_mismatches+=got[i]!=expected[i];
            double error=std::abs(double(got[i])-expected[i]);
            max_abs=std::max(max_abs,error);
            max_normalized=std::max(max_normalized,error/std::max(1.0,double(magnitude[i])));
        }
        std::printf("{\"gate\":\"wmma_candidate\",\"variant\":%d,\"mismatches\":%zu}\n",variant,mismatches);
        // Native WMMA has empirically non-IEEE internal accumulation for mixed
        // signs on this target. Layout requires all impulses exact. Arithmetic
        // is checked separately against a conservative, reported error envelope.
        bool accepted=impulse_mismatches==0 && max_normalized<=16*0x1p-24;
        if(accepted) {
            winner=variant;
            std::printf("{\"gate\":\"wmma_arithmetic\",\"variant\":%d,\"impulse_mismatches\":%zu,\"max_abs_error\":%.12g,\"max_error_over_sum_abs_products\":%.12g,\"envelope\":%.12g}\n",variant,impulse_mismatches,max_abs,max_normalized,16*0x1p-24);
        }
    }
    if(winner<0) {std::fprintf(stderr,"No WMMA layout candidate passed\n");std::exit(2);}
    std::printf("{\"gate\":\"wmma_layout\",\"cases\":513,\"variant\":%d,\"outputs\":%zu,\"impulse_exact_and_arithmetic_within_envelope\":true}\n",winner,got.size());
    HIP(hipFree(da)); HIP(hipFree(db)); HIP(hipFree(dc));
}

template<int Bits> static void split_probe(int input,int output) {
    size_t tile_count=size_t(input/16)*(output/16), words=tile_count*Bits*8;
    std::vector<uint32_t> packed(words); std::vector<float> x(input),got(output); std::vector<double> expected(output),magnitudes(output);
    uint32_t rng=42; for(auto& w:packed) {rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;w=rng;}
    for(int i=0;i<input;++i) x[i]=float((i*19%31)-15)/16;
    for(int tk=0;tk<input/16;++tk) for(int tn=0;tn<output/16;++tn) {
        auto* tile=packed.data()+(size_t(tk)*(output/16)+tn)*Bits*8;
        for(int r=0;r<16;++r) for(int n=0;n<16;++n) {
            int position=((n&7)<<5)|((r&6)<<2)|(r&1)|((r&8)>>2)|((n&8)>>1);
            double term=double(x[tk*16+r])*host_half(host_codebook(host_state(tile,Bits,position),2));
            expected[tn*16+n]+=term; magnitudes[tn*16+n]+=std::abs(term);
        }
    }
    uint32_t* dp; float *dx,*partial,*dy;
    HIP(hipMalloc(&dp,words*4)); HIP(hipMalloc(&dx,input*4)); HIP(hipMalloc(&partial,64*output*4)); HIP(hipMalloc(&dy,output*4));
    HIP(hipMemcpy(dp,packed.data(),words*4,hipMemcpyHostToDevice)); HIP(hipMemcpy(dx,x.data(),input*4,hipMemcpyHostToDevice));
    hipEvent_t start,stop;HIP(hipEventCreate(&start));HIP(hipEventCreate(&stop));
    for(int split: {1,2,4,8,16,32,64}) {
        auto launch=[&](){core_probe<Bits><<<dim3((output/16+3)/4,split),128>>>(dp,dx,partial,input,output,split); finish_probe<<<(output+255)/256,256>>>(partial,dy,output,split); HIP(hipGetLastError());};
        launch(); HIP(hipMemcpy(got.data(),dy,output*4,hipMemcpyDeviceToHost));
        double max_error=0,normalized=0;
        for(int n=0;n<output;++n) {double error=std::abs(double(got[n])-expected[n]); max_error=std::max(max_error,error); normalized=std::max(normalized,error/std::max(1.0,magnitudes[n]));}
        // Very conservative FP32 error envelope scaled by absolute products, independent of output cancellation.
        if(normalized>input*0x1p-23) {std::fprintf(stderr,"Split probe mismatch %.9g\n",normalized);std::exit(2);}
        std::vector<float> times;
        for(int repeat=0;repeat<11;++repeat) {HIP(hipEventRecord(start));launch();HIP(hipEventRecord(stop));HIP(hipEventSynchronize(stop));float ms;HIP(hipEventElapsedTime(&ms,start,stop));times.push_back(ms);}
        std::sort(times.begin(),times.end());
        std::printf("{\"gate\":\"split_probe\",\"bits\":%d,\"input\":%d,\"output\":%d,\"split\":%d,\"median_ms\":%.6f,\"packed_GBs\":%.3f,\"max_abs_error\":%.9g,\"max_error_over_sum_abs_products\":%.9g,\"cache\":\"warm_single_matrix\"}\n",Bits,input,output,split,times[5],double(words*4)/times[5]/1e6,max_error,normalized);
        std::fflush(stdout);
    }
    HIP(hipEventDestroy(start)); HIP(hipEventDestroy(stop)); HIP(hipFree(dp));HIP(hipFree(dx));HIP(hipFree(partial));HIP(hipFree(dy));
}
int main(int argc,char**) { if(argc==1) {codebook_gate();wmma_gate();} split_probe<6>(5120,512);split_probe<5>(1280,32768); }
