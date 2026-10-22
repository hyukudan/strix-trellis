#include "backend.hpp"
#include "gemv_fused_both.hpp"
#include "fused_both_policy.hpp"
#include "trellis_device.hpp"
#include "transforms.hpp"
#include "gemm.hpp"
#include "gemv.hpp"
#include "gemv_lds.hpp"
#include "gemv_fused.hpp"
#include "launch_policy.hpp"
#include "gemv_policy.hpp"
#include "fused_output_policy.hpp"
#include "gemv_output_direct.hpp"
#include <algorithm>
#include <limits>

namespace strix_trellis {
static_assert(sizeof(size_t)==8,"The HIP backend requires a 64-bit host");
// Split policy uses measured cold vector shapes and enough WMMA workgroups for
// narrow outputs. It is shared by query and submission, with no mutable tuning.
static int split_count(const DeviceWeights& weights,int rows) {
    int requested;
    if(rows>8) {
        int tile_m=wmma_tile_rows(weights,rows);
        size_t groups=((size_t(rows)+tile_m-1)/tile_m)*((size_t(weights.output)+63)/64);
        if(medium_prefill_candidate(weights,rows)) {
            // A narrow M64 grid needs more K partitions. A partial second/third
            // M64 block benefits from additional parallelism in the measured tails.
            if(weights.output>=32768) requested=rows<64?4:1;
            else {
                size_t target=rows>64 && rows%64?512:256;
                requested=1;
                while(groups*size_t(requested)<target && requested<16) requested*=2;
            }
        } else {
            requested=1;
            while(groups*size_t(requested)<160 && requested<32) requested*=2;
        }
    } else {
        requested=measured_vector_splits(weights.input,weights.output,
            weights.bits,weights.codebook,rows);
    }
    return std::min(requested,weights.input/16);
}
static size_t align256(size_t value) { return (value+255)&~size_t(255); }
size_t linear_workspace_bytes(const DeviceWeights& w,int rows) {
    if(rows==0) return 0;
    if(rows<0 || w.input<=0 || w.output<=0 || w.input%128 || w.output%128) return SIZE_MAX;
    const size_t limit=std::numeric_limits<size_t>::max();
    size_t input_elements=size_t(rows)*size_t(w.input);
    size_t partial_elements=size_t(rows)*size_t(w.output);
    int splits=split_count(w,rows);
    if(input_elements>(limit-255)/2 || partial_elements>limit/4/size_t(splits)) return SIZE_MAX;
    // The transform dispatch processes four 128-value groups per block.
    if(input_elements/128>size_t(UINT32_MAX)*4 || partial_elements/128>size_t(UINT32_MAX)*4) return SIZE_MAX;
    size_t input_bytes=align256(input_elements*2),partial_bytes=partial_elements*4*size_t(splits);
    if(partial_bytes>limit-input_bytes) return SIZE_MAX;
    return input_bytes+partial_bytes;
}

hipError_t launch_linear(const DeviceWeights& w,const float* input,float* output,int rows,
    void* workspace,hipStream_t stream,float alpha,float beta,bool prefer_wmma) {
    if(rows==0) return hipSuccess;
    size_t bytes=linear_workspace_bytes(w,rows);
    if(bytes==SIZE_MAX || !workspace || w.bits<1 || w.bits>8 || w.codebook<0 || w.codebook>2) return hipErrorInvalidValue;
    FusedBothLayout direct{};
    if(fused_both_layout(w.input,w.output,w.bits,w.codebook,rows,prefer_wmma,
            w.fused_both_capable,w.trellis,input,w.suh,w.svh,w.bias,output,direct))
        return launch_fused_both_trusted(w,input,output,rows,prefer_wmma,stream,alpha,beta);
    auto* transformed=reinterpret_cast<_Float16*>(workspace);
    auto* partial=reinterpret_cast<float*>(reinterpret_cast<char*>(workspace)+align256(size_t(rows)*w.input*2));
    int splits=split_count(w,rows);
    // Only measured dense mul1 geometries select the original-input R1 fusion.
    // Larger heads, other formats and GEMM retain the separately staged path.
    bool measured_fusion=(w.bits==6 && w.input==5120 && w.output==1280) ||
        (w.bits==5 && ((w.input==4096 && w.output==1024) ||
          (w.input==8192 && w.output==5120) || (w.input==5120 && w.output==2304) ||
          (w.input==2304 && w.output==5120)));
    bool fused=!prefer_wmma && rows==1 && w.codebook==2 && measured_fusion;
    hipError_t status;
    if(fused) {
        status=launch_gemv_fused_input(w,input,partial,rows,rows,splits,stream);
        if(status!=hipSuccess)return status;
    } else {
        input_transform<<<(size_t(rows)*(w.input/128)+3)/4,128,0,stream>>>(input,w.suh,transformed,w.input,rows);
        status=hipGetLastError();if(status!=hipSuccess)return status;
        if(fused_output_candidate(w.input,w.output,w.bits,w.codebook,rows,splits,
                prefer_wmma,w.fused_output_capable,
                transformed && !(reinterpret_cast<uintptr_t>(transformed)&3u)))
            return launch_fused_output_trusted(w,transformed,output,stream,alpha,beta);
        if(prefer_wmma) {
            status=launch_wmma_core(w,transformed,partial,rows,splits,stream);
            if(status!=hipSuccess)return status;
        } else {
            for(int begin=0;begin<rows;begin+=8) {
                status=launch_gemv_lds_core(w,transformed+size_t(begin)*w.input,partial+size_t(begin)*w.output,
                    std::min(8,rows-begin),rows,splits,stream);
                if(status!=hipSuccess)return status;
            }
        }
    }
    output_transform<<<(size_t(rows)*(w.output/128)+3)/4,128,0,stream>>>(partial,w.svh,w.bias,output,w.output,rows,splits,alpha,beta);
    return hipGetLastError();
}
}
