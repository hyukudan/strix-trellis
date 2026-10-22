#pragma once
#include "backend.hpp"
namespace strix_trellis {
// Optional R1 only. I/N positive multiples of 128, I <=8192.
// Original F32 input, signed suh, RN32 -> RN16 -> H128F32 -> RN16.
// Partial layout [splits,total_rows,N]; caller supplies the row offset.
// H128-aligned splits differ from the legacy I16 split reduction tree.
// No allocation, transfer or synchronization; output transform is separate.
hipError_t launch_gemv_fused_input(const DeviceWeights&,const float*,float*,
    int rows,int total_rows,int splits,hipStream_t);
// Diagnostic variant: block x=0 emits the entire transformed input once.
hipError_t launch_gemv_fused_input_debug(const DeviceWeights&,const float*,float*,
    int rows,int total_rows,int splits,hipStream_t,_Float16* a,float* h);
}
