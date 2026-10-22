#pragma once
#include "backend.hpp"
namespace strix_trellis {
// Contiguous FP16 rows; deterministic FP32 partial [splits,total_rows,output].
// Two-byte aligned input is supported through a runtime fallback.
hipError_t launch_gemv_lds_core(const DeviceWeights&,const _Float16*,float*,
    int rows,int total_rows,int splits,hipStream_t);
// Benchmark controls: 0 original memcpy, 1 aligned packed, 2 cooperative LDS.
hipError_t launch_gemv_lds_variant(const DeviceWeights&,const _Float16*,float*,
    int rows,int total_rows,int splits,hipStream_t,int mode);
}
