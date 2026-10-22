#pragma once
#include "backend.hpp"
namespace strix_trellis {
// Private R1 core. Output is [split,total_rows,N] relative to caller's row pointer.
// X must be four-byte aligned; dispatchers must route offset2 through v9.
hipError_t launch_gemv_k2_core(const DeviceWeights&,const _Float16*,float*,
    int total_rows,int splits,hipStream_t);
}
