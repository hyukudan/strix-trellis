#pragma once
#include "backend.hpp"
namespace strix_trellis {
// Core only: transformed FP16 [rows,input] times decoded Q. Writes deterministic
// split-major FP32 [splits,rows,output] partials. No Hadamard or signed scales.
hipError_t launch_wmma_core(const DeviceWeights& weights,const _Float16* input,
    float* partial,int rows,int splits,hipStream_t stream);
}
