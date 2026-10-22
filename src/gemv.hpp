#pragma once
#include "backend.hpp"
namespace strix_trellis {
// A contiguous chunk of 1..8 FP16 rows. partial points at that chunk's row
// offset inside split-major [splits,total_rows,output]. No transforms/scales.
hipError_t launch_gemv_core(const DeviceWeights& weights,const _Float16* input,
    float* partial,int rows,int total_rows,int splits,hipStream_t stream);
}
