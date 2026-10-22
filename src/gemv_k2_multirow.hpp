#pragma once
#include "backend.hpp"
namespace strix_trellis {
hipError_t launch_gemv_k2_multirow(const DeviceWeights&,const _Float16*,float*,int rows,int total_rows,int splits,hipStream_t);
}
