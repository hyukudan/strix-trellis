#pragma once
#include "backend.hpp"
namespace strix_trellis {
hipError_t launch_wmma_packed(const DeviceWeights&,const _Float16*,float*,int rows,int splits,hipStream_t);
}
