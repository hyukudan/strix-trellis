#pragma once
#include "backend.hpp"
namespace strix_trellis_combined {
using DeviceWeights = ::strix_trellis::DeviceWeights;
// Isolated core experiment: 0 v8, 1 hash, 2 SADDR, 3 both. Only mul1 K5/K6.
hipError_t launch(const DeviceWeights&,const _Float16*,float*,int,int,int,hipStream_t,int);
}
