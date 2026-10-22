#pragma once
#include "backend.hpp"
namespace strix_trellis::static_combined {
// 0: exact current v9 policy; 1: exact global/original decode;
// 2: exact global/hash+SADDR; 3: exact LDS8/hash+SADDR.
// Mul1 K5/K6, rows 1..8, I<=8192, N<=131072, S<=32 only.
hipError_t launch(const DeviceWeights&,const _Float16*,float*,int,int,int,hipStream_t,int);
}
