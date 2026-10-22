#pragma once
#include "backend.hpp"
namespace strix_trellis {
// Addresses of the two new images; no resource queries or mutable cache.
void grouped_direct_functions(const void** k5g1,const void** k6g2);
hipError_t launch_grouped_direct_trusted(const DeviceWeights& weights,
    const _Float16* a,float* y,hipStream_t stream,float alpha,float beta,int groups);
// Trusted backend entry: public validation owns capacities, device and lifetime.
hipError_t launch_fused_output_trusted(const DeviceWeights& weights,
    const _Float16* a,float* y,hipStream_t stream,float alpha,float beta);
}
