#pragma once
#include "hadamard.hpp"
#include <cmath>
namespace strix_trellis {
__device__ __forceinline__ float tp_stable_swiglu(float g, float u) {
    const float e = expf(g < 0.0f ? g : -g);
    const float sigmoid = g < 0.0f ? e / (1.0f + e) : 1.0f / (1.0f + e);
    return materialize_fp32(materialize_fp32(g * sigmoid) * u);
}
}
