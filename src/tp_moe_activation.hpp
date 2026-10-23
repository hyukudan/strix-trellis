#pragma once
// SwiGLU of the routed-expert operator, shared by every gate epilogue.
// Default: the original stable sigmoid with the device expf (bits unchanged).
// STRIX_TRELLIS_TP_MOE_DECLARED_SWIGLU: the DeepSeek V4.1 engine semantics,
// gate clamped from above and up on both sides at 10, silu(g) = g / (1 + exp(-g))
// with an exp built from basic IEEE operations only, so host and device give the
// same bits (same arithmetic as strix::ds41::swiglu_clamped / declared_exp).
#include <cstdint>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif
#if defined(__HIPCC__)
#define STRIX_TP_ACT_HD __host__ __device__
#else
#define STRIX_TP_ACT_HD
#endif
namespace strix_trellis {
STRIX_TP_ACT_HD inline float tp_moe_pow2(int k) {
    return __builtin_bit_cast(float, static_cast<uint32_t>(k + 127) << 23U);
}
STRIX_TP_ACT_HD inline float tp_moe_declared_exp(float x) {
    if (!(x == x)) return x;
    if (x > 88.0F) return __builtin_huge_valf();
    if (x < -103.0F) return 0.0F;
    const float n = __builtin_rintf(x * 1.44269504F);
    const float r = (x - n * 0.693359375F) - n * -2.12194440e-4F;
    float p = 1.98412698e-4F;
    p = p * r + 1.38888889e-3F;
    p = p * r + 8.33333333e-3F;
    p = p * r + 4.16666667e-2F;
    p = p * r + 1.66666667e-1F;
    p = p * r + 0.5F;
    p = p * r + 1.0F;
    p = p * r + 1.0F;
    const int k = static_cast<int>(n), k1 = k / 2, k2 = k - k1;
    return (p * tp_moe_pow2(k1)) * tp_moe_pow2(k2);
}
STRIX_TP_ACT_HD inline float tp_moe_declared_swiglu(float gate, float up) {
    constexpr float limit = 10.0F;
    const float g = gate > limit ? limit : gate;
    const float u = up < -limit ? -limit : (up > limit ? limit : up);
    return (g / (1.0F + tp_moe_declared_exp(-g))) * u;
}
#if defined(__HIPCC__)
// hidden = materialized SwiGLU(g, u) for the configured identity.
__device__ __forceinline__ float tp_moe_swiglu(float g, float u) {
#ifdef STRIX_TRELLIS_TP_MOE_DECLARED_SWIGLU
    float h = tp_moe_declared_swiglu(g, u);
    asm volatile("" : "+v"(h));
    return h;
#else
    const float ex = expf(g < 0.0f ? g : -g);
    const float sigmoid = g < 0.0f ? ex / (1.0f + ex) : 1.0f / (1.0f + ex);
    float activated = g * sigmoid;
    asm volatile("" : "+v"(activated));
    float h = activated * u;
    asm volatile("" : "+v"(h));
    return h;
#endif
}
#endif
} // namespace strix_trellis
#undef STRIX_TP_ACT_HD
