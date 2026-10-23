#pragma once
// Project-owned candidate A8R arithmetic. This is a new identity, never S4.
// Host/device compilation must disable FP contraction and fast math.
#include <cstdint>

#if defined(__HIPCC__)
#define STRIX_A8R_HD __host__ __device__ __forceinline__
#else
#define STRIX_A8R_HD inline
#endif
namespace strix_trellis::a8r {
STRIX_A8R_HD float rounded(float value) {
#if defined(__HIP_DEVICE_COMPILE__)
    asm volatile("" : "+v"(value));
    return value;
#else
    volatile float result = value;
    return result;
#endif
}
STRIX_A8R_HD float add(float a, float b) { return rounded(a + b); }
STRIX_A8R_HD float mul(float a, float b) { return rounded(a * b); }
STRIX_A8R_HD float divide(float a, float b) { return rounded(a / b); }
STRIX_A8R_HD uint32_t bits(float v) {
    uint32_t result; __builtin_memcpy(&result, &v, 4); return result;
}
STRIX_A8R_HD float from_bits(uint32_t v) {
    float result; __builtin_memcpy(&result, &v, 4); return result;
}
STRIX_A8R_HD float canonical_nan() { return from_bits(0x7fc00000u); }
STRIX_A8R_HD bool finite(float v) { return (bits(v) & 0x7fffffffu) < 0x7f800000u; }
STRIX_A8R_HD float absolute(float v) { return from_bits(bits(v) & 0x7fffffffu); }
// A scalar host reference for the same ascending-stride, normalized H128.
// Used by qualification, not by the wave implementation in device kernels.
STRIX_A8R_HD void hadamard(float* values) {
    for (unsigned stride = 1; stride < 128; stride *= 2)
        for (unsigned i = 0; i < 128; ++i) if (!(i & stride)) {
            const float left = values[i], right = values[i + stride];
            values[i] = add(left, right);
            values[i + stride] = add(left, -right);
        }
    for (unsigned i = 0; i < 128; ++i) values[i] = mul(values[i], 0x1.6a09e6p-4f);
}
// Only finite inputs in [-127,127] are admitted. Conversion truncates exactly;
// the explicit half-way rule is independent of the host rounding-mode API.
STRIX_A8R_HD int nearest_even(float v) {
    const bool negative = v < 0.0f;
    const float magnitude = absolute(v);
    int n = int(magnitude);
    const float fraction = add(magnitude, -float(n));
    if (fraction > 0.5f || (fraction == 0.5f && (n & 1))) ++n;
    return negative ? -n : n;
}
STRIX_A8R_HD float scale(float maximum) {
    return maximum < 0x1p-119f ? 1.0f : divide(maximum, 127.0f);
}
STRIX_A8R_HD int quantize(float value, float maximum, float delta) {
    if (maximum < 0x1p-119f) return 0;
    float ratio = divide(value, delta);
    if (ratio > 127.0f) ratio = 127.0f;
    if (ratio < -127.0f) ratio = -127.0f;
    return nearest_even(ratio);
}
// BQ=sum(byte_sum(hash)*q), Q=sum(q), for exactly 32 signed A8 codes.
// The final value is bounded by 1,839,398,912. Intermediates can overflow
// signed32, so perform explicitly defined unsigned modulo32 arithmetic.
STRIX_A8R_HD int32_t numerator(int32_t bq, int32_t q) {
    const uint32_t u = uint32_t(bq) * 887u - uint32_t(q) * 452608u;
    int32_t result; __builtin_memcpy(&result, &u, 4); return result;
}
STRIX_A8R_HD float group(int32_t bq, int32_t q, float delta) {
    return mul(mul(float(numerator(bq, q)), 0x1p-17f), delta);
}
// A declared deterministic exp approximation for z in [-80,0]. Integer
// range reduction and degree-seven Horner evaluation use separate RN32 ops.
// No platform exp implementation is part of this candidate identity.
STRIX_A8R_HD float exp_negative(float z) {
    const int k = nearest_even(mul(z, 0x1.715476p+0f));
    float r = add(z, -mul(float(k), 0x1.62e4p-1f));
    r = add(r, -mul(float(k), 0x1.7f7d1cp-20f));
    float p = 1.0f / 5040.0f;
    p = add(mul(p, r), 1.0f / 720.0f);
    p = add(mul(p, r), 1.0f / 120.0f);
    p = add(mul(p, r), 1.0f / 24.0f);
    p = add(mul(p, r), 1.0f / 6.0f);
    p = add(mul(p, r), 0.5f);
    p = add(mul(p, r), 1.0f);
    p = add(mul(p, r), 1.0f);
    return mul(p, from_bits(uint32_t(k + 127) << 23));
}
STRIX_A8R_HD float silu(float g) {
    if (!finite(g)) return canonical_nan();
    if (g >= 80.0f) return g;
    if (g <= -80.0f) return mul(g, 0.0f);
    const float ex = exp_negative(-absolute(g));
    const float denominator = add(1.0f, ex);
    const float sigmoid = g < 0.0f ? divide(ex, denominator) : divide(1.0f, denominator);
    return mul(g, sigmoid);
}
} // namespace strix_trellis::a8r
#undef STRIX_A8R_HD
