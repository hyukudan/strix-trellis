#pragma once
// Independent scalar reference for bounded K2 A8R qualification fixtures.
// It decodes circular bit positions, not the GPU lane/window reader.
#include "tp_moe_a8r_math.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace strix_trellis::a8r_reference {
struct Weights {
    const uint32_t* trellis;
    const uint16_t *suh, *svh;
    const float* bias;
    unsigned input, output;
};
inline float half(uint16_t h) {
    const bool negative = h >> 15;
    const unsigned exponent = (h >> 10) & 31, fraction = h & 1023;
    if (exponent == 31) return a8r::from_bits((uint32_t(negative) << 31) | 0x7f800000u | (fraction ? 0x400000u : 0u));
    float value = exponent ? std::ldexp(1.f + float(fraction) / 1024.f, int(exponent) - 15)
                           : std::ldexp(float(fraction), -24);
    return negative ? -value : value;
}
inline uint16_t byte_sum(const Weights& w, unsigned k, unsigned n) {
    const auto* tile = w.trellis + (size_t(k / 16) * (w.output / 16) + n / 16) * 16;
    const unsigned row = k & 15, column = n & 15;
    const unsigned position = ((column & 7) << 5) | ((row & 6) << 2) |
        (row & 1) | ((row & 8) >> 2) | ((column & 8) >> 1);
    const unsigned end = (position + 1) * 2;
    uint32_t state = 0;
    for (unsigned bit = 0; bit < 16; ++bit) {
        const unsigned address = (end + 512 - 16 + bit) & 511;
        state = (state << 1) | ((tile[address / 32] >> (31 - address % 32)) & 1);
    }
    const uint32_t hash = state * 0x83dcd12du;
    return (hash & 255) + ((hash >> 8) & 255) + ((hash >> 16) & 255) + (hash >> 24);
}
// Optional output-major decoded fixture cache. No full model materialization
// is required or implied by this qualification helper.
inline std::vector<uint16_t> decode(const Weights& w) {
    std::vector<uint16_t> b(size_t(w.input) * w.output);
    for (unsigned n = 0; n < w.output; ++n)
        for (unsigned k = 0; k < w.input; ++k) b[size_t(n) * w.input + k] = byte_sum(w, k, n);
    return b;
}
struct Quantized {
    std::vector<int8_t> q;
    std::vector<float> delta;
};
inline Quantized prepare(const Weights& w, const float* input) {
    Quantized result{std::vector<int8_t>(w.input), std::vector<float>(w.input / 32)};
    for (unsigned h = 0; h < w.input / 128; ++h) {
        float values[128];
        for (unsigned j = 0; j < 128; ++j)
            values[j] = float(_Float16(a8r::mul(input[h * 128 + j], half(w.suh[h * 128 + j]))));
        a8r::hadamard(values);
        for (unsigned group = 0; group < 4; ++group) {
            float maximum = 0.f;bool invalid = false;
            for (unsigned j = 0; j < 32; ++j) {
                const float value = values[group * 32 + j];
                invalid |= !std::isfinite(value);maximum = std::max(maximum, std::fabs(value));
            }
            const bool tiny = maximum < 0x1p-119f;
            const float delta = invalid ? a8r::canonical_nan() : tiny ? 1.f : a8r::rounded(maximum / 127.f);
            result.delta[h * 4 + group] = delta;
            for (unsigned j = 0; j < 32; ++j) {
                float code = invalid || tiny ? 0.f : a8r::rounded(values[group * 32 + j] / delta);
                code = std::max(-127.f, std::min(127.f, code));
                result.q[h * 128 + group * 32 + j] = int(std::nearbyint(code));
            }
        }
    }
    return result;
}
inline std::vector<float> partials(const Weights& w, const Quantized& x, const uint16_t* decoded = nullptr) {
    if (!w.input || !w.output || w.input % 128 || w.output % 128)
        throw std::invalid_argument("A8R reference needs complete H128 blocks");
    std::vector<float> result(size_t(4) * w.output);
    for (unsigned n = 0; n < w.output; ++n) for (unsigned split = 0; split < 4; ++split) {
        float total = 0.f;
        for (unsigned group = (w.input / 32) * split / 4; group < (w.input / 32) * (split + 1) / 4; ++group) {
            // Sum exact affine integer weights directly, independently of
            // the device's BQ/Q decomposition and modulo32 implementation.
            int64_t numerator = 0;
            for (unsigned k = group * 32; k < (group + 1) * 32; ++k) {
                const unsigned b = decoded ? decoded[size_t(n) * w.input + k] : byte_sum(w, k, n);
                numerator += (int64_t(887) * b - 452608) * x.q[k];
            }
            const float rounded_integer = float(numerator);
            const float exact_scaled = std::ldexp(rounded_integer, -17);
            const float product = float(double(exact_scaled) * double(x.delta[group]));
            total = a8r::add(total, product);
        }
        result[size_t(split) * w.output + n] = total;
    }
    return result;
}
inline std::vector<float> finish(const Weights& w, const std::vector<float>& p, float bias_offset = 0.f) {
    std::vector<float> result(w.output);
    for (unsigned h = 0; h < w.output / 128; ++h) {
        float values[128] = {};
        for (unsigned split = 0; split < 4; ++split)
            for (unsigned j = 0; j < 128; ++j)
                values[j] = a8r::add(values[j], p[size_t(split) * w.output + h * 128 + j]);
        a8r::hadamard(values);
        for (unsigned j = 0; j < 128; ++j) {
            const unsigned n = h * 128 + j;
            result[n] = a8r::mul(values[j], half(w.svh[n]));
            if (w.bias) {
                float bias = w.bias[n];
                if (bias_offset != 0.f) bias = a8r::add(bias, bias_offset);
                result[n] = a8r::add(result[n], bias);
            }
        }
    }
    return result;
}
} // namespace strix_trellis::a8r_reference
