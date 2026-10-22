#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>
static uint16_t host_state(const uint32_t* data, int bits, int position) {
    int width = bits * 256;
    int end = (position + 1) * bits;
    unsigned state = 0;
    for (int j = 16; j > 0; --j) {
        int bit = (end - j + width) % width;
        state = (state << 1) | ((data[bit / 32] >> (31 - bit % 32)) & 1);
    }
    return uint16_t(state);
}
static float host_half(uint16_t bits) {
    bool negative = bits >> 15;
    int exponent = (bits >> 10) & 31;
    int fraction = bits & 1023;
    float value = exponent ? std::ldexp(1.0f + fraction / 1024.0f, exponent - 15) : std::ldexp(float(fraction), -24);
    return negative ? -value : value;
}
static uint16_t host_codebook(uint16_t state, int cb) {
    float value;
    if (cb == 2) {
        uint32_t p = uint32_t(state) * 0x83dcd12du;
        unsigned sum = 0;
        for (int i = 0; i < 4; ++i) sum += (p >> (8 * i)) & 255;
        value = std::fma(float(1024 + sum), host_half(0x1eee), host_half(0xc931));
    } else {
        uint32_t mixed = cb == 1 ? uint32_t(state) * 0xcbac1fedu : uint32_t(state) * 89226354u + 64248484u;
        uint32_t h = (mixed & 0x8fff8fffu) ^ 0x3b603b60u;
        value = host_half(uint16_t(h)) + host_half(uint16_t(h >> 16));
    }
    _Float16 rounded = (_Float16)value;
    uint16_t result; std::memcpy(&result, &rounded, sizeof(result)); return result;
}

