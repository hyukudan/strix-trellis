// CPU lane-exact reference of the K2/mul1 S4 expert operator.
// Emulates, per wave lane, the GPU order: RN32/RN16 prepare, H128 butterflies,
// the K2 window/hash/sad decode, the half FMA, native v_dot2acc_f32_f16 (fitted
// model, cpu_dot2acc.hpp) on two FP32 chains, the 4-lane shuffle reduction,
// the fixed split order and the finish epilogues. Host only, no HIP.
#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include "dot2acc_reference.hpp"

namespace cpu_s4 {
inline float h2f(uint16_t h) { _Float16 v; std::memcpy(&v, &h, 2); return float(v); }
inline uint16_t f2h(float f) { _Float16 v = (_Float16)f; uint16_t h; std::memcpy(&h, &v, 2); return h; }
inline uint32_t fbits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }
// One H128 block held as GPU lanes: value i = lane + 32 * j, lane = i % 32, j = i / 32.
inline void hadamard128(float* v) {
  for (int stride = 1; stride < 32; stride *= 2) {
    float next[128];
    for (int i = 0; i < 128; ++i) {
      const int lane = i & 31, j = i >> 5;
      const float peer = v[((lane ^ stride) & 31) + 32 * j];
      next[i] = (lane & stride) ? peer - v[i] : v[i] + peer;
    }
    std::memcpy(v, next, sizeof next);
  }
  for (int stride = 1; stride < 4; stride *= 2)
    for (int lane = 0; lane < 32; ++lane)
      for (int j = 0; j < 4; ++j) if (!(j & stride)) {
        const float l = v[lane + 32 * j], r = v[lane + 32 * (j + stride)];
        v[lane + 32 * j] = l + r; v[lane + 32 * (j + stride)] = l - r;
      }
  for (int i = 0; i < 128; ++i) v[i] = v[i] * 0x1.6a09e6p-4f;
}
// pipeline_prepare: x*suh (RN32) -> RN16 -> H128 -> RN16, per 128 block.
inline std::vector<uint16_t> prepare(const float* x, const uint16_t* suh, unsigned width) {
  std::vector<uint16_t> out(width);
  for (unsigned b = 0; b < width / 128; ++b) {
    float v[128];
    for (int i = 0; i < 128; ++i) { const unsigned c = b * 128 + i; v[i] = h2f(f2h(x[c] * h2f(suh[c]))); }
    hadamard128(v);
    for (int i = 0; i < 128; ++i) out[b * 128 + i] = f2h(v[i]);
  }
  return out;
}
// __hfma2 lane: RN16(a*b+c) computed exactly in double (all terms fit).
inline uint16_t hfma(uint16_t a, uint16_t b, uint16_t c) {
  const double r = double(h2f(a)) * double(h2f(b)) + double(h2f(c));
  // r is exact in double; one direct double -> binary16 rounding (RNE).
  const _Float16 v = (_Float16)r; uint16_t o; std::memcpy(&o, &v, 2); return o;
}
inline uint32_t hash16(uint32_t s) { return (s & 65535u) * 0x83dcd12du; }
inline uint32_t sad(uint32_t h) { return (h & 255) + ((h >> 8) & 255) + ((h >> 16) & 255) + (h >> 24); }
// Partial sums of one split for all N outputs: out[n] = lane-group reduction.
// trellis: uint32 words, tile (k, tn) at (k*(N/16)+tn)*16; act: RN16 bits [I].
inline void split_partials(const uint32_t* trellis, const uint16_t* act, unsigned I, unsigned N,
                           unsigned split, float* out) {
  const unsigned tiles = I / 16, begin = tiles * split / 4, end = tiles * (split + 1) / 4;
  for (unsigned tn = 0; tn < N / 16; ++tn) {
    float lo[32] = {}, hi[32] = {};
    for (unsigned k = begin; k < end; ++k) {
      const uint32_t* base = trellis + (size_t(k) * (N / 16) + tn) * 16;
      for (int lane = 0; lane < 32; ++lane) {
        const int index = ((lane * 8 + 1) * 2 + 512 - 16) / 32 - 16;
        const int ia = index < 0 ? index + 16 : index, ib = index + 1 >= 16 ? index + 1 - 16 : index + 1;
        const int shift = ((lane * 8 + 1) * 2 + 512 - 16) & 31;
        const uint32_t a = base[ia], b = base[ib];
        const uint32_t w = shift == 2 ? a : ((a << 16) | (b >> 16));
        uint32_t h[8];
        for (int j = 0; j < 8; ++j) h[j] = hash16(w >> (14 - 2 * j));
        const uint16_t* xa = act + k * 16 + 2 * (lane & 3);
        const uint16_t* xb = xa + 8;
        for (int pair = 0; pair < 4; ++pair) {
          const uint32_t both = 0x64006400u + sad(h[2 * pair]) + (sad(h[2 * pair + 1]) << 16);
          const uint16_t w0 = hfma(uint16_t(both), 0x1eee, 0xc931), w1 = hfma(uint16_t(both >> 16), 0x1eee, 0xc931);
          const uint16_t* xv = (pair & 1) ? xb : xa;
          float& acc = pair < 2 ? lo[lane] : hi[lane];
          acc = cpu_dot2acc::dot2acc(w0, w1, xv[0], xv[1], acc);
        }
      }
    }
    for (int half = 0; half < 2; ++half) {
      const float* s = half ? hi : lo;
      for (int g = 0; g < 8; ++g) {
        // v += shfl_down(v,2,4); v += shfl_down(v,1,4) in a group of four lanes.
        const float v0 = s[4 * g], v1 = s[4 * g + 1], v2 = s[4 * g + 2], v3 = s[4 * g + 3];
        const float u0 = v0 + v2, u1 = v1 + v3;
        out[tn * 16 + g + half * 8] = u0 + u1;
      }
    }
  }
}
// Sum of four splits in fixed order, H128, svh, bias: the projected value.
inline void project_finish(const float* const parts[4], const uint16_t* svh, const float* bias,
                           unsigned N, float* out) {
  for (unsigned b = 0; b < N / 128; ++b) {
    float v[128] = {};
    for (int s = 0; s < 4; ++s) for (int i = 0; i < 128; ++i) v[i] += parts[s][b * 128 + i];
    hadamard128(v);
    for (int i = 0; i < 128; ++i) {
      const unsigned c = b * 128 + i;
      float value = v[i] * h2f(svh[c]);
      if (bias) value += bias[c];
      out[c] = value;
    }
  }
}
} // namespace cpu_s4
