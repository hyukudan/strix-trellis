// CPU model of native gfx1151 v_dot2acc_f32_f16 / v_dot2_f32_f16.
// Arithmetic model characterized in ROCm/rocm-systems issue #12056 and merged in
// rocjitsu gfx11_dot2.h (PR #12120, MIT): the three addends (FP32 accumulator,
// two exact F16 products) are aligned on one integer grid
//   grid = max(max_lane(E(a) + E(b)) - 24, max(E(c), E(p0), E(p1)) - 26)
// (E(a) = operand exponent floored at the F16 minimum normal, E(x) of a term =
// its leading-bit exponent), truncated toward zero onto that grid and summed in
// the accumulator's sign frame, where an opposing term contributes ~m (no +1).
// The sum is rounded once to 24 bits (RNE); results below 2^-126 become +0, a
// subnormal accumulator is flushed (its sign still sets the frame), and the
// result is never -0. Revalidated bitwise against our gfx1151 dump (dot2cal).
#pragma once
#include <cstdint>
#include <cstring>
#include <initializer_list>
namespace cpu_dot2acc {
inline const char* model_name() { return "gfx11-dot2 (rocm-systems#12056 / rocjitsu gfx11_dot2.h)"; }
namespace detail {
constexpr int empty = -1024;
inline int bit_width(uint64_t v) { return v ? 64 - __builtin_clzll(v) : 0; }
struct Term { uint64_t m; int e; bool neg; int lead() const { return m ? e + bit_width(m) - 1 : empty; } };
struct Half {
  uint16_t b;
  bool nan() const { return (b & 0x7fff) > 0x7c00; }
  bool inf() const { return (b & 0x7fff) == 0x7c00; }
  bool neg() const { return b >> 15; }
  unsigned field() const { return (b & 0x7fff) >> 10; }
  uint64_t sig() const { return field() ? (1024u | (b & 1023u)) : (b & 1023u); }
  int align() const { return int(field() ? field() : 1u) - 15; }
  Term mul(Half o) const { return {sig() * o.sig(), align() + o.align() - 20, neg() != o.neg()}; }
};
inline uint32_t pack(int64_t units, int grid, bool frame_negative) {
  if (!units) return 0;
  const uint32_t sign = uint32_t((units < 0) != frame_negative) << 31;
  const uint64_t mag = units < 0 ? uint64_t(-units) : uint64_t(units);
  const int top = bit_width(mag) - 1;
  int exponent = top + grid;
  const int shift = top - 23;
  uint64_t sig;
  if (shift <= 0) sig = mag << -shift;
  else {
    const uint64_t tail = mag & ((uint64_t(1) << shift) - 1), half = uint64_t(1) << (shift - 1);
    sig = (mag >> shift) + (tail > half || (tail == half && ((mag >> shift) & 1)));
  }
  if (sig == 0x1000000) { sig >>= 1; ++exponent; }
  if (exponent < -126) return 0;
  if (exponent > 127) return sign | 0x7f800000u;
  return sign | (uint32_t(exponent + 127) << 23) | (uint32_t(sig) & 0x7fffff);
}
} // namespace detail
inline uint32_t dot2acc_bits(uint16_t a0b, uint16_t a1b, uint16_t b0b, uint16_t b1b, uint32_t acc) {
  using namespace detail;
  const Half a0{a0b}, b0{b0b}, a1{a1b}, b1{b1b};
  if (a0.nan() || b0.nan() || a1.nan() || b1.nan()) return 0xffc00a3du;
  if ((a0.inf() && !b0.sig()) || (b0.inf() && !a0.sig()) || (a1.inf() && !b1.sig()) || (b1.inf() && !a1.sig())) return 0xffc00000u;
  bool pinf = false, ninf = false;
  auto lane_inf = [&](Half l, Half r) { if (l.inf() || r.inf()) (l.neg() != r.neg() ? ninf : pinf) = true; };
  lane_inf(a0, b0); lane_inf(a1, b1);
  if (pinf && ninf) return 0xffc00000u;
  if ((acc & 0x7fffffffu) > 0x7f800000u) return acc | 0x00400000u;
  pinf |= acc == 0x7f800000u; ninf |= acc == 0xff800000u;
  if (pinf || ninf) return pinf && ninf ? 0xffc00000u : ninf ? 0xff800000u : 0x7f800000u;
  const Term p0 = a0.mul(b0), p1 = a1.mul(b1);
  const unsigned ce = (acc >> 23) & 255;
  const Term c{ce ? (acc & 0x7fffffu) | 0x800000u : 0, int(ce) - 150, bool(acc >> 31)};
  if (!(c.m | p0.m | p1.m)) return 0;
  int product_grid = empty;
  if (p0.m && a0.align() + b0.align() > product_grid) product_grid = a0.align() + b0.align();
  if (p1.m && a1.align() + b1.align() > product_grid) product_grid = a1.align() + b1.align();
  product_grid -= 24;
  int lead = c.lead(); if (p0.lead() > lead) lead = p0.lead(); if (p1.lead() > lead) lead = p1.lead();
  const int grid = product_grid > lead - 26 ? product_grid : lead - 26;
  int64_t total = 0;
  for (const Term& t : {c, p0, p1}) {
    const int s = t.e - grid;
    const uint64_t m = !t.m || s <= -64 ? 0 : s < 0 ? t.m >> -s : t.m << s;
    total += t.neg != c.neg ? -int64_t(m) - 1 : int64_t(m);
  }
  return pack(total, grid, c.neg);
}
// acc + a0*b0 + a1*b1 with the native accumulation semantics.
inline float dot2acc(uint16_t a0, uint16_t a1, uint16_t b0, uint16_t b1, float acc) {
  uint32_t ab; std::memcpy(&ab, &acc, 4);
  const uint32_t o = dot2acc_bits(a0, a1, b0, b1, ab); float f; std::memcpy(&f, &o, 4); return f;
}
} // namespace cpu_dot2acc
