#pragma once
// Alternate instruction selection for the same unsigned16 -> modulo32 hash.
// Included inside tp_moe_split after the project-owned K2 decoder.
__device__ __forceinline__ void pipeline_hash(const uint32_t (&state)[8],
                                             uint32_t (&hash)[8]) {
#ifdef STRIX_TRELLIS_TP_MOE_HASH32
#pragma unroll
  for (unsigned j = 0; j < 8; ++j) {
    const uint32_t input = state[j] & 65535u;
    asm("v_mul_lo_u32 %0, %1, %2"
        : "=v"(hash[j]) : "v"(input), "s"(0x83dcd12du));
  }
#else
  k2_local_kernel::staged(state, hash);
#endif
}

__device__ __forceinline__ void pipeline_consume(k2_local_kernel::Raw raw,
                                                float& low, float& high) {
#ifdef STRIX_TRELLIS_TP_MOE_HASH32
  uint32_t state[8], hash[8];
  k2_local_kernel::states(raw, state);
  pipeline_hash(state, hash);
  const Pair a = __builtin_bit_cast(Pair, raw.x);
  const Pair b = __builtin_bit_cast(Pair, raw.y);
#pragma unroll
  for (unsigned pair = 0; pair < 4; ++pair) {
    const Pair weight = k2_local_kernel::sad_pair(hash[2 * pair], hash[2 * pair + 1]);
    if (pair < 2)
      low = __builtin_amdgcn_fdot2(weight, pair & 1 ? b : a, low, false);
    else
      high = __builtin_amdgcn_fdot2(weight, pair & 1 ? b : a, high, false);
  }
#else
  k2_local_kernel::consume(raw, low, high);
#endif
}

#ifdef STRIX_TRELLIS_TP_MOE_LEAN_CORE
// Same bits as read_raw/consume with fewer VALU per tile: the activation pair
// is read through a wave-uniform base plus a constant lane offset (no 64-bit
// VGPR address per tile), and the lane window is one alignbit by a per-lane
// constant (odd lanes read word ia twice and shift by 0), with no select.
__device__ __forceinline__ k2_local_kernel::Raw lean_read(const uint32_t* p, const _Float16* x,
                                                       int k, int n, int tn) {
  const int lane = threadIdx.x & 31;
  const int index = ((lane * 8 + 1) * 2 + 512 - 16) / 32 - 16;
  const int ia = index < 0 ? index + 16 : index;
  const int ib = (lane & 1) ? ia : (index + 1 >= 16 ? index + 1 - 16 : index + 1);
  auto* base = reinterpret_cast<const unsigned char*>(p + (size_t(k) * (n / 16) + tn) * 16);
  auto* xb = reinterpret_cast<const unsigned char*>(x) + size_t(k) * 32;
  k2_local_kernel::Raw r;
  // Global address space with 32-bit lane offsets: saddr form, no VGPR pointer.
  using G = const __attribute__((address_space(1))) unsigned char;
  G* gw = (G*)base;
  uint32_t oa = uint32_t(ia) * 4u, ob = uint32_t(ib) * 4u;
  asm("" : "+v"(oa), "+v"(ob));
  r.a = *(const __attribute__((address_space(1))) uint32_t*)(gw + oa);
  r.b = *(const __attribute__((address_space(1))) uint32_t*)(gw + ob);
  G* gx = (G*)xb;
  uint32_t ox = uint32_t(lane & 3) * 4u;
  asm("" : "+v"(ox));
  r.x = *(const __attribute__((address_space(1))) uint32_t*)(gx + ox);
  r.y = *(const __attribute__((address_space(1))) uint32_t*)(gx + ox + 16);
  return r;
}
__device__ __forceinline__ void lean_consume(k2_local_kernel::Raw r, float& lo, float& hi) {
  const uint32_t shift = (threadIdx.x & 1) ? 0u : 16u;
  const uint32_t w = __builtin_amdgcn_alignbit(r.a, r.b, shift);
  uint32_t s[8], hsh[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) s[j] = w >> (14 - 2 * j);
  k2_local_kernel::staged(s, hsh);
  const Pair a = __builtin_bit_cast(Pair, r.x), b = __builtin_bit_cast(Pair, r.y);
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    const Pair weight = k2_local_kernel::sad_pair(hsh[2 * pair], hsh[2 * pair + 1]);
    if (pair < 2) lo = __builtin_amdgcn_fdot2(weight, (pair & 1) ? b : a, lo, false);
    else hi = __builtin_amdgcn_fdot2(weight, (pair & 1) ? b : a, hi, false);
  }
}
#endif
