#pragma once
// Independent scheduling experiments. Included inside tp_moe_split, so these
// helpers share its types without changing the numerical S4 core contract.

template <int Rows, int Depth>
__device__ __forceinline__ void pipeline_prefetch_rows(
    const DeviceWeights& w, const _Float16* const (&x)[Rows], unsigned n,
    int begin, int end, float (&sum)[Rows][2]) {
  static_assert(Depth == 1 || Depth == 2 || Depth == 4);
  const unsigned lane = threadIdx.x & 31;
  const auto window = tp_moe_reader_window<2>(lane);
  for (int k = begin; k < end; k += Depth) {
    k2_local_kernel::Raw raw[Depth];
    Pair input[Depth][Rows][2];
    // Both weights and all row operands are visible loads before the hash.
    // Depth changes only the lookahead, never the per-output K/pair order.
#pragma unroll
    for (int j = 0; j < Depth; ++j) {
      if (k + j < end) {
        const auto* base = reinterpret_cast<const unsigned char*>(
            w.trellis + (size_t(k + j) * (w.output / 16) + n) * 16);
        raw[j].a = k2_local_kernel::word_load(base, window.a);
        raw[j].b = k2_local_kernel::word_load(base, window.b);
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
          const int column = (k + j) * 16 + 2 * (lane & 3);
          __builtin_memcpy(&input[j][row][0], x[row] + column, 4);
          __builtin_memcpy(&input[j][row][1], x[row] + column + 8, 4);
        }
      }
    }
#pragma unroll
    for (int j = 0; j < Depth; ++j) {
      if (k + j < end) {
        uint32_t states[8], hashes[8];
        k2_local_kernel::states(raw[j], states);
        pipeline_hash(states, hashes);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
          const Pair weight = k2_local_kernel::sad_pair(
              hashes[2 * pair], hashes[2 * pair + 1]);
#pragma unroll
          for (int row = 0; row < Rows; ++row)
            sum[row][pair / 2] = __builtin_amdgcn_fdot2(
                weight, input[j][row][pair & 1], sum[row][pair / 2], false);
        }
      }
    }
  }
}

__device__ __forceinline__ bool pipeline_resolve_route(
    const TpMoeLaunch& l, TpMoeDeviceRows& m, bool down) {
  const unsigned row = blockIdx.y / 6, slot = blockIdx.y % 6;
  const auto* ids = l.bank ? l.device_route_ids + row * 6
                          : l.table->expert_indices[row];
  const auto* scores = l.bank ? l.device_route_scores + row * 6
                             : l.table->route_scores[row];
  const unsigned expert = ids[slot];
  if (l.bank) {
    if (expert >= l.bank_count) return false;
    // Match first-slot ownership in gate/up even for malformed route rows.
    for (unsigned previous = 0; previous < slot; ++previous)
      if (ids[previous] == expert) return false;
    if (down && !tp_moe_device_row_valid(ids, scores, l.bank_count)) return false;
  }
  if (down && scores[slot] == 0.0f) return false;
  m.bank_index = expert;
  m.row_count = 1;
  m.token_rows[0] = row;
  m.route_slots[0] = slot;
  return true;
}
