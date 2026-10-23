#pragma once
// Project-owned pipeline using the native cooperative grid primitive. Creation
// rejects private frames and bounds the grid by occupancy; the default uses one
// CTA per reported compute unit.

template <bool Down>
__device__ __forceinline__ void persistent_prepare(TpMoeLaunch l, TpMoeDeviceRows& m, bool& owner, unsigned tile, unsigned candidate, unsigned logical_grid_x) {
  if (threadIdx.x == 0)
    owner = pipeline_resolve(l, m, Down, candidate);
  __syncthreads();
  if (!owner)
    return;
  constexpr unsigned Width = Down ? 1152 : 5120;
  constexpr unsigned Blocks = Width / 128;
  const unsigned lane = threadIdx.x & 31;
  const unsigned wave = __builtin_amdgcn_readfirstlane(unsigned(threadIdx.x) / 32);
  const unsigned bank_index = __builtin_amdgcn_readfirstlane(m.bank_index);
  const auto &e = l.bank ? l.bank[bank_index] : l.table->experts[bank_index];
  // Grid size depends on the projection, not global window capacity. Each
  // wave visits the remaining expert rows without launching empty row CTAs.
  for (unsigned unit = tile * 4 + wave;
       unit < m.row_count * Blocks; unit += logical_grid_x * 4) {
    const unsigned row = unit / Blocks, h = unit % Blocks;
#pragma unroll
    for (unsigned projection = 0; projection < (Down ? 1u : 2u); ++projection) {
      const auto &w = Down ? e.down : projection ? e.up : e.gate;
      float v[4];
#pragma unroll
      for (unsigned j = 0; j < 4; ++j) {
        const unsigned column = h * 128 + lane + j * 32;
        const float scaled =
            materialize_fp32(Input{l, m, Down}.load(row, column) *
                             __half2float(__ushort_as_half(w.suh[column])));
        v[j] = float(_Float16(scaled));
      }
      hadamard128(v);
      auto *x = pipeline_activation(l, m.token_rows[row], m.route_slots[row],
                                    Down, projection);
#pragma unroll
      for (unsigned j = 0; j < 4; ++j)
        x[h * 128 + lane + j * 32] = _Float16(v[j]);
    }
  }
}

template <int Capacity, bool Down>
__device__ __forceinline__ void persistent_core(TpMoeLaunch l, TpMoeDeviceRows& m, bool& owner, unsigned tile, unsigned candidate, unsigned coordinate_z) {
  if (threadIdx.x == 0) {
#ifdef STRIX_TRELLIS_TP_MOE_SINGLE_ROW
    owner = pipeline_resolve_route(l, m, Down);
#else
    owner = pipeline_resolve(l, m, Down, candidate);
#endif
  }
  __syncthreads();
  if (!owner)
    return;
  const unsigned bank_index = __builtin_amdgcn_readfirstlane(m.bank_index);
  const auto &e = l.bank ? l.bank[bank_index] : l.table->experts[bank_index];
  const unsigned projection = Down ? 0 : coordinate_z / 4;
  const auto &w = Down ? e.down : projection ? e.up : e.gate;
#ifdef STRIX_TRELLIS_TP_MOE_SINGLE_ROW
  pipeline_project<1, Down>(w, l, m, projection, tile, coordinate_z & 3);
#elif STRIX_TRELLIS_TP_MOE_ROW_CAP > 0
  // Separate the numerical row core from the window's maximum capacity.
  // Splitting row groups repeats decode but preserves every output chain.
  constexpr unsigned Chunk = STRIX_TRELLIS_TP_MOE_ROW_CAP;
  if constexpr (Capacity <= Chunk) {
    if (m.row_count == 1)
      pipeline_project<1, Down>(w, l, m, projection, tile, coordinate_z & 3);
    else
      pipeline_project<Capacity, Down>(w, l, m, projection, tile, coordinate_z & 3);
  } else for (unsigned first = 0; first < m.row_count; first += Chunk) {
    TpMoeDeviceRows slice{};
    slice.bank_index = m.bank_index;
    slice.row_count = min(Chunk, m.row_count - first);
#pragma unroll
    for (unsigned row = 0; row < Chunk; ++row) {
      const unsigned source = first + (row < slice.row_count ? row : 0);
      slice.token_rows[row] = m.token_rows[source];
      slice.route_slots[row] = m.route_slots[source];
    }
    if (slice.row_count == 1)
      pipeline_project<1, Down>(w, l, slice, projection, tile, coordinate_z & 3);
    else
      pipeline_project<Chunk, Down>(w, l, slice, projection, tile, coordinate_z & 3);
  }
#else
  if (m.row_count == 1)
    pipeline_project<1, Down>(w, l, m, projection, tile, coordinate_z & 3);
  else if constexpr (Capacity >= 2) {
    if (m.row_count <= 2)
      pipeline_project<2, Down>(w, l, m, projection, tile, coordinate_z & 3);
    else if constexpr (Capacity >= 4) {
      if (m.row_count <= 4)
        pipeline_project<4, Down>(w, l, m, projection, tile, coordinate_z & 3);
      else if constexpr (Capacity >= 8)
        pipeline_project<8, Down>(w, l, m, projection, tile, coordinate_z & 3);
    }
  }
#endif
}

template <bool PrepareDown>
__device__ __forceinline__ void persistent_gate_finish(TpMoeLaunch l, unsigned tile, unsigned route) {
  const unsigned row = route / 6, slot = route % 6,
                 lane = threadIdx.x & 31;
  unsigned expert;
  if (l.bank) {
    const auto *ids = l.device_route_ids + row * 6;
    expert = ids[slot];
    if (expert >= l.bank_count)
      return;
    // Gate/up, like the original owner map, ignores scores and writes only
    // the first duplicate slot. Down/C validate malformed token rows.
    for (unsigned previous = 0; previous < slot; ++previous)
      if (ids[previous] == expert)
        return;
  } else
    expert = l.table->expert_indices[row][slot];
  const auto &e = l.bank ? l.bank[expert] : l.table->experts[expert];
  float gate[4] = {}, up[4] = {};
#pragma unroll
  for (unsigned s = 0; s < 4; ++s) {
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      const unsigned column = tile * 128 + lane + j * 32;
      gate[j] += l.partials[tp_moe_pipeline_gate_index(l.rows, 0, row, slot, s,
                                                       column)];
      up[j] += l.partials[tp_moe_pipeline_gate_index(l.rows, 1, row, slot, s,
                                                     column)];
    }
  }
  hadamard128(gate);
  hadamard128(up);
  float hidden[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const unsigned column = tile * 128 + lane + j * 32;
    float g = gate[j] * __half2float(__ushort_as_half(e.gate.svh[column]));
    float u = up[j] * __half2float(__ushort_as_half(e.up.svh[column]));
    if (e.gate.bias)
      g += e.gate.bias[column];
    if (e.up.bias)
      u += e.up.bias[column];
        hidden[j] = tp_moe_swiglu(g, u);
    l.hidden[(size_t(row) * 6 + slot) * 1152 + column] = hidden[j];
  }
  if constexpr (PrepareDown) {
    // Every down H128 uses exactly the hidden tile produced by this wave.
    // Keep its F32 materialization and both RN16 steps; no cross-CTA read
    // or grid barrier is needed before the next ordered core launch.
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      const unsigned column = tile * 128 + lane + j * 32;
      const float scaled = materialize_fp32(hidden[j] *
          __half2float(__ushort_as_half(e.down.suh[column])));
      hidden[j] = float(_Float16(scaled));
    }
    hadamard128(hidden);
    auto* x = pipeline_activation(l, row, slot, true);
#pragma unroll
    for (unsigned j = 0; j < 4; ++j)
      x[tile * 128 + lane + j * 32] = _Float16(hidden[j]);
  }
}

__device__ __forceinline__ void persistent_combine(TpMoeLaunch l, unsigned tile, unsigned token) {
    const unsigned row = token, lane = threadIdx.x & 31;
    uint32_t selected[6]; float scores[6];
    #pragma unroll
    for (int s = 0; s < 6; ++s) {
        selected[s] = l.bank ? l.device_route_ids[row * 6 + s] : l.table->expert_indices[row][s];
        scores[s] = l.bank ? l.device_route_scores[row * 6 + s] : l.table->route_scores[row][s];
    }
    const bool valid = !l.bank || tp_moe_device_row_valid(selected, scores, l.bank_count);
    float combined[4] = {};
    if (valid) {
        #pragma unroll
        for (int slot = 0; slot < 6; ++slot) {
            if (scores[slot] == 0.0f) continue;
            const auto& w = l.bank ? l.bank[selected[slot]].down : l.table->experts[selected[slot]].down;
            float v[4] = {};
            #pragma unroll
            for (int split = 0; split < 4; ++split) {
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    v[j] += l.partials[tp_moe_split_index(row, slot, split, tile * 128 + lane + j * 32)];
            }
            hadamard128(v);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned column = tile * 128 + lane + j * 32;
                float result = v[j] * __half2float(__ushort_as_half(w.svh[column]));
                if (w.bias) result += w.bias[column];
                const float weighted = materialize_fp32(scores[slot] * result);
                combined[j] = materialize_fp32(combined[j] + weighted);
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint64_t output = uint64_t(row) * l.y_stride + tile * 128 + lane + j * 32;
        float v = materialize_fp32(l.alpha * combined[j]);
        if (l.beta != 0.0f) v = materialize_fp32(v + materialize_fp32(l.beta * l.y[output]));
        l.y[output] = valid ? v : __builtin_bit_cast(float, uint32_t(0x7fc00000u));
    }
}

template <int Capacity>
__global__ __launch_bounds__(128) void persistent_full(TpMoeLaunch l) {
  __shared__ TpMoeDeviceRows map;
  __shared__ TpMoeDeviceRows down_map;
  __shared__ bool owner;
  auto grid = cooperative_groups::this_grid();
  const unsigned owners = l.bank ? l.rows * 6 : l.expert_count;
  const unsigned wave = __builtin_amdgcn_readfirstlane(unsigned(threadIdx.x) / 32);
  // Build both score-independent gate and score-filtered down maps once.
  for (unsigned candidate = blockIdx.x; candidate < owners; candidate += gridDim.x) {
    if (threadIdx.x == 0) {
      // Only the owner thread touches metadata before grid.sync; reuse shared
      // maps instead of giving every work item a 148-byte private frame.
      map = {};
      down_map = {};
      if (!resolve(l, map, false, candidate)) map.row_count = 0;
      if (!resolve(l, down_map, true, candidate)) down_map.row_count = 0;
      pipeline_maps(l, false)[candidate] = map;
      pipeline_maps(l, true)[candidate] = down_map;
    }
  }
  grid.sync();
  for (unsigned unit = blockIdx.x; unit < owners * 10; unit += gridDim.x) {
    persistent_prepare<false>(l, map, owner, unit % 10, unit / 10, 10);
    __syncthreads(); // Metadata cannot be replaced while another wave reads it.
  }
  grid.sync();
  for (unsigned unit = blockIdx.x; unit < owners * 18 * 8; unit += gridDim.x) {
    persistent_core<Capacity, false>(l, map, owner, unit % 18,
        (unit / 18) % owners, unit / (18 * owners));
    __syncthreads();
  }
  grid.sync();
  for (unsigned unit = blockIdx.x * 4 + wave; unit < l.rows * 6 * 9; unit += gridDim.x * 4)
    persistent_gate_finish<true>(l, unit % 9, unit / 9);
  grid.sync();
  for (unsigned unit = blockIdx.x; unit < owners * 80 * 4; unit += gridDim.x) {
    persistent_core<Capacity, true>(l, map, owner, unit % 80,
        (unit / 80) % owners, unit / (80 * owners));
    __syncthreads();
  }
  grid.sync();
  for (unsigned unit = blockIdx.x * 4 + wave; unit < l.rows * 40; unit += gridDim.x * 4)
    persistent_combine(l, unit % 40, unit / 40);
}
const void* persistent_function(unsigned rows) {
  if (rows == 1) return reinterpret_cast<const void*>(persistent_full<1>);
  if (rows == 2) return reinterpret_cast<const void*>(persistent_full<2>);
  if (rows <= 4) return reinterpret_cast<const void*>(persistent_full<4>);
  return reinterpret_cast<const void*>(persistent_full<8>);
}
