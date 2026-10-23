#pragma once
#include "tp_moe_hash.hpp"
#include "tp_moe_schedule.hpp"
#ifndef STRIX_TRELLIS_TP_MOE_SINGLE_DEPTH
#define STRIX_TRELLIS_TP_MOE_SINGLE_DEPTH 4
#endif
#ifndef STRIX_TRELLIS_TP_MOE_DOWN_DEPTH
#define STRIX_TRELLIS_TP_MOE_DOWN_DEPTH 0
#endif

// Included inside tp_moe_split after its routing/input helpers. No launch,
// allocation, copy or synchronization outside the caller's ordered stream.

__device__ __forceinline__ _Float16 *
pipeline_activation(const TpMoeLaunch &l, unsigned row, unsigned slot,
                    bool down, unsigned projection = 0) {
  auto *base = reinterpret_cast<_Float16 *>(
      reinterpret_cast<unsigned char *>(l.partials) +
      tp_moe_split_partial_bytes(l.rows));
  return base + tp_moe_pipeline_activation_index(row, slot, down, projection);
}

#ifdef STRIX_TRELLIS_TP_MOE_PREPARE_ROUTES
__device__ __forceinline__ TpMoeDeviceRows* pipeline_maps(const TpMoeLaunch& l, bool down) {
  auto* base = reinterpret_cast<TpMoeDeviceRows*>(
      reinterpret_cast<unsigned char*>(l.partials) +
      tp_moe_split_partial_bytes(l.rows) + tp_moe_split_activation_bytes(l.rows));
  return base + (down ? 48 : 0);
}
__global__ __launch_bounds__(32) void pipeline_prepare_routes(TpMoeLaunch l) {
  if (threadIdx.x != 0) return;
  TpMoeDeviceRows map{};
  const bool down = blockIdx.z != 0;
  if (!resolve(l, map, down)) map.row_count = 0;
  pipeline_maps(l, down)[blockIdx.y] = map;
}
#endif
__device__ __forceinline__ bool pipeline_resolve(const TpMoeLaunch& l,
                                               TpMoeDeviceRows& map, bool down,
                                               unsigned candidate = blockIdx.y) {
#ifdef STRIX_TRELLIS_TP_MOE_PREPARE_ROUTES
  map = pipeline_maps(l, down)[candidate];
  return map.row_count != 0;
#else
  return resolve(l, map, down, candidate);
#endif
}

template <bool Down>
__global__ __launch_bounds__(128) void pipeline_prepare(TpMoeLaunch l) {
  __shared__ TpMoeDeviceRows m;
  __shared__ bool owner;
  if (threadIdx.x == 0)
    owner = pipeline_resolve(l, m, Down);
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
  for (unsigned unit = blockIdx.x * 4 + wave;
       unit < m.row_count * Blocks; unit += gridDim.x * 4) {
    const unsigned row = unit / Blocks, h = unit % Blocks;
#ifdef STRIX_TRELLIS_TP_MOE_SPLIT_PREPARE
    constexpr unsigned Projections = 1;
#else
    constexpr unsigned Projections = Down ? 1 : 2;
#endif
#pragma unroll
    for (unsigned step = 0; step < Projections; ++step) {
#ifdef STRIX_TRELLIS_TP_MOE_SPLIT_PREPARE
      const unsigned projection = Down ? 0 : blockIdx.z;
#else
      const unsigned projection = step;
#endif
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

template <int Bits, int Rows, bool Down>
__device__ __forceinline__ void
pipeline_dot(const DeviceWeights &w, const TpMoeLaunch &l,
             const TpMoeDeviceRows &m, unsigned projection, unsigned tile, unsigned split) {
  const unsigned lane = threadIdx.x & 31;
  const unsigned n = __builtin_amdgcn_readfirstlane(tile * tp_moe_pipeline_core_waves + unsigned(threadIdx.x) / 32);
  const auto p = tp_fused_partition(w.input, split);
  float sum[Rows][2] = {};
  const _Float16 *x[Rows];
#pragma unroll
  for (unsigned row = 0; row < Rows; ++row) {
    // Padding rows consume a valid input; their outputs are discarded.
    const unsigned mapped = row < m.row_count ? row : 0;
    const unsigned token = __builtin_amdgcn_readfirstlane(m.token_rows[mapped]);
    const unsigned slot = __builtin_amdgcn_readfirstlane(m.route_slots[mapped]);
    x[row] = pipeline_activation(l, token, slot, Down, projection);
  }
  if constexpr (Bits == 2 && Rows == 1) {
    // The default preserves the qualified U4 wide core. Other depths change
    // only how many weight/activation loads precede the first hash.
    constexpr int Depth = Down && STRIX_TRELLIS_TP_MOE_DOWN_DEPTH != 0
        ? STRIX_TRELLIS_TP_MOE_DOWN_DEPTH : STRIX_TRELLIS_TP_MOE_SINGLE_DEPTH;
    int k = p.begin_tile;
    for (; k + Depth <= p.end_tile; k += Depth) {
      k2_local_kernel::Raw raw[Depth];
#ifdef STRIX_TRELLIS_TP_MOE_LEAN_CORE
#pragma unroll
      for (int j = 0; j < Depth; ++j)
        raw[j] = lean_read(w.trellis, x[0], k + j, w.output, n);
#pragma unroll
      for (int j = 0; j < Depth; ++j)
        lean_consume(raw[j], sum[0][0], sum[0][1]);
#else
#pragma unroll
      for (int j = 0; j < Depth; ++j)
        raw[j] = k2_local_kernel::read_raw(w.trellis, x[0], k + j, w.output, n);
#pragma unroll
      for (int j = 0; j < Depth; ++j)
        pipeline_consume(raw[j], sum[0][0], sum[0][1]);
#endif
    }
    for (; k < p.end_tile; ++k)
#ifdef STRIX_TRELLIS_TP_MOE_LEAN_CORE
      lean_consume(lean_read(w.trellis, x[0], k, w.output, n), sum[0][0], sum[0][1]);
#else
      pipeline_consume(k2_local_kernel::read_raw(w.trellis, x[0], k, w.output, n),
                               sum[0][0], sum[0][1]);
#endif
  } else if constexpr (Bits == 2) {
#if STRIX_TRELLIS_TP_MOE_PREFETCH_ROWS > 0
    pipeline_prefetch_rows<Rows, STRIX_TRELLIS_TP_MOE_PREFETCH_ROWS>(w, x, n, p.begin_tile, p.end_tile, sum);
#else
    for (int k = p.begin_tile; k < p.end_tile; k += 4) {
      k2_local_kernel::Raw raw[4]{};
      const auto window = tp_moe_reader_window<2>(lane);
#pragma unroll
      for (int j = 0; j < 4; ++j)
        if (k + j < p.end_tile) {
          const auto *base = reinterpret_cast<const unsigned char *>(
              w.trellis + (size_t(k + j) * (w.output / 16) + n) * 16);
          raw[j].a = k2_local_kernel::word_load(base, window.a);
          raw[j].b = k2_local_kernel::word_load(base, window.b);
        }
#pragma unroll
      for (int j = 0; j < 4; ++j)
        if (k + j < p.end_tile) {
          uint32_t states[8], hashes[8];
          k2_local_kernel::states(raw[j], states);
          pipeline_hash(states, hashes);
          Pair input[Rows][2];
#pragma unroll
          for (unsigned row = 0; row < Rows; ++row) {
            const int column = (k + j) * 16 + 2 * (lane & 3);
            __builtin_memcpy(&input[row][0], x[row] + column, 4);
            __builtin_memcpy(&input[row][1], x[row] + column + 8, 4);
          }
#pragma unroll
          for (int pair = 0; pair < 4; ++pair) {
            const Pair weight = k2_local_kernel::sad_pair(hashes[2 * pair],
                                                          hashes[2 * pair + 1]);
#pragma unroll
            for (unsigned row = 0; row < Rows; ++row)
              sum[row][pair / 2] = __builtin_amdgcn_fdot2(
                  weight, input[row][pair & 1], sum[row][pair / 2], false);
          }
        }
    }

#endif
  } else {
    for (int k = p.begin_tile; k < p.end_tile; ++k) {
      TpMoeWeightTile<Bits> reader(w, k, n);
      Pair input[Rows][2];
#pragma unroll
      for (unsigned row = 0; row < Rows; ++row) {
        const int column = k * 16 + 2 * (lane & 3);
        __builtin_memcpy(&input[row][0], x[row] + column, 4);
        __builtin_memcpy(&input[row][1], x[row] + column + 8, 4);
      }
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        __half2 decoded;
        if (pair == 0)
          decoded = reader.template pair<0>();
        else if (pair == 1)
          decoded = reader.template pair<1>();
        else if (pair == 2)
          decoded = reader.template pair<2>();
        else
          decoded = reader.template pair<3>();
        const Pair weight = __builtin_bit_cast(Pair, decoded);
#pragma unroll
        for (unsigned row = 0; row < Rows; ++row)
          sum[row][pair / 2] = __builtin_amdgcn_fdot2(
              weight, input[row][pair & 1], sum[row][pair / 2], false);
      }
    }
  }
#pragma unroll
  for (unsigned row = 0; row < Rows; ++row) {
#pragma unroll
    for (unsigned half = 0; half < 2; ++half) {
      float v = sum[row][half];
      v += __shfl_down(v, 2, 4);
      v += __shfl_down(v, 1, 4);
      if (!(lane & 3) && row < m.row_count) {
        const unsigned column = n * 16 + lane / 4 + half * 8;
        const uint64_t index =
            Down
                ? tp_moe_split_index(m.token_rows[row], m.route_slots[row],
                                     split, column)
                : tp_moe_pipeline_gate_index(l.rows, projection,
                                             m.token_rows[row],
                                             m.route_slots[row], split, column);
        l.partials[index] = v;
      }
    }
  }
}

template <int Rows, bool Down>
__device__ __forceinline__ void
pipeline_project(const DeviceWeights &w, const TpMoeLaunch &l,
                 const TpMoeDeviceRows &m, unsigned projection,
                 unsigned tile = blockIdx.x, unsigned split = blockIdx.z & 3) {
  if (w.bits == 2)
    pipeline_dot<2, Rows, Down>(w, l, m, projection, tile, split);
  else
    pipeline_dot<3, Rows, Down>(w, l, m, projection, tile, split);
}

template <int Capacity, bool Down, int RowClass = 0>
__global__ __launch_bounds__(tp_moe_pipeline_core_threads) void pipeline_core(TpMoeLaunch l) {
  __shared__ TpMoeDeviceRows m;
  __shared__ bool owner;
  if (threadIdx.x == 0) {
#ifdef STRIX_TRELLIS_TP_MOE_SINGLE_ROW
    owner = pipeline_resolve_route(l, m, Down);
#else
    owner = pipeline_resolve(l, m, Down);
#endif
  }
  __syncthreads();
  if (!owner)
    return;
  if constexpr (RowClass == 1) { if (m.row_count > 2) return; }
  if constexpr (RowClass == 2) { if (m.row_count <= 2) return; }
  const unsigned bank_index = __builtin_amdgcn_readfirstlane(m.bank_index);
  const auto &e = l.bank ? l.bank[bank_index] : l.table->experts[bank_index];
  const unsigned projection = Down ? 0 : blockIdx.z / 4;
  const auto &w = Down ? e.down : projection ? e.up : e.gate;
#ifdef STRIX_TRELLIS_TP_MOE_SINGLE_ROW
  pipeline_project<1, Down>(w, l, m, projection);
#elif STRIX_TRELLIS_TP_MOE_ROW_CAP > 0
  // Separate the numerical row core from the window's maximum capacity.
  // Splitting row groups repeats decode but preserves every output chain.
  constexpr unsigned Chunk = STRIX_TRELLIS_TP_MOE_ROW_CAP;
  if constexpr (RowClass == 2) {
    // Keep all rows of a shared expert together, without raising the register
    // allocation of the separate one/two-row kernel.
    if (m.row_count <= 4)
      pipeline_project<4, Down>(w, l, m, projection);
    else
      pipeline_project<8, Down>(w, l, m, projection);
  } else if constexpr (Capacity <= Chunk) {
    if (m.row_count == 1)
      pipeline_project<1, Down>(w, l, m, projection);
    else
      pipeline_project<Capacity, Down>(w, l, m, projection);
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
      pipeline_project<1, Down>(w, l, slice, projection);
    else
      pipeline_project<Chunk, Down>(w, l, slice, projection);
  }
#else
  if (m.row_count == 1)
    pipeline_project<1, Down>(w, l, m, projection);
  else if constexpr (Capacity >= 2) {
    if (m.row_count <= 2)
      pipeline_project<2, Down>(w, l, m, projection);
    else if constexpr (Capacity >= 4) {
      if (m.row_count <= 4)
        pipeline_project<4, Down>(w, l, m, projection);
      else if constexpr (Capacity >= 8)
        pipeline_project<8, Down>(w, l, m, projection);
    }
  }
#endif
}

template <bool PrepareDown = false>
__global__ __launch_bounds__(32) void pipeline_gate_finish(TpMoeLaunch l) {
  const unsigned row = blockIdx.y / 6, slot = blockIdx.y % 6,
                 lane = threadIdx.x;
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
      const unsigned column = blockIdx.x * 128 + lane + j * 32;
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
    const unsigned column = blockIdx.x * 128 + lane + j * 32;
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
      const unsigned column = blockIdx.x * 128 + lane + j * 32;
      const float scaled = materialize_fp32(hidden[j] *
          __half2float(__ushort_as_half(e.down.suh[column])));
      hidden[j] = float(_Float16(scaled));
    }
    hadamard128(hidden);
    auto* x = pipeline_activation(l, row, slot, true);
#pragma unroll
    for (unsigned j = 0; j < 4; ++j)
      x[blockIdx.x * 128 + lane + j * 32] = _Float16(hidden[j]);
  }
}

#ifdef STRIX_TRELLIS_TP_MOE_SPLIT_GATE_FINISH
#include "tp_moe_gate_finish.hpp"
#endif
template <bool PrepareDown> const void* pipeline_gate_finish_function() {
#ifdef STRIX_TRELLIS_TP_MOE_SPLIT_GATE_FINISH
  return reinterpret_cast<const void*>(split_gate_finish<PrepareDown>);
#else
  return reinterpret_cast<const void*>(pipeline_gate_finish<PrepareDown>);
#endif
}
template <bool PrepareDown>
void pipeline_gate_finish_launch(const TpMoeLaunch& l, hipStream_t stream) {
#ifdef STRIX_TRELLIS_TP_MOE_SPLIT_GATE_FINISH
  split_gate_finish<PrepareDown><<<dim3((9 + tp_moe_gate_finish_tiles - 1) / tp_moe_gate_finish_tiles, l.rows * 6), tp_moe_gate_finish_threads, 0, stream>>>(l);
#else
  pipeline_gate_finish<PrepareDown><<<dim3(9, l.rows * 6), 32, 0, stream>>>(l);
#endif
}
template <bool Down> const void *pipeline_function(unsigned rows) {
#ifdef STRIX_TRELLIS_TP_MOE_ADAPTIVE_ROWS
  if (rows > 2)
    return reinterpret_cast<const void *>(pipeline_core<2, Down, 1>);
#endif
  if (rows == 1)
    return reinterpret_cast<const void *>(pipeline_core<1, Down>);
  if (rows == 2)
    return reinterpret_cast<const void *>(pipeline_core<2, Down>);
  if (rows <= 4)
    return reinterpret_cast<const void *>(pipeline_core<4, Down>);
  return reinterpret_cast<const void *>(pipeline_core<8, Down>);
}

template <bool Down>
hipError_t pipeline_launch(const TpMoeLaunch &l, hipStream_t stream,
                           bool prepare = true, bool prepare_down = false) {
  const unsigned owners = l.bank ? l.rows * 6 : l.expert_count;
  constexpr unsigned Width = Down ? 1152 : 5120;
  auto status = hipSuccess;
#ifdef STRIX_TRELLIS_TP_MOE_PREPARE_ROUTES
  if (prepare) {
    pipeline_prepare_routes<<<dim3(1, owners, 2), 32, 0, stream>>>(l);
    status = hipGetLastError();
    if (status != hipSuccess) return status;
  }
#endif
  if (prepare) {
    pipeline_prepare<Down>
        <<<dim3(((Width / 128) + 3) / 4, owners,
#ifdef STRIX_TRELLIS_TP_MOE_SPLIT_PREPARE
                Down ? 1 : 2
#else
                1
#endif
                ), 128, 0, stream>>>(l);
    status = hipGetLastError();
  }
  if (status != hipSuccess)
    return status;
#ifdef STRIX_TRELLIS_TP_MOE_SINGLE_ROW
  const unsigned core_owners = l.rows * 6;
#else
  const unsigned core_owners = owners;
#endif
  const dim3 grid((Down ? 320 : 72) / tp_moe_pipeline_core_waves, core_owners, Down ? 4 : 8);
  if (l.rows == 1)
    pipeline_core<1, Down><<<grid, tp_moe_pipeline_core_threads, 0, stream>>>(l);
  else if (l.rows == 2)
    pipeline_core<2, Down><<<grid, tp_moe_pipeline_core_threads, 0, stream>>>(l);
#ifdef STRIX_TRELLIS_TP_MOE_ADAPTIVE_ROWS
  else {
    const unsigned classes = l.bank ? 3u : Down ? l.down_row_classes : l.gate_row_classes;
    if (classes & 1u) {
      pipeline_core<2, Down, 1><<<grid, tp_moe_pipeline_core_threads, 0, stream>>>(l);
      status = hipGetLastError();
      if (status != hipSuccess) return status;
    }
    if (classes & 2u)
      pipeline_core<8, Down, 2><<<grid, tp_moe_pipeline_core_threads, 0, stream>>>(l);
  }
#else
  else if (l.rows <= 4)
    pipeline_core<4, Down><<<grid, tp_moe_pipeline_core_threads, 0, stream>>>(l);
  else
    pipeline_core<8, Down><<<grid, tp_moe_pipeline_core_threads, 0, stream>>>(l);
#endif
  status = hipGetLastError();
  if (status != hipSuccess)
    return status;
  if constexpr (!Down) {
    if (prepare_down)
      pipeline_gate_finish_launch<true>(l, stream);
    else
      pipeline_gate_finish_launch<false>(l, stream);
    status = hipGetLastError();
  }
  return status;
}
