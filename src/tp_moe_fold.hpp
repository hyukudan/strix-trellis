#pragma once
#ifndef STRIX_TRELLIS_TP_MOE_FOLD_PREPARE
#define STRIX_TRELLIS_TP_MOE_FOLD_PREPARE 0
#endif
#ifndef STRIX_TRELLIS_TP_MOE_FOLD_GATE_FINISH
#define STRIX_TRELLIS_TP_MOE_FOLD_GATE_FINISH 1
#endif
// R1 full submission in two launches. The gate kernel prepares its own input
// slice in LDS (same RN32/RN16/H128 boundaries as pipeline_prepare), runs the
// unchanged S4 core and, in the last CTA to finish each H128 output block,
// performs split_gate_finish<true> for that block. The down kernel runs the
// unchanged down core and, in the last CTA of each output block, the
// grouped_route_finish of that block. Partials are still materialized in P
// and summed in the original fixed order, so H, P and Y keep their bits.
// Arrival counters live in plan-owned memory and are reset by their last CTA.
// Every CTA of the grid arrives, including non-owners, so the expected count
// is fixed by the grid and does not depend on the routes.

__device__ __forceinline__ float fold_coherent_load(const float* p) {
  return __hip_atomic_load(const_cast<float*>(p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ bool fold_arrive(unsigned* counter, unsigned expected) {
  __shared__ bool last;
  // Release this CTA's P stores before arriving.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const unsigned previous = atomicAdd(counter, 1u);
    last = previous + 1 == expected;
    if (last) __hip_atomic_store(counter, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  __syncthreads();
  if (last) __threadfence();
  return last;
}

// split_gate_finish<true> for one (row, slot) and one H128 tile, 128 threads:
// wave 0 = gate, wave 1 = up, then wave 0 completes SwiGLU and the down input.
__device__ __forceinline__ void fold_gate_finish_tile(const TpMoeLaunch& l, const TpMoeExpert& e,
    unsigned row, unsigned slot, unsigned tile, float (&projected)[2][128]) {
  const unsigned lane = threadIdx.x & 31;
  const unsigned wave = __builtin_amdgcn_readfirstlane(unsigned(threadIdx.x) / 32);
  if (wave < 2) {
    const unsigned projection = wave;
    const auto& w = projection ? e.up : e.gate;
    float v[4] = {};
#pragma unroll
    for (unsigned split = 0; split < 4; ++split) {
#pragma unroll
      for (unsigned j = 0; j < 4; ++j)
        v[j] += fold_coherent_load(l.partials + tp_moe_pipeline_gate_index(l.rows, projection, row, slot, split, tile * 128 + lane + j * 32));
    }
    hadamard128(v);
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      const unsigned column = tile * 128 + lane + j * 32;
      float value = v[j] * __half2float(__ushort_as_half(w.svh[column]));
      if (w.bias) value += w.bias[column];
      projected[projection][lane + j * 32] = value;
    }
  }
  __syncthreads();
  if (wave == 0) {
    float hidden[4];
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      const unsigned column = tile * 128 + lane + j * 32;
      const float g = projected[0][lane + j * 32], u = projected[1][lane + j * 32];
            hidden[j] = tp_moe_swiglu(g, u);
      l.hidden[(uint64_t(row) * 6 + slot) * 1152 + column] = hidden[j];
    }
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
      const unsigned column = tile * 128 + lane + j * 32;
      const float scaled = materialize_fp32(hidden[j] * __half2float(__ushort_as_half(e.down.suh[column])));
      hidden[j] = float(_Float16(scaled));
    }
    hadamard128(hidden);
    auto* x = pipeline_activation(l, row, slot, true);
#pragma unroll
    for (unsigned j = 0; j < 4; ++j) x[tile * 128 + lane + j * 32] = _Float16(hidden[j]);
  }
  __syncthreads();
}

// K2 single-row core over an LDS input slice; identical arithmetic to
// lean_consume/pipeline_dot<2, 1, false>.
__device__ __forceinline__ void fold_dot_lds(const DeviceWeights& w, const TpMoeLaunch& l,
    const TpMoeDeviceRows& m, unsigned projection, unsigned tile, unsigned split,
    const _Float16* xs) {
  const unsigned lane = threadIdx.x & 31;
  const unsigned n = __builtin_amdgcn_readfirstlane(tile * tp_moe_pipeline_core_waves + unsigned(threadIdx.x) / 32);
  const auto p = tp_fused_partition(w.input, split);
  float sum[2] = {};
  constexpr int Depth = STRIX_TRELLIS_TP_MOE_SINGLE_DEPTH;
  const int index = ((int(lane) * 8 + 1) * 2 + 512 - 16) / 32 - 16;
  const int ia = index < 0 ? index + 16 : index;
  const int ib = (lane & 1) ? ia : (index + 1 >= 16 ? index + 1 - 16 : index + 1);
  using G = const __attribute__((address_space(1))) unsigned char;
  uint32_t oa = uint32_t(ia) * 4u, ob = uint32_t(ib) * 4u;
  asm("" : "+v"(oa), "+v"(ob));
  auto weight = [&](int k, k2_local_kernel::Raw& r) {
    G* gw = (G*)reinterpret_cast<const unsigned char*>(w.trellis + (size_t(k) * (w.output / 16) + n) * 16);
    r.a = *(const __attribute__((address_space(1))) uint32_t*)(gw + oa);
    r.b = *(const __attribute__((address_space(1))) uint32_t*)(gw + ob);
  };
  auto input = [&](int k, k2_local_kernel::Raw& r) {
    const unsigned char* xb = reinterpret_cast<const unsigned char*>(xs) + (k - p.begin_tile) * 32;
    __builtin_memcpy(&r.x, xb + 4 * (lane & 3), 4);
    __builtin_memcpy(&r.y, xb + 4 * (lane & 3) + 16, 4);
  };
  int k = p.begin_tile;
  for (; k + Depth <= p.end_tile; k += Depth) {
    k2_local_kernel::Raw raw[Depth];
#pragma unroll
    for (int j = 0; j < Depth; ++j) { weight(k + j, raw[j]); input(k + j, raw[j]); }
#pragma unroll
    for (int j = 0; j < Depth; ++j) lean_consume(raw[j], sum[0], sum[1]);
  }
  for (; k < p.end_tile; ++k) {
    k2_local_kernel::Raw raw;
    weight(k, raw); input(k, raw);
    lean_consume(raw, sum[0], sum[1]);
  }
#pragma unroll
  for (unsigned half = 0; half < 2; ++half) {
    float v = sum[half];
    v += __shfl_down(v, 2, 4);
    v += __shfl_down(v, 1, 4);
    if (!(lane & 3) && m.row_count > 0) {
      const unsigned column = n * 16 + lane / 4 + half * 8;
      l.partials[tp_moe_pipeline_gate_index(l.rows, projection, m.token_rows[0], m.route_slots[0], split, column)] = v;
    }
  }
}

// grid (72 / waves, owners, 8), core threads. R1 only.
__global__ __launch_bounds__(tp_moe_pipeline_core_threads) void fold_gate(TpMoeLaunch l) {
  static_assert(tp_moe_pipeline_core_waves == 4, "fold assumes four waves per CTA");
  __shared__ TpMoeDeviceRows m;
  __shared__ bool owner;
  __shared__ _Float16 xs[1280];
  __shared__ float projected[2][128];
  if (threadIdx.x == 0) owner = pipeline_resolve(l, m, false);
  __syncthreads();
  const unsigned projection = blockIdx.z / 4, split = blockIdx.z & 3;
#if !STRIX_TRELLIS_TP_MOE_FOLD_PREPARE
  // The separate pipeline_prepare<false> launch already wrote the activation.
  if (owner) {
    const unsigned bank_index = __builtin_amdgcn_readfirstlane(m.bank_index);
    const auto& e = l.bank ? l.bank[bank_index] : l.table->experts[bank_index];
    pipeline_project<1, false>(projection ? e.up : e.gate, l, m, projection, blockIdx.x, split);
  }
#else
  if (owner) {
    const unsigned bank_index = __builtin_amdgcn_readfirstlane(m.bank_index);
    const auto& e = l.bank ? l.bank[bank_index] : l.table->experts[bank_index];
    const auto& w = projection ? e.up : e.gate;
    const auto part = tp_fused_partition(5120, split);
    const unsigned lane = threadIdx.x & 31;
    const unsigned wave = __builtin_amdgcn_readfirstlane(unsigned(threadIdx.x) / 32);
    const unsigned blocks = part.end_h128 - part.first_h128;
    // pipeline_prepare<false> for this split's H128 blocks of one projection.
    for (unsigned b = wave; b < blocks; b += 4) {
      const unsigned h = part.first_h128 + b;
      float v[4];
#pragma unroll
      for (unsigned j = 0; j < 4; ++j) {
        const unsigned column = h * 128 + lane + j * 32;
        const float scaled = materialize_fp32(Input{l, m, false}.load(0, column) *
                                              __half2float(__ushort_as_half(w.suh[column])));
        v[j] = float(_Float16(scaled));
      }
      hadamard128(v);
#pragma unroll
      for (unsigned j = 0; j < 4; ++j) xs[b * 128 + lane + j * 32] = _Float16(v[j]);
    }
    if (w.bits != 2) {
      // K3 consumes the global activation image, as pipeline_dot<3> does.
      auto* x = pipeline_activation(l, m.token_rows[0], m.route_slots[0], false, projection);
      for (unsigned i = threadIdx.x; i < blocks * 128; i += blockDim.x)
        x[part.first_h128 * 128 + i] = xs[i];
      __threadfence_block();
    }
    __syncthreads();
    if (w.bits == 2) fold_dot_lds(w, l, m, projection, blockIdx.x, split, xs);
    else pipeline_dot<3, 1, false>(w, l, m, projection, blockIdx.x, split);
  }
#endif
#if !STRIX_TRELLIS_TP_MOE_FOLD_GATE_FINISH
  return;
#endif
  // Each H128 output tile is covered by two CTAs of every projection/split.
  const unsigned tile = blockIdx.x / 2;
  if (!fold_arrive(l.fold_counters + blockIdx.y * 9 + tile, 16)) return;
  if (!owner) return;
  const unsigned bank_index = __builtin_amdgcn_readfirstlane(m.bank_index);
  const auto& e = l.bank ? l.bank[bank_index] : l.table->experts[bank_index];
  for (unsigned i = 0; i < m.row_count; ++i)
    fold_gate_finish_tile(l, e, m.token_rows[i], m.route_slots[i], tile, projected);
}

// grid (320 / waves, owners, 4), core threads. R1 only.
__global__ __launch_bounds__(tp_moe_pipeline_core_threads) void fold_down(TpMoeLaunch l) {
  __shared__ TpMoeDeviceRows m;
  __shared__ bool owner;
  __shared__ float weighted[6][128];
  if (threadIdx.x == 0) owner = pipeline_resolve(l, m, true);
  __syncthreads();
  if (owner && m.row_count != 0) {
    const unsigned bank_index = __builtin_amdgcn_readfirstlane(m.bank_index);
    const auto& e = l.bank ? l.bank[bank_index] : l.table->experts[bank_index];
    pipeline_project<1, true>(e.down, l, m, 0, blockIdx.x, blockIdx.z & 3);
  }
  const unsigned block = blockIdx.x / 2;
  if (!fold_arrive(l.fold_counters + 64 + block, 2 * gridDim.y * 4)) return;
  // grouped_route_finish for (block, row) with four waves: slots 0-3, then 4-5.
  const unsigned lane = threadIdx.x & 31;
  const unsigned wave = __builtin_amdgcn_readfirstlane(unsigned(threadIdx.x) / 32);
  for (unsigned row = 0; row < l.rows; ++row) {
    const auto* ids = l.bank ? l.device_route_ids + row * 6 : l.table->expert_indices[row];
    const auto* scores = l.bank ? l.device_route_scores + row * 6 : l.table->route_scores[row];
    const bool valid = !l.bank || tp_moe_device_row_valid(ids, scores, l.bank_count);
    for (unsigned slot = wave; slot < 6; slot += 4) {
      float v[4] = {};
      if (valid && scores[slot] != 0.0f) {
        const auto& w = l.bank ? l.bank[ids[slot]].down : l.table->experts[ids[slot]].down;
#pragma unroll
        for (unsigned split = 0; split < 4; ++split) {
#pragma unroll
          for (unsigned j = 0; j < 4; ++j)
            v[j] += fold_coherent_load(l.partials + tp_moe_split_index(row, slot, split, block * 128 + lane + j * 32));
        }
        hadamard128(v);
#pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
          const unsigned column = block * 128 + lane + j * 32;
          float value = v[j] * __half2float(__ushort_as_half(w.svh[column]));
          if (w.bias) value += w.bias[column];
          v[j] = materialize_fp32(scores[slot] * value);
        }
      }
#pragma unroll
      for (unsigned j = 0; j < 4; ++j) weighted[slot][lane + j * 32] = v[j];
    }
    __syncthreads();
    float sum = 0.0f;
    if (valid) {
#pragma unroll
      for (unsigned slot = 0; slot < 6; ++slot)
        if (scores[slot] != 0.0f) sum = materialize_fp32(sum + weighted[slot][threadIdx.x]);
    }
    const uint64_t output = uint64_t(row) * l.y_stride + block * 128 + threadIdx.x;
    float value = materialize_fp32(l.alpha * sum);
    if (l.beta != 0.0f) value = materialize_fp32(value + materialize_fp32(l.beta * l.y[output]));
    l.y[output] = valid ? value : __builtin_bit_cast(float, uint32_t(0x7fc00000u));
    __syncthreads();
  }
}

inline hipError_t fold_launch(const TpMoeLaunch& l, hipStream_t stream) {
  const unsigned owners = l.bank ? l.rows * 6 : l.expert_count;
  hipError_t status;
#if !STRIX_TRELLIS_TP_MOE_FOLD_PREPARE
#ifdef STRIX_TRELLIS_TP_MOE_SPLIT_PREPARE
  constexpr unsigned projections = 2;
#else
  constexpr unsigned projections = 1;
#endif
  pipeline_prepare<false><<<dim3((5120 / 128 + 3) / 4, owners, projections), 128, 0, stream>>>(l);
  status = hipGetLastError();
  if (status != hipSuccess) return status;
#endif
  fold_gate<<<dim3(72 / tp_moe_pipeline_core_waves, owners, 8), tp_moe_pipeline_core_threads, 0, stream>>>(l);
  status = hipGetLastError();
  if (status != hipSuccess) return status;
#if !STRIX_TRELLIS_TP_MOE_FOLD_GATE_FINISH
  pipeline_gate_finish_launch<true>(l, stream);
  status = hipGetLastError();
  if (status != hipSuccess) return status;
#endif
  fold_down<<<dim3(320 / tp_moe_pipeline_core_waves, owners, 4), tp_moe_pipeline_core_threads, 0, stream>>>(l);
  return hipGetLastError();
}
