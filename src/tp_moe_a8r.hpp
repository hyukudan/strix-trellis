#pragma once
#ifndef STRIX_TRELLIS_TP_MOE_A8R_CLUSTER_UNROLL
#define STRIX_TRELLIS_TP_MOE_A8R_CLUSTER_UNROLL 1
#endif
#ifndef STRIX_TRELLIS_TP_MOE_A8R_HASH_FENCE
#define STRIX_TRELLIS_TP_MOE_A8R_HASH_FENCE 0
#endif
#ifndef STRIX_TRELLIS_TP_MOE_A8R_OUTPUTS_PER_LANE
#define STRIX_TRELLIS_TP_MOE_A8R_OUTPUTS_PER_LANE 2
#endif
// Included inside tp_moe_split. Only explicit A8R plans use these kernels.
namespace a8r_kernel {
namespace arithmetic = ::strix_trellis::a8r;
constexpr unsigned wide_outputs = STRIX_TRELLIS_TP_MOE_A8R_OUTPUTS_PER_LANE;
static_assert(wide_outputs == 1 || wide_outputs == 2);
__device__ __forceinline__ void hash4(const uint32_t (&s)[4], uint32_t (&h)[4]) {
#ifdef STRIX_TRELLIS_TP_MOE_HASH32
    #pragma unroll
    for (unsigned j=0;j<4;++j) h[j]=(s[j]&65535u)*0x83dcd12du;
#else
    uint32_t t[4];
    asm("v_pk_mul_lo_u16 %0, %8, %12 op_sel:[0,0] op_sel_hi:[0,1]\n\t"
        "v_pk_mul_lo_u16 %1, %9, %12 op_sel:[0,0] op_sel_hi:[0,1]\n\t"
        "v_pk_mul_lo_u16 %2, %10, %12 op_sel:[0,0] op_sel_hi:[0,1]\n\t"
        "v_pk_mul_lo_u16 %3, %11, %12 op_sel:[0,0] op_sel_hi:[0,1]\n\t"
        "v_mad_u32_u16 %4, %8, %13, %0\n\t"
        "v_mad_u32_u16 %5, %9, %13, %1\n\t"
        "v_mad_u32_u16 %6, %10, %13, %2\n\t"
        "v_mad_u32_u16 %7, %11, %13, %3"
        : "=&v"(t[0]),"=&v"(t[1]),"=&v"(t[2]),"=&v"(t[3]),
          "=&v"(h[0]),"=&v"(h[1]),"=&v"(h[2]),"=&v"(h[3])
        : "v"(s[0]),"v"(s[1]),"v"(s[2]),"v"(s[3]),"s"(0x83dc0000u),"s"(0xd12du));
#endif
}
template<unsigned Count>
__device__ __forceinline__ void wide_hash(const uint32_t (&s)[Count], uint32_t (&h)[Count]) {
    if constexpr (Count == 8) pipeline_hash(s, h);
    else hash4(s, h);
}
__device__ __forceinline__ signed char* codes(const TpMoeLaunch& l, unsigned row,
    unsigned slot, bool down, unsigned projection = 0) {
    return reinterpret_cast<signed char*>(pipeline_activation(l, row, slot, down, projection));
}
__device__ __forceinline__ float* scales(signed char* q, bool down) { return reinterpret_cast<float*>(q + (down ? 1152 : 5120)); }
__device__ __forceinline__ int32_t* sums(signed char* q, bool down) { return reinterpret_cast<int32_t*>(q + (down ? 1296 : 5760)); }
__device__ __forceinline__ float affine_group(int32_t bq, int32_t correction, float delta) {
#ifdef STRIX_TRELLIS_TP_MOE_A8R_PREPARE_AFFINE
    const uint32_t u = uint32_t(bq) * 887u + uint32_t(correction);
    int32_t n;__builtin_memcpy(&n, &u, 4);
    return arithmetic::mul(float(n), delta);
#else
    return arithmetic::group(bq, correction, delta);
#endif
}
// Each wave publishes four groups of 32. Q sum and integer dot reduction are
// exact; the host reference need not reproduce any lane reduction tree.
__device__ __forceinline__ void quantize(float (&v)[4], signed char* q, unsigned h, bool down) {
    const unsigned lane = threadIdx.x & 31;
    #pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
        const bool invalid = __ballot(!arithmetic::finite(v[j])) != 0;
        float maximum = arithmetic::absolute(v[j]);
        #pragma unroll
        for (unsigned distance = 16; distance; distance /= 2)
            maximum = fmaxf(maximum, __shfl_xor(maximum, distance, 32));
        const float delta = invalid ? arithmetic::canonical_nan() : arithmetic::scale(maximum);
        const int code = invalid ? 0 : arithmetic::quantize(v[j], maximum, delta);
        q[h * 128 + j * 32 + lane] = code;
        int total = code;
        #pragma unroll
        for (unsigned distance = 16; distance; distance /= 2)
            total += __shfl_xor(total, distance, 32);
        if (!lane) {
#ifdef STRIX_TRELLIS_TP_MOE_A8R_PREPARE_AFFINE
            // Finite RN16/H128 inputs give delta in [2^-35,2^13]. The
            // power-of-two adjustment stays normal and is exact, so moving
            // it across the final multiplication preserves the RN32 result.
            scales(q, down)[h * 4 + j] = arithmetic::mul(delta, 0x1p-17f);
            sums(q, down)[h * 4 + j] = -452608 * total;
#else
            scales(q, down)[h * 4 + j] = delta;sums(q, down)[h * 4 + j] = total;
#endif
        }
    }
}
template<bool Down>
__global__ __launch_bounds__(128) void prepare(TpMoeLaunch l) {
    __shared__ TpMoeDeviceRows m; __shared__ bool owner;
    if (!threadIdx.x) owner = resolve(l, m, Down);
    __syncthreads(); if (!owner) return;
    constexpr unsigned Blocks = Down ? 9 : 40;
    const unsigned lane = threadIdx.x & 31;
    const unsigned wave = __builtin_amdgcn_readfirstlane(unsigned(threadIdx.x) / 32);
    const unsigned bank_index = __builtin_amdgcn_readfirstlane(m.bank_index);
    const auto& e = l.bank ? l.bank[bank_index] : l.table->experts[bank_index];
    const unsigned projection = Down ? 0 : blockIdx.z;
    const auto& w = Down ? e.down : projection ? e.up : e.gate;
    for (unsigned unit = blockIdx.x * 4 + wave; unit < m.row_count * Blocks; unit += gridDim.x * 4) {
        const unsigned row = unit / Blocks, h = unit % Blocks;
        const unsigned token = __builtin_amdgcn_readfirstlane(m.token_rows[row]);
        const unsigned slot = __builtin_amdgcn_readfirstlane(m.route_slots[row]);
        float v[4];
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            const unsigned column = h * 128 + lane + j * 32;
            const float value = Down ? l.hidden[(uint64_t(token) * 6 + slot) * 1152 + column]
                : l.x[uint64_t(token) * l.x_stride + column];
            v[j] = float(_Float16(arithmetic::mul(value,
                __half2float(__ushort_as_half(w.suh[column])))));
        }
        hadamard128(v);
        quantize(v, codes(l, token, slot, Down, projection), h, Down);
    }
}
template<unsigned Rows, bool Down>
__device__ __forceinline__ void dot_narrow(const DeviceWeights& w, const TpMoeLaunch& l,
    const TpMoeDeviceRows& m, unsigned projection) {
    const unsigned lane = threadIdx.x & 31;
    const unsigned n = __builtin_amdgcn_readfirstlane(unsigned(blockIdx.x) * 4 + unsigned(threadIdx.x) / 32);
    const unsigned split = blockIdx.z & 3;
    const auto partition = tp_fused_partition(w.input, split);
    float value[Rows][2] = {};
    signed char* input[Rows];
    #pragma unroll
    for (unsigned row = 0; row < Rows; ++row) {
        const unsigned mapped = row < m.row_count ? row : 0;
        const unsigned token = __builtin_amdgcn_readfirstlane(m.token_rows[mapped]);
        const unsigned slot = __builtin_amdgcn_readfirstlane(m.route_slots[mapped]);
        input[row] = codes(l, token, slot, Down, projection);
    }
    for (unsigned group = partition.begin_tile / 2; group < unsigned(partition.end_tile / 2); ++group) {
        int32_t bq[Rows][2] = {};
        int32_t qsum[Rows];float delta[Rows];
        #pragma unroll
        for (unsigned row = 0; row < Rows; ++row) {
            qsum[row] = sums(input[row], Down)[group];
            delta[row] = scales(input[row], Down)[group];
        }
        #pragma unroll
        for (unsigned step = 0; step < 2; ++step) {
            const unsigned k = group * 2 + step;
            const auto window = tp_moe_reader_window<2>(lane);
            const auto* base = reinterpret_cast<const unsigned char*>(w.trellis + (size_t(k) * (w.output / 16) + n) * 16);
            const k2_local_kernel::Raw raw{k2_local_kernel::word_load(base, window.a),
                k2_local_kernel::word_load(base, window.b), 0, 0};
            uint32_t states[8], hash[8];k2_local_kernel::states(raw, states);pipeline_hash(states, hash);
            #pragma unroll
            for (unsigned pair = 0; pair < 4; ++pair) {
                #pragma unroll
                for (unsigned row = 0; row < Rows; ++row) {
                    const unsigned column = k * 16 + 2 * (lane & 3) + (pair & 1) * 8;
                    const uint32_t first = (uint32_t(input[row][column]) & 255u) * 0x01010101u;
                    const uint32_t second = (uint32_t(input[row][column + 1]) & 255u) * 0x01010101u;
                    bq[row][pair / 2] = __builtin_amdgcn_sudot4(false, hash[2 * pair], true, first, bq[row][pair / 2], false);
                    bq[row][pair / 2] = __builtin_amdgcn_sudot4(false, hash[2 * pair + 1], true, second, bq[row][pair / 2], false);
                }
            }
        }
        #pragma unroll
        for (unsigned row = 0; row < Rows; ++row) {
            #pragma unroll
            for (unsigned half = 0; half < 2; ++half) {
                int32_t total = bq[row][half];total += __shfl_down(total, 2, 4);total += __shfl_down(total, 1, 4);
                // Only quad leaders are stored. Computing the discarded lane
                // values avoids a masked execution transition every group.
                value[row][half] = arithmetic::add(value[row][half],
                    affine_group(total, qsum[row], delta[row]));
            }
        }
    }
    #pragma unroll
    for (unsigned row = 0; row < Rows; ++row) if (row < m.row_count && !(lane & 3)) {
        #pragma unroll
        for (unsigned half = 0; half < 2; ++half) {
            const unsigned column = n * 16 + lane / 4 + half * 8;
            const uint64_t index = Down ? tp_moe_split_index(m.token_rows[row], m.route_slots[row], split, column)
                : tp_moe_pipeline_gate_index(l.rows, projection, m.token_rows[row], m.route_slots[row], split, column);
            l.partials[index] = value[row][half];
        }
    }
}
// One lane owns both columns of a complete output pair. Its three words per
// K16 tile cover the same four overlapping reader windows previously owned
// by a quad. Exact integer accumulation removes every per-group lane shuffle.
template<unsigned Rows, bool Down>
__device__ __forceinline__ void dot_wide(const DeviceWeights& w, const TpMoeLaunch& l,
    const TpMoeDeviceRows& m, unsigned projection) {
    const unsigned lane = threadIdx.x & 31;
    constexpr unsigned Outputs = wide_outputs;
    const unsigned tile = blockIdx.x * (8 * Outputs) + unsigned(threadIdx.x) / (16 / Outputs);
    if (tile >= unsigned(w.output / 16)) return;
    const unsigned column_pair = lane & 7, split = blockIdx.z & 3;
    const auto partition = tp_fused_partition(w.input, split);
    float value[Rows][Outputs] = {};
    signed char* input[Rows];
    #pragma unroll
    for (unsigned row = 0; row < Rows; ++row) {
        const unsigned mapped = row < m.row_count ? row : 0;
        const unsigned token = __builtin_amdgcn_readfirstlane(m.token_rows[mapped]);
        const unsigned slot = __builtin_amdgcn_readfirstlane(m.route_slots[mapped]);
        input[row] = codes(l, token, slot, Down, projection);
    }
    for (unsigned group = partition.begin_tile / 2; group < unsigned(partition.end_tile / 2); ++group) {
        int32_t bq[Rows][Outputs] = {}, qsum[Rows];float delta[Rows];
        #pragma unroll
        for (unsigned row = 0; row < Rows; ++row) {
            qsum[row] = sums(input[row], Down)[group];delta[row] = scales(input[row], Down)[group];
#ifdef STRIX_TRELLIS_TP_MOE_A8R_SCALAR_INPUT
            qsum[row] = __builtin_amdgcn_readfirstlane(qsum[row]);
            delta[row] = arithmetic::from_bits(__builtin_amdgcn_readfirstlane(arithmetic::bits(delta[row])));
#endif
        }
        #pragma unroll
        for (unsigned step = 0; step < 2; ++step) {
            const unsigned k = group * 2 + step;
            const auto* base = reinterpret_cast<const unsigned char*>(w.trellis + (size_t(k) * (w.output / 16) + tile) * 16);
            const uint32_t a = k2_local_kernel::word_load(base, (2 * column_pair + 15) & 15);
            const uint32_t b = k2_local_kernel::word_load(base, 2 * column_pair);
            const uint32_t c = k2_local_kernel::word_load(base, 2 * column_pair + 1);
#if STRIX_TRELLIS_TP_MOE_A8R_CLUSTER_UNROLL == 1
            #pragma clang loop unroll(disable)
#elif STRIX_TRELLIS_TP_MOE_A8R_CLUSTER_UNROLL == 2
            #pragma clang loop unroll_count(2)
#else
            #pragma clang loop unroll(full)
#endif
            for (unsigned cluster = 0; cluster < 4; ++cluster) {
                uint32_t window = cluster == 0 ? (a << 16) | (b >> 16)
                    : cluster == 1 ? b : cluster == 2 ? (b << 16) | (c >> 16) : c;
#if STRIX_TRELLIS_TP_MOE_A8R_HASH_FENCE > 0
                if (cluster % STRIX_TRELLIS_TP_MOE_A8R_HASH_FENCE == 0) {
                    // A zero-instruction compiler dependency limits live hash
                    // batches. Neither the window nor integer sums change.
                    asm volatile("" : "+v"(window) : "v"(bq[0][0]), "v"(bq[0][Outputs-1]) : "memory");
                }
#endif
                if constexpr (Outputs == 1) window >>= (lane & 8) ? 0 : 8;
                uint32_t states[4 * Outputs], hash[4 * Outputs];
                #pragma unroll
                for (unsigned j = 0; j < 4 * Outputs; ++j) states[j] = window >> (8 * Outputs - 2 - 2 * j);
                wide_hash(states, hash);
                #pragma unroll
                for (unsigned row = 0; row < Rows; ++row) {
                    uint32_t packed[4];
                    #pragma unroll
                    for (unsigned j = 0; j < 4; ++j) {
                        const unsigned index = k * 16 + cluster * 2 + (j & 1) + (j / 2) * 8;
                        uint32_t code = uint32_t(input[row][index]) & 255u;
#ifdef STRIX_TRELLIS_TP_MOE_A8R_SCALAR_INPUT
                        code = __builtin_amdgcn_readfirstlane(code);
#endif
                        packed[j] = code * 0x01010101u;
                    }
                    #pragma unroll
                    for (unsigned j = 0; j < 4 * Outputs; ++j)
                        bq[row][j / 4] = __builtin_amdgcn_sudot4(false, hash[j], true, packed[j & 3], bq[row][j / 4], false);
                }
            }
        }
        #pragma unroll
        for (unsigned row = 0; row < Rows; ++row)
            for (unsigned half = 0; half < Outputs; ++half)
                value[row][half] = arithmetic::add(value[row][half], affine_group(bq[row][half], qsum[row], delta[row]));
    }
    #pragma unroll
    for (unsigned row = 0; row < Rows; ++row) if (row < m.row_count) {
        #pragma unroll
        for (unsigned half = 0; half < Outputs; ++half) {
            const unsigned column = tile * 16 + (Outputs == 2 ? column_pair : lane & 15) + half * 8;
            const uint64_t index = Down ? tp_moe_split_index(m.token_rows[row], m.route_slots[row], split, column)
                : tp_moe_pipeline_gate_index(l.rows, projection, m.token_rows[row], m.route_slots[row], split, column);
            l.partials[index] = value[row][half];
        }
    }
}
template<unsigned Rows, bool Down>
__device__ __forceinline__ void dot(const DeviceWeights& w, const TpMoeLaunch& l,
    const TpMoeDeviceRows& m, unsigned projection) {
#ifdef STRIX_TRELLIS_TP_MOE_A8R_WIDE
    dot_wide<Rows, Down>(w, l, m, projection);
#else
    dot_narrow<Rows, Down>(w, l, m, projection);
#endif
}
template<unsigned Capacity, bool Down>
__global__ __launch_bounds__(128) void core(TpMoeLaunch l) {
    __shared__ TpMoeDeviceRows m; __shared__ bool owner;
    if (!threadIdx.x) owner = resolve(l, m, Down);
    __syncthreads(); if (!owner) return;
    const unsigned bank_index = __builtin_amdgcn_readfirstlane(m.bank_index);
    const auto& e = l.bank ? l.bank[bank_index] : l.table->experts[bank_index];
    const unsigned projection = Down ? 0 : blockIdx.z / 4;
    const auto& w = Down ? e.down : projection ? e.up : e.gate;
    if (m.row_count == 1) dot<1, Down>(w, l, m, projection);
    else if constexpr (Capacity >= 2) {
        if (m.row_count == 2) dot<2, Down>(w, l, m, projection);
        else if constexpr (Capacity >= 4) {
            if (m.row_count <= 4) dot<4, Down>(w, l, m, projection);
            else if constexpr (Capacity >= 8) dot<8, Down>(w, l, m, projection);
        }
    }
}
template<bool PrepareDown>
__global__ __launch_bounds__(64) void gate_finish(TpMoeLaunch l) {
    __shared__ float projected[2][128];
    const unsigned row = blockIdx.y / 6, slot = blockIdx.y % 6, lane = threadIdx.x & 31, projection = threadIdx.x / 32;
    unsigned expert;
    if (l.bank) {
        const auto* ids = l.device_route_ids + row * 6;expert = ids[slot];
        if (expert >= l.bank_count) return;
        for (unsigned s = 0; s < slot; ++s) if (ids[s] == expert) return;
    } else expert = l.table->expert_indices[row][slot];
    const auto& e = l.bank ? l.bank[expert] : l.table->experts[expert];
    const auto& w = projection ? e.up : e.gate;
    float v[4] = {};
    for (unsigned split = 0; split < 4; ++split)
        for (unsigned j = 0; j < 4; ++j) v[j] = arithmetic::add(v[j],
            l.partials[tp_moe_pipeline_gate_index(l.rows, projection, row, slot, split, blockIdx.x * 128 + lane + j * 32)]);
    hadamard128(v);
    for (unsigned j = 0; j < 4; ++j) {
        const unsigned column = blockIdx.x * 128 + lane + j * 32;
        float result = arithmetic::mul(v[j], __half2float(__ushort_as_half(w.svh[column])));
        if (w.bias) result = arithmetic::add(result, w.bias[column]);
        projected[projection][lane + j * 32] = result;
    }
    __syncthreads();if (projection) return;
    for (unsigned j = 0; j < 4; ++j) {
        const unsigned column = blockIdx.x * 128 + lane + j * 32;
        float hidden = arithmetic::mul(arithmetic::silu(projected[0][lane + j * 32]), projected[1][lane + j * 32]);
        if (!arithmetic::finite(hidden)) hidden = arithmetic::canonical_nan();
        l.hidden[(uint64_t(row) * 6 + slot) * 1152 + column] = hidden;
        if constexpr (PrepareDown) v[j] = float(_Float16(arithmetic::mul(hidden, __half2float(__ushort_as_half(e.down.suh[column])))));
    }
    if constexpr (PrepareDown) { hadamard128(v);quantize(v, codes(l, row, slot, true), blockIdx.x, true); }
}
__global__ __launch_bounds__(192) void finish(TpMoeLaunch l) {
    __shared__ float weighted[6][128];
    const unsigned row = blockIdx.y, lane = threadIdx.x & 31;
    const unsigned slot = __builtin_amdgcn_readfirstlane(unsigned(threadIdx.x) / 32);
    const auto* ids = l.bank ? l.device_route_ids + row * 6 : l.table->expert_indices[row];
    const auto* scores = l.bank ? l.device_route_scores + row * 6 : l.table->route_scores[row];
    const bool valid = !l.bank || tp_moe_device_row_valid(ids, scores, l.bank_count);
    float v[4] = {};
    if (valid && scores[slot] != 0.0f) {
        const unsigned expert = __builtin_amdgcn_readfirstlane(ids[slot]);
        const auto& w = l.bank ? l.bank[expert].down : l.table->experts[expert].down;
        for (unsigned split = 0; split < 4; ++split)
            for (unsigned j = 0; j < 4; ++j) v[j] = arithmetic::add(v[j],
                l.partials[tp_moe_split_index(row, slot, split, blockIdx.x * 128 + lane + j * 32)]);
        hadamard128(v);
        for (unsigned j = 0; j < 4; ++j) {
            const unsigned column = blockIdx.x * 128 + lane + j * 32;
            float result = arithmetic::mul(v[j], __half2float(__ushort_as_half(w.svh[column])));
            if (w.bias) result = arithmetic::add(result, w.bias[column]);
            v[j] = arithmetic::mul(result, scores[slot]);
        }
    }
    for (unsigned j = 0; j < 4; ++j) weighted[slot][lane + j * 32] = v[j];
    __syncthreads();
    if (threadIdx.x < 128) {
        float total = 0.0f;
        if (valid) for (unsigned s = 0; s < 6; ++s)
            if (scores[s] != 0.0f) total = arithmetic::add(total, weighted[s][threadIdx.x]);
        const uint64_t index = uint64_t(row) * l.y_stride + blockIdx.x * 128 + threadIdx.x;
        total = arithmetic::mul(l.alpha, total);
        if (l.beta != 0.0f) total = arithmetic::add(total, arithmetic::mul(l.beta, l.y[index]));
        l.y[index] = valid && arithmetic::finite(total) ? total : arithmetic::canonical_nan();
    }
}
template<bool Down> const void* core_function(unsigned rows) {
    if (rows == 1) return reinterpret_cast<const void*>(core<1, Down>);
    if (rows == 2) return reinterpret_cast<const void*>(core<2, Down>);
    if (rows <= 4) return reinterpret_cast<const void*>(core<4, Down>);
    return reinterpret_cast<const void*>(core<8, Down>);
}
template<bool Down> hipError_t launch_core(const TpMoeLaunch& l, hipStream_t stream) {
    const dim3 grid(
#ifdef STRIX_TRELLIS_TP_MOE_A8R_WIDE
        Down ? 320 / (8 * wide_outputs) : (72 + 8 * wide_outputs - 1) / (8 * wide_outputs),
#else
        Down ? 80 : 18,
#endif
        l.bank ? l.rows * 6 : l.expert_count, Down ? 4 : 8);
    if (l.rows == 1) core<1, Down><<<grid, 128, 0, stream>>>(l);
    else if (l.rows == 2) core<2, Down><<<grid, 128, 0, stream>>>(l);
    else if (l.rows <= 4) core<4, Down><<<grid, 128, 0, stream>>>(l);
    else core<8, Down><<<grid, 128, 0, stream>>>(l);
    return hipGetLastError();
}
template<bool PrepareDown> hipError_t launch_gate(const TpMoeLaunch& l, hipStream_t stream) {
    prepare<false><<<dim3(10, l.bank ? l.rows * 6 : l.expert_count, 2), 128, 0, stream>>>(l);
    auto status = hipGetLastError();if (status != hipSuccess) return status;
    status = launch_core<false>(l, stream);if (status != hipSuccess) return status;
    gate_finish<PrepareDown><<<dim3(9, l.rows * 6), 64, 0, stream>>>(l);
    return hipGetLastError();
}
} // namespace a8r_kernel
