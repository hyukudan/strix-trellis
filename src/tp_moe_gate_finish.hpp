#pragma once
// The two independent H128 projections use one wave each. SwiGLU and the
// optional down input transform retain the original F32/RN16 boundaries.
template <bool PrepareDown>
__global__ __launch_bounds__(tp_moe_gate_finish_threads) void split_gate_finish(TpMoeLaunch l) {
    __shared__ float projected[tp_moe_gate_finish_tiles][2][128];
    const unsigned row = blockIdx.y / 6, slot = blockIdx.y % 6;
    const unsigned lane = threadIdx.x & 31;
    const unsigned projection = __builtin_amdgcn_readfirstlane((unsigned(threadIdx.x) / 32) & 1);
    const unsigned local_tile = __builtin_amdgcn_readfirstlane(unsigned(threadIdx.x) / 64);
    const unsigned tile = blockIdx.x * tp_moe_gate_finish_tiles + local_tile;
    unsigned expert;
    if (l.bank) {
        const auto* ids = l.device_route_ids + row * 6;
        expert = ids[slot];
        if (expert >= l.bank_count) return;
        for (unsigned s = 0; s < slot; ++s) if (ids[s] == expert) return;
    } else expert = l.table->expert_indices[row][slot];
    const auto& e = l.bank ? l.bank[expert] : l.table->experts[expert];
    const auto& w = projection ? e.up : e.gate;
    float v[4] = {};
    if (tile < 9) {
        #pragma unroll
        for (unsigned split = 0; split < 4; ++split) {
            #pragma unroll
            for (unsigned j = 0; j < 4; ++j)
                v[j] += l.partials[tp_moe_pipeline_gate_index(l.rows, projection, row, slot, split, tile * 128 + lane + j * 32)];
        }
        hadamard128(v);
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            const unsigned column = tile * 128 + lane + j * 32;
            float value = v[j] * __half2float(__ushort_as_half(w.svh[column]));
            if (w.bias) value += w.bias[column];
            projected[local_tile][projection][lane + j * 32] = value;
        }
    }
    __syncthreads();
    if (projection || tile >= 9) return;
    float hidden[4];
    #pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
        const unsigned column = tile * 128 + lane + j * 32;
        const float g = projected[local_tile][0][lane + j * 32], u = projected[local_tile][1][lane + j * 32];
                hidden[j] = tp_moe_swiglu(g, u);
        l.hidden[(uint64_t(row) * 6 + slot) * 1152 + column] = hidden[j];
    }
    if constexpr (PrepareDown) {
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
}
