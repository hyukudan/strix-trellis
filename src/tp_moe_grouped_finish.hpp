#pragma once
// Six waves finish the original route partials concurrently. The ordered slot
// combine then consumes materialized weighted values in LDS, without a global
// intermediate or a second launch. H and P remain unchanged.
__global__ __launch_bounds__(tp_moe_grouped_finish_threads) void grouped_route_finish(TpMoeLaunch l) {
    __shared__ float weighted[6][128];
    const unsigned row = blockIdx.y, lane = threadIdx.x & 31;
    const unsigned wave = __builtin_amdgcn_readfirstlane(unsigned(threadIdx.x) / 32);
    const auto* ids = l.bank ? l.device_route_ids + row * 6 : l.table->expert_indices[row];
    const auto* scores = l.bank ? l.device_route_scores + row * 6 : l.table->route_scores[row];
    const bool valid = !l.bank || tp_moe_device_row_valid(ids, scores, l.bank_count);
    if (wave < 6) {
        float v[4] = {};
        if (valid && scores[wave] != 0.0f) {
            const auto& w = l.bank ? l.bank[ids[wave]].down : l.table->experts[ids[wave]].down;
            #pragma unroll
            for (unsigned split = 0; split < 4; ++split) {
                #pragma unroll
                for (unsigned j = 0; j < 4; ++j)
                    v[j] += l.partials[tp_moe_split_index(row, wave, split, blockIdx.x * 128 + lane + j * 32)];
            }
            hadamard128(v);
            #pragma unroll
            for (unsigned j = 0; j < 4; ++j) {
                const unsigned column = blockIdx.x * 128 + lane + j * 32;
                float value = v[j] * __half2float(__ushort_as_half(w.svh[column]));
                if (w.bias) value += w.bias[column];
                v[j] = materialize_fp32(scores[wave] * value);
            }
        }
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j) weighted[wave][lane + j * 32] = v[j];
    }
    __syncthreads();
    if (threadIdx.x < 128) {
        float sum = 0.0f;
        if (valid) {
            #pragma unroll
            for (unsigned slot = 0; slot < 6; ++slot) {
                if (scores[slot] != 0.0f)
                    sum = materialize_fp32(sum + weighted[slot][threadIdx.x]);
            }
        }
        const uint64_t output = uint64_t(row) * l.y_stride + blockIdx.x * 128 + threadIdx.x;
        float value = materialize_fp32(l.alpha * sum);
        if (l.beta != 0.0f) value = materialize_fp32(value + materialize_fp32(l.beta * l.y[output]));
        l.y[output] = valid ? value : __builtin_bit_cast(float, uint32_t(0x7fc00000u));
    }
}
