#pragma once
// Down inputs are dead after B. Reuse their allocation for one F32 vector per
// route, keeping the exported H/P regions and every S4/slot addition unchanged.
__device__ __forceinline__ float* finish_values(const TpMoeLaunch& l) {
    return reinterpret_cast<float*>(pipeline_activation(l, 0, 0, true));
}
__global__ __launch_bounds__(32) void parallel_route_finish(TpMoeLaunch l) {
    const unsigned row = blockIdx.y / 6, slot = blockIdx.y % 6, lane = threadIdx.x;
    const auto* ids = l.bank ? l.device_route_ids + row * 6 : l.table->expert_indices[row];
    const auto* scores = l.bank ? l.device_route_scores + row * 6 : l.table->route_scores[row];
    if ((l.bank && !tp_moe_device_row_valid(ids, scores, l.bank_count)) || scores[slot] == 0.0f) return;
    const auto& w = l.bank ? l.bank[ids[slot]].down : l.table->experts[ids[slot]].down;
    float v[4] = {};
    #pragma unroll
    for (unsigned split = 0; split < 4; ++split) {
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j)
            v[j] += l.partials[tp_moe_split_index(row, slot, split, blockIdx.x * 128 + lane + j * 32)];
    }
    hadamard128(v);
    #pragma unroll
    for (unsigned j = 0; j < 4; ++j) {
        const unsigned column = blockIdx.x * 128 + lane + j * 32;
        float value = v[j] * __half2float(__ushort_as_half(w.svh[column]));
        if (w.bias) value += w.bias[column];
        finish_values(l)[(uint64_t(row) * 6 + slot) * 5120 + column] = value;
    }
}
__global__ __launch_bounds__(128) void parallel_slot_combine(TpMoeLaunch l) {
    const unsigned row = blockIdx.y, column = blockIdx.x * 128 + threadIdx.x;
    const auto* ids = l.bank ? l.device_route_ids + row * 6 : l.table->expert_indices[row];
    const auto* scores = l.bank ? l.device_route_scores + row * 6 : l.table->route_scores[row];
    const bool valid = !l.bank || tp_moe_device_row_valid(ids, scores, l.bank_count);
    const uint64_t output = uint64_t(row) * l.y_stride + column;
    float sum = 0.0f;
    if (valid) {
        #pragma unroll
        for (unsigned slot = 0; slot < 6; ++slot) {
            if (scores[slot] == 0.0f) continue;
            const float value = finish_values(l)[(uint64_t(row) * 6 + slot) * 5120 + column];
            const float weighted = materialize_fp32(scores[slot] * value);
            sum = materialize_fp32(sum + weighted);
        }
    }
    float value = materialize_fp32(l.alpha * sum);
    if (l.beta != 0.0f) value = materialize_fp32(value + materialize_fp32(l.beta * l.y[output]));
    l.y[output] = valid ? value : __builtin_bit_cast(float, uint32_t(0x7fc00000u));
}
