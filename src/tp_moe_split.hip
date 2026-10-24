// Independent S4 expert-TP split schedule. The legacy two-kernel path is kept.
#include "tp_moe.hpp"
#include "tp_moe_activation.hpp"
#include "tp_moe_split_layout.hpp"
#include "tp_moe_reader.hpp"
#include "tp_fused_device.hpp"
#include "tp_moe_weight_tile.hpp"
#ifdef STRIX_TRELLIS_TP_MOE_PIPELINED_K2
#include "k2_local_decode.hpp"
#endif
#include <cmath>
#ifdef STRIX_TRELLIS_TP_MOE_A8R
#include "tp_moe_a8r_math.hpp"
#endif
#ifdef STRIX_TRELLIS_TP_MOE_PERSISTENT
#include <hip/hip_cooperative_groups.h>
#endif

namespace strix_trellis::tp_moe_split {
using tp_fused::Pair;

// A group owns one complete mathematical S4 partition. Four waves each own
// two N16 tiles in A; eight waves each own one tile in B. Neither schedule
// changes the per-tile fdot2 chain or the down2 -> down1 lane reduction.
template<int Bits, int Rows, int Waves, class Access, class Store>
__device__ __forceinline__ void core(const DeviceWeights& w, const Access& access,
    const Store& store, unsigned split, unsigned wave, _Float16* cache) {
    const int lane = threadIdx.x & 31;
    const auto p = tp_fused_partition(w.input, split);
    constexpr int Tiles = 8 / Waves;
    float sum[Rows][Tiles][2] = {};
    // A has four groups with possibly different H128 extents. All threads
    // must execute the same number of barriers even for a cropped boundary.
    for (int step = 0; step < tp_fused_maximum_h128(w.input); ++step) {
        const int h = p.first_h128 + step;
        if (h < p.end_h128) {
            for (int row = int(wave); row < Rows; row += Waves) {
                float v[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int column = h * 128 + lane + j * 32;
                    const float scaled = materialize_fp32(access.load(row, column) *
                        __half2float(__ushort_as_half(w.suh[column])));
                    v[j] = float(_Float16(scaled));
                }
                hadamard128(v);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    cache[row * 128 + lane + j * 32] = _Float16(v[j]);
                }
            }
        }
        __syncthreads();
        if (h < p.end_h128) {
            const int begin = p.begin_tile > h * 8 ? p.begin_tile : h * 8;
            const int end = p.end_tile < (h + 1) * 8 ? p.end_tile : (h + 1) * 8;
            for (int k = begin; k < end; ++k) {
                #pragma unroll
                for (int tile = 0; tile < Tiles; ++tile) {
                    const int n = blockIdx.x * 8 + wave + tile * Waves;
                    {
                    TpMoeWeightTile<Bits> reader(w,k,n);
                    Pair input[Rows][2];
                    const int local = (k - h * 8) * 16 + 2 * (lane & 3);
                    #pragma unroll
                    for (int row = 0; row < Rows; ++row) {
                        __builtin_memcpy(&input[row][0], cache + row * 128 + local, 4);
                        __builtin_memcpy(&input[row][1], cache + row * 128 + local + 8, 4);
                    }
                    #pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        __half2 decoded;
                        if (pair == 0) decoded=reader.template pair<0>();
                        else if (pair == 1) decoded=reader.template pair<1>();
                        else if (pair == 2) decoded=reader.template pair<2>();
                        else decoded=reader.template pair<3>();
                        const Pair weight = __builtin_bit_cast(Pair,decoded);
                        #pragma unroll
                        for (int row = 0; row < Rows; ++row)
                            sum[row][tile][pair / 2] = __builtin_amdgcn_fdot2(
                                weight, input[row][pair & 1], sum[row][tile][pair / 2], false);
                    }
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int row = 0; row < Rows; ++row) {
        #pragma unroll
        for (int tile = 0; tile < Tiles; ++tile) {
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                float v = sum[row][tile][half];
                v += __shfl_down(v, 2, 4);
                v += __shfl_down(v, 1, 4);
                if (!(lane & 3))
                    store.put(row, split, (wave + tile * Waves) * 16 + lane / 4 + half * 8, v);
            }
        }
    }
}
struct Input {
    const TpMoeLaunch& l;
    const TpMoeDeviceRows& m;
    bool down;
    __device__ __forceinline__ float load(unsigned row, unsigned column) const {
        if (row >= m.row_count) return 0.0f;
        return down ? l.hidden[(uint64_t(m.token_rows[row]) * 6 + m.route_slots[row]) * 1152 + column]
            : l.x[uint64_t(m.token_rows[row]) * l.x_stride + column];
    }
};
template<int Rows> struct LocalStore {
    float* partial;
    __device__ __forceinline__ void put(unsigned row, unsigned split, unsigned column, float v) const {
        partial[(split * Rows + row) * 128 + column] = v;
    }
};
struct GlobalStore {
    const TpMoeLaunch& l;
    const TpMoeDeviceRows& m;
    __device__ __forceinline__ void put(unsigned row, unsigned split, unsigned column, float v) const {
        if (row < m.row_count)
            l.partials[tp_moe_split_index(m.token_rows[row], m.route_slots[row], split,
                blockIdx.x * 128 + column)] = v;
    }
};
// One thread constructs metadata; all projection pointers are wave-uniform.
__device__ __forceinline__ bool resolve(const TpMoeLaunch& l, TpMoeDeviceRows& m,
    bool down, unsigned candidate = blockIdx.y) {
    if (l.bank) {
        if (!tp_moe_resolve_owner(l.device_route_ids, l.rows, l.bank_count, candidate, m)) return false;
        if (down) tp_moe_split_active_rows(m, l.device_route_ids, l.device_route_scores, l.bank_count);
    } else {
        const auto& e = l.table->experts[candidate];
        m.bank_index = candidate; m.row_count = 0;
        for (unsigned i = 0; i < e.row_count; ++i) {
            const unsigned row = e.token_rows[i], slot = e.route_slots[i];
            if (down && l.table->route_scores[row][slot] == 0.0f) continue;
            const unsigned n = m.row_count++;
            m.token_rows[n] = row; m.route_slots[n] = slot;
        }
    }
    return m.row_count != 0;
}

template<int Bits, int Rows>
__device__ __forceinline__ void gate_project(const DeviceWeights& w,
    const TpMoeLaunch& l, const TpMoeDeviceRows& m, float* gate, float* partial,
    _Float16* cache, bool up, _Float16* prepared = nullptr) {
    const unsigned group = __builtin_amdgcn_readfirstlane(unsigned(threadIdx.x) / 128);
    const unsigned wave = __builtin_amdgcn_readfirstlane((unsigned(threadIdx.x) & 127) / 32);
    core<Bits, Rows, 4>(w, Input{l, m, false}, LocalStore<Rows>{partial}, group, wave,
        cache + group * Rows * 128);
    __syncthreads();
    const unsigned row = threadIdx.x / 32, lane = threadIdx.x & 31;
    if (row < m.row_count) {
        float v[4] = {};
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) v[j] += partial[(s * Rows + row) * 128 + lane + j * 32];
        }
        hadamard128(v);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned local = lane + j * 32, column = blockIdx.x * 128 + local;
            float value = v[j] * __half2float(__ushort_as_half(w.svh[column]));
            if (w.bias) value += w.bias[column];
            if (!up) gate[row * 128 + local] = value;
            else {
                const float g = gate[row * 128 + local];
                l.hidden[(uint64_t(m.token_rows[row]) * 6 + m.route_slots[row]) * 1152 + column]
                    = tp_moe_swiglu(g, value);
            }
        }
    }
    __syncthreads();
}
template<int Rows>
__device__ __forceinline__ void gates(const TpMoeExpert& e, const TpMoeLaunch& l,
    const TpMoeDeviceRows& m, float* g, float* p, _Float16* c, _Float16* prepared = nullptr) {
    if (e.gate.bits == 2) gate_project<2, Rows>(e.gate, l, m, g, p, c, false, prepared);
    else gate_project<3, Rows>(e.gate, l, m, g, p, c, false, prepared);
    if (e.up.bits == 2) gate_project<2, Rows>(e.up, l, m, g, p, c, true, prepared);
    else gate_project<3, Rows>(e.up, l, m, g, p, c, true, prepared);
}
template<int Capacity>
__global__ __launch_bounds__(512) void kernel_a(TpMoeLaunch l) {
    __shared__ float gate[Capacity * 128];
    __shared__ float partial[4 * Capacity * 128];
    _Float16* prepared = nullptr;
    __shared__ _Float16 cache[4 * Capacity * 128];
    __shared__ TpMoeDeviceRows m;
    __shared__ bool owner;
    if (threadIdx.x == 0) owner = resolve(l, m, false);
    __syncthreads();
    if (!owner) return;
    const auto& e = l.bank ? l.bank[m.bank_index] : l.table->experts[m.bank_index];
    if (m.row_count == 1) gates<1>(e, l, m, gate, partial, cache, prepared);
    else if constexpr (Capacity >= 2) {
        if (m.row_count <= 2) gates<2>(e, l, m, gate, partial, cache, prepared);
        else if constexpr (Capacity >= 4) {
            if (m.row_count <= 4) gates<4>(e, l, m, gate, partial, cache, prepared);
            else if constexpr (Capacity >= 8) gates<8>(e, l, m, gate, partial, cache, prepared);
        }
    }
}
template<int Rows>
__device__ __forceinline__ void down_core(const DeviceWeights& w, const TpMoeLaunch& l,
    const TpMoeDeviceRows& m, _Float16* cache) {
    const unsigned wave = __builtin_amdgcn_readfirstlane(unsigned(threadIdx.x) / 32);
    if (w.bits == 2) {
        core<2, Rows, 8>(w, Input{l, m, true}, GlobalStore{l, m}, blockIdx.z, wave, cache);
    }
    else core<3, Rows, 8>(w, Input{l, m, true}, GlobalStore{l, m}, blockIdx.z, wave, cache);
}
template<int Capacity>
__global__ __launch_bounds__(256) void kernel_b(TpMoeLaunch l) {
    __shared__ _Float16 cache[Capacity * 128];
    __shared__ TpMoeDeviceRows m;
    __shared__ bool owner;
    if (threadIdx.x == 0) owner = resolve(l, m, true);
    __syncthreads();
    if (!owner) return;
    const auto& w = l.bank ? l.bank[m.bank_index].down : l.table->experts[m.bank_index].down;
    if (m.row_count == 1) down_core<1>(w, l, m, cache);
    else if constexpr (Capacity >= 2) {
        if (m.row_count <= 2) down_core<2>(w, l, m, cache);
        else if constexpr (Capacity >= 4) {
            if (m.row_count <= 4) down_core<4>(w, l, m, cache);
            else if constexpr (Capacity >= 8) down_core<8>(w, l, m, cache);
        }
    }
}
#ifdef STRIX_TRELLIS_TP_MOE_PIPELINED_K2
#include "tp_moe_pipeline.hpp"
#endif
#ifdef STRIX_TRELLIS_TP_MOE_A8R
#include "tp_moe_a8r.hpp"
#endif
__global__ __launch_bounds__(32) void kernel_c(TpMoeLaunch l) {
    const unsigned row = blockIdx.y, lane = threadIdx.x;
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
                    v[j] += l.partials[tp_moe_split_index(row, slot, split, blockIdx.x * 128 + lane + j * 32)];
            }
            hadamard128(v);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned column = blockIdx.x * 128 + lane + j * 32;
                float result = v[j] * __half2float(__ushort_as_half(w.svh[column]));
                if (w.bias) result += w.bias[column];
                const float weighted = materialize_fp32(scores[slot] * result);
                combined[j] = materialize_fp32(combined[j] + weighted);
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint64_t output = uint64_t(row) * l.y_stride + blockIdx.x * 128 + lane + j * 32;
        float v = materialize_fp32(l.alpha * combined[j]);
        if (l.beta != 0.0f) v = materialize_fp32(v + materialize_fp32(l.beta * l.y[output]));
        l.y[output] = valid ? v : __builtin_bit_cast(float, uint32_t(0x7fc00000u));
    }
}
#ifdef STRIX_TRELLIS_TP_MOE_PERSISTENT
#include "tp_moe_persistent.hpp"
#endif
#ifdef STRIX_TRELLIS_TP_MOE_PARALLEL_FINISH
#include "tp_moe_finish.hpp"
#endif
#ifdef STRIX_TRELLIS_TP_MOE_GROUPED_FINISH
#include "tp_moe_grouped_finish.hpp"
#endif
#ifdef STRIX_TRELLIS_TP_MOE_FOLD_R1
#include "tp_moe_fold.hpp"
#endif
__global__ __launch_bounds__(256) void route_probe(unsigned rows, unsigned bank_count,
    const uint32_t* ids, const float* scores, uint32_t* output) {
    if (threadIdx.x != 0) return;
    TpMoeDeviceRows m{};
    uint32_t hash = 0;
    if (tp_moe_resolve_owner(ids, rows, bank_count, blockIdx.y, m)) {
        tp_moe_split_active_rows(m, ids, scores, bank_count);
        hash = m.row_count;
        for (unsigned i = 0; i < m.row_count; ++i)
            hash = hash * 31 + m.token_rows[i] * 6 + m.route_slots[i];
    }
    output[(blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x] = hash;
}
const void* a_function(unsigned rows) {
#ifdef STRIX_TRELLIS_TP_MOE_PIPELINED_K2
    return pipeline_function<false>(rows);
#else
    if (rows == 1) return reinterpret_cast<const void*>(kernel_a<1>);
    if (rows == 2) return reinterpret_cast<const void*>(kernel_a<2>);
    if (rows <= 4) return reinterpret_cast<const void*>(kernel_a<4>);
    return reinterpret_cast<const void*>(kernel_a<8>);
#endif
}
const void* b_function(unsigned rows) {
#ifdef STRIX_TRELLIS_TP_MOE_PIPELINED_K2
    return pipeline_function<true>(rows);
#else
    if (rows == 1) return reinterpret_cast<const void*>(kernel_b<1>);
    if (rows == 2) return reinterpret_cast<const void*>(kernel_b<2>);
    if (rows <= 4) return reinterpret_cast<const void*>(kernel_b<4>);
    return reinterpret_cast<const void*>(kernel_b<8>);
#endif
}
} // namespace strix_trellis::tp_moe_split
namespace strix_trellis {
hipError_t launch_tp_moe_route_probe(unsigned rows, unsigned bank_count,
    const uint32_t* ids, const float* scores, uint32_t* output, hipStream_t stream) {
    if (rows < 1 || rows > 8 || bank_count < 6 || bank_count > 256 || !ids || !scores || !output)
        return hipErrorInvalidValue;
    tp_moe_split::route_probe<<<dim3(40, rows * 6, 4),256,0,stream>>>(rows,bank_count,ids,scores,output);
    return hipGetLastError();
}
const void* tp_moe_split_function(unsigned stage, unsigned rows) {
#if defined(STRIX_TRELLIS_TP_MOE_PIPELINED_K2) && defined(STRIX_TRELLIS_TP_MOE_DEVICE_ROUTE_MAPS)
    if (rows >= 1 && rows <= 8 && stage == 13)
        return reinterpret_cast<const void*>(tp_moe_split::pipeline_prepare_routes);
#endif
#if defined(STRIX_TRELLIS_TP_MOE_PIPELINED_K2) && defined(STRIX_TRELLIS_TP_MOE_DEVICE_MERGED_CLASSES)
    if (rows >= 1 && rows <= 8 && stage == 14)
        return rows <= 4 ? reinterpret_cast<const void*>(tp_moe_split::pipeline_core<4, false, 3>)
                         : reinterpret_cast<const void*>(tp_moe_split::pipeline_core<8, false, 3>);
    if (rows >= 1 && rows <= 8 && stage == 15)
        return rows <= 4 ? reinterpret_cast<const void*>(tp_moe_split::pipeline_core<4, true, 3>)
                         : reinterpret_cast<const void*>(tp_moe_split::pipeline_core<8, true, 3>);
#endif
#ifdef STRIX_TRELLIS_TP_MOE_PARALLEL_FINISH
    if (rows >= 1 && rows <= 8 && stage == 11)
        return reinterpret_cast<const void*>(tp_moe_split::parallel_route_finish);
    if (rows >= 1 && rows <= 8 && stage == 12)
        return reinterpret_cast<const void*>(tp_moe_split::parallel_slot_combine);
#endif
#ifdef STRIX_TRELLIS_TP_MOE_ADAPTIVE_ROWS
    if (rows >= 1 && rows <= 8 && stage == 9)
        return reinterpret_cast<const void*>(tp_moe_split::pipeline_core<8, false, 2>);
    if (rows >= 1 && rows <= 8 && stage == 10)
        return reinterpret_cast<const void*>(tp_moe_split::pipeline_core<8, true, 2>);
#endif
#ifdef STRIX_TRELLIS_TP_MOE_PERSISTENT
    if (rows >= 1 && rows <= 8 && stage == 8) return tp_moe_split::persistent_function(rows);
#endif
#ifdef STRIX_TRELLIS_TP_MOE_PIPELINED_K2
#ifdef STRIX_TRELLIS_TP_MOE_FUSE_DOWN_PREPARE
#ifdef STRIX_TRELLIS_TP_MOE_PREPARE_ROUTES
    if (rows < 1 || rows > 8 || stage > 7) return nullptr;
    if (stage == 7) return reinterpret_cast<const void*>(tp_moe_split::pipeline_prepare_routes);
#else
    if (rows < 1 || rows > 8 || stage > 6) return nullptr;
#endif
    if (stage == 6) return tp_moe_split::pipeline_gate_finish_function<true>();
#else
#ifdef STRIX_TRELLIS_TP_MOE_PREPARE_ROUTES
    if (rows < 1 || rows > 8 || stage > 6) return nullptr;
    if (stage == 6) return reinterpret_cast<const void*>(tp_moe_split::pipeline_prepare_routes);
#else
    if (rows < 1 || rows > 8 || stage > 5) return nullptr;
#endif
#endif
    if (stage == 3) return reinterpret_cast<const void*>(tp_moe_split::pipeline_prepare<false>);
    if (stage == 4) return tp_moe_split::pipeline_gate_finish_function<false>();
    if (stage == 5) return reinterpret_cast<const void*>(tp_moe_split::pipeline_prepare<true>);
#else
    if (rows < 1 || rows > 8 || stage > 2) return nullptr;
#endif
    if (stage == 0) return tp_moe_split::a_function(rows);
    if (stage == 1) return tp_moe_split::b_function(rows);
#ifdef STRIX_TRELLIS_TP_MOE_GROUPED_FINISH
    return reinterpret_cast<const void*>(tp_moe_split::grouped_route_finish);
#else
    return reinterpret_cast<const void*>(tp_moe_split::kernel_c);
#endif
}
hipError_t query_tp_moe_split_resources(unsigned rows, size_t available, bool* supported) {
    if (!supported || rows < 1 || rows > 8) return hipErrorInvalidValue;
    *supported = false;
#ifdef STRIX_TRELLIS_TP_MOE_PARALLEL_FINISH
    for (unsigned stage : {11u, 12u}) {
        hipFuncAttributes a{};
        const auto e = hipFuncGetAttributes(&a, tp_moe_split_function(stage, rows));
        if (e != hipSuccess) return e;
        const int threads = stage == 11 ? 32 : 128;
        if (a.maxThreadsPerBlock < threads || a.sharedSizeBytes > available) return hipSuccess;
    }
#endif
#if defined(STRIX_TRELLIS_TP_MOE_PIPELINED_K2) && defined(STRIX_TRELLIS_TP_MOE_DEVICE_ROUTE_MAPS)
    {
        hipFuncAttributes a{};
        const auto e = hipFuncGetAttributes(&a, tp_moe_split_function(13, rows));
        if (e != hipSuccess) return e;
        if (a.maxThreadsPerBlock < 32 || a.sharedSizeBytes > available) return hipSuccess;
    }
#endif
#if defined(STRIX_TRELLIS_TP_MOE_PIPELINED_K2) && defined(STRIX_TRELLIS_TP_MOE_DEVICE_MERGED_CLASSES)
    for (unsigned stage : {14u, 15u}) {
        hipFuncAttributes a{};
        const auto e = hipFuncGetAttributes(&a, tp_moe_split_function(stage, rows));
        if (e != hipSuccess) return e;
        if (a.maxThreadsPerBlock < int(tp_moe_pipeline_core_threads) || a.sharedSizeBytes > available)
            return hipSuccess;
    }
#endif
#ifdef STRIX_TRELLIS_TP_MOE_ADAPTIVE_ROWS
    for (unsigned stage : {9u, 10u}) {
        hipFuncAttributes a{};
        const auto e = hipFuncGetAttributes(&a, tp_moe_split_function(stage, rows));
        if (e != hipSuccess) return e;
        if (a.maxThreadsPerBlock < int(tp_moe_pipeline_core_threads) || a.sharedSizeBytes > available)
            return hipSuccess;
    }
#endif
#ifdef STRIX_TRELLIS_TP_MOE_PIPELINED_K2
#ifdef STRIX_TRELLIS_TP_MOE_FUSE_DOWN_PREPARE
    const int threads[]{int(tp_moe_pipeline_core_threads), int(tp_moe_pipeline_core_threads), 32, 128, 32, 128, 32
#ifdef STRIX_TRELLIS_TP_MOE_PREPARE_ROUTES
        , 32
#endif
#ifdef STRIX_TRELLIS_TP_MOE_PERSISTENT
        , 128
#endif
    };
#else
    const int threads[]{int(tp_moe_pipeline_core_threads), int(tp_moe_pipeline_core_threads), 32, 128, 32, 128
#ifdef STRIX_TRELLIS_TP_MOE_PREPARE_ROUTES
        , 32
#endif
    };
#endif
#else
    const int threads[]{512, 256, 32};
#endif
    for (unsigned i = 0; i < sizeof(threads) / sizeof(threads[0]); ++i) {
        hipFuncAttributes a{};
        const auto e = hipFuncGetAttributes(&a, tp_moe_split_function(i, rows));
        if (e != hipSuccess) return e;
        const int required_threads =
#ifdef STRIX_TRELLIS_TP_MOE_SPLIT_GATE_FINISH
            i == 4
#ifdef STRIX_TRELLIS_TP_MOE_FUSE_DOWN_PREPARE
                || i == 6
#endif
                ? int(tp_moe_gate_finish_threads) :
#endif
#ifdef STRIX_TRELLIS_TP_MOE_GROUPED_FINISH
            i == 2 ? int(tp_moe_grouped_finish_threads) :
#endif
            threads[i];
        if (a.maxThreadsPerBlock < required_threads || a.sharedSizeBytes > available) return hipSuccess;
    }
    *supported = true;
    return hipSuccess;
}
hipError_t launch_tp_moe_split_gate_up(const TpMoeLaunch& l, hipStream_t stream) {
    if (!l.partials || l.rows < 1 || l.rows > 8) return hipErrorInvalidValue;
#ifdef STRIX_TRELLIS_TP_MOE_PIPELINED_K2
    return tp_moe_split::pipeline_launch<false>(l, stream);
#else
    const dim3 grid(9, l.bank ? l.rows * 6 : l.expert_count);
    if (l.rows == 1) tp_moe_split::kernel_a<1><<<grid, 512, 0, stream>>>(l);
    else if (l.rows == 2) tp_moe_split::kernel_a<2><<<grid, 512, 0, stream>>>(l);
    else if (l.rows <= 4) tp_moe_split::kernel_a<4><<<grid, 512, 0, stream>>>(l);
    else tp_moe_split::kernel_a<8><<<grid, 512, 0, stream>>>(l);
    return hipGetLastError();
#endif
}
hipError_t launch_tp_moe_split_core(const TpMoeLaunch& l, hipStream_t stream) {
    if (!l.partials || l.rows < 1 || l.rows > 8) return hipErrorInvalidValue;
#ifdef STRIX_TRELLIS_TP_MOE_PIPELINED_K2
    return tp_moe_split::pipeline_launch<true>(l, stream);
#else
    const dim3 grid(40, l.bank ? l.rows * 6 : l.expert_count, 4);
    if (l.rows == 1) tp_moe_split::kernel_b<1><<<grid, 256, 0, stream>>>(l);
    else if (l.rows == 2) tp_moe_split::kernel_b<2><<<grid, 256, 0, stream>>>(l);
    else if (l.rows <= 4) tp_moe_split::kernel_b<4><<<grid, 256, 0, stream>>>(l);
    else tp_moe_split::kernel_b<8><<<grid, 256, 0, stream>>>(l);
    return hipGetLastError();
#endif
}
hipError_t launch_tp_moe_split_finish(const TpMoeLaunch& l, hipStream_t stream) {
    if (!l.partials || l.rows < 1 || l.rows > 8) return hipErrorInvalidValue;
#ifdef STRIX_TRELLIS_TP_MOE_GROUPED_FINISH
    tp_moe_split::grouped_route_finish<<<dim3(40, l.rows), tp_moe_grouped_finish_threads, 0, stream>>>(l);
#elif defined(STRIX_TRELLIS_TP_MOE_PARALLEL_FINISH)
    tp_moe_split::parallel_route_finish<<<dim3(40, l.rows * 6), 32, 0, stream>>>(l);
    const auto status = hipGetLastError();
    if (status != hipSuccess) return status;
    tp_moe_split::parallel_slot_combine<<<dim3(40, l.rows), 128, 0, stream>>>(l);
#else
    tp_moe_split::kernel_c<<<dim3(40, l.rows), 32, 0, stream>>>(l);
#endif
    return hipGetLastError();
}
} // namespace strix_trellis

#ifdef STRIX_TRELLIS_TP_MOE_A8R
namespace strix_trellis {
namespace a8k = tp_moe_split::a8r_kernel;
const void* tp_moe_a8r_function(unsigned stage, unsigned rows) {
    if (rows < 1 || rows > 8) return nullptr;
    switch (stage) {
    case 0: return reinterpret_cast<const void*>(a8k::prepare<false>);
    case 1: return a8k::core_function<false>(rows);
    case 2: return reinterpret_cast<const void*>(a8k::gate_finish<false>);
    case 3: return reinterpret_cast<const void*>(a8k::prepare<true>);
    case 4: return a8k::core_function<true>(rows);
    case 5: return reinterpret_cast<const void*>(a8k::finish);
    case 6: return reinterpret_cast<const void*>(a8k::gate_finish<true>);
    default: return nullptr;
    }
}
hipError_t query_tp_moe_a8r_resources(unsigned rows, size_t available, bool* supported) {
    if (!supported || rows < 1 || rows > 8) return hipErrorInvalidValue;
    *supported = false;
    const int threads[]{128, 128, 64, 128, 128, 192, 64};
    for (unsigned stage = 0; stage < 7; ++stage) {
        hipFuncAttributes attributes{};
        const auto error = hipFuncGetAttributes(&attributes, tp_moe_a8r_function(stage, rows));
        if (error != hipSuccess) return error;
        if (attributes.maxThreadsPerBlock < threads[stage] || attributes.sharedSizeBytes > available) return hipSuccess;
    }
    *supported = true;return hipSuccess;
}
hipError_t launch_tp_moe_a8r_gate_up(const TpMoeLaunch& l, hipStream_t stream) {
    if (!l.partials || l.rows < 1 || l.rows > 8) return hipErrorInvalidValue;
    return a8k::launch_gate<false>(l, stream);
}
hipError_t launch_tp_moe_a8r_core(const TpMoeLaunch& l, hipStream_t stream) {
    if (!l.partials || l.rows < 1 || l.rows > 8) return hipErrorInvalidValue;
    a8k::prepare<true><<<dim3(3, l.bank ? l.rows * 6 : l.expert_count), 128, 0, stream>>>(l);
    const auto status = hipGetLastError();
    return status == hipSuccess ? a8k::launch_core<true>(l, stream) : status;
}
hipError_t launch_tp_moe_a8r_finish(const TpMoeLaunch& l, hipStream_t stream) {
    if (!l.partials || l.rows < 1 || l.rows > 8) return hipErrorInvalidValue;
    a8k::finish<<<dim3(40, l.rows), 192, 0, stream>>>(l);
    return hipGetLastError();
}
hipError_t launch_tp_moe_a8r_full(const TpMoeLaunch& l, hipStream_t stream) {
    if (!l.partials || l.rows < 1 || l.rows > 8) return hipErrorInvalidValue;
    auto status = a8k::launch_gate<true>(l, stream);
    if (status == hipSuccess) status = a8k::launch_core<true>(l, stream);
    if (status == hipSuccess) status = launch_tp_moe_a8r_finish(l, stream);
    return status;
}
} // namespace strix_trellis
#endif

#if defined(STRIX_TRELLIS_TP_MOE_PIPELINED_K2) && defined(STRIX_TRELLIS_TP_MOE_FUSE_DOWN_PREPARE)
namespace strix_trellis {
hipError_t launch_tp_moe_split_full(const TpMoeLaunch& l, hipStream_t stream) {
    if (!l.partials || l.rows < 1 || l.rows > 8) return hipErrorInvalidValue;
#ifdef STRIX_TRELLIS_TP_MOE_PERSISTENT
    if (!l.persistent_grid) return hipErrorInvalidValue;
    auto arguments = l;
    void* argv[]{&arguments};
    return hipLaunchCooperativeKernel(tp_moe_split::persistent_function(l.rows),
        dim3(l.persistent_grid), dim3(128), argv, 0, stream);
#else
#ifdef STRIX_TRELLIS_TP_MOE_FOLD_R1
    if (l.rows == 1 && l.fold_counters) return tp_moe_split::fold_launch(l, stream);
#endif
    auto status = tp_moe_split::pipeline_launch<false>(l, stream, true, true);
    if (status == hipSuccess)
        status = tp_moe_split::pipeline_launch<true>(l, stream, false);
    if (status == hipSuccess) status = launch_tp_moe_split_finish(l, stream);
    return status;
#endif
}
} // namespace strix_trellis
#endif

#ifdef STRIX_TRELLIS_TP_MOE_PERSISTENT
namespace strix_trellis {
hipError_t query_tp_moe_persistent_grid(unsigned rows, unsigned compute_units, unsigned* result) {
    if (!result || rows < 1 || rows > 8 || !compute_units) return hipErrorInvalidValue;
    *result = 0;
    int device = 0, cooperative = 0, blocks = 0;
    auto status = hipGetDevice(&device);
    if (status != hipSuccess) return status;
    status = hipDeviceGetAttribute(&cooperative, hipDeviceAttributeCooperativeLaunch, device);
    if (status != hipSuccess || !cooperative) return status;
    hipFuncAttributes attributes{};
    status = hipFuncGetAttributes(&attributes, tp_moe_split::persistent_function(rows));
    if (status != hipSuccess) return status;
    // Private-frame-backed cooperative residency is not qualified. The first
    // experimental image with local routing arrays stalled in native graph QA.
    if (attributes.localSizeBytes != 0) return hipSuccess;
    status = hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,
        tp_moe_split::persistent_function(rows), 128, 0);
    if (status == hipSuccess && blocks > 0) {
        // Start with one resident CTA per reported compute unit. The occupancy
        // ceiling alone is not a qualified cooperative scheduling policy.
        *result = compute_units;
#if STRIX_TRELLIS_TP_MOE_PERSISTENT_GRID > 0
        const unsigned limit = compute_units * unsigned(blocks);
        const unsigned requested = STRIX_TRELLIS_TP_MOE_PERSISTENT_GRID;
        *result = requested < limit ? requested : limit;
#endif
    }
    return status;
}
} // namespace strix_trellis
#endif
