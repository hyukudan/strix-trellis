// Included by api.hip to share the validated context/weight implementation.
// The experimental entry points never change the established linear dispatch.
extern "C" void strix_tp_moe_desc_init(strix_tp_moe_desc* d) {
    if (!d) return;
    *d = {};
    d->struct_size = sizeof(*d);
    d->abi_version = STRIX_TRELLIS_ABI_VERSION;
    d->alpha = 1.0f;
    d->input.struct_size = d->output.struct_size = sizeof(strix_tensor_desc);
    d->input.abi_version = d->output.abi_version = STRIX_TRELLIS_ABI_VERSION;
    d->input.dtype = d->output.dtype = STRIX_DTYPE_F32;
    d->input.rank = d->output.rank = 2;
}
extern "C" void strix_tp_moe_device_desc_init(strix_tp_moe_device_desc* d) {
    if (!d) return;
    *d = {};
    d->struct_size = sizeof(*d);
    d->abi_version = STRIX_TRELLIS_ABI_VERSION;
    d->alpha = 1.0f;
    d->input.struct_size = d->output.struct_size = sizeof(strix_tensor_desc);
    d->input.abi_version = d->output.abi_version = STRIX_TRELLIS_ABI_VERSION;
    d->input.dtype = d->output.dtype = STRIX_DTYPE_F32;
    d->input.rank = d->output.rank = 2;
}

#ifdef STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP
#include "tp_dense.hpp"
#include "tp_splitk.hpp"
#include "dense_r1.hpp"
#include "tp_moe.hpp"
#include "tp_moe_policy.hpp"
#include "tp_moe_split_layout.hpp"
#ifdef STRIX_TRELLIS_TP_MOE_FOLD_R1
// Arrival counters of the folded R1 submission: 9 per gate owner, 40 down blocks.
static constexpr size_t tp_moe_fold_counter_bytes = 512;
#else
static constexpr size_t tp_moe_fold_counter_bytes = 0;
#endif

struct strix_tp_dense_plan_impl {
    strix_context owner;
    strix_weights weights;
    strix_trellis::TpDenseArguments arguments;
    hipStream_t stream;
    strix_trellis::TpSplitLaunch split{};
    bool use_split = false;
    strix_trellis::DenseR1Launch r1{};
    bool use_r1 = false;
    bool use_owner = false;  // dense-r1 with the partials folded by the owning CTA (same bits)
    strix_trellis::DenseR1Pair pair{};  // dense-r1 q_a + kv in one dispatch (second member's weights retained)
    strix_weights second = nullptr;
    bool use_pair = false;
};
struct strix_tp_moe_plan_impl {
    strix_context owner = nullptr;
    strix_trellis::TpMoeLaunch arguments{};
    void* table = nullptr;
    strix_weights retained[STRIX_TP_MAX_BANK_EXPERTS * 3]{};
    unsigned retained_count = 0;
    hipStream_t stream = nullptr;
    bool a8r = false;
};

namespace tp_api {
using namespace strix_trellis;

strix_status tensor(const strix_tensor_desc& t, uint64_t width, uint32_t rank,
                    uint64_t rows, uint64_t& bytes, uint64_t& stride) {
    const auto s = header(&t);
    if (s != STRIX_SUCCESS) return s;
    if (enum_bits(t.dtype) != STRIX_DTYPE_F32 || t.rank != rank)
        return STRIX_ERROR_UNSUPPORTED_FORMAT;
    if (rank == 1) {
        if (rows != 1 || t.shape[0] != width || t.shape[1] ||
            t.strides[0] != 1 || t.strides[1]) return STRIX_ERROR_UNSUPPORTED_FORMAT;
        stride = width;
    } else if (rank == 2) {
        if (t.shape[0] != rows || t.shape[1] != width ||
            t.strides[0] < width || t.strides[1] != 1)
            return STRIX_ERROR_UNSUPPORTED_FORMAT;
        stride = t.strides[0];
    } else return STRIX_ERROR_UNSUPPORTED_FORMAT;
    if (!tp_dense_span(rows, width, stride, bytes) || !address(t.data, bytes))
        return STRIX_ERROR_INVALID_ARGUMENT;
    if (bytes > t.buffer_bytes) return STRIX_ERROR_INSUFFICIENT_BUFFER;
    if (!t.data || reinterpret_cast<uintptr_t>(t.data) % alignof(float))
        return STRIX_ERROR_INVALID_ARGUMENT;
    return STRIX_SUCCESS;
}

strix_status properties(strix_context c, hipDeviceProp_t& p) {
    const auto e = hipGetDeviceProperties(&p, c->device);
    if (e != hipSuccess) return hip_status(e);
    if (p.maxThreadsPerBlock < 512 || p.maxThreadsDim[0] < 512 ||
        p.warpSize != 32) return STRIX_ERROR_UNSUPPORTED_DEVICE;
    return STRIX_SUCCESS;
}

bool touches_weights(const void* data, uint64_t bytes, strix_weights w) {
    return overlap(data, bytes, w->allocation, w->allocation_bytes);
}

struct MoeValidated {
    TpMoeTable table{};
    TpMoeLaunch args{};
    strix_weights retained[STRIX_TP_MAX_EXPERTS * 3]{};
    unsigned retained_count = 0;
    uint64_t x_bytes = 0, y_bytes = 0, hidden_bytes = 0;
};

// Host-only shape, lifetime-owner and alias checks. No allocation/upload occurs
// until every expert, route and tensor in the window has passed validation.
strix_status moe_check(strix_context c, const strix_tp_moe_desc* d,
                       MoeValidated& v) {
    auto s = header(d);
    if (s != STRIX_SUCCESS) return s;
    if (!std::isfinite(d->alpha) || !std::isfinite(d->beta))
        return STRIX_ERROR_INVALID_ARGUMENT;
    if (d->input.rank != 2 || d->input.shape[0] < 1 ||
        d->input.shape[0] > STRIX_TP_MAX_ROWS) return STRIX_ERROR_UNSUPPORTED_FORMAT;
    const unsigned rows = static_cast<unsigned>(d->input.shape[0]);
    s = tensor(d->input, tp_moe_input, 2, rows, v.x_bytes, v.args.x_stride);
    if (s != STRIX_SUCCESS) return s;
    s = tensor(d->output, tp_moe_output, 2, rows, v.y_bytes, v.args.y_stride);
    if (s != STRIX_SUCCESS) return s;
    if (overlap(d->input.data, v.x_bytes, d->output.data, v.y_bytes))
        return STRIX_ERROR_INVALID_ARGUMENT;
    if (d->expert_count < STRIX_TP_TOP_K || d->expert_count > STRIX_TP_MAX_EXPERTS ||
        d->expert_count > rows * STRIX_TP_TOP_K || !d->experts ||
        !address(d->experts, d->expert_count * sizeof(strix_tp_expert_desc)) ||
        d->route_ids_count != rows * STRIX_TP_TOP_K ||
        d->route_scores_count != rows * STRIX_TP_TOP_K || !d->route_ids || !d->route_scores ||
        !address(d->route_ids, d->route_ids_count * sizeof(uint32_t)) ||
        !address(d->route_scores, d->route_scores_count * sizeof(float)))
        return STRIX_ERROR_INVALID_ARGUMENT;
    uint32_t expert_ids[STRIX_TP_MAX_EXPERTS]{};
    const unsigned count = static_cast<unsigned>(d->expert_count);
    for (unsigned i = 0; i < count; ++i) {
        const auto& e = d->experts[i];
        s = header(&e);
        if (s != STRIX_SUCCESS) return s;
        if (e.flags) return STRIX_ERROR_INVALID_ARGUMENT;
        expert_ids[i] = e.expert_id;
        const strix_weights weights[] = {e.gate, e.up, e.down};
        for (unsigned k = 0; k < 3; ++k) {
            const auto w = weights[k];
            if (!w || w->owner != c) return STRIX_ERROR_INVALID_ARGUMENT;
            const auto& native = w->device;
            if (!tp_moe_weight_shape(native.input, native.output, native.bits,
                                     native.codebook, k == 2))
                return STRIX_ERROR_UNSUPPORTED_FORMAT;
            if (touches_weights(d->input.data, v.x_bytes, w) ||
                touches_weights(d->output.data, v.y_bytes, w))
                return STRIX_ERROR_INVALID_ARGUMENT;
            unsigned retained = 0;
            while (retained < v.retained_count && v.retained[retained] != w) ++retained;
            if (retained == v.retained_count) {
                if (w->plan_refs == SIZE_MAX) return STRIX_ERROR_OUT_OF_MEMORY;
                v.retained[v.retained_count++] = w;
            }
        }
        v.table.experts[i].gate = e.gate->device;
        v.table.experts[i].up = e.up->device;
        v.table.experts[i].down = e.down->device;
    }
    TpMoeRouting routing{};
    if (!tp_moe_build_routing(expert_ids, count, d->route_ids, d->route_scores, rows, routing))
        return STRIX_ERROR_INVALID_ARGUMENT;
    for (unsigned i = 0; i < count; ++i) {
        auto& target = v.table.experts[i];
        target.row_count = routing.row_count[i];
        std::memcpy(target.token_rows, routing.token_rows[i], sizeof(target.token_rows));
        std::memcpy(target.route_slots, routing.route_slots[i], sizeof(target.route_slots));
        if (target.row_count)
            v.args.gate_row_classes |= target.row_count <= 2 ? 1u : 2u;
        unsigned active = 0;
        for (unsigned j = 0; j < target.row_count; ++j)
            active += d->route_scores[target.token_rows[j] * 6 + target.route_slots[j]] != 0.0f;
        if (active) v.args.down_row_classes |= active <= 2 ? 1u : 2u;
    }
    std::memcpy(v.table.expert_indices, routing.expert_indices, sizeof(v.table.expert_indices));
    std::memcpy(v.table.route_scores, routing.route_scores, sizeof(v.table.route_scores));
    v.hidden_bytes = tp_moe_hidden_bytes(rows);
    v.args.x = static_cast<const float*>(d->input.data);
    v.args.y = static_cast<float*>(d->output.data);
    v.args.hidden = static_cast<float*>(d->workspace);
    v.args.rows = rows;
    v.args.expert_count = count;
    v.args.alpha = d->alpha;
    v.args.beta = d->beta;
    return STRIX_SUCCESS;
}
} // namespace tp_api

extern "C" {
strix_status strix_tp_dense_plan_create(strix_context c, const strix_matmul_desc* d,
    const strix_stream* stream, strix_tp_dense_plan* out) {
    return strix_tp_dense_plan_create_tiled(c, d, stream, 0, out);
}
strix_status strix_tp_dense_plan_create_tiled(strix_context c, const strix_matmul_desc* d,
    const strix_stream* stream, uint32_t rows_per_block, strix_tp_dense_plan* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = nullptr;
    auto s = context_check(c);
    if (s != STRIX_SUCCESS) return s;
    hipStream_t hs{};
    s = stream_check(c, stream, hs);
    if (s != STRIX_SUCCESS) return s;
    s = header(d);
    if (s != STRIX_SUCCESS) return s;
    if (!d->weights || d->weights->owner != c || d->workspace || d->workspace_bytes ||
        !std::isfinite(d->alpha) || !std::isfinite(d->beta))
        return STRIX_ERROR_INVALID_ARGUMENT;
    const auto& w = d->weights->device;
    const uint32_t rank = d->input.rank;
    if (rank != 1 && rank != 2) return STRIX_ERROR_UNSUPPORTED_FORMAT;
    const uint64_t rows = rank == 1 ? 1 : d->input.shape[0];
    if (rows < 1 || rows > STRIX_TP_MAX_ROWS) return STRIX_ERROR_UNSUPPORTED_FORMAT;
    if (rows_per_block > rows) return STRIX_ERROR_INVALID_ARGUMENT;
    const int tile = rows_per_block ? static_cast<int>(rows_per_block) : static_cast<int>(rows);
    uint64_t xb = 0, yb = 0, xs = 0, ys = 0;
    s = tp_api::tensor(d->input, w.input, rank, rows, xb, xs);
    if (s != STRIX_SUCCESS) return s;
    s = tp_api::tensor(d->output, w.output, rank, rows, yb, ys);
    if (s != STRIX_SUCCESS) return s;
    strix_trellis::TpDenseShape shape{};
    if (!strix_trellis::tp_dense_shape(w.input, w.output, w.bits, w.codebook,
                                      static_cast<int>(rows), xs, ys, shape, tile))
        return STRIX_ERROR_UNSUPPORTED_FORMAT;
    if (overlap(d->input.data, xb, d->output.data, yb) ||
        tp_api::touches_weights(d->input.data, xb, d->weights) ||
        tp_api::touches_weights(d->output.data, yb, d->weights))
        return STRIX_ERROR_INVALID_ARGUMENT;
    s = device_buffer(c, d->input.data, xb, alignof(float));
    if (s != STRIX_SUCCESS) return s;
    s = device_buffer(c, d->output.data, yb, alignof(float));
    if (s != STRIX_SUCCESS) return s;
    hipDeviceProp_t props{};
    s = tp_api::properties(c, props);
    if (s != STRIX_SUCCESS) return s;
    bool supported = false;
    const auto e = strix_trellis::query_tp_dense_resources(w.bits, tile,
                                                         props.sharedMemPerBlock, &supported);
    if (e != hipSuccess) return hip_status(e);
    if (!supported) return STRIX_ERROR_UNSUPPORTED_DEVICE;
    if (d->weights->plan_refs == SIZE_MAX) return STRIX_ERROR_OUT_OF_MEMORY;
    auto p = new (std::nothrow) strix_tp_dense_plan_impl{c, d->weights,
        {w, static_cast<const float*>(d->input.data), static_cast<float*>(d->output.data),
         xs, ys, static_cast<int>(rows), d->alpha, d->beta, tile}, hs};
    if (!p) return STRIX_ERROR_OUT_OF_MEMORY;
    ++d->weights->plan_refs;
    *out = p;
    return STRIX_SUCCESS;
}
strix_status strix_tp_dense_plan_submit(strix_tp_dense_plan p) {
    if (!p) return STRIX_ERROR_INVALID_ARGUMENT;
    const auto s = context_check(p->owner);
    if (s != STRIX_SUCCESS) return s;
    if (p->use_pair) return hip_status(strix_trellis::launch_dense_r1_pair_trusted(p->pair, p->stream));
    if (p->use_owner) return hip_status(strix_trellis::launch_dense_r1_owner_trusted(p->r1, p->stream));
    if (p->use_r1) return hip_status(strix_trellis::launch_dense_r1_trusted(p->r1, p->stream));
    return hip_status(p->use_split ? strix_trellis::launch_tp_split_trusted(p->split, p->stream)
        : strix_trellis::launch_tp_dense_trusted(p->arguments, p->stream));
}
strix_status strix_tp_dense_plan_destroy(strix_tp_dense_plan p) {
    if (!p) return STRIX_ERROR_INVALID_ARGUMENT;
    --p->weights->plan_refs;
    if (p->second) --p->second->plan_refs;
    delete p;
    return STRIX_SUCCESS;
}
strix_status strix_tp_moe_workspace_bytes(strix_context c, const strix_tp_moe_desc* d,
    uint64_t* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = 0;
    auto s = context_check(c);
    if (s != STRIX_SUCCESS) return s;
    tp_api::MoeValidated v{};
    s = tp_api::moe_check(c, d, v);
    if (s == STRIX_SUCCESS) *out = v.hidden_bytes;
    return s;
}
static strix_status tp_moe_create(strix_context c, const strix_tp_moe_desc* d,
    const strix_stream* stream, strix_tp_moe_plan* out, bool split, bool a8r = false) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = nullptr;
    auto s = context_check(c);
    if (s != STRIX_SUCCESS) return s;
    hipStream_t hs{};
    s = stream_check(c, stream, hs);
    if (s != STRIX_SUCCESS) return s;
    tp_api::MoeValidated v{};
    s = tp_api::moe_check(c, d, v);
    if (s != STRIX_SUCCESS) return s;
    if (a8r) for (unsigned i = 0; i < v.retained_count; ++i)
        if (v.retained[i]->device.bits != 2 || v.retained[i]->device.codebook != 2)
            return STRIX_ERROR_UNSUPPORTED_FORMAT;
    const uint64_t required = split ? strix_trellis::tp_moe_split_workspace_bytes(v.args.rows) : v.hidden_bytes;
    if (d->workspace_bytes < required) return STRIX_ERROR_INSUFFICIENT_BUFFER;
    if (!d->workspace || !address(d->workspace, d->workspace_bytes) ||
        reinterpret_cast<uintptr_t>(d->workspace) % alignment ||
        overlap(d->workspace, d->workspace_bytes, d->input.data, v.x_bytes) ||
        overlap(d->workspace, d->workspace_bytes, d->output.data, v.y_bytes))
        return STRIX_ERROR_INVALID_ARGUMENT;
    for (unsigned i = 0; i < v.retained_count; ++i)
        if (tp_api::touches_weights(d->workspace, d->workspace_bytes, v.retained[i]))
            return STRIX_ERROR_INVALID_ARGUMENT;
    s = device_buffer(c, d->input.data, v.x_bytes, alignof(float));
    if (s != STRIX_SUCCESS) return s;
    s = device_buffer(c, d->output.data, v.y_bytes, alignof(float));
    if (s != STRIX_SUCCESS) return s;
    s = device_buffer(c, d->workspace, d->workspace_bytes, alignment);
    if (s != STRIX_SUCCESS) return s;
    hipDeviceProp_t props{};
    s = tp_api::properties(c, props);
    if (s != STRIX_SUCCESS) return s;
    bool supported = false;
    hipError_t e;
#ifdef STRIX_TRELLIS_TP_MOE_A8R
    if (a8r) e = strix_trellis::query_tp_moe_a8r_resources(v.args.rows, props.sharedMemPerBlock, &supported);
    else
#endif
    e = split
        ? strix_trellis::query_tp_moe_split_resources(v.args.rows, props.sharedMemPerBlock, &supported)
        : strix_trellis::query_tp_moe_resources(v.args.rows, props.sharedMemPerBlock, &supported);
    if (e != hipSuccess) return hip_status(e);
    if (!supported) return STRIX_ERROR_UNSUPPORTED_DEVICE;
#ifdef STRIX_TRELLIS_TP_MOE_PERSISTENT
    unsigned persistent_grid = 0;
    if (split) {
        e = strix_trellis::query_tp_moe_persistent_grid(v.args.rows, props.multiProcessorCount, &persistent_grid);
        if (e != hipSuccess) return hip_status(e);
        if (!persistent_grid) return STRIX_ERROR_UNSUPPORTED_DEVICE;
    }
#endif
    auto p = new (std::nothrow) strix_tp_moe_plan_impl;
    if (!p) return STRIX_ERROR_OUT_OF_MEMORY;
    void* table = nullptr;
    // Fold counters follow the table at a 256-byte boundary; no padding without them.
    const size_t table_bytes = tp_moe_fold_counter_bytes ? (sizeof(v.table) + 255) / 256 * 256 : sizeof(v.table);
    e = hipMalloc(&table, table_bytes + tp_moe_fold_counter_bytes);
    if (e != hipSuccess) { delete p; return hip_status(e); }
    e = hipMemcpyAsync(table, &v.table, sizeof(v.table), hipMemcpyHostToDevice, hs);
#ifdef STRIX_TRELLIS_TP_MOE_FOLD_R1
    if (e == hipSuccess && tp_moe_fold_counter_bytes)
        e = hipMemsetAsync(static_cast<unsigned char*>(table) + table_bytes, 0, tp_moe_fold_counter_bytes, hs);
#endif
    // Also drain a failed enqueue before releasing the borrowed stack table.
    const auto completion = hipStreamSynchronize(hs);
    if (e != hipSuccess || completion != hipSuccess) {
        (void)hipFree(table);
        delete p;
        return hip_status(e != hipSuccess ? e : completion);
    }
    p->owner = c;
    p->stream = hs;
    p->table = table;
    p->arguments = v.args;
    p->a8r = a8r;
#ifdef STRIX_TRELLIS_TP_MOE_PERSISTENT
    p->arguments.persistent_grid = persistent_grid;
#endif
    if (split) p->arguments.partials = reinterpret_cast<float*>(
        static_cast<unsigned char*>(d->workspace) + v.hidden_bytes);
    p->arguments.table = static_cast<const strix_trellis::TpMoeTable*>(table);
    if (split && tp_moe_fold_counter_bytes) p->arguments.fold_counters =
        reinterpret_cast<unsigned*>(static_cast<unsigned char*>(table) + table_bytes);
    p->retained_count = v.retained_count;
    for (unsigned i = 0; i < v.retained_count; ++i) {
        p->retained[i] = v.retained[i];
        ++p->retained[i]->plan_refs;
    }
    *out = p;
    return STRIX_SUCCESS;
}
static strix_status tp_moe_create_device(strix_context c, const strix_tp_moe_device_desc* d,
    const strix_stream* stream, strix_tp_moe_plan* out, bool split, bool a8r = false) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = nullptr;
    auto s = context_check(c);
    if (s != STRIX_SUCCESS) return s;
    hipStream_t hs{};
    s = stream_check(c, stream, hs);
    if (s != STRIX_SUCCESS) return s;
    s = header(d);
    if (s != STRIX_SUCCESS) return s;
    if (!std::isfinite(d->alpha) || !std::isfinite(d->beta))
        return STRIX_ERROR_INVALID_ARGUMENT;
    if (d->input.rank != 2 || d->input.shape[0] < 1 || d->input.shape[0] > STRIX_TP_MAX_ROWS)
        return STRIX_ERROR_UNSUPPORTED_FORMAT;
    const auto rows = static_cast<unsigned>(d->input.shape[0]);
    strix_trellis::TpMoeLaunch args{};
    uint64_t xb = 0, yb = 0;
    s = tp_api::tensor(d->input, strix_trellis::tp_moe_input, 2, rows, xb, args.x_stride);
    if (s != STRIX_SUCCESS) return s;
    s = tp_api::tensor(d->output, strix_trellis::tp_moe_output, 2, rows, yb, args.y_stride);
    if (s != STRIX_SUCCESS) return s;
    const uint64_t route_bytes = uint64_t(rows) * STRIX_TP_TOP_K * sizeof(uint32_t);
    const uint64_t hidden_bytes = strix_trellis::tp_moe_hidden_bytes(rows);
    if (d->expert_count < STRIX_TP_TOP_K || d->expert_count > STRIX_TP_MAX_BANK_EXPERTS ||
        !d->experts || !address(d->experts, d->expert_count * sizeof(strix_tp_expert_desc)))
        return STRIX_ERROR_INVALID_ARGUMENT;
    if (d->route_ids_bytes != route_bytes || d->route_scores_bytes != route_bytes)
        return STRIX_ERROR_INVALID_ARGUMENT;
    const uint64_t required = split ? strix_trellis::tp_moe_split_workspace_bytes(rows) : hidden_bytes;
    if (d->workspace_bytes < required) return STRIX_ERROR_INSUFFICIENT_BUFFER;
    const void* buffers[] = {d->input.data, d->output.data, d->workspace,
                             d->route_ids, d->route_scores};
    const uint64_t bytes[] = {xb, yb, d->workspace_bytes, route_bytes, route_bytes};
    for (unsigned i = 0; i < 5; ++i) {
        if (!buffers[i] || !address(buffers[i], bytes[i]) ||
            reinterpret_cast<uintptr_t>(buffers[i]) % (i == 2 ? alignment : alignof(float)))
            return STRIX_ERROR_INVALID_ARGUMENT;
        for (unsigned j = 0; j < i; ++j)
            if (overlap(buffers[i], bytes[i], buffers[j], bytes[j]))
                return STRIX_ERROR_INVALID_ARGUMENT;
    }
    // Retain each distinct handle once even when several experts share it.
    strix_weights retained[STRIX_TP_MAX_BANK_EXPERTS * 3]{};
    unsigned retained_count = 0;
    const unsigned count = static_cast<unsigned>(d->expert_count);
    for (unsigned i = 0; i < count; ++i) {
        const auto& expert = d->experts[i];
        s = header(&expert);
        if (s != STRIX_SUCCESS) return s;
        if (expert.flags || expert.expert_id != i) return STRIX_ERROR_INVALID_ARGUMENT;
        const strix_weights weights[] = {expert.gate, expert.up, expert.down};
        for (unsigned k = 0; k < 3; ++k) {
            const auto w = weights[k];
            if (!w || w->owner != c) return STRIX_ERROR_INVALID_ARGUMENT;
            const auto& native = w->device;
            if (a8r && (native.bits != 2 || native.codebook != 2))
                return STRIX_ERROR_UNSUPPORTED_FORMAT;
            if (!strix_trellis::tp_moe_weight_shape(native.input, native.output,
                                                   native.bits, native.codebook, k == 2))
                return STRIX_ERROR_UNSUPPORTED_FORMAT;
            for (unsigned b = 0; b < 5; ++b)
                if (tp_api::touches_weights(buffers[b], bytes[b], w))
                    return STRIX_ERROR_INVALID_ARGUMENT;
            unsigned n = 0;
            while (n < retained_count && retained[n] != w) ++n;
            if (n == retained_count) {
                if (w->plan_refs == SIZE_MAX) return STRIX_ERROR_OUT_OF_MEMORY;
                retained[retained_count++] = w;
            }
        }
    }
    for (unsigned i = 0; i < 5; ++i) {
        s = device_buffer(c, buffers[i], bytes[i], i == 2 ? alignment : alignof(float));
        if (s != STRIX_SUCCESS) return s;
    }
    hipDeviceProp_t props{};
    s = tp_api::properties(c, props);
    if (s != STRIX_SUCCESS) return s;
    bool supported = false;
    hipError_t e;
#ifdef STRIX_TRELLIS_TP_MOE_A8R
    if (a8r) e = strix_trellis::query_tp_moe_a8r_resources(rows, props.sharedMemPerBlock, &supported);
    else
#endif
    e = split
        ? strix_trellis::query_tp_moe_split_resources(rows, props.sharedMemPerBlock, &supported)
        : strix_trellis::query_tp_moe_resources(rows, props.sharedMemPerBlock, &supported);
    if (e != hipSuccess) return hip_status(e);
    if (!supported) return STRIX_ERROR_UNSUPPORTED_DEVICE;
#ifdef STRIX_TRELLIS_TP_MOE_PERSISTENT
    unsigned persistent_grid = 0;
    if (split) {
        e = strix_trellis::query_tp_moe_persistent_grid(rows, props.multiProcessorCount, &persistent_grid);
        if (e != hipSuccess) return hip_status(e);
        if (!persistent_grid) return STRIX_ERROR_UNSUPPORTED_DEVICE;
    }
#endif
    auto p = new (std::nothrow) strix_tp_moe_plan_impl;
    if (!p) return STRIX_ERROR_OUT_OF_MEMORY;
    auto bank = new (std::nothrow) strix_trellis::TpMoeExpert[count]{};
    if (!bank) { delete p; return STRIX_ERROR_OUT_OF_MEMORY; }
    for (unsigned i = 0; i < count; ++i) {
        bank[i].gate = d->experts[i].gate->device;
        bank[i].up = d->experts[i].up->device;
        bank[i].down = d->experts[i].down->device;
    }
    void* table = nullptr;
    const size_t bank_bytes = count * sizeof(*bank);
    const size_t padded_bank_bytes = tp_moe_fold_counter_bytes ? (bank_bytes + 255) / 256 * 256 : bank_bytes;
    e = hipMalloc(&table, padded_bank_bytes + tp_moe_fold_counter_bytes);
    if (e != hipSuccess) { delete[] bank; delete p; return hip_status(e); }
    e = hipMemcpyAsync(table, bank, bank_bytes, hipMemcpyHostToDevice, hs);
#ifdef STRIX_TRELLIS_TP_MOE_FOLD_R1
    if (e == hipSuccess && tp_moe_fold_counter_bytes)
        e = hipMemsetAsync(static_cast<unsigned char*>(table) + padded_bank_bytes, 0, tp_moe_fold_counter_bytes, hs);
#endif
    const auto completion = hipStreamSynchronize(hs);
    delete[] bank;
    if (e != hipSuccess || completion != hipSuccess) {
        (void)hipFree(table);
        delete p;
        return hip_status(e != hipSuccess ? e : completion);
    }
    args.bank = static_cast<const strix_trellis::TpMoeExpert*>(table);
    args.bank_count = count;
    args.device_route_ids = d->route_ids;
    args.device_route_scores = d->route_scores;
    args.x = static_cast<const float*>(d->input.data);
    args.y = static_cast<float*>(d->output.data);
    args.hidden = static_cast<float*>(d->workspace);
    if (split) args.partials = reinterpret_cast<float*>(
        static_cast<unsigned char*>(d->workspace) + hidden_bytes);
    if (split && tp_moe_fold_counter_bytes) args.fold_counters =
        reinterpret_cast<unsigned*>(static_cast<unsigned char*>(table) + padded_bank_bytes);
    args.rows = rows;
    args.expert_count = rows * STRIX_TP_TOP_K;
    args.alpha = d->alpha;
    args.beta = d->beta;
    p->owner = c;
    p->stream = hs;
    p->table = table;
    p->arguments = args;
    p->a8r = a8r;
#ifdef STRIX_TRELLIS_TP_MOE_PERSISTENT
    p->arguments.persistent_grid = persistent_grid;
#endif
    p->retained_count = retained_count;
    for (unsigned i = 0; i < retained_count; ++i) {
        p->retained[i] = retained[i];
        ++retained[i]->plan_refs;
    }
    *out = p;
    return STRIX_SUCCESS;
}
strix_status strix_tp_moe_plan_create(strix_context c, const strix_tp_moe_desc* d,
    const strix_stream* stream, strix_tp_moe_plan* out) {
    return tp_moe_create(c, d, stream, out, false);
}
strix_status strix_tp_moe_plan_create_split(strix_context c, const strix_tp_moe_desc* d,
    const strix_stream* stream, strix_tp_moe_plan* out) {
    return tp_moe_create(c, d, stream, out, true);
}
strix_status strix_tp_moe_plan_create_device(strix_context c, const strix_tp_moe_device_desc* d,
    const strix_stream* stream, strix_tp_moe_plan* out) {
    return tp_moe_create_device(c, d, stream, out, false);
}
strix_status strix_tp_moe_plan_create_device_split(strix_context c, const strix_tp_moe_device_desc* d,
    const strix_stream* stream, strix_tp_moe_plan* out) {
    return tp_moe_create_device(c, d, stream, out, true);
}
strix_status strix_tp_moe_plan_create_a8r(strix_context c, const strix_tp_moe_desc* d,
    const strix_stream* stream, strix_tp_moe_plan* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = nullptr;
#ifdef STRIX_TRELLIS_TP_MOE_A8R
    return tp_moe_create(c, d, stream, out, true, true);
#else
    (void)c; (void)d; (void)stream;return STRIX_ERROR_NOT_IMPLEMENTED;
#endif
}
strix_status strix_tp_moe_plan_create_device_a8r(strix_context c, const strix_tp_moe_device_desc* d,
    const strix_stream* stream, strix_tp_moe_plan* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = nullptr;
#ifdef STRIX_TRELLIS_TP_MOE_A8R
    return tp_moe_create_device(c, d, stream, out, true, true);
#else
    (void)c; (void)d; (void)stream;return STRIX_ERROR_NOT_IMPLEMENTED;
#endif
}
strix_status strix_tp_moe_split_workspace_bytes(strix_context c, const strix_tp_moe_desc* d,
    uint64_t* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = 0;
    auto s = context_check(c);
    if (s != STRIX_SUCCESS) return s;
    tp_api::MoeValidated v{};
    s = tp_api::moe_check(c, d, v);
    if (s == STRIX_SUCCESS) *out = strix_trellis::tp_moe_split_workspace_bytes(v.args.rows);
    return s;
}
strix_status strix_tp_moe_plan_down_core(strix_tp_moe_plan p) {
    if (!p || !p->arguments.partials) return STRIX_ERROR_INVALID_ARGUMENT;
    const auto s = context_check(p->owner);
    if (s != STRIX_SUCCESS) return s;
#ifdef STRIX_TRELLIS_TP_MOE_A8R
    if (p->a8r) return hip_status(strix_trellis::launch_tp_moe_a8r_core(p->arguments, p->stream));
#endif
    return hip_status(strix_trellis::launch_tp_moe_split_core(p->arguments, p->stream));
}
strix_status strix_tp_moe_plan_down_finish(strix_tp_moe_plan p) {
    if (!p || !p->arguments.partials) return STRIX_ERROR_INVALID_ARGUMENT;
    const auto s = context_check(p->owner);
    if (s != STRIX_SUCCESS) return s;
#ifdef STRIX_TRELLIS_TP_MOE_A8R
    if (p->a8r) return hip_status(strix_trellis::launch_tp_moe_a8r_finish(p->arguments, p->stream));
#endif
    return hip_status(strix_trellis::launch_tp_moe_split_finish(p->arguments, p->stream));
}
strix_status strix_tp_moe_plan_down_reference_core(strix_tp_moe_plan p) {
    if (!p || !p->arguments.partials) return STRIX_ERROR_INVALID_ARGUMENT;
    const auto s = context_check(p->owner);
    if (s != STRIX_SUCCESS) return s;
    if (p->a8r) return STRIX_ERROR_UNSUPPORTED_FORMAT;
    return hip_status(strix_trellis::launch_tp_moe_reference_partials(p->arguments, p->stream));
}
strix_status strix_tp_moe_plan_gate_up(strix_tp_moe_plan p) {
    if (!p) return STRIX_ERROR_INVALID_ARGUMENT;
    const auto s = context_check(p->owner);
    if (s != STRIX_SUCCESS) return s;
#ifdef STRIX_TRELLIS_TP_MOE_A8R
    if (p->a8r) return hip_status(strix_trellis::launch_tp_moe_a8r_gate_up(p->arguments, p->stream));
#endif
    return hip_status(p->arguments.partials
        ? strix_trellis::launch_tp_moe_split_gate_up(p->arguments, p->stream)
        : strix_trellis::launch_tp_moe_gate_up(p->arguments, p->stream));
}
strix_status strix_tp_moe_plan_down(strix_tp_moe_plan p) {
    if (!p) return STRIX_ERROR_INVALID_ARGUMENT;
    const auto s = context_check(p->owner);
    if (s != STRIX_SUCCESS) return s;
#ifdef STRIX_TRELLIS_TP_MOE_A8R
    if (p->a8r) {
        auto error = strix_trellis::launch_tp_moe_a8r_core(p->arguments, p->stream);
        if (error == hipSuccess) error = strix_trellis::launch_tp_moe_a8r_finish(p->arguments, p->stream);
        return hip_status(error);
    }
#endif
    if (!p->arguments.partials)
        return hip_status(strix_trellis::launch_tp_moe_down(p->arguments, p->stream));
    auto e = strix_trellis::launch_tp_moe_split_core(p->arguments, p->stream);
    if (e == hipSuccess) e = strix_trellis::launch_tp_moe_split_finish(p->arguments, p->stream);
    return hip_status(e);
}
strix_status strix_tp_moe_plan_submit(strix_tp_moe_plan p) {
    if (!p) return STRIX_ERROR_INVALID_ARGUMENT;
    const auto s = context_check(p->owner);
    if (s != STRIX_SUCCESS) return s;
#ifdef STRIX_TRELLIS_TP_MOE_A8R
    if (p->a8r) return hip_status(strix_trellis::launch_tp_moe_a8r_full(p->arguments, p->stream));
#endif
#ifdef STRIX_TRELLIS_TP_MOE_FUSE_DOWN_PREPARE
    if (p->arguments.partials)
        return hip_status(strix_trellis::launch_tp_moe_split_full(p->arguments, p->stream));
#endif
    auto e = p->arguments.partials
        ? strix_trellis::launch_tp_moe_split_gate_up(p->arguments, p->stream)
        : strix_trellis::launch_tp_moe_gate_up(p->arguments, p->stream);
    if (p->arguments.partials) {
        if (e == hipSuccess) e = strix_trellis::launch_tp_moe_split_core(p->arguments, p->stream);
        if (e == hipSuccess) e = strix_trellis::launch_tp_moe_split_finish(p->arguments, p->stream);
    } else if (e == hipSuccess) e = strix_trellis::launch_tp_moe_down(p->arguments, p->stream);
    return hip_status(e);
}
strix_status strix_tp_moe_plan_destroy(strix_tp_moe_plan p) {
    if (!p) return STRIX_ERROR_INVALID_ARGUMENT;
    const auto s = context_check(p->owner);
    if (s != STRIX_SUCCESS) return s;
    const auto e = hipFree(p->table);
    if (e != hipSuccess) return hip_status(e);
    for (unsigned i = 0; i < p->retained_count; ++i) --p->retained[i]->plan_refs;
    delete p;
    return STRIX_SUCCESS;
}
} // extern "C"
#else
extern "C" {
strix_status strix_tp_dense_plan_create(strix_context, const strix_matmul_desc*,
    const strix_stream*, strix_tp_dense_plan* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = nullptr;
    return STRIX_ERROR_NOT_IMPLEMENTED;
}
strix_status strix_tp_dense_plan_create_tiled(strix_context, const strix_matmul_desc*,
    const strix_stream*, uint32_t, strix_tp_dense_plan* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = nullptr;
    return STRIX_ERROR_NOT_IMPLEMENTED;
}
strix_status strix_tp_dense_plan_submit(strix_tp_dense_plan) { return STRIX_ERROR_NOT_IMPLEMENTED; }
strix_status strix_tp_dense_plan_destroy(strix_tp_dense_plan) { return STRIX_ERROR_NOT_IMPLEMENTED; }
strix_status strix_tp_moe_workspace_bytes(strix_context, const strix_tp_moe_desc*, uint64_t* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = 0;
    return STRIX_ERROR_NOT_IMPLEMENTED;
}
strix_status strix_tp_moe_plan_create(strix_context, const strix_tp_moe_desc*,
    const strix_stream*, strix_tp_moe_plan* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = nullptr;
    return STRIX_ERROR_NOT_IMPLEMENTED;
}
strix_status strix_tp_moe_plan_create_device(strix_context, const strix_tp_moe_device_desc*,
    const strix_stream*, strix_tp_moe_plan* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = nullptr;
    return STRIX_ERROR_NOT_IMPLEMENTED;
}
strix_status strix_tp_moe_plan_submit(strix_tp_moe_plan) { return STRIX_ERROR_NOT_IMPLEMENTED; }
strix_status strix_tp_moe_plan_gate_up(strix_tp_moe_plan) { return STRIX_ERROR_NOT_IMPLEMENTED; }
strix_status strix_tp_moe_plan_down(strix_tp_moe_plan) { return STRIX_ERROR_NOT_IMPLEMENTED; }
strix_status strix_tp_moe_plan_destroy(strix_tp_moe_plan) { return STRIX_ERROR_NOT_IMPLEMENTED; }
strix_status strix_tp_moe_plan_create_split(strix_context c, const strix_tp_moe_desc* d,
    const strix_stream* stream, strix_tp_moe_plan* out) {
    return strix_tp_moe_plan_create(c, d, stream, out);
}
strix_status strix_tp_moe_plan_create_device_split(strix_context c, const strix_tp_moe_device_desc* d,
    const strix_stream* stream, strix_tp_moe_plan* out) {
    return strix_tp_moe_plan_create_device(c, d, stream, out);
}
strix_status strix_tp_moe_plan_create_a8r(strix_context c, const strix_tp_moe_desc* d,
    const strix_stream* stream, strix_tp_moe_plan* out) {
    return strix_tp_moe_plan_create(c, d, stream, out);
}
strix_status strix_tp_moe_plan_create_device_a8r(strix_context c, const strix_tp_moe_device_desc* d,
    const strix_stream* stream, strix_tp_moe_plan* out) {
    return strix_tp_moe_plan_create_device(c, d, stream, out);
}
strix_status strix_tp_moe_split_workspace_bytes(strix_context c, const strix_tp_moe_desc* d,
    uint64_t* out) { return strix_tp_moe_workspace_bytes(c, d, out); }
strix_status strix_tp_moe_plan_down_core(strix_tp_moe_plan) { return STRIX_ERROR_NOT_IMPLEMENTED; }
strix_status strix_tp_moe_plan_down_finish(strix_tp_moe_plan) { return STRIX_ERROR_NOT_IMPLEMENTED; }
strix_status strix_tp_moe_plan_down_reference_core(strix_tp_moe_plan) { return STRIX_ERROR_NOT_IMPLEMENTED; }
} // extern "C"
#endif
