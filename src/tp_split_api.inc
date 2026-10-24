// Additive dense split-K API; included after the existing TP validation helpers.
extern "C" void strix_tp_dense_pair_desc_init(strix_tp_dense_pair_desc* d) {
    if (!d) return;
    *d = {}; d->struct_size = sizeof(*d); d->abi_version = STRIX_TRELLIS_ABI_VERSION;
    strix_matmul_desc m; strix_matmul_desc_init(&m);
    d->input = m.input;
    for (auto& member : d->member) { member.output = m.output; member.alpha = 1; }
}
extern "C" void strix_tp_swiglu_desc_init(strix_tp_swiglu_desc* d) {
    if (!d) return;
    *d = {}; d->struct_size = sizeof(*d); d->abi_version = STRIX_TRELLIS_ABI_VERSION;
    strix_matmul_desc m; strix_matmul_desc_init(&m);
    d->input = m.input; d->hidden_output = m.output;
}
#ifdef STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP
struct strix_tp_dense_pair_plan_impl {
    strix_context owner;
    strix_weights retained[2];
    strix_trellis::TpSplitLaunch launch;
    hipStream_t stream;
};
namespace tp_split_api {
using namespace strix_trellis;
struct Validated {
    TpSplitLaunch launch{};
    strix_weights weights[2]{};
    uint64_t xb = 0, yb[2]{};
    size_t partial_bytes[2]{}, counter_offset = 0, counter_bytes = 0, bytes = 0;
    const void* x = nullptr;
    void* y[2]{};
};
strix_status member(strix_context c, const strix_matmul_desc& d, unsigned tile,
                    Validated& v, int m) {
    auto s = header(&d);
    if (s != STRIX_SUCCESS) return s;
    if (!d.weights || d.weights->owner != c || !std::isfinite(d.alpha) || !std::isfinite(d.beta))
        return STRIX_ERROR_INVALID_ARGUMENT;
    const auto& w = d.weights->device;
    const unsigned rank = d.input.rank;
    const uint64_t rows = rank == 1 ? 1 : d.input.shape[0];
    if ((rank != 1 && rank != 2) || rows < 1 || rows > 8) return STRIX_ERROR_UNSUPPORTED_FORMAT;
    if (tile > rows) return STRIX_ERROR_INVALID_ARGUMENT;
    uint64_t xs = 0, ys = 0;
    s = tp_api::tensor(d.input, w.input, rank, rows, v.xb, xs);
    if (s != STRIX_SUCCESS) return s;
    s = tp_api::tensor(d.output, w.output, rank, rows, v.yb[m], ys);
    if (s != STRIX_SUCCESS) return s;
    TpDenseShape shape{};
    if (!tp_dense_shape(w.input, w.output, w.bits, w.codebook, int(rows), xs, ys, shape, int(tile)))
        return STRIX_ERROR_UNSUPPORTED_FORMAT;
    v.weights[m] = d.weights; v.x = d.input.data; v.y[m] = d.output.data;
    v.launch.member[m] = {w, static_cast<const float*>(d.input.data), static_cast<float*>(d.output.data),
        xs, ys, int(rows), d.alpha, d.beta, int(shape.rows_per_block)};
    TpSplitLayout layout{};
    if (tp_split_layout(w.input, w.output, int(rows), int(tile), layout)) {
        v.partial_bytes[m] = layout.partial_bytes;
        v.counter_bytes += layout.counter_bytes;
    }
    return STRIX_SUCCESS;
}
strix_status aliases(Validated& v) {
    for (int m = 0; m < v.launch.members; ++m) {
        if (overlap(v.x, v.xb, v.y[m], v.yb[m])) return STRIX_ERROR_INVALID_ARGUMENT;
        for (int w = 0; w < v.launch.members; ++w)
            if (tp_api::touches_weights(v.x, v.xb, v.weights[w]) ||
                tp_api::touches_weights(v.y[m], v.yb[m], v.weights[w])) return STRIX_ERROR_INVALID_ARGUMENT;
        if (v.weights[m]->plan_refs == SIZE_MAX) return STRIX_ERROR_OUT_OF_MEMORY;
    }
    if (v.launch.members == 2 && !v.launch.swiglu && overlap(v.y[0], v.yb[0], v.y[1], v.yb[1]))
        return STRIX_ERROR_INVALID_ARGUMENT;
    v.counter_offset = v.partial_bytes[0] + v.partial_bytes[1];
    // Compact counter array shared by both members, padded only once.
    const int tile = v.launch.member[0].rows_per_block;
    const int row_blocks = (v.launch.member[0].rows + tile - 1) / tile;
    int nb = v.launch.member[0].weights.output / 128;
    if (v.launch.members == 2 && !v.launch.swiglu) nb += v.launch.member[1].weights.output / 128;
    v.counter_bytes = v.counter_offset ? (size_t(row_blocks * nb) * 4 + 255) & ~size_t(255) : 0;
    v.bytes = v.counter_offset + v.counter_bytes;
    return STRIX_SUCCESS;
}
strix_status single(strix_context c, const strix_matmul_desc* d, unsigned tile, Validated& v) {
    if (!d) return STRIX_ERROR_INVALID_ARGUMENT;
    auto s = member(c, *d, tile, v, 0);
    return s == STRIX_SUCCESS ? aliases(v) : s;
}
strix_status pair(strix_context c, const strix_tp_dense_pair_desc* d, Validated& v, bool swiglu) {
    auto s = header(d);
    if (s != STRIX_SUCCESS) return s;
    if (d->strategy > STRIX_TP_SPLITK_TWO) return STRIX_ERROR_INVALID_ARGUMENT;
    v.launch.members = 2; v.launch.swiglu = swiglu; v.launch.two_kernel = d->strategy == STRIX_TP_SPLITK_TWO;
    for (int m = 0; m < 2; ++m) {
        strix_matmul_desc one; strix_matmul_desc_init(&one);
        one.weights = d->member[m].weights; one.input = d->input; one.output = d->member[m].output;
        one.alpha = d->member[m].alpha; one.beta = d->member[m].beta;
        s = member(c, one, d->rows_per_block, v, m);
        if (s != STRIX_SUCCESS) return s;
        const auto& w = v.launch.member[m].weights;
        if (w.input != 5120 || w.output != (swiglu ? 1152 : m == 0 ? 1280 : 512))
            return STRIX_ERROR_UNSUPPORTED_FORMAT;
    }
    return aliases(v);
}
strix_tp_dense_pair_desc convert(const strix_tp_swiglu_desc& d) {
    strix_tp_dense_pair_desc p; strix_tp_dense_pair_desc_init(&p);
    p.input = d.input; p.member[0] = {d.gate, d.hidden_output, 1, 0};
    p.member[1] = {d.up, d.hidden_output, 1, 0};
    p.workspace = d.workspace; p.workspace_bytes = d.workspace_bytes;
    p.rows_per_block = d.rows_per_block; p.strategy = d.strategy;
    return p;
}
strix_status diagnostics(strix_context c, const strix_tp_swiglu_desc& d, Validated& v, bool device) {
    float* ptr[] = {d.diagnostic_gate, d.diagnostic_up};
    const uint64_t needed = uint64_t(v.launch.member[0].rows) * 1152 * 4;
    if (!ptr[0] && !ptr[1] && d.diagnostic_bytes) return STRIX_ERROR_INVALID_ARGUMENT;
    for (int i = 0; i < 2; ++i) if (ptr[i]) {
        if (d.diagnostic_bytes < needed) return STRIX_ERROR_INSUFFICIENT_BUFFER;
        if (!address(ptr[i], d.diagnostic_bytes) || (reinterpret_cast<uintptr_t>(ptr[i]) & 3u) ||
            overlap(ptr[i], d.diagnostic_bytes, v.x, v.xb) ||
            overlap(ptr[i], d.diagnostic_bytes, v.y[0], v.yb[0]) ||
            overlap(ptr[i], d.diagnostic_bytes, d.workspace, d.workspace_bytes) ||
            (i && overlap(ptr[0], ptr[0] ? d.diagnostic_bytes : 0, ptr[1], d.diagnostic_bytes)))
            return STRIX_ERROR_INVALID_ARGUMENT;
        for (auto w : v.weights) if (tp_api::touches_weights(ptr[i], d.diagnostic_bytes, w))
            return STRIX_ERROR_INVALID_ARGUMENT;
        if (device) { const auto s = device_buffer(c, ptr[i], d.diagnostic_bytes, 4); if (s != STRIX_SUCCESS) return s; }
        v.launch.diagnostic[i] = ptr[i];
    }
    return STRIX_SUCCESS;
}
strix_status storage(strix_context c, void* workspace, uint64_t bytes, Validated& v, hipStream_t stream) {
    if (bytes < v.bytes) return STRIX_ERROR_INSUFFICIENT_BUFFER;
    if (!address(workspace, bytes) || !workspace || reinterpret_cast<uintptr_t>(workspace) % alignment ||
        overlap(workspace, bytes, v.x, v.xb)) return STRIX_ERROR_INVALID_ARGUMENT;
    for (int m = 0; m < v.launch.members; ++m)
        if (overlap(workspace, bytes, v.y[m], v.yb[m]) || tp_api::touches_weights(workspace, bytes, v.weights[m]))
            return STRIX_ERROR_INVALID_ARGUMENT;
    auto s = device_buffer(c, workspace, bytes, alignment);
    if (s != STRIX_SUCCESS) return s;
    s = device_buffer(c, v.x, v.xb, 4);
    if (s != STRIX_SUCCESS) return s;
    for (int m = 0; m < v.launch.members; ++m) {
        s = device_buffer(c, v.y[m], v.yb[m], 4);
        if (s != STRIX_SUCCESS) return s;
    }
    v.launch.partial[0] = static_cast<float*>(workspace);
    v.launch.partial[1] = reinterpret_cast<float*>(static_cast<char*>(workspace) + v.partial_bytes[0]);
    v.launch.counters = reinterpret_cast<uint32_t*>(static_cast<char*>(workspace) + v.counter_offset);
    hipDeviceProp_t props{}; s = tp_api::properties(c, props);
    if (s != STRIX_SUCCESS) return s;
    bool supported = false;
    auto e = query_tp_split_resources(v.launch, props.sharedMemPerBlock, &supported);
    if (e != hipSuccess) return hip_status(e);
    if (!supported) return STRIX_ERROR_UNSUPPORTED_DEVICE;
    // Creation only; no clearing is performed by submit or graph replay.
    e = hipMemsetAsync(v.launch.counters, 0, v.counter_bytes, stream);
    const auto completion = hipStreamSynchronize(stream);
    if (e != hipSuccess || completion != hipSuccess) return hip_status(e != hipSuccess ? e : completion);
    return STRIX_SUCCESS;
}
strix_status create_pair(strix_context c, Validated& v, void* ws, uint64_t bytes,
                        hipStream_t stream, strix_tp_dense_pair_plan* out) {
    auto p = new (std::nothrow) strix_tp_dense_pair_plan_impl;
    if (!p) return STRIX_ERROR_OUT_OF_MEMORY;
    const auto s = storage(c, ws, bytes, v, stream);
    if (s != STRIX_SUCCESS) { delete p; return s; }
    *p = {c, {v.weights[0], v.weights[1]}, v.launch, stream};
    ++p->retained[0]->plan_refs;
    if (p->retained[1] != p->retained[0]) ++p->retained[1]->plan_refs;
    *out = p;
    return STRIX_SUCCESS;
}
} // namespace tp_split_api
extern "C" {
strix_status strix_tp_dense_workspace_bytes(strix_context c, const strix_matmul_desc* d, uint32_t tile, uint64_t* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = 0; auto s = context_check(c); if (s != STRIX_SUCCESS) return s;
    tp_split_api::Validated v; s = tp_split_api::single(c, d, tile, v);
    if (s == STRIX_SUCCESS) *out = v.bytes;
    return s;
}
strix_status strix_tp_dense_plan_create_splitk(strix_context c, const strix_matmul_desc* d,
    const strix_stream* stream, uint32_t tile, uint32_t strategy, strix_tp_dense_plan* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = nullptr; auto s = context_check(c); if (s != STRIX_SUCCESS) return s;
    if (strategy > STRIX_TP_SPLITK_TWO) return STRIX_ERROR_INVALID_ARGUMENT;
    hipStream_t hs{}; s = stream_check(c, stream, hs); if (s != STRIX_SUCCESS) return s;
    tp_split_api::Validated v; s = tp_split_api::single(c, d, tile, v); if (s != STRIX_SUCCESS) return s;
    if (!v.bytes) return strix_tp_dense_plan_create_tiled(c, d, stream, tile, out);
    v.launch.two_kernel = strategy == STRIX_TP_SPLITK_TWO;
    auto p = new (std::nothrow) strix_tp_dense_plan_impl;
    if (!p) return STRIX_ERROR_OUT_OF_MEMORY;
    s = tp_split_api::storage(c, d->workspace, d->workspace_bytes, v, hs);
    if (s != STRIX_SUCCESS) { delete p; return s; }
    *p = {c, d->weights, v.launch.member[0], hs, v.launch, true};
    ++p->weights->plan_refs; *out = p; return STRIX_SUCCESS;
}
strix_status strix_tp_dense_pair_workspace_bytes(strix_context c, const strix_tp_dense_pair_desc* d, uint64_t* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = 0; auto s = context_check(c); if (s != STRIX_SUCCESS) return s;
    tp_split_api::Validated v; s = tp_split_api::pair(c, d, v, false);
    if (s == STRIX_SUCCESS) *out = v.bytes;
    return s;
}
strix_status strix_tp_dense_pair_plan_create(strix_context c, const strix_tp_dense_pair_desc* d,
    const strix_stream* stream, strix_tp_dense_pair_plan* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = nullptr; auto s = context_check(c); if (s != STRIX_SUCCESS) return s;
    hipStream_t hs{}; s = stream_check(c, stream, hs); if (s != STRIX_SUCCESS) return s;
    tp_split_api::Validated v; s = tp_split_api::pair(c, d, v, false); if (s != STRIX_SUCCESS) return s;
    return tp_split_api::create_pair(c, v, d->workspace, d->workspace_bytes, hs, out);
}
strix_status strix_tp_swiglu_workspace_bytes(strix_context c, const strix_tp_swiglu_desc* d, uint64_t* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = 0; auto s = context_check(c); if (s != STRIX_SUCCESS) return s;
    s = header(d); if (s != STRIX_SUCCESS) return s;
    const auto pair = tp_split_api::convert(*d);
    tp_split_api::Validated v; s = tp_split_api::pair(c, &pair, v, true); if (s != STRIX_SUCCESS) return s;
    s = tp_split_api::diagnostics(c, *d, v, false); if (s == STRIX_SUCCESS) *out = v.bytes;
    return s;
}
strix_status strix_tp_swiglu_plan_create(strix_context c, const strix_tp_swiglu_desc* d,
    const strix_stream* stream, strix_tp_swiglu_plan* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = nullptr; auto s = context_check(c); if (s != STRIX_SUCCESS) return s;
    hipStream_t hs{}; s = stream_check(c, stream, hs); if (s != STRIX_SUCCESS) return s;
    s = header(d); if (s != STRIX_SUCCESS) return s;
    const auto pair = tp_split_api::convert(*d);
    tp_split_api::Validated v; s = tp_split_api::pair(c, &pair, v, true); if (s != STRIX_SUCCESS) return s;
    // Validate diagnostics before any counter initialization or other work.
    s = tp_split_api::diagnostics(c, *d, v, true); if (s != STRIX_SUCCESS) return s;
    return tp_split_api::create_pair(c, v, d->workspace, d->workspace_bytes, hs, out);
}
strix_status strix_tp_dense_pair_plan_submit(strix_tp_dense_pair_plan p) {
    if (!p) return STRIX_ERROR_INVALID_ARGUMENT;
    const auto s = context_check(p->owner); if (s != STRIX_SUCCESS) return s;
    return hip_status(strix_trellis::launch_tp_split_trusted(p->launch, p->stream));
}
strix_status strix_tp_dense_pair_plan_destroy(strix_tp_dense_pair_plan p) {
    if (!p) return STRIX_ERROR_INVALID_ARGUMENT;
    --p->retained[0]->plan_refs;
    if (p->retained[1] != p->retained[0]) --p->retained[1]->plan_refs;
    delete p; return STRIX_SUCCESS;
}
strix_status strix_tp_swiglu_plan_submit(strix_tp_swiglu_plan p) { return strix_tp_dense_pair_plan_submit(p); }
strix_status strix_tp_swiglu_plan_destroy(strix_tp_swiglu_plan p) { return strix_tp_dense_pair_plan_destroy(p); }
}
#else
extern "C" {
strix_status strix_tp_dense_workspace_bytes(strix_context, const strix_matmul_desc*, uint32_t, uint64_t* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = 0; return STRIX_ERROR_NOT_IMPLEMENTED;
}
strix_status strix_tp_dense_plan_create_splitk(strix_context, const strix_matmul_desc*, const strix_stream*, uint32_t, uint32_t, strix_tp_dense_plan* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = nullptr; return STRIX_ERROR_NOT_IMPLEMENTED;
}
#define STRIX_PAIR_STUB(PREFIX, DESC, PLAN) \
strix_status PREFIX##_workspace_bytes(strix_context, const DESC*, uint64_t* out) { \
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT; \
    *out = 0; return STRIX_ERROR_NOT_IMPLEMENTED; } \
strix_status PREFIX##_plan_create(strix_context, const DESC*, const strix_stream*, PLAN* out) { \
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT; \
    *out = nullptr; return STRIX_ERROR_NOT_IMPLEMENTED; } \
strix_status PREFIX##_plan_submit(PLAN) { return STRIX_ERROR_NOT_IMPLEMENTED; } \
strix_status PREFIX##_plan_destroy(PLAN) { return STRIX_ERROR_NOT_IMPLEMENTED; }
STRIX_PAIR_STUB(strix_tp_dense_pair, strix_tp_dense_pair_desc, strix_tp_dense_pair_plan)
STRIX_PAIR_STUB(strix_tp_swiglu, strix_tp_swiglu_desc, strix_tp_swiglu_plan)
#undef STRIX_PAIR_STUB
}
#endif
