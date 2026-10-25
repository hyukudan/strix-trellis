// Grouped strided GEMV over the dense-r1 kernels; included after dense_r1_api.inc.
#ifdef STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP
struct strix_grouped_gemv_plan_impl {
    strix_context owner;
    strix_weights retained[strix_trellis::dense_r1_group_max]{};
    int count = 0;
    strix_trellis::DenseR1Group group{};
    hipStream_t stream = nullptr;
    bool owner_fold = false;  // owner-fold kernel (same bits)
};
namespace dense_r1_group_api {
using namespace strix_trellis;
struct Validated {
    DenseR1Group group{};
    strix_weights weights[dense_r1_group_max]{};
    int count = 0;
    uint64_t input_bytes = 0, output_bytes = 0;
    size_t member_bytes = 0, counter_offset = 0, counter_bytes = 0, bytes = 0;
};
// Last F32 element touched + 1 by `count` views of `rows` rows of `width` values.
bool span(uint64_t rows, uint64_t row_stride, uint64_t count, uint64_t group_stride, uint64_t width, uint64_t& out) {
    uint64_t a = 0, b = 0;
    if (!multiply(rows - 1, row_stride, a) || !multiply(count - 1, group_stride, b)) return false;
    if (a > UINT64_MAX - b || a + b > UINT64_MAX - width) return false;
    return multiply(a + b + width, 4, out);
}
strix_status validate(strix_context c, const strix_grouped_gemv_desc* d, Validated& v) {
    auto s = header(d);
    if (s != STRIX_SUCCESS) return s;
    if (!d->weights || d->group_count < 1 || d->group_count > uint64_t(dense_r1_group_max) || d->rows < 1 || d->rows > 8 ||
        d->flags != STRIX_GEMV_ROW_INVARIANT_R1 || !d->input || !d->output) return STRIX_ERROR_INVALID_ARGUMENT;
    v.count = int(d->group_count);
    const uint64_t rows = d->rows;
    int bits = 0, input = 0, output = 0;
    for (int g = 0; g < v.count; ++g) {
        const auto w = d->weights[g];
        if (!w || w->owner != c) return STRIX_ERROR_INVALID_ARGUMENT;
        const auto& dw = w->device;
        if (!dense_r1_group_geometry(dw.input, dw.output) || (dw.bits != 4 && dw.bits != 5) || dw.codebook != 2 || dw.bias)
            return STRIX_ERROR_UNSUPPORTED_FORMAT;
        if (g == 0) { bits = dw.bits; input = dw.input; output = dw.output; }
        else if (dw.bits != bits || dw.input != input || dw.output != output) return STRIX_ERROR_UNSUPPORTED_FORMAT;
        if (w->plan_refs == SIZE_MAX) return STRIX_ERROR_OUT_OF_MEMORY;
        v.weights[g] = w;
    }
    const uint64_t G = uint64_t(v.count), I = uint64_t(input), N = uint64_t(output);
    const uint64_t xrs = rows > 1 ? d->input_row_stride : 0, xgs = G > 1 ? d->input_group_stride : 0;
    const uint64_t yrs = rows > 1 ? d->output_row_stride : 0, ygs = G > 1 ? d->output_group_stride : 0;
    if ((rows > 1 && xrs < I) || (G > 1 && ygs < N) || (rows > 1 && yrs < (G - 1) * ygs + N)) return STRIX_ERROR_INVALID_ARGUMENT;
    if (!span(rows, xrs, G, xgs, I, v.input_bytes) || !span(rows, yrs, G, ygs, N, v.output_bytes) ||
        v.input_bytes > d->input_buffer_bytes || v.output_bytes > d->output_buffer_bytes ||
        reinterpret_cast<uintptr_t>(d->input) % 4 || reinterpret_cast<uintptr_t>(d->output) % 4 ||
        !address(d->input, v.input_bytes) || !address(d->output, v.output_bytes) ||
        overlap(d->input, v.input_bytes, d->output, v.output_bytes)) return STRIX_ERROR_INVALID_ARGUMENT;
    for (int g = 0; g < v.count; ++g)
        if (tp_api::touches_weights(d->input, v.input_bytes, v.weights[g]) ||
            tp_api::touches_weights(d->output, v.output_bytes, v.weights[g])) return STRIX_ERROR_INVALID_ARGUMENT;
    const int splits = dense_r1_group_splits(input, output, bits);
    // Counters first, at offsets that depend only on the geometry (room for the maximum member count), then the
    // members' partial planes: plans of other row counts sharing this workspace never write over these counters.
    v.counter_bytes = size_t(N / 128) * 4;
    v.counter_offset = (size_t(dense_r1_group_max) * v.counter_bytes + 255) & ~size_t(255);
    v.member_bytes = (size_t(splits) * rows * N * 4 + 255) & ~size_t(255);
    v.bytes = v.counter_offset + v.member_bytes * size_t(v.count);
    const auto sched = dense_r1_group_schedule(input, output, bits, int(rows), splits);
    for (int g = 0; g < v.count; ++g) {
        auto& a = v.group.member[g];
        a.projection = {v.weights[g]->device, d->input + size_t(g) * xgs, d->output + size_t(g) * ygs,
            xrs, yrs, int(rows), 1.0f, 0.0f, int(rows)};
        a.splits = splits;
        a.columns_per_block = sched.columns;
        a.prefetch = sched.prefetch;
        a.outputs_per_wave = sched.outputs_per_wave;
        a.pipelined = dense_r1_group_pipelined(input, output, bits, int(rows), splits);
    }
    return STRIX_SUCCESS;
}
}
extern "C" strix_status strix_grouped_gemv_workspace_bytes(strix_context c, const strix_grouped_gemv_desc* d, uint64_t* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = 0; auto s = context_check(c); if (s != STRIX_SUCCESS) return s;
    dense_r1_group_api::Validated v; s = dense_r1_group_api::validate(c, d, v);
    if (s == STRIX_SUCCESS) *out = v.bytes;
    return s;
}
extern "C" strix_status strix_grouped_gemv_plan_create(strix_context c, const strix_grouped_gemv_desc* d,
    const strix_stream* stream, strix_grouped_gemv_plan* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = nullptr; auto s = context_check(c); if (s != STRIX_SUCCESS) return s;
    hipStream_t hs{}; s = stream_check(c, stream, hs); if (s != STRIX_SUCCESS) return s;
    dense_r1_group_api::Validated v; s = dense_r1_group_api::validate(c, d, v); if (s != STRIX_SUCCESS) return s;
    if (d->workspace_bytes < v.bytes) return STRIX_ERROR_INSUFFICIENT_BUFFER;
    if (!d->workspace || !address(d->workspace, d->workspace_bytes) ||
        reinterpret_cast<uintptr_t>(d->workspace) % alignment ||
        overlap(d->workspace, d->workspace_bytes, d->input, v.input_bytes) ||
        overlap(d->workspace, d->workspace_bytes, d->output, v.output_bytes)) return STRIX_ERROR_INVALID_ARGUMENT;
    for (int g = 0; g < v.count; ++g)
        if (tp_api::touches_weights(d->workspace, d->workspace_bytes, v.weights[g])) return STRIX_ERROR_INVALID_ARGUMENT;
    s = device_buffer(c, d->input, v.input_bytes, 4); if (s != STRIX_SUCCESS) return s;
    s = device_buffer(c, d->output, v.output_bytes, 4); if (s != STRIX_SUCCESS) return s;
    s = device_buffer(c, d->workspace, d->workspace_bytes, alignment); if (s != STRIX_SUCCESS) return s;
    for (int g = 0; g < v.count; ++g) {
        auto& a = v.group.member[g];
        char* base = static_cast<char*>(d->workspace);
        a.counters = reinterpret_cast<uint32_t*>(base + v.counter_bytes * size_t(g));
        a.partial = reinterpret_cast<float*>(base + v.counter_offset + v.member_bytes * size_t(g));
    }
    hipDeviceProp_t props{}; s = tp_api::properties(c, props); if (s != STRIX_SUCCESS) return s;
    bool supported = false;
    auto e = strix_trellis::query_dense_r1_group_resources(v.group.member[0], props.sharedMemPerBlock, &supported);
    if (e != hipSuccess) return hip_status(e);
    if (!supported) return STRIX_ERROR_UNSUPPORTED_DEVICE;
    auto p = new (std::nothrow) strix_grouped_gemv_plan_impl;
    if (!p) return STRIX_ERROR_OUT_OF_MEMORY;
    e = hipMemsetAsync(d->workspace, 0, v.counter_bytes * size_t(v.count), hs);
    const auto completion = hipStreamSynchronize(hs);
    if (e != hipSuccess || completion != hipSuccess) { delete p; return hip_status(e != hipSuccess ? e : completion); }
    p->owner = c; p->count = v.count; p->group = v.group; p->stream = hs;
    {
        const auto& m0 = v.group.member[0];
        bool owner_ok = false;
        if (strix_trellis::dense_r1_owner(m0.projection.weights.input, m0.projection.weights.output, m0.projection.weights.bits,
                                          m0.projection.rows, m0.splits) &&
            strix_trellis::query_dense_r1_group_owner_resources(m0, props.sharedMemPerBlock, &owner_ok) == hipSuccess && owner_ok)
            p->owner_fold = true;
    }
    for (int g = 0; g < v.count; ++g) { p->retained[g] = v.weights[g]; ++v.weights[g]->plan_refs; }
    *out = p; return STRIX_SUCCESS;
}
extern "C" strix_status strix_grouped_gemv_plan_submit(strix_grouped_gemv_plan p) {
    if (!p) return STRIX_ERROR_INVALID_ARGUMENT;
    const auto s = context_check(p->owner);
    if (s != STRIX_SUCCESS) return s;
    if (p->owner_fold) return hip_status(strix_trellis::launch_dense_r1_group_owner_trusted(p->group, p->count, p->stream));
    return hip_status(strix_trellis::launch_dense_r1_group_trusted(p->group, p->count, p->stream));
}
extern "C" strix_status strix_grouped_gemv_plan_destroy(strix_grouped_gemv_plan p) {
    if (!p) return STRIX_ERROR_INVALID_ARGUMENT;
    for (int g = 0; g < p->count; ++g) --p->retained[g]->plan_refs;
    delete p;
    return STRIX_SUCCESS;
}
namespace dense_r1_pair_api {
using namespace strix_trellis;
struct Validated {
    DenseR1Pair pair{};
    strix_weights weights[2]{};
    const void* x = nullptr;
    void* y[2]{};
    uint64_t xb = 0, yb[2]{};
    size_t partial_offset[2]{}, bytes = 0;
};
constexpr size_t counters_bytes = 256;  // member 0 counters at 0 (1280 / 128 words), member 1 at 64 (512 / 128 words)
strix_status validate(strix_context c, const strix_tp_dense_pair_desc* d, uint32_t identity, Validated& v) {
    auto s = header(d);
    if (s != STRIX_SUCCESS) return s;
    if (identity != STRIX_DENSE_R1_LEGACY || d->rows_per_block) return STRIX_ERROR_UNSUPPORTED_FORMAT;
    for (int m = 0; m < 2; ++m) {
        strix_matmul_desc md; strix_matmul_desc_init(&md);
        md.weights = d->member[m].weights; md.input = d->input; md.output = d->member[m].output;
        md.alpha = d->member[m].alpha; md.beta = d->member[m].beta;
        tp_split_api::Validated one;
        s = dense_r1_api::validate(c, &md, 0, identity, one);
        if (s != STRIX_SUCCESS) return s;
        auto& a = v.pair.member[m];
        a.projection = one.launch.member[0];
        const auto& w = a.projection.weights;
        if (w.input != 5120 || w.output != (m == 0 ? 1280 : 512)) return STRIX_ERROR_UNSUPPORTED_FORMAT;
        a.splits = dense_r1_splits(w.input, w.output, true);
        const auto sched = dense_r1_schedule(w.input, w.output, w.bits, a.projection.rows_per_block, a.splits);
        a.columns_per_block = sched.columns; a.prefetch = sched.prefetch; a.outputs_per_wave = sched.outputs_per_wave;
        if (sched.use_splitk04) return STRIX_ERROR_UNSUPPORTED_FORMAT;
        v.weights[m] = md.weights; v.x = one.x; v.xb = one.xb; v.y[m] = one.y[0]; v.yb[m] = one.yb[0];
    }
    if (v.pair.member[0].projection.rows != v.pair.member[1].projection.rows ||
        v.pair.member[0].projection.weights.bits != v.pair.member[1].projection.weights.bits ||
        overlap(v.y[0], v.yb[0], v.y[1], v.yb[1])) return STRIX_ERROR_INVALID_ARGUMENT;
    const size_t rows = size_t(v.pair.member[0].projection.rows);
    v.partial_offset[0] = counters_bytes;
    v.partial_offset[1] = v.partial_offset[0] + ((size_t(16) * rows * 1280 * 4 + 255) & ~size_t(255));
    v.bytes = v.partial_offset[1] + ((size_t(32) * rows * 512 * 4 + 255) & ~size_t(255));
    return STRIX_SUCCESS;
}
}
extern "C" strix_status strix_tp_dense_pair_r1_workspace_bytes(strix_context c, const strix_tp_dense_pair_desc* d,
    uint32_t identity, uint64_t* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = 0; auto s = context_check(c); if (s != STRIX_SUCCESS) return s;
    dense_r1_pair_api::Validated v; s = dense_r1_pair_api::validate(c, d, identity, v);
    if (s == STRIX_SUCCESS) *out = v.bytes;
    return s;
}
extern "C" strix_status strix_tp_dense_pair_plan_create_r1(strix_context c, const strix_tp_dense_pair_desc* d,
    const strix_stream* stream, uint32_t identity, strix_tp_dense_plan* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = nullptr; auto s = context_check(c); if (s != STRIX_SUCCESS) return s;
    hipStream_t hs{}; s = stream_check(c, stream, hs); if (s != STRIX_SUCCESS) return s;
    dense_r1_pair_api::Validated v; s = dense_r1_pair_api::validate(c, d, identity, v); if (s != STRIX_SUCCESS) return s;
    if (d->workspace_bytes < v.bytes) return STRIX_ERROR_INSUFFICIENT_BUFFER;
    if (!d->workspace || !address(d->workspace, d->workspace_bytes) || reinterpret_cast<uintptr_t>(d->workspace) % alignment ||
        overlap(d->workspace, d->workspace_bytes, v.x, v.xb) || overlap(d->workspace, d->workspace_bytes, v.y[0], v.yb[0]) ||
        overlap(d->workspace, d->workspace_bytes, v.y[1], v.yb[1]) ||
        tp_api::touches_weights(d->workspace, d->workspace_bytes, v.weights[0]) ||
        tp_api::touches_weights(d->workspace, d->workspace_bytes, v.weights[1])) return STRIX_ERROR_INVALID_ARGUMENT;
    s = device_buffer(c, v.x, v.xb, 4); if (s != STRIX_SUCCESS) return s;
    s = device_buffer(c, v.y[0], v.yb[0], 4); if (s != STRIX_SUCCESS) return s;
    s = device_buffer(c, v.y[1], v.yb[1], 4); if (s != STRIX_SUCCESS) return s;
    s = device_buffer(c, d->workspace, d->workspace_bytes, alignment); if (s != STRIX_SUCCESS) return s;
    char* base = static_cast<char*>(d->workspace);
    v.pair.member[0].counters = reinterpret_cast<uint32_t*>(base);
    v.pair.member[1].counters = reinterpret_cast<uint32_t*>(base + 64);
    v.pair.member[0].partial = reinterpret_cast<float*>(base + v.partial_offset[0]);
    v.pair.member[1].partial = reinterpret_cast<float*>(base + v.partial_offset[1]);
    hipDeviceProp_t props{}; s = tp_api::properties(c, props); if (s != STRIX_SUCCESS) return s;
    bool supported = false;
    auto e = strix_trellis::query_dense_r1_pair_resources(v.pair, props.sharedMemPerBlock, &supported);
    if (e != hipSuccess) return hip_status(e);
    if (!supported) return STRIX_ERROR_UNSUPPORTED_DEVICE;
    auto p = new (std::nothrow) strix_tp_dense_plan_impl;
    if (!p) return STRIX_ERROR_OUT_OF_MEMORY;
    e = hipMemsetAsync(base, 0, dense_r1_pair_api::counters_bytes, hs);
    const auto completion = hipStreamSynchronize(hs);
    if (e != hipSuccess || completion != hipSuccess) { delete p; return hip_status(e != hipSuccess ? e : completion); }
    *p = {c, v.weights[0], v.pair.member[0].projection, hs};
    p->pair = v.pair; p->use_pair = true; p->second = v.weights[1];
    ++v.weights[0]->plan_refs; ++v.weights[1]->plan_refs;
    *out = p; return STRIX_SUCCESS;
}
#else
extern "C" strix_status strix_grouped_gemv_workspace_bytes(strix_context, const strix_grouped_gemv_desc*, uint64_t* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = 0; return STRIX_ERROR_NOT_IMPLEMENTED;
}
extern "C" strix_status strix_grouped_gemv_plan_create(strix_context, const strix_grouped_gemv_desc*, const strix_stream*,
    strix_grouped_gemv_plan* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = nullptr; return STRIX_ERROR_NOT_IMPLEMENTED;
}
extern "C" strix_status strix_grouped_gemv_plan_submit(strix_grouped_gemv_plan) { return STRIX_ERROR_NOT_IMPLEMENTED; }
extern "C" strix_status strix_grouped_gemv_plan_destroy(strix_grouped_gemv_plan) { return STRIX_ERROR_NOT_IMPLEMENTED; }
extern "C" strix_status strix_tp_dense_pair_r1_workspace_bytes(strix_context, const strix_tp_dense_pair_desc*, uint32_t, uint64_t* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = 0; return STRIX_ERROR_NOT_IMPLEMENTED;
}
extern "C" strix_status strix_tp_dense_pair_plan_create_r1(strix_context, const strix_tp_dense_pair_desc*, const strix_stream*,
    uint32_t, strix_tp_dense_plan* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = nullptr; return STRIX_ERROR_NOT_IMPLEMENTED;
}
#endif
