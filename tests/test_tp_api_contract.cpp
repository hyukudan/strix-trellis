// Exercise the actual experimental C API with the established fake HIP memory
// registry. Kernel arithmetic is not emulated; native QA remains a separate gate.
#define STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP 1
#define main legacy_contract_main
#include "test_api_contract.cpp"
#undef main
#include <limits>

namespace tp_fake {
int resource_error = 0, dense_error = 0, gate_error = 0, down_error = 0;
bool supported = true;
unsigned dense_resources = 0, moe_resources = 0, dense_calls = 0, gate_calls = 0, down_calls = 0;
int resource_bits = 0, resource_rows = 0;
strix_trellis::TpDenseArguments dense{};
strix_trellis::TpMoeLaunch moe{};
std::vector<char> order;
}

namespace strix_trellis {
hipError_t query_tp_dense_resources(int bits, int rows, size_t shared, bool* supported) {
    ++tp_fake::dense_resources;
    assert(shared == fake::shared_bytes);
    tp_fake::resource_bits = bits; tp_fake::resource_rows = rows;
    *supported = tp_fake::supported;
    return tp_fake::resource_error;
}
hipError_t query_tp_moe_resources(unsigned rows, size_t shared, bool* supported) {
    ++tp_fake::moe_resources;
    assert(shared == fake::shared_bytes && rows >= 1 && rows <= 8);
    *supported = tp_fake::supported;
    return tp_fake::resource_error;
}
hipError_t launch_tp_dense_trusted(const TpDenseArguments& args, hipStream_t stream) {
    ++tp_fake::dense_calls; ++fake::launches;
    tp_fake::dense = args; fake::last_stream = stream;
    return tp_fake::dense_error;
}
hipError_t launch_tp_moe_gate_up(const TpMoeLaunch& args, hipStream_t stream) {
    ++tp_fake::gate_calls; ++fake::launches;
    tp_fake::moe = args; fake::last_stream = stream; tp_fake::order.push_back('g');
    return tp_fake::gate_error;
}
hipError_t launch_tp_moe_down(const TpMoeLaunch& args, hipStream_t stream) {
    ++tp_fake::down_calls; ++fake::launches;
    tp_fake::moe = args; fake::last_stream = stream; tp_fake::order.push_back('d');
    return tp_fake::down_error;
}
}

namespace {
struct Counters {
    int allocs = fake::allocs, copies = fake::copies, syncs = fake::syncs;
    int launches = fake::launches, pointers = fake::pointer_queries;
    int ranges = fake::range_queries, streams = fake::stream_queries;
    unsigned dense_resources = tp_fake::dense_resources, moe_resources = tp_fake::moe_resources;
    size_t buffers = fake::buffers.size();
    void no_work() const {
        assert(fake::allocs == allocs && fake::copies == copies && fake::syncs == syncs);
        assert(fake::launches == launches && fake::buffers.size() == buffers);
    }
    void hot_path(int launched) const {
        assert(fake::allocs == allocs && fake::copies == copies && fake::syncs == syncs);
        assert(fake::pointer_queries == pointers && fake::range_queries == ranges && fake::stream_queries == streams);
        assert(tp_fake::dense_resources == dense_resources && tp_fake::moe_resources == moe_resources);
        assert(fake::launches == launches + launched && fake::buffers.size() == buffers);
    }
};

struct Buffer {
    void* pointer = nullptr;
    size_t bytes;
    explicit Buffer(size_t n) : bytes(n) { assert(hipMalloc(&pointer, n) == hipSuccess); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { assert(hipFree(pointer) == hipSuccess); }
    void* at(size_t offset = 0) const { return static_cast<unsigned char*>(pointer) + offset; }
};

struct Environment {
    strix_context context = nullptr;
    strix_stream stream = descriptor<strix_stream>();
    Environment() {
        auto d = descriptor<strix_context_desc>();
        assert(strix_context_create(&d, &context) == STRIX_SUCCESS);
        stream.hip_stream = reinterpret_cast<void*>(1);
    }
    ~Environment() { assert(strix_context_destroy(context) == STRIX_SUCCESS); }
};

strix_weights prepare(Environment& env, unsigned input, unsigned output, unsigned bits) {
    std::vector<unsigned char> packed(size_t(input) * output * bits / 8, 0x93);
    std::vector<uint16_t> suh(input, 0x3800), svh(output, 0x3400);
    std::vector<float> bias(output, 0.25f);
    auto d = descriptor<strix_exl3_parts_desc>();
    d.transport_version = 1; d.columns = input; d.rows = output; d.bits_x2 = bits * 2;
    d.codebook = STRIX_CODEBOOK_MUL1; d.marker = STRIX_EXL3_MARKER_MUL1;
    d.trellis = {packed.data(), packed.size()};
    d.suh = {suh.data(), suh.size() * 2}; d.svh = {svh.data(), svh.size() * 2};
    d.bias = {bias.data(), bias.size() * 4};
    strix_weights weights = nullptr;
    Counters before;
    assert(strix_weights_prepare_parts(env.context, &d, &env.stream, &weights) == STRIX_SUCCESS);
    assert(weights && fake::allocs == before.allocs + 1 && fake::copies == before.copies + 4 &&
           fake::syncs == before.syncs + 1);
    assert(std::memcmp(weights->device.trellis, packed.data(), packed.size()) == 0);
    return weights;
}

strix_tensor_desc tensor(void* data, unsigned rows, unsigned columns, uint64_t stride) {
    auto t = descriptor<strix_tensor_desc>();
    t.data = data; t.dtype = STRIX_DTYPE_F32; t.rank = 2;
    t.shape[0] = rows; t.shape[1] = columns; t.strides[0] = stride; t.strides[1] = 1;
    t.buffer_bytes = ((rows - 1) * stride + columns) * 4;
    return t;
}

strix_matmul_desc dense_desc(strix_weights weights, unsigned rows, void* x, void* y) {
    strix_matmul_desc d; strix_matmul_desc_init(&d);
    d.weights = weights;
    d.input = tensor(x, rows, unsigned(weights->device.input), uint64_t(weights->device.input) + 17);
    d.output = tensor(y, rows, unsigned(weights->device.output), uint64_t(weights->device.output) + 23);
    d.alpha = 0.75f; d.beta = -0.5f;
    return d;
}

void dense_positive() {
    Environment env;
    const unsigned shapes[][2] = {{1280,16384}, {4096,5120}, {5120,1152},
                                 {1152,5120}, {5120,1280}, {5120,512}};
    unsigned plans = 0;
    for (const auto& shape : shapes) for (unsigned bits : {4u,5u}) {
        auto weights = prepare(env, shape[0], shape[1], bits);
        {
            Buffer x(size_t(shape[0] + 17) * 8 * 4 + 256);
            Buffer y(size_t(shape[1] + 23) * 8 * 4 + 256);
            for (unsigned rows = 1; rows <= 8; ++rows) for (unsigned tile = 1; tile <= rows; ++tile) {
                auto d = dense_desc(weights, rows, x.at(12), y.at(20));
                const auto expected = d;
                strix_tp_dense_plan plan = nullptr;
                Counters created;
                assert(strix_tp_dense_plan_create_tiled(env.context, &d, &env.stream, tile, &plan) == STRIX_SUCCESS);
                created.no_work();
                assert(weights->plan_refs == 1 && strix_weights_destroy(weights) == STRIX_ERROR_INVALID_ARGUMENT);
                assert(tp_fake::resource_bits == int(bits) && tp_fake::resource_rows == int(tile));
                d.weights = nullptr; d.input.data = nullptr; d.output.data = nullptr; d.alpha = 99;
                auto frozen_stream = env.stream; env.stream.hip_stream = reinterpret_cast<void*>(2);
                Counters submission;
                for (unsigned repeat = 0; repeat < 3; ++repeat) {
                    assert(strix_tp_dense_plan_submit(plan) == STRIX_SUCCESS);
                    assert(tp_fake::dense.weights.trellis == weights->device.trellis);
                    assert(tp_fake::dense.x == expected.input.data && tp_fake::dense.y == expected.output.data);
                    assert(tp_fake::dense.rows == int(rows) && tp_fake::dense.rows_per_block == int(tile));
                    assert(tp_fake::dense.x_row_stride == expected.input.strides[0]);
                    assert(tp_fake::dense.y_row_stride == expected.output.strides[0]);
                    assert(tp_fake::dense.alpha == expected.alpha && tp_fake::dense.beta == expected.beta);
                    assert(fake::last_stream == frozen_stream.hip_stream);
                }
                submission.hot_path(3); env.stream = frozen_stream;
                assert(strix_tp_dense_plan_destroy(plan) == STRIX_SUCCESS && weights->plan_refs == 0);
                ++plans;
            }
            auto d = dense_desc(weights, 1, x.at(), y.at());
            d.input.rank = d.output.rank = 1;
            d.input.shape[0] = shape[0]; d.output.shape[0] = shape[1];
            d.input.shape[1] = d.output.shape[1] = 0;
            d.input.strides[0] = d.output.strides[0] = 1;
            d.input.strides[1] = d.output.strides[1] = 0;
            strix_tp_dense_plan plan = nullptr;
            assert(strix_tp_dense_plan_create(env.context, &d, &env.stream, &plan) == STRIX_SUCCESS);
            assert(strix_tp_dense_plan_submit(plan) == STRIX_SUCCESS && tp_fake::dense.rows_per_block == 1);
            assert(strix_tp_dense_plan_destroy(plan) == STRIX_SUCCESS);
        }
        assert(strix_weights_destroy(weights) == STRIX_SUCCESS);
    }
    assert(plans == 432);
}

void dense_negative() {
    Environment env, other;
    auto weights = prepare(env, 5120, 1152, 4);
    auto foreign = prepare(other, 5120, 1152, 4);
    {
        Buffer x((5120 + 17) * 8 * 4 + 256), y((1152 + 23) * 8 * 4 + 256);
        const auto good = dense_desc(weights, 3, x.at(12), y.at(20));
        auto d = good;
        auto reject = [&](strix_status expected, unsigned tile = 0) {
            strix_tp_dense_plan plan = reinterpret_cast<strix_tp_dense_plan>(uintptr_t(1));
            Counters before;
            const auto retained_before = weights->plan_refs;
            assert(strix_tp_dense_plan_create_tiled(env.context, &d, &env.stream, tile, &plan) == expected && !plan);
            before.no_work(); assert(weights->plan_refs == retained_before);
            d = good;
        };
        d.struct_size--; reject(STRIX_ERROR_ABI_MISMATCH);
        d.reserved[1] = 1; reject(STRIX_ERROR_INVALID_ARGUMENT);
        d.input.reserved[1] = 1; reject(STRIX_ERROR_INVALID_ARGUMENT);
        d.workspace = x.at(); reject(STRIX_ERROR_INVALID_ARGUMENT);
        d.workspace_bytes = 1; reject(STRIX_ERROR_INVALID_ARGUMENT);
        d.weights = foreign; reject(STRIX_ERROR_INVALID_ARGUMENT);
        d.weights = nullptr; reject(STRIX_ERROR_INVALID_ARGUMENT);
        d.alpha = std::numeric_limits<float>::infinity(); reject(STRIX_ERROR_INVALID_ARGUMENT);
        d.beta = std::numeric_limits<float>::quiet_NaN(); reject(STRIX_ERROR_INVALID_ARGUMENT);
        for (uint32_t bits : {UINT32_MAX, 0x10003u}) {
            std::memcpy(&d.input.dtype, &bits, sizeof(bits)); reject(STRIX_ERROR_UNSUPPORTED_FORMAT);
            std::memcpy(&d.output.dtype, &bits, sizeof(bits)); reject(STRIX_ERROR_UNSUPPORTED_FORMAT);
        }
        d.input.shape[0] = 0; reject(STRIX_ERROR_UNSUPPORTED_FORMAT);
        d.input.shape[0] = 9; reject(STRIX_ERROR_UNSUPPORTED_FORMAT);
        d.output.shape[0] = 2; reject(STRIX_ERROR_UNSUPPORTED_FORMAT);
        d.input.strides[1] = 2; reject(STRIX_ERROR_UNSUPPORTED_FORMAT);
        d.input.strides[0] = 5119; reject(STRIX_ERROR_UNSUPPORTED_FORMAT);
        d.input.strides[0] = UINT64_MAX; reject(STRIX_ERROR_INVALID_ARGUMENT);
        d.input.buffer_bytes--; reject(STRIX_ERROR_INSUFFICIENT_BUFFER);
        d.input.data = x.at(x.bytes - 4); reject(STRIX_ERROR_INSUFFICIENT_BUFFER);
        d.input.data = x.at(1); reject(STRIX_ERROR_INVALID_ARGUMENT);
        d.input.data = reinterpret_cast<void*>(UINTPTR_MAX - 3); reject(STRIX_ERROR_INVALID_ARGUMENT);
        d.output.data = d.input.data; reject(STRIX_ERROR_INVALID_ARGUMENT);
        d.input.data = weights->allocation; reject(STRIX_ERROR_INVALID_ARGUMENT);
        d.output.data = weights->allocation; reject(STRIX_ERROR_INVALID_ARGUMENT);
        reject(STRIX_ERROR_INVALID_ARGUMENT, 4); reject(STRIX_ERROR_INVALID_ARGUMENT, UINT32_MAX);
        weights->plan_refs = SIZE_MAX; reject(STRIX_ERROR_OUT_OF_MEMORY);
        weights->plan_refs = 0;
        fake::device = 1; reject(STRIX_ERROR_INVALID_ARGUMENT); fake::device = 0;
        env.stream.hip_stream = reinterpret_cast<void*>(2); reject(STRIX_ERROR_INVALID_ARGUMENT);
        env.stream.hip_stream = reinterpret_cast<void*>(1);
        fake::max_threads = 511; reject(STRIX_ERROR_UNSUPPORTED_DEVICE); fake::max_threads = 1024;
        fake::max_dimension = 511; reject(STRIX_ERROR_UNSUPPORTED_DEVICE); fake::max_dimension = 1024;
        fake::wave = 64; reject(STRIX_ERROR_UNSUPPORTED_DEVICE); fake::wave = 32;
        tp_fake::supported = false; reject(STRIX_ERROR_UNSUPPORTED_DEVICE); tp_fake::supported = true;
        tp_fake::resource_error = 99; reject(STRIX_ERROR_BACKEND);
        tp_fake::resource_error = hipErrorOutOfMemory; reject(STRIX_ERROR_OUT_OF_MEMORY); tp_fake::resource_error = 0;
        strix_tp_dense_plan plan = nullptr;
        assert(strix_tp_dense_plan_create(env.context, &good, &env.stream, &plan) == STRIX_SUCCESS);
        Counters before;
        fake::device = 1; assert(strix_tp_dense_plan_submit(plan) == STRIX_ERROR_INVALID_ARGUMENT);
        before.hot_path(0); fake::device = 0;
        fake::fail_get_device = 1; assert(strix_tp_dense_plan_submit(plan) == STRIX_ERROR_BACKEND);
        before.hot_path(0); fake::fail_get_device = 0;
        tp_fake::dense_error = 99; assert(strix_tp_dense_plan_submit(plan) == STRIX_ERROR_BACKEND);
        tp_fake::dense_error = hipErrorOutOfMemory; assert(strix_tp_dense_plan_submit(plan) == STRIX_ERROR_OUT_OF_MEMORY);
        tp_fake::dense_error = 0; before.hot_path(2);
        assert(strix_tp_dense_plan_destroy(plan) == STRIX_SUCCESS);
        assert(strix_tp_dense_plan_submit(nullptr) == STRIX_ERROR_INVALID_ARGUMENT);
        assert(strix_tp_dense_plan_destroy(nullptr) == STRIX_ERROR_INVALID_ARGUMENT);
    }
    assert(strix_weights_destroy(foreign) == STRIX_SUCCESS);
    assert(strix_weights_destroy(weights) == STRIX_SUCCESS);
}

struct MoeCase {
    unsigned rows, count;
    Buffer x, y, hidden;
    std::vector<strix_tp_expert_desc> experts;
    std::vector<uint32_t> ids;
    std::vector<float> scores;
    strix_tp_moe_desc d{};
    MoeCase(unsigned r, unsigned n, const std::vector<strix_weights>& weights) :
        rows(r), count(n), x(size_t(5120 + 17) * r * 4 + 256),
        y(size_t(5120 + 23) * r * 4 + 256), hidden(size_t(r) * 6 * 1152 * 4),
        experts(n), ids(r * 6), scores(r * 6) {
        for (unsigned i = 0; i < n; ++i) {
            experts[i] = descriptor<strix_tp_expert_desc>();
            experts[i].expert_id = 37 + 31 * i;
            experts[i].gate = weights[i % 2]; experts[i].up = weights[(i + 1) % 2];
            experts[i].down = weights[2 + i % 2];
        }
        for (unsigned i = 0; i < r * 6; ++i) {
            ids[i] = experts[i % n].expert_id;
            scores[i] = i % 7 == 0 ? 0.0f : float(i + 1) / 49.0f;
        }
        strix_tp_moe_desc_init(&d);
        d.input = tensor(x.at(12), r, 5120, 5137);
        d.output = tensor(y.at(20), r, 5120, 5143);
        d.experts = experts.data(); d.expert_count = n;
        d.route_ids = ids.data(); d.route_ids_count = ids.size();
        d.route_scores = scores.data(); d.route_scores_count = scores.size();
        d.workspace = hidden.pointer; d.workspace_bytes = hidden.bytes;
        d.alpha = 0.625f; d.beta = -0.25f;
    }
};

void moe_positive(Environment& env, const std::vector<strix_weights>& weights) {
    unsigned cases = 0;
    for (unsigned rows = 1; rows <= 8; ++rows) for (unsigned count = 6; count <= rows * 6; ++count) {
        MoeCase m(rows, count, weights);
        Counters query;
        uint64_t workspace = 0;
        assert(strix_tp_moe_workspace_bytes(env.context, &m.d, &workspace) == STRIX_SUCCESS);
        assert(workspace == m.hidden.bytes); query.no_work();
        strix_tp_moe_plan plan = nullptr;
        Counters creation;
        assert(strix_tp_moe_plan_create(env.context, &m.d, &env.stream, &plan) == STRIX_SUCCESS);
        assert(fake::allocs == creation.allocs + 1 && fake::copies == creation.copies + 1 &&
               fake::syncs == creation.syncs + 1 && fake::launches == creation.launches);
        for (auto w : weights) {
            assert(w->plan_refs == 1);
            assert(strix_weights_destroy(w) == STRIX_ERROR_INVALID_ARGUMENT);
        }
        assert(plan->retained_count == 4);
        const auto* table = static_cast<const strix_trellis::TpMoeTable*>(plan->table);
        for (unsigned row = 0; row < rows; ++row) for (unsigned slot = 0; slot < 6; ++slot) {
            const auto index = (row * 6 + slot) % count;
            assert(table->expert_indices[row][slot] == index);
            assert(table->route_scores[row][slot] == m.scores[row * 6 + slot]);
            assert(table->experts[index].gate.trellis == m.experts[index].gate->device.trellis);
        }
        unsigned routes = 0;
        for (unsigned i = 0; i < count; ++i) routes += table->experts[i].row_count;
        assert(routes == rows * 6);
        const auto original = m.d;
        m.d.alpha = 99; m.d.input.data = nullptr; m.d.output.data = nullptr; m.d.workspace = nullptr;
        m.ids.assign(m.ids.size(), UINT32_MAX); m.scores.assign(m.scores.size(), -1);
        for (auto& expert : m.experts) expert.gate = nullptr;
        Counters submission;
        const auto order_start = tp_fake::order.size();
        assert(strix_tp_moe_plan_submit(plan) == STRIX_SUCCESS);
        assert(tp_fake::order.size() == order_start + 2 && tp_fake::order[order_start] == 'g' &&
               tp_fake::order[order_start + 1] == 'd');
        assert(strix_tp_moe_plan_gate_up(plan) == STRIX_SUCCESS);
        assert(strix_tp_moe_plan_down(plan) == STRIX_SUCCESS);
        submission.hot_path(4);
        assert(tp_fake::moe.x == original.input.data && tp_fake::moe.y == original.output.data);
        assert(tp_fake::moe.hidden == original.workspace && tp_fake::moe.rows == rows);
        assert(tp_fake::moe.expert_count == count && tp_fake::moe.alpha == original.alpha &&
               tp_fake::moe.beta == original.beta);
        assert(tp_fake::moe.x_stride == 5137 && tp_fake::moe.y_stride == 5143);
        assert(fake::last_stream == env.stream.hip_stream);
        assert(strix_tp_moe_plan_destroy(plan) == STRIX_SUCCESS);
        for (auto w : weights) assert(w->plan_refs == 0);
        assert(fake::buffers.size() == creation.buffers);
        ++cases;
    }
    assert(cases == 176);
}

void moe_negative(Environment& env, const std::vector<strix_weights>& weights) {
    MoeCase m(2, 7, weights);
    const auto good = m.d;
    auto reject = [&](strix_status status) {
        Counters before;
        strix_tp_moe_plan plan = reinterpret_cast<strix_tp_moe_plan>(uintptr_t(1));
        assert(strix_tp_moe_plan_create(env.context, &m.d, &env.stream, &plan) == status && !plan);
        before.no_work();
        for (auto w : weights) assert(w->plan_refs == 0);
        m.d = good;
    };
    m.d.struct_size--; reject(STRIX_ERROR_ABI_MISMATCH);
    m.d.reserved[0] = 1; reject(STRIX_ERROR_INVALID_ARGUMENT);
    m.d.expert_count = 5; reject(STRIX_ERROR_INVALID_ARGUMENT);
    m.d.expert_count = 49; reject(STRIX_ERROR_INVALID_ARGUMENT);
    m.d.route_ids_count--; reject(STRIX_ERROR_INVALID_ARGUMENT);
    m.d.route_scores_count++; reject(STRIX_ERROR_INVALID_ARGUMENT);
    m.d.route_ids = nullptr; reject(STRIX_ERROR_INVALID_ARGUMENT);
    m.d.route_scores = reinterpret_cast<const float*>(UINTPTR_MAX - 3); reject(STRIX_ERROR_INVALID_ARGUMENT);
    m.d.workspace_bytes--; reject(STRIX_ERROR_INSUFFICIENT_BUFFER);
    m.d.workspace = m.x.pointer; reject(STRIX_ERROR_INVALID_ARGUMENT);
    m.d.workspace = weights[0]->allocation; reject(STRIX_ERROR_INVALID_ARGUMENT);
    m.d.input.data = m.y.at(20); reject(STRIX_ERROR_INVALID_ARGUMENT);
    m.d.output.data = weights[2]->allocation; reject(STRIX_ERROR_INVALID_ARGUMENT);
    m.d.input.strides[0] = UINT64_MAX; reject(STRIX_ERROR_INVALID_ARGUMENT);
    m.d.output.buffer_bytes--; reject(STRIX_ERROR_INSUFFICIENT_BUFFER);
    m.d.beta = std::numeric_limits<float>::quiet_NaN(); reject(STRIX_ERROR_INVALID_ARGUMENT);
    const auto saved_id = m.ids[1]; m.ids[1] = m.ids[0]; reject(STRIX_ERROR_INVALID_ARGUMENT); m.ids[1] = saved_id;
    m.ids[1] = UINT32_MAX; reject(STRIX_ERROR_INVALID_ARGUMENT); m.ids[1] = saved_id;
    const auto saved_expert = m.experts[6];
    m.experts[6].expert_id = m.experts[0].expert_id; reject(STRIX_ERROR_INVALID_ARGUMENT); m.experts[6] = saved_expert;
    m.experts[6].expert_id = UINT32_MAX; reject(STRIX_ERROR_INVALID_ARGUMENT); m.experts[6] = saved_expert;
    m.experts[6].flags = 1; reject(STRIX_ERROR_INVALID_ARGUMENT); m.experts[6] = saved_expert;
    m.experts[6].down = m.experts[6].gate; reject(STRIX_ERROR_UNSUPPORTED_FORMAT); m.experts[6] = saved_expert;
    m.experts[6].gate = nullptr; reject(STRIX_ERROR_INVALID_ARGUMENT); m.experts[6] = saved_expert;
    const float saved_score = m.scores[0];
    for (float score : {-1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
        m.scores[0] = score; reject(STRIX_ERROR_INVALID_ARGUMENT);
    }
    m.scores[0] = saved_score;
    tp_fake::supported = false; reject(STRIX_ERROR_UNSUPPORTED_DEVICE); tp_fake::supported = true;
    tp_fake::resource_error = 99; reject(STRIX_ERROR_BACKEND); tp_fake::resource_error = 0;
    for (bool copy_failure : {false, true}) {
        Counters before;
        strix_tp_moe_plan plan = nullptr;
        fake::fail_copy = copy_failure; fake::fail_sync = !copy_failure;
        assert(strix_tp_moe_plan_create(env.context, &m.d, &env.stream, &plan) == STRIX_ERROR_BACKEND && !plan);
        fake::fail_copy = fake::fail_sync = 0;
        assert(fake::allocs == before.allocs + 1 && fake::copies == before.copies + 1 &&
               fake::syncs == before.syncs + 1 && fake::launches == before.launches &&
               fake::buffers.size() == before.buffers);
        for (auto w : weights) assert(w->plan_refs == 0);
    }
    strix_tp_moe_plan plan = nullptr;
    assert(strix_tp_moe_plan_create(env.context, &m.d, &env.stream, &plan) == STRIX_SUCCESS);
    Counters before;
    fake::device = 1;
    assert(strix_tp_moe_plan_submit(plan) == STRIX_ERROR_INVALID_ARGUMENT);
    assert(strix_tp_moe_plan_destroy(plan) == STRIX_ERROR_INVALID_ARGUMENT);
    before.hot_path(0); fake::device = 0;
    const auto downs = tp_fake::down_calls;
    tp_fake::gate_error = 99;
    assert(strix_tp_moe_plan_submit(plan) == STRIX_ERROR_BACKEND && tp_fake::down_calls == downs);
    tp_fake::gate_error = 0; tp_fake::down_error = 99;
    assert(strix_tp_moe_plan_submit(plan) == STRIX_ERROR_BACKEND);
    tp_fake::down_error = 0; before.hot_path(3);
    assert(strix_tp_moe_plan_destroy(plan) == STRIX_SUCCESS);
    assert(strix_tp_moe_plan_submit(nullptr) == STRIX_ERROR_INVALID_ARGUMENT);
    assert(strix_tp_moe_plan_gate_up(nullptr) == STRIX_ERROR_INVALID_ARGUMENT);
    assert(strix_tp_moe_plan_down(nullptr) == STRIX_ERROR_INVALID_ARGUMENT);
    assert(strix_tp_moe_plan_destroy(nullptr) == STRIX_ERROR_INVALID_ARGUMENT);
}

struct DeviceMoeCase {
    unsigned rows, count;
    Buffer x, y, hidden, ids, scores;
    std::vector<strix_tp_expert_desc> experts;
    strix_tp_moe_device_desc d{};
    DeviceMoeCase(unsigned r, unsigned n, const std::vector<strix_weights>& weights) :
        rows(r), count(n), x(size_t(5120 + 17) * r * 4), y(size_t(5120 + 23) * r * 4),
        hidden(size_t(r) * 6 * 1152 * 4), ids(size_t(r) * 6 * 4), scores(size_t(r) * 6 * 4),
        experts(n) {
        for (unsigned i = 0; i < n; ++i) {
            experts[i] = descriptor<strix_tp_expert_desc>(); experts[i].expert_id = i;
            experts[i].gate = weights[i % 2]; experts[i].up = weights[(i + 1) % 2];
            experts[i].down = weights[2 + i % 2];
        }
        for (unsigned i = 0; i < r * 6; ++i) {
            static_cast<uint32_t*>(ids.pointer)[i] = i % n;
            static_cast<float*>(scores.pointer)[i] = float(i + 1) / 49.0f;
        }
        strix_tp_moe_device_desc_init(&d);
        d.input = tensor(x.pointer, r, 5120, 5137);
        d.output = tensor(y.pointer, r, 5120, 5143);
        d.experts = experts.data(); d.expert_count = n;
        d.route_ids = static_cast<const uint32_t*>(ids.pointer); d.route_ids_bytes = ids.bytes;
        d.route_scores = static_cast<const float*>(scores.pointer); d.route_scores_bytes = scores.bytes;
        d.workspace = hidden.pointer; d.workspace_bytes = hidden.bytes;
        d.alpha = 0.375f; d.beta = -0.125f;
    }
};

void device_moe_positive(Environment& env, const std::vector<strix_weights>& weights) {
    for (unsigned rows : {1u,3u,8u}) for (unsigned count : {6u,7u,48u,49u,255u,256u}) {
        DeviceMoeCase m(rows, count, weights);
        strix_tp_moe_plan plan = nullptr;
        Counters creation;
        assert(strix_tp_moe_plan_create_device(env.context, &m.d, &env.stream, &plan) == STRIX_SUCCESS);
        assert(fake::allocs == creation.allocs + 1 && fake::copies == creation.copies + 1 &&
               fake::syncs == creation.syncs + 1 && fake::launches == creation.launches);
        assert(fake::copy_sources[size_t(creation.copies)] != m.ids.pointer &&
               fake::copy_sources[size_t(creation.copies)] != m.scores.pointer);
        assert(fake::find(plan->table)->second == count * sizeof(strix_trellis::TpMoeExpert));
        assert(plan->retained_count == 4);
        for (auto w : weights) {
            assert(w->plan_refs == 1);
            assert(strix_weights_destroy(w) == STRIX_ERROR_INVALID_ARGUMENT);
        }
        const auto* bank = static_cast<const strix_trellis::TpMoeExpert*>(plan->table);
        for (unsigned i = 0; i < count; ++i) {
            assert(bank[i].gate.trellis == m.experts[i].gate->device.trellis);
            assert(bank[i].up.bits == m.experts[i].up->device.bits);
            assert(bank[i].down.trellis == m.experts[i].down->device.trellis);
        }
        const auto frozen = m.d;
        m.d.route_ids = nullptr; m.d.route_scores = nullptr; m.d.alpha = 77;
        m.d.input.data = nullptr; m.d.output.data = nullptr; m.d.workspace = nullptr;
        for (auto& expert : m.experts) expert.gate = nullptr;
        const auto saved_stream = env.stream; env.stream.hip_stream = reinterpret_cast<void*>(2);
        Counters submission;
        for (unsigned window = 0; window < 4; ++window) {
            // Model ordinary device-content updates through this fake memory.
            // The API must retain addresses and never recopy either route array.
            for (unsigned i = 0; i < rows * 6; ++i) {
                static_cast<uint32_t*>(m.ids.pointer)[i] = (i + window) % count;
                static_cast<float*>(m.scores.pointer)[i] = float(i + window + 1) / 53.0f;
            }
            assert(strix_tp_moe_plan_submit(plan) == STRIX_SUCCESS);
            assert(tp_fake::moe.table == nullptr && tp_fake::moe.bank == bank);
            assert(tp_fake::moe.bank_count == count && tp_fake::moe.rows == rows);
            assert(tp_fake::moe.device_route_ids == frozen.route_ids &&
                   tp_fake::moe.device_route_scores == frozen.route_scores);
            assert(tp_fake::moe.x == frozen.input.data && tp_fake::moe.y == frozen.output.data);
            assert(tp_fake::moe.hidden == frozen.workspace && tp_fake::moe.alpha == frozen.alpha &&
                   tp_fake::moe.beta == frozen.beta);
            assert(fake::last_stream == saved_stream.hip_stream);
        }
        submission.hot_path(8); env.stream = saved_stream;
        assert(strix_tp_moe_plan_destroy(plan) == STRIX_SUCCESS);
        for (auto w : weights) assert(w->plan_refs == 0);
        assert(fake::buffers.size() == creation.buffers);
    }
}

void device_moe_negative(Environment& env, const std::vector<strix_weights>& weights) {
    DeviceMoeCase m(8, 256, weights);
    const auto good = m.d;
    auto reject = [&](strix_status status) {
        strix_tp_moe_plan plan = reinterpret_cast<strix_tp_moe_plan>(uintptr_t(1));
        Counters before;
        assert(strix_tp_moe_plan_create_device(env.context, &m.d, &env.stream, &plan) == status && !plan);
        before.no_work(); for (auto w : weights) assert(w->plan_refs == 0);
        m.d = good;
    };
    m.d.struct_size--; reject(STRIX_ERROR_ABI_MISMATCH);
    m.d.reserved[0] = 1; reject(STRIX_ERROR_INVALID_ARGUMENT);
    m.d.expert_count = 5; reject(STRIX_ERROR_INVALID_ARGUMENT);
    m.d.expert_count = 257; reject(STRIX_ERROR_INVALID_ARGUMENT);
    m.d.route_ids_bytes--; reject(STRIX_ERROR_INVALID_ARGUMENT);
    m.d.route_scores_bytes++; reject(STRIX_ERROR_INVALID_ARGUMENT);
    m.d.route_ids = nullptr; reject(STRIX_ERROR_INVALID_ARGUMENT);
    m.d.route_scores = reinterpret_cast<const float*>(UINTPTR_MAX - 3); reject(STRIX_ERROR_INVALID_ARGUMENT);
    m.d.workspace_bytes--; reject(STRIX_ERROR_INSUFFICIENT_BUFFER);
    m.d.input.strides[0] = UINT64_MAX; reject(STRIX_ERROR_INVALID_ARGUMENT);
    m.d.output.buffer_bytes--; reject(STRIX_ERROR_INSUFFICIENT_BUFFER);
    m.d.alpha = std::numeric_limits<float>::infinity(); reject(STRIX_ERROR_INVALID_ARGUMENT);
    uint32_t invalid_dtype = UINT32_MAX;
    std::memcpy(&m.d.output.dtype, &invalid_dtype, sizeof(invalid_dtype)); reject(STRIX_ERROR_UNSUPPORTED_FORMAT);
    const auto saved = m.experts.back();
    m.experts.back().expert_id--; reject(STRIX_ERROR_INVALID_ARGUMENT); m.experts.back() = saved;
    m.experts.back().flags = 1; reject(STRIX_ERROR_INVALID_ARGUMENT); m.experts.back() = saved;
    m.experts.back().reserved[0] = 1; reject(STRIX_ERROR_INVALID_ARGUMENT); m.experts.back() = saved;
    m.experts.back().down = m.experts.back().gate; reject(STRIX_ERROR_UNSUPPORTED_FORMAT); m.experts.back() = saved;
    m.experts.back().up = nullptr; reject(STRIX_ERROR_INVALID_ARGUMENT); m.experts.back() = saved;
    const void* buffers[] = {m.x.pointer, m.y.pointer, m.hidden.pointer, m.ids.pointer, m.scores.pointer};
    for (unsigned i = 1; i < 5; ++i) for (unsigned j = 0; j < i; ++j) {
        const void* p = buffers[j];
        if (i == 1) m.d.output.data = const_cast<void*>(p);
        else if (i == 2) m.d.workspace = const_cast<void*>(p);
        else if (i == 3) m.d.route_ids = static_cast<const uint32_t*>(p);
        else m.d.route_scores = static_cast<const float*>(p);
        reject(STRIX_ERROR_INVALID_ARGUMENT);
    }
    m.d.route_ids = static_cast<const uint32_t*>(weights[0]->allocation); reject(STRIX_ERROR_INVALID_ARGUMENT);
    m.d.route_scores = static_cast<const float*>(weights[3]->allocation); reject(STRIX_ERROR_INVALID_ARGUMENT);
    uint32_t host_ids[48]{};
    m.d.route_ids = host_ids; reject(STRIX_ERROR_INVALID_ARGUMENT);
    {
        Buffer short_ids(4);
        m.d.route_ids = static_cast<const uint32_t*>(short_ids.pointer);
        reject(STRIX_ERROR_INSUFFICIENT_BUFFER);
    }
    fake::device = 1; reject(STRIX_ERROR_INVALID_ARGUMENT); fake::device = 0;
    env.stream.hip_stream = reinterpret_cast<void*>(2); reject(STRIX_ERROR_INVALID_ARGUMENT);
    env.stream.hip_stream = reinterpret_cast<void*>(1);
    tp_fake::supported = false; reject(STRIX_ERROR_UNSUPPORTED_DEVICE); tp_fake::supported = true;
    tp_fake::resource_error = hipErrorOutOfMemory; reject(STRIX_ERROR_OUT_OF_MEMORY); tp_fake::resource_error = 0;
    for (bool copy_failure : {false,true}) {
        Counters before;
        strix_tp_moe_plan plan = nullptr;
        fake::fail_copy = copy_failure; fake::fail_sync = !copy_failure;
        assert(strix_tp_moe_plan_create_device(env.context, &m.d, &env.stream, &plan) == STRIX_ERROR_BACKEND && !plan);
        fake::fail_copy = fake::fail_sync = 0;
        assert(fake::allocs == before.allocs + 1 && fake::copies == before.copies + 1 &&
               fake::syncs == before.syncs + 1 && fake::launches == before.launches &&
               fake::buffers.size() == before.buffers);
        for (auto w : weights) assert(w->plan_refs == 0);
    }
}
} // namespace

int main() {
    dense_positive(); dense_negative();
    {
        Environment env;
        std::vector<strix_weights> weights = {
            prepare(env, 5120, 1152, 2), prepare(env, 5120, 1152, 3),
            prepare(env, 1152, 5120, 2), prepare(env, 1152, 5120, 3)};
        moe_positive(env, weights); moe_negative(env, weights);
        device_moe_positive(env, weights); device_moe_negative(env, weights);
        for (auto w : weights) assert(strix_weights_destroy(w) == STRIX_SUCCESS);
    }
    assert(fake::buffers.empty());
    std::cout << "TP CPU API contract passed: dense 432 tiled plans, 176 exact MoE unions, "
                 "18 device banks through 256 experts; mock HIP only.\n";
}
