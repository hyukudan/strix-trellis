// Native qualification of explicit TP plans. No mock backend is linked here.
// Timings use warm fixed addresses; they do not measure full-model tokens/s or
// sustained DRAM bandwidth. Run only through the repository GPU turn wrapper.
#include "strix_trellis.h"
#include "rank_fixture.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <set>

namespace {
using rank_qa::Fixture;
using rank_qa::require;
void hip_ok(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
void api_ok(strix_status status) {
    if (status != STRIX_SUCCESS) throw std::runtime_error(std::string("C API: ") + strix_status_string(status));
}
struct Runtime {
    hipStream_t stream = nullptr;
    strix_stream stream_desc{};
    strix_context context = nullptr;
    Runtime() {
        hip_ok(hipSetDevice(0));
        hipDeviceProp_t properties{}; hip_ok(hipGetDeviceProperties(&properties, 0));
        std::printf("TP_DEVICE name=%s arch=%s wave=%d CUs=%d\n", properties.name,
                    properties.gcnArchName, properties.warpSize, properties.multiProcessorCount);
        hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        stream_desc.struct_size = sizeof(stream_desc); stream_desc.abi_version = STRIX_TRELLIS_ABI_VERSION;
        stream_desc.hip_stream = reinterpret_cast<void*>(stream);
        strix_context_desc d{}; d.struct_size = sizeof(d); d.abi_version = STRIX_TRELLIS_ABI_VERSION;
        api_ok(strix_context_create(&d, &context));
    }
    ~Runtime() {
        if (stream) (void)hipStreamSynchronize(stream);
        if (context) (void)strix_context_destroy(context);
        if (stream) (void)hipStreamDestroy(stream);
    }
};
struct Buffer {
    Runtime& rt;
    void* allocation = nullptr;
    unsigned char* data = nullptr;
    size_t bytes;
    Buffer(Runtime& runtime, size_t count) : rt(runtime), bytes(count) {
        hip_ok(hipMalloc(&allocation, bytes + 512));
        data = static_cast<unsigned char*>(allocation) + 256;
        hip_ok(hipMemsetAsync(allocation, 0xa5, bytes + 512, rt.stream));
    }
    ~Buffer() { if (allocation) { (void)hipStreamSynchronize(rt.stream); (void)hipFree(allocation); } }
    void upload(const void* source, size_t count) {
        require(count <= bytes, "Rank upload exceeds allocation");
        hip_ok(hipMemcpyAsync(data, source, count, hipMemcpyHostToDevice, rt.stream));
    }
    std::vector<unsigned char> download() const {
        std::vector<unsigned char> result(bytes);
        hip_ok(hipMemcpyAsync(result.data(), data, bytes, hipMemcpyDeviceToHost, rt.stream));
        hip_ok(hipStreamSynchronize(rt.stream)); return result;
    }
    void canaries() const {
        std::array<unsigned char, 512> result{};
        hip_ok(hipMemcpyAsync(result.data(), allocation, 256, hipMemcpyDeviceToHost, rt.stream));
        hip_ok(hipMemcpyAsync(result.data() + 256, data + bytes, 256, hipMemcpyDeviceToHost, rt.stream));
        hip_ok(hipStreamSynchronize(rt.stream));
        for (auto byte : result) require(byte == 0xa5, "Rank allocation canary changed");
    }
};
struct View {
    unsigned rows, width, stride, offset;
    Buffer buffer;
    std::vector<unsigned char> initial;
    View(Runtime& rt, unsigned r, unsigned w, bool sliced = true)
        : rows(r), width(w), stride(sliced ? 2 * w + 17 : w), offset(sliced ? w : 0),
          buffer(rt, size_t(r) * stride * 4), initial(buffer.bytes, 0xa5) {}
    strix_tensor_desc descriptor() const {
        strix_tensor_desc d{}; d.struct_size = sizeof(d); d.abi_version = STRIX_TRELLIS_ABI_VERSION;
        d.data = buffer.data + size_t(offset) * 4;
        d.buffer_bytes = buffer.bytes - size_t(offset) * 4;
        d.dtype = STRIX_DTYPE_F32; d.rank = 2; d.shape[0] = rows; d.shape[1] = width;
        d.strides[0] = stride; d.strides[1] = 1; return d;
    }
    void reset(const std::vector<float>& values, bool poison = false) {
        require(values.size() >= size_t(rows) * width, "Short rank tensor source");
        std::fill(initial.begin(), initial.end(), 0xa5);
        for (unsigned row = 0; row < rows; ++row) {
            auto* target = initial.data() + size_t(row * stride + offset) * 4;
            if (poison) std::memset(target, 0xff, size_t(width) * 4);
            else std::memcpy(target, values.data() + size_t(row) * width, size_t(width) * 4);
        }
        buffer.upload(initial.data(), initial.size());
    }
    std::vector<float> values() const {
        const auto bytes = buffer.download();
        std::vector<float> result(size_t(rows) * width);
        for (unsigned row = 0; row < rows; ++row)
            std::memcpy(result.data() + size_t(row) * width, bytes.data() + size_t(row * stride + offset) * 4, size_t(width) * 4);
        for (size_t byte = 0; byte < bytes.size(); ++byte) {
            const size_t column = (byte / 4) % stride;
            if (column < offset || column >= offset + width) require(bytes[byte] == 0xa5, "Rank view padding changed");
        }
        buffer.canaries(); return result;
    }
    void unchanged() const { require(buffer.download() == initial, "Rank input changed"); buffer.canaries(); }
};
struct Weight {
    Runtime& rt;
    strix_weights handle = nullptr;
    Weight(Runtime& runtime, const Fixture& fixture, const std::vector<float>* bias = nullptr) : rt(runtime) {
        strix_exl3_parts_desc p{}; p.struct_size = sizeof(p); p.abi_version = STRIX_TRELLIS_ABI_VERSION;
        p.transport_version = STRIX_TRELLIS_TRANSPORT_VERSION; p.bits_x2 = 2 * fixture.bits;
        p.rows = fixture.output; p.columns = fixture.input; p.codebook = STRIX_CODEBOOK_MUL1;
        p.marker = STRIX_EXL3_MARKER_MUL1;
        const char* keys[] = {"trellis", "suh", "svh", "bias"};
        strix_host_region* regions[] = {&p.trellis, &p.suh, &p.svh, &p.bias};
        for (unsigned k = 0; k < 4; ++k) {
            const auto& bytes = fixture.sections.at(keys[k]);
            *regions[k] = {bytes.data(), uint64_t(bytes.size())};
        }
        if (bias) p.bias = {bias->data(), bias->size() * sizeof(float)};
        api_ok(strix_weights_prepare_parts(rt.context, &p, &rt.stream_desc, &handle));
    }
    ~Weight() { if (handle) { (void)hipStreamSynchronize(rt.stream); (void)strix_weights_destroy(handle); } }
};
struct DensePlan {
    Runtime& rt; strix_tp_dense_plan handle = nullptr;
    DensePlan(Runtime& runtime, const strix_matmul_desc& source, unsigned tile) : rt(runtime) {
        auto d = source; auto stream = rt.stream_desc;
        const auto started = std::chrono::steady_clock::now();
        api_ok(strix_tp_dense_plan_create_tiled(rt.context, &d, &stream, tile, &handle));
        const double elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count();
        std::printf("TP_PREPARE kind=dense rows=%llu tile=%u observed_host_us=%.9g\n",
                    static_cast<unsigned long long>(d.input.shape[0]), tile, elapsed);
        d.input.data = d.output.data = nullptr; d.weights = nullptr; stream.reserved[0] = 1;
    }
    ~DensePlan() { if (handle) { (void)hipStreamSynchronize(rt.stream); (void)strix_tp_dense_plan_destroy(handle); } }
};
struct MoePlan {
    Runtime& rt; strix_tp_moe_plan handle = nullptr;
    explicit MoePlan(Runtime& runtime) : rt(runtime) {}
    ~MoePlan() { if (handle) { (void)hipStreamSynchronize(rt.stream); (void)strix_tp_moe_plan_destroy(handle); } }
};
struct LegacyPlan {
    Runtime& rt; strix_linear_plan handle = nullptr;
    LegacyPlan(Runtime& runtime, const strix_matmul_desc& d) : rt(runtime) {
        api_ok(strix_linear_plan_create(rt.context, &d, &rt.stream_desc, STRIX_OPERATION_GEMV, &handle));
    }
    ~LegacyPlan() { if (handle) { (void)hipStreamSynchronize(rt.stream); (void)strix_linear_plan_destroy(handle); } }
};
struct Graph {
    hipGraph_t graph = nullptr;
    hipGraphExec_t execution = nullptr;
    size_t kernels = 0;
    ~Graph() { if (execution) (void)hipGraphExecDestroy(execution); if (graph) (void)hipGraphDestroy(graph); }
    void capture(Runtime& rt, const std::function<void()>& call, size_t wanted) {
        hip_ok(hipStreamBeginCapture(rt.stream, hipStreamCaptureModeThreadLocal));
        call(); hip_ok(hipStreamEndCapture(rt.stream, &graph));
        size_t count = 0; hip_ok(hipGraphGetNodes(graph, nullptr, &count));
        require(wanted ? count == wanted : count >= 1 && count <= 3, "Unexpected TP dispatch count");
        kernels = count;
        std::vector<hipGraphNode_t> nodes(count); hip_ok(hipGraphGetNodes(graph, nodes.data(), &count));
        for (auto node : nodes) {
            hipGraphNodeType type{}; hip_ok(hipGraphNodeGetType(node, &type));
            require(type == hipGraphNodeTypeKernel, "Copy/allocation/non-kernel node inside TP submission");
        }
        hip_ok(hipGraphInstantiate(&execution, graph, nullptr, nullptr, 0));
    }
    void replay(Runtime& rt) { hip_ok(hipGraphLaunch(execution, rt.stream)); hip_ok(hipStreamSynchronize(rt.stream)); }
};
void compare(const std::string& label, const std::vector<float>& actual,
             const std::vector<float>& reference, float alpha = 1, float beta = 0,
             const std::vector<float>* old = nullptr, double max_relative = 5e-4, double rms_limit = 3e-5,
             const std::vector<double>* bounds = nullptr) {
    require(actual.size() <= reference.size(), "Rank reference extent mismatch");
    double max_error = 0, max_ref = 0, error2 = 0, reference2 = 0;
    for (size_t k = 0; k < actual.size(); ++k) {
        const float scaled = alpha * reference[k];
        const float expected = beta == 0 ? scaled : scaled + beta * old->at(k);
        require(std::isfinite(actual[k]) && std::isfinite(expected), "Nonfinite TP output");
        const double error = std::abs(double(actual[k]) - expected);
        max_error = std::max(max_error, error); max_ref = std::max(max_ref, std::abs(double(expected)));
        error2 += error * error; reference2 += double(expected) * expected;
        if (bounds) require(error <= std::abs(alpha) * bounds->at(k) +
                    2e-7 * (1 + std::abs(double(expected))), "TP analytical output envelope exceeded");
    }
    const double rms = std::sqrt(error2 / std::max(reference2, 1e-30));
    std::printf("TP_CHECK label=%s values=%zu max_abs=%.9g nrmse=%.9g\n", label.c_str(), actual.size(), max_error, rms);
    require(max_error <= max_relative * std::max(1., max_ref) && rms <= rms_limit, "TP numerical gate exceeded");
}
void exact(const std::vector<float>& a, const std::vector<float>& b, const char* reason) {
    require(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0, reason);
}
void measure(Runtime& rt, const std::string& label, const std::function<void()>& call, int repetitions,
             const char* protocol = "hot_fixed_addresses") {
    hipEvent_t begin{}, end{}; hip_ok(hipEventCreate(&begin)); hip_ok(hipEventCreate(&end));
    std::vector<double> host, gpu, completion;
    for (int sweep = -1; sweep < 11; ++sweep) {
        hip_ok(hipStreamSynchronize(rt.stream)); hip_ok(hipEventRecord(begin, rt.stream));
        const auto t0 = std::chrono::steady_clock::now();
        for (int repeat = 0; repeat < repetitions; ++repeat) call();
        const auto submitted = std::chrono::steady_clock::now();
        hip_ok(hipEventRecord(end, rt.stream)); hip_ok(hipEventSynchronize(end));
        const auto completed = std::chrono::steady_clock::now(); float elapsed = 0;
        hip_ok(hipEventElapsedTime(&elapsed, begin, end));
        const double host_us = std::chrono::duration<double, std::micro>(submitted - t0).count() / repetitions;
        const double gpu_us = elapsed * 1000. / repetitions;
        const double complete_us = std::chrono::duration<double, std::micro>(completed - t0).count() / repetitions;
        require(std::isfinite(gpu_us) && gpu_us > 0 && host_us > 0, "Invalid TP timing");
        if (sweep >= 0) {
            host.push_back(host_us); gpu.push_back(gpu_us); completion.push_back(complete_us);
            std::printf("TP_SAMPLE %s sweep=%d repetitions=%d host_us=%.9g gpu_us=%.9g completion_us=%.9g\n",
                        label.c_str(), sweep, repetitions, host_us, gpu_us, complete_us);
        }
    }
    hip_ok(hipEventDestroy(begin)); hip_ok(hipEventDestroy(end));
    std::sort(host.begin(), host.end()); std::sort(gpu.begin(), gpu.end()); std::sort(completion.begin(), completion.end());
    std::printf("TP_RESULT %s protocol=%s samples=11 host_median_us=%.9g gpu_median_us=%.9g completion_median_us=%.9g\n",
                label.c_str(), protocol, host[5], gpu[5], completion[5]);
}
void dense_case(Runtime& rt, const Fixture& fixture, const std::string& site, int repetitions) {
    require(!fixture.moe && (fixture.bits == 4 || fixture.bits == 5), "Dense probe needs K4/K5 fixtures");
    Weight weights(rt, fixture);
    const auto x = fixture.values<float>("x"), old = fixture.values<float>("old"), ref = fixture.values<float>("ref");
    const auto bounds = fixture.values<double>("bound");
    for (unsigned rows = 1; rows <= 8; ++rows) {
        View input(rt, rows, fixture.input), output(rt, rows, fixture.output);
        input.reset(x);
        {
            // Optimistic existing-backend control: input/output are compact,
            // populated before timing. Packing/scatter adapter costs are not
            // included, and its addresses differ from the strided candidate.
            View baseline_input(rt, rows, fixture.input, false), baseline_output(rt, rows, fixture.output, false);
            baseline_input.reset(x); baseline_output.reset(old, true);
            strix_matmul_desc d{}; strix_matmul_desc_init(&d);
            d.weights = weights.handle; d.input = baseline_input.descriptor(); d.output = baseline_output.descriptor();
            uint64_t needed = 0; api_ok(strix_matmul_workspace_bytes(rt.context, &d, &needed));
            Buffer workspace(rt, needed); d.workspace = needed ? workspace.data : nullptr; d.workspace_bytes = needed;
            LegacyPlan plan(rt, d); auto call = [&] { api_ok(strix_linear_plan_submit(plan.handle)); };
            call(); const auto actual = baseline_output.values();
            const std::string label = "kind=dense_baseline layout=contiguous backend=legacy_plan site=" + site +
                " bits=" + std::to_string(fixture.bits) + " rows=" + std::to_string(rows);
            compare(label, actual, ref, 1, 0, nullptr, 5e-4, 3e-5, &bounds);
            size_t kernels = 0;
            {
                Graph graph; graph.capture(rt, call, 0); kernels = graph.kernels;
                baseline_output.reset(old, true); graph.replay(rt);
                exact(baseline_output.values(), actual, "Legacy baseline graph changed output");
            }
            measure(rt, label, call, repetitions); exact(baseline_output.values(), actual, "Legacy baseline timed output changed");
            baseline_input.unchanged(); workspace.canaries();
            std::printf("TP_DENSE_BASELINE_PASS %s dispatches=%zu adapter_cost_included=0 separate_addresses=1\n", label.c_str(), kernels);
        }
        std::set<unsigned> tiles{rows}; for (unsigned value : {1u, 2u, 4u, 8u}) if (value <= rows) tiles.insert(value);
        std::vector<float> first;
        for (unsigned tile : tiles) {
            strix_matmul_desc d{}; strix_matmul_desc_init(&d);
            d.weights = weights.handle; d.input = input.descriptor(); d.output = output.descriptor();
            DensePlan plan(rt, d, tile);
            require(strix_weights_destroy(weights.handle) == STRIX_ERROR_INVALID_ARGUMENT, "Dense TP plan did not retain weights");
            auto call = [&] { api_ok(strix_tp_dense_plan_submit(plan.handle)); };
            output.reset(old, true); call(); const auto actual = output.values();
            const std::string label = "kind=dense site=" + site + " bits=" + std::to_string(fixture.bits) +
                " rows=" + std::to_string(rows) + " tile=" + std::to_string(tile);
            compare(label, actual, ref, 1, 0, nullptr, 5e-4, 3e-5, &bounds);
            if (first.empty()) first = actual; else exact(actual, first, "Dense row tiling changed output bits");
            {
                Graph graph; graph.capture(rt, call, 1); output.reset(old, true); graph.replay(rt);
                exact(output.values(), actual, "Dense graph output changed");
            }
            measure(rt, label, call, repetitions);
            exact(output.values(), actual, "Dense timed output changed");
            input.unchanged();
            // Scalar epilogue is validated with finite old output on the same
            // direct strided views; it is not used in repeated timing.
            d.alpha = -.5f; d.beta = .25f;
            DensePlan epilogue(rt, d, tile); output.reset(old);
            api_ok(strix_tp_dense_plan_submit(epilogue.handle));
            compare(label + " epilogue", output.values(), ref, -.5f, .25f, &old, 5e-4, 3e-5, &bounds);
            std::printf("TP_DENSE_PASS %s dispatches=1 adapter_copies=0 row_padding=1 graph=1 retention=1\n", label.c_str());
        }
    }
}
unsigned bits_for(unsigned index, unsigned projection) {
    return projection == 0 ? 2 + index % 2 : projection == 1 ? (index % 3 == 0 ? 3 : 2) : (index % 3 == 1 ? 2 : 3);
}
struct ExpertBank {
    std::array<std::array<std::unique_ptr<Fixture>, 2>, 3> fixtures;
    std::vector<std::unique_ptr<Weight>> weights;
    std::array<strix_tp_expert_desc, 48> descriptors{};
    ExpertBank(Runtime& rt, const std::string& directory) {
        const char* names[] = {"gate", "up", "down"};
        for (unsigned p = 0; p < 3; ++p) for (unsigned bits = 2; bits <= 3; ++bits)
            fixtures[p][bits - 2] = std::make_unique<Fixture>(directory + "/expert-" + names[p] + "-k" + std::to_string(bits) + ".strank");
        const auto input = fixtures[0][0]->sections.at("x");
        for (unsigned p = 0; p < 2; ++p) for (unsigned bits = 0; bits < 2; ++bits)
            require(fixtures[p][bits]->sections.at("x") == input, "Expert gate/up fixtures have different input rows");
        for (unsigned index = 0; index < 48; ++index) {
            auto& d = descriptors[index]; d.struct_size = sizeof(d); d.abi_version = STRIX_TRELLIS_ABI_VERSION;
            d.expert_id = 17 + 3 * index;
            strix_weights* destinations[] = {&d.gate, &d.up, &d.down};
            for (unsigned p = 0; p < 3; ++p) {
                const auto& f = *fixtures[p][bits_for(index, p) - 2];
                auto bias = f.values<float>("bias");
                const float factors[] = {1, -.5f, .25f};
                const float offset = (d.expert_id + 1) * factors[p] / 512;
                for (auto& value : bias) value += offset;
                weights.push_back(std::make_unique<Weight>(rt, f, &bias));
                *destinations[p] = weights.back()->handle;
            }
        }
    }
};
void moe_case(Runtime& rt, ExpertBank& bank, const Fixture& oracle, const std::string& name, int repetitions) {
    require(oracle.moe, "MoE probe needs a routing oracle");
    const auto all_ids = oracle.values<uint32_t>("routes"); const auto all_scores = oracle.values<float>("scores");
    const auto ref = oracle.values<float>("ref"), old = oracle.values<float>("old"), hidden_ref = oracle.values<float>("hidden");
    const auto x = bank.fixtures[0][0]->values<float>("x");
    for (unsigned rows = 1; rows <= 8; ++rows) {
        View input(rt, rows, 5120), output(rt, rows, 5120); input.reset(x);
        Buffer workspace(rt, size_t(rows) * 6 * 1152 * 4);
        strix_tp_moe_desc d{}; strix_tp_moe_desc_init(&d);
        d.input = input.descriptor(); d.output = output.descriptor();
        std::vector<uint32_t> ids(all_ids.begin(), all_ids.begin() + rows * 6);
        std::vector<float> scores(all_scores.begin(), all_scores.begin() + rows * 6);
        std::set<uint32_t> union_ids(ids.begin(), ids.end());
        std::vector<strix_tp_expert_desc> experts;
        // Reverse descriptor order: route IDs must be resolved, not used as
        // offsets or mistaken for the host expert array's order.
        for (auto it = union_ids.rbegin(); it != union_ids.rend(); ++it) experts.push_back(bank.descriptors[(*it - 17) / 3]);
        d.experts = experts.data(); d.expert_count = experts.size();
        d.route_ids = ids.data(); d.route_ids_count = ids.size();
        d.route_scores = scores.data(); d.route_scores_count = scores.size();
        uint64_t needed = 0; api_ok(strix_tp_moe_workspace_bytes(rt.context, &d, &needed));
        require(needed == workspace.bytes, "MoE workspace extent mismatch");
        d.workspace = workspace.data; d.workspace_bytes = workspace.bytes;
        MoePlan plan(rt);
        const auto create_started = std::chrono::steady_clock::now();
        api_ok(strix_tp_moe_plan_create(rt.context, &d, &rt.stream_desc, &plan.handle));
        const double create_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - create_started).count();
        std::printf("TP_PREPARE kind=moe case=%s rows=%u union=%zu observed_host_us=%.9g includes_validation_upload_sync=1\n",
                    name.c_str(), rows, union_ids.size(), create_us);
        require(strix_weights_destroy(experts[0].gate) == STRIX_ERROR_INVALID_ARGUMENT, "MoE plan did not retain weights");
        // Host metadata expires/mutates immediately: plan creation owns its
        // descriptor/route copies and cannot reread these arrays on submit.
        std::fill(ids.begin(), ids.end(), UINT32_MAX); std::fill(scores.begin(), scores.end(), -1);
        for (auto& expert : experts) expert.gate = expert.up = expert.down = nullptr;
        auto gate_up = [&] { api_ok(strix_tp_moe_plan_gate_up(plan.handle)); };
        auto down = [&] { api_ok(strix_tp_moe_plan_down(plan.handle)); };
        auto complete = [&] { api_ok(strix_tp_moe_plan_submit(plan.handle)); };
        const std::string label = "kind=moe case=" + name + " rows=" + std::to_string(rows) + " union=" + std::to_string(union_ids.size());
        gate_up(); auto hidden_bytes = workspace.download(); std::vector<float> hidden(hidden_bytes.size() / 4);
        std::memcpy(hidden.data(), hidden_bytes.data(), hidden_bytes.size());
        compare(label + " stage=gate_up", hidden, hidden_ref, 1, 0, nullptr, 5e-4, 3e-5);
        // Isolate the down projection from upstream FP16 rounding boundaries.
        workspace.upload(hidden_ref.data(), workspace.bytes); output.reset(old, true); down();
        compare(label + " stage=down_cpu_hidden", output.values(), ref);
        auto poisoned = std::vector<float>(hidden_ref.begin(), hidden_ref.begin() + workspace.bytes / 4);
        bool zero_score = false;
        for (unsigned row = 0; row < rows; ++row) for (unsigned slot = 0; slot < 6; ++slot)
            if (all_scores[row * 6 + slot] == 0) {
                zero_score = true;
                std::fill(poisoned.begin() + size_t(row * 6 + slot) * 1152,
                          poisoned.begin() + size_t(row * 6 + slot + 1) * 1152,
                          std::numeric_limits<float>::quiet_NaN());
            }
        if (zero_score) {
            workspace.upload(poisoned.data(), workspace.bytes); output.reset(old, true); down();
            compare(label + " stage=down_zero_score_nan_hidden", output.values(), ref);
        }
        output.reset(old, true); complete(); const auto full = output.values();
        // Gate error can cross the documented RN16 boundary before down;
        // this end-to-end gate is explicitly wider than the isolated stages.
        compare(label + " stage=full", full, ref, 1, 0, nullptr, 2e-3, 1e-3);
        {
            Graph gate_graph, down_graph, full_graph;
            gate_graph.capture(rt, gate_up, 1); down_graph.capture(rt, down, 1); full_graph.capture(rt, complete, 2);
            output.reset(old, true); full_graph.replay(rt); exact(output.values(), full, "MoE full graph changed result");
            gate_graph.replay(rt); output.reset(old, true); down_graph.replay(rt); exact(output.values(), full, "MoE staged graph changed result");
        }
        measure(rt, label + " stage=gate_up", gate_up, repetitions);
        measure(rt, label + " stage=down", down, repetitions);
        measure(rt, label + " stage=full", complete, repetitions);
        exact(output.values(), full, "MoE timed result changed");
        workspace.canaries(); input.unchanged();
        // Recreate with the original frozen metadata for alpha/beta and the
        // all-zero-score contract. The first plan continues to own its tables.
        ids.assign(all_ids.begin(), all_ids.begin() + rows * 6);
        scores.assign(all_scores.begin(), all_scores.begin() + rows * 6);
        experts.clear();
        for (auto it = union_ids.rbegin(); it != union_ids.rend(); ++it) experts.push_back(bank.descriptors[(*it - 17) / 3]);
        d.experts = experts.data(); d.route_ids = ids.data(); d.route_scores = scores.data();
        d.alpha = -.5f; d.beta = .25f;
        {
            MoePlan epilogue(rt); api_ok(strix_tp_moe_plan_create(rt.context, &d, &rt.stream_desc, &epilogue.handle));
            output.reset(old); api_ok(strix_tp_moe_plan_submit(epilogue.handle));
            compare(label + " stage=full_epilogue", output.values(), ref, -.5f, .25f, &old, 2e-3, 1e-3);
        }
        std::fill(scores.begin(), scores.end(), 0.f);
        {
            MoePlan zero(rt); api_ok(strix_tp_moe_plan_create(rt.context, &d, &rt.stream_desc, &zero.handle));
            std::fill(poisoned.begin(), poisoned.end(), std::numeric_limits<float>::quiet_NaN());
            workspace.upload(poisoned.data(), workspace.bytes); output.reset(old);
            api_ok(strix_tp_moe_plan_down(zero.handle));
            compare(label + " stage=down_all_zero_scores", output.values(), std::vector<float>(size_t(rows) * 5120, 0),
                    -.5f, .25f, &old, 1e-7, 1e-7);
        }
        workspace.canaries(); input.unchanged();
        std::printf("TP_MOE_PASS %s dispatches=2 adapter_copies=0 top_k=6 deterministic_sum=1 graph=1 frozen_routes=1\n", label.c_str());
    }
}
void device_moe(Runtime& rt, ExpertBank& bank, const std::string& directory, int repetitions) {
    std::array<std::unique_ptr<Fixture>, 3> oracles;
    const char* names[] = {"shared", "overlap", "disjoint"};
    for (unsigned k = 0; k < 3; ++k) oracles[k] = std::make_unique<Fixture>(directory + "/moe-" + names[k] + ".strank");
    const auto x = bank.fixtures[0][0]->values<float>("x");
    for (unsigned rows : {1u, 3u, 5u, 8u}) {
        View input(rt, rows, 5120), output(rt, rows, 5120); input.reset(x);
        Buffer workspace(rt, size_t(rows) * 6 * 1152 * 4), ids_buffer(rt, size_t(rows) * 6 * 4), scores_buffer(rt, size_t(rows) * 6 * 4);
        auto experts = bank.descriptors;
        // Device banks use dense router IDs. Weight biases remain those of the
        // original fixture identity, so the same independently generated oracle
        // can check both public routing modes without rebuilding any weights.
        for (unsigned k = 0; k < experts.size(); ++k) experts[k].expert_id = k;
        strix_tp_moe_device_desc d{}; strix_tp_moe_device_desc_init(&d);
        d.input = input.descriptor(); d.output = output.descriptor();
        d.experts = experts.data(); d.expert_count = experts.size();
        d.route_ids = reinterpret_cast<const uint32_t*>(ids_buffer.data); d.route_ids_bytes = ids_buffer.bytes;
        d.route_scores = reinterpret_cast<const float*>(scores_buffer.data); d.route_scores_bytes = scores_buffer.bytes;
        d.workspace = workspace.data; d.workspace_bytes = workspace.bytes;
        MoePlan plan(rt);
        const auto started = std::chrono::steady_clock::now();
        api_ok(strix_tp_moe_plan_create_device(rt.context, &d, &rt.stream_desc, &plan.handle));
        const double create_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count();
        std::printf("TP_PREPARE kind=moe_device rows=%u bank=48 observed_host_us=%.9g bank_upload_once=1\n", rows, create_us);
        require(strix_weights_destroy(experts.back().down) == STRIX_ERROR_INVALID_ARGUMENT,
                "Device plan did not retain the whole bank");
        for (auto& expert : experts) expert.gate = expert.up = expert.down = nullptr;
        auto gate_up = [&] { api_ok(strix_tp_moe_plan_gate_up(plan.handle)); };
        auto down = [&] { api_ok(strix_tp_moe_plan_down(plan.handle)); };
        auto complete = [&] { api_ok(strix_tp_moe_plan_submit(plan.handle)); };
        Graph graph; graph.capture(rt, complete, 2);
        std::vector<uint32_t> route_ids;
        std::vector<float> route_scores;
        std::vector<float> last_full, last_ref, last_old;
        for (unsigned state : {0u, 1u, 2u, 0u}) {
            const auto& oracle = *oracles[state];
            const auto all_ids = oracle.values<uint32_t>("routes"); const auto all_scores = oracle.values<float>("scores");
            route_ids.assign(all_ids.begin(), all_ids.begin() + rows * 6);
            route_scores.assign(all_scores.begin(), all_scores.begin() + rows * 6);
            for (auto& id : route_ids) id = (id - 17) / 3;
            const auto union_size = std::set<uint32_t>(route_ids.begin(), route_ids.end()).size();
            ids_buffer.upload(route_ids.data(), ids_buffer.bytes); scores_buffer.upload(route_scores.data(), scores_buffer.bytes);
            last_ref = oracle.values<float>("ref"); last_old = oracle.values<float>("old");
            output.reset(last_old, true); graph.replay(rt); last_full = output.values();
            const std::string label = "kind=moe_device case=" + std::string(names[state]) + " rows=" +
                std::to_string(rows) + " union=" + std::to_string(union_size) + " bank=48";
            compare(label + " stage=full_graph", last_full, last_ref, 1, 0, nullptr, 2e-3, 1e-3);
            const auto hidden_ref = oracle.values<float>("hidden");
            const auto dynamic_hidden_bytes = workspace.download();
            std::vector<float> hidden(dynamic_hidden_bytes.size() / 4);
            std::memcpy(hidden.data(), dynamic_hidden_bytes.data(), dynamic_hidden_bytes.size());
            compare(label + " stage=gate_up", hidden, hidden_ref);
            {
                // Same weights/rows/scores through the independent host-union
                // route path. Its preparation is deliberately outside timing.
                std::vector<uint32_t> frozen_ids(all_ids.begin(), all_ids.begin() + rows * 6);
                std::set<uint32_t> selected(frozen_ids.begin(), frozen_ids.end());
                std::vector<strix_tp_expert_desc> frozen_experts;
                for (auto it = selected.rbegin(); it != selected.rend(); ++it)
                    frozen_experts.push_back(bank.descriptors[(*it - 17) / 3]);
                strix_tp_moe_desc frozen{}; strix_tp_moe_desc_init(&frozen);
                frozen.input = input.descriptor(); frozen.output = output.descriptor();
                frozen.experts = frozen_experts.data(); frozen.expert_count = frozen_experts.size();
                frozen.route_ids = frozen_ids.data(); frozen.route_ids_count = frozen_ids.size();
                frozen.route_scores = route_scores.data(); frozen.route_scores_count = route_scores.size();
                frozen.workspace = workspace.data; frozen.workspace_bytes = workspace.bytes;
                MoePlan control(rt); api_ok(strix_tp_moe_plan_create(rt.context, &frozen, &rt.stream_desc, &control.handle));
                output.reset(last_old, true); api_ok(strix_tp_moe_plan_submit(control.handle));
                exact(output.values(), last_full, "Device and immutable routing outputs differ");
                require(workspace.download() == dynamic_hidden_bytes, "Device and immutable hidden results differ");
            }
            workspace.upload(hidden_ref.data(), workspace.bytes); output.reset(last_old, true); down();
            compare(label + " stage=down_cpu_hidden", output.values(), last_ref);
            complete(); exact(output.values(), last_full, "Device routing direct/graph disagreement");
            measure(rt, label + " stage=gate_up", gate_up, repetitions);
            measure(rt, label + " stage=down", down, repetitions);
            measure(rt, label + " stage=full", complete, repetitions);
            exact(output.values(), last_full, "Device routing timed output changed");
            input.unchanged(); workspace.canaries(); ids_buffer.canaries(); scores_buffer.canaries();
            std::printf("TP_MOE_DEVICE_PASS %s dispatches=2 adapter_copies=0 graph_reroute=1 immutable_exact=1\n", label.c_str());
        }
        const auto original_ids = route_ids; const auto original_scores = route_scores;
        {
            std::vector<float> zero_scores(size_t(rows) * 6, 0.f);
            std::vector<float> nan_hidden(workspace.bytes / 4, std::numeric_limits<float>::quiet_NaN());
            scores_buffer.upload(zero_scores.data(), scores_buffer.bytes);
            workspace.upload(nan_hidden.data(), workspace.bytes); output.reset(last_old);
            down();
            exact(output.values(), std::vector<float>(size_t(rows) * 5120, 0.f), "All-zero device scores read NaN hidden");
        }
        for (unsigned negative = 0; negative < 7; ++negative) {
            route_ids = original_ids; route_scores = original_scores;
            if (negative == 0) route_ids[0] = 48;
            if (negative == 1) route_ids[0] = UINT32_MAX;
            if (negative == 2) route_ids[0] = route_ids[1];
            if (negative == 3) route_scores[0] = -1;
            if (negative == 4) route_scores[0] = std::numeric_limits<float>::quiet_NaN();
            if (negative == 5) route_scores[0] = std::numeric_limits<float>::infinity();
            if (negative == 6) { route_ids[0] = 48; route_scores[0] = 0; }
            ids_buffer.upload(route_ids.data(), ids_buffer.bytes); scores_buffer.upload(route_scores.data(), scores_buffer.bytes);
            // Finite old Y ensures a missing write cannot masquerade as the
            // documented NaN signal for invalid device routing contents.
            output.reset(last_old); graph.replay(rt); const auto actual = output.values();
            for (unsigned column = 0; column < 5120; ++column)
                require(std::isnan(actual[column]), "Invalid device route failed to signal NaN");
            if (rows > 1) {
                const std::vector<float> valid(actual.begin() + 5120, actual.end());
                const std::vector<float> expected(last_ref.begin() + 5120, last_ref.begin() + size_t(rows) * 5120);
                compare("device_invalid_valid_tokens", valid, expected, 1, 0, nullptr, 2e-3, 1e-3);
            }
            workspace.canaries(); ids_buffer.canaries(); scores_buffer.canaries();
        }
        ids_buffer.upload(original_ids.data(), ids_buffer.bytes); scores_buffer.upload(original_scores.data(), scores_buffer.bytes);
        output.reset(last_old, true); graph.replay(rt); exact(output.values(), last_full, "Device route recovery changed valid output");
        input.unchanged();
        std::printf("TP_MOE_DEVICE_NEGATIVE_PASS rows=%u cases=7 affected_token_nan=1 valid_tokens=1 recovery=1\n", rows);
    }
}
} // namespace
int main(int argc, char** argv) {
    try {
        setvbuf(stdout, nullptr, _IOLBF, 0);
        require(argc == 3 || argc == 4, "usage: strix_trellis_rank_fused_probe dense|moe|moe-device FIXTURE_DIRECTORY [REPETITIONS=4]");
        const std::string suite = argv[1], directory = argv[2];
        require(suite == "dense" || suite == "moe" || suite == "moe-device", "Unknown TP probe suite");
        int repetitions = 4;
        if (argc == 4) { size_t consumed = 0; repetitions = std::stoi(argv[3], &consumed); require(consumed == std::strlen(argv[3]), "Invalid repetition count"); }
        require(repetitions > 0 && repetitions <= 64, "Repetitions must be in 1..64");
        Runtime runtime;
        if (suite == "dense") {
            for (const char* site : {"q_b", "o_b", "shared_gate", "shared_up", "shared_down", "q_a", "kv"})
                for (unsigned bits : {4u, 5u}) {
                    Fixture fixture(directory + "/dense-" + site + "-k" + std::to_string(bits) + ".strank");
                    dense_case(runtime, fixture, site, repetitions);
                }
            std::puts("PASS rank_fused_dense_probe");
        } else {
            ExpertBank bank(runtime, directory);
            if (suite == "moe-device") {
                device_moe(runtime, bank, directory, repetitions);
                std::puts("PASS rank_fused_moe_device_probe");
            } else {
                for (const char* name : {"shared", "overlap", "disjoint"}) {
                    Fixture oracle(directory + "/moe-" + name + ".strank");
                    moe_case(runtime, bank, oracle, name, repetitions);
                }
                std::puts("PASS rank_fused_moe_probe");
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL rank_fused_probe: %s\n", error.what()); return 1;
    }
}
