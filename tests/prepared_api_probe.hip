// Native checks for region preparation and immutable execution plans. Uses this
// project's fixture parser/reference, while all execution uses the real C API.
#define main strix_original_runtime_main
#include "gpu_runtime.hip"
#undef main
#include <array>
#include <chrono>
#include <functional>

namespace {
void insist(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct GuardedPrepared {
    Device allocation;
    unsigned char* p;
    size_t bytes;
    Runtime& runtime;
    GuardedPrepared(Runtime& rt, size_t count) : allocation(count + 512),
        p(static_cast<unsigned char*>(allocation.p) + 256), bytes(count), runtime(rt) {
        hip_check(hipMemsetAsync(allocation.p, 0xa5, count + 512, rt.stream));
    }
    std::vector<unsigned char> download() {
        std::vector<unsigned char> result(bytes);
        if (bytes) hip_check(hipMemcpyAsync(result.data(), p, bytes, hipMemcpyDeviceToHost, runtime.stream));
        hip_check(hipStreamSynchronize(runtime.stream)); return result;
    }
    void check() {
        unsigned char before[256], after[256];
        hip_check(hipMemcpyAsync(before, allocation.p, 256, hipMemcpyDeviceToHost, runtime.stream));
        hip_check(hipMemcpyAsync(after, p + bytes, 256, hipMemcpyDeviceToHost, runtime.stream));
        hip_check(hipStreamSynchronize(runtime.stream));
        for (unsigned i = 0; i < 256; ++i) insist(before[i] == 0xa5 && after[i] == 0xa5, "prepared buffer canary");
    }
};
struct PartsWeight {
    Runtime& runtime;
    strix_weights w = nullptr;
    explicit PartsWeight(Runtime& r, const Fixture& f) : runtime(r) {
        strix_exl3_parts_desc d{};
        d.struct_size = sizeof(d); d.abi_version = STRIX_TRELLIS_ABI_VERSION;
        d.transport_version = STRIX_TRELLIS_TRANSPORT_VERSION;
        d.rows = f.n; d.columns = f.i; d.bits_x2 = 2 * f.bits;
        d.codebook = static_cast<strix_codebook>(f.cb); d.marker = f.marker;
        std::array<std::vector<unsigned char>, 4> source;
        strix_host_region* regions[] = {&d.trellis, &d.suh, &d.svh, &d.bias};
        const char* keys[] = {"trellis", "suh", "svh", "bias"};
        for (int index = 0; index < 4; ++index) {
            const auto& original = f.parts.at(keys[index]);
            if (original.empty()) continue;
            source[index].resize(original.size() + 1);
            std::memcpy(source[index].data() + 1, original.data(), original.size());
            *regions[index] = {source[index].data() + 1, original.size()};
        }
        api_check(strix_weights_prepare_parts(r.context, &d, &r.s, &w));
        // Every host pointer was deliberately unaligned. Overwrite and release
        // the source after return to verify the synchronous ownership contract.
        for (auto& region : source) std::fill(region.begin(), region.end(), 0x5a);
    }
    ~PartsWeight() { if (w) { (void)hipStreamSynchronize(runtime.stream); (void)strix_weights_destroy(w); } }
    void close() { hip_check(hipStreamSynchronize(runtime.stream)); api_check(strix_weights_destroy(w)); w = nullptr; }
};
struct PreparedPlan {
    Runtime& runtime;
    strix_linear_plan plan = nullptr;
    PreparedPlan(Runtime& rt, const strix_matmul_desc& d, uint32_t operation) : runtime(rt) {
        auto copied_desc = d; auto copied_stream = rt.s;
        api_check(strix_linear_plan_create(rt.context, &copied_desc, &copied_stream, operation, &plan));
        // These temporary descriptors are mutated and expire immediately.
        copied_desc.weights = nullptr; copied_desc.alpha = std::numeric_limits<float>::quiet_NaN();
        copied_desc.input.data = copied_desc.output.data = copied_desc.workspace = nullptr;
        copied_stream.hip_stream = nullptr; copied_stream.reserved[0] = 1;
    }
    ~PreparedPlan() { if (plan) { (void)hipStreamSynchronize(runtime.stream); (void)strix_linear_plan_destroy(plan); } }
    void close() { hip_check(hipStreamSynchronize(runtime.stream)); api_check(strix_linear_plan_destroy(plan)); plan = nullptr; }
};
struct PreparedGraph {
    hipGraph_t graph = nullptr; hipGraphExec_t executable = nullptr;
    ~PreparedGraph() { if (executable) (void)hipGraphExecDestroy(executable); if (graph) (void)hipGraphDestroy(graph); }
    size_t capture(Runtime& r, const std::function<void()>& submit) {
        hip_check(hipStreamBeginCapture(r.stream, hipStreamCaptureModeThreadLocal));
        submit(); hip_check(hipStreamEndCapture(r.stream, &graph));
        size_t count = 0; hip_check(hipGraphGetNodes(graph, nullptr, &count));
        insist(count > 0 && count <= 3, "prepared graph kernel count");
        std::vector<hipGraphNode_t> nodes(count); hip_check(hipGraphGetNodes(graph, nodes.data(), &count));
        for (auto node : nodes) { hipGraphNodeType type{}; hip_check(hipGraphNodeGetType(node, &type)); insist(type == hipGraphNodeTypeKernel, "prepared graph non-kernel node"); }
        hip_check(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0)); return count;
    }
    void replay(Runtime& r) { hip_check(hipGraphLaunch(executable, r.stream)); hip_check(hipStreamSynchronize(r.stream)); }
};
void check_cpu(const std::string& label, const std::vector<unsigned char>& bytes,
               const Fixture& fixture, uint32_t rows, float alpha, float beta) {
    std::vector<float> actual(bytes.size() / 4); std::memcpy(actual.data(), bytes.data(), bytes.size());
    compare(label, actual, expected(fixture, rows, alpha, beta));
}
void prepared_case(Runtime& r, const Fixture& f, strix_weights blob, strix_weights parts, uint32_t rows) {
    const uint32_t operation = rows <= 8 ? STRIX_OPERATION_GEMV : STRIX_OPERATION_GEMM;
    GuardedPrepared x(r, size_t(rows) * f.i * 4), y_blob(r, size_t(rows) * f.n * 4), y_parts(r, y_blob.bytes);
    upload(x.p, f.parts.at("x").data(), x.bytes, r);
    auto canonical = desc(blob, x.p, y_blob.p, rows, f, rows == 1, 1, 0);
    auto regional = desc(parts, x.p, y_parts.p, rows, f, rows == 1, 1, 0);
    uint64_t bytes = 0, regional_bytes = 0;
    api_check(strix_matmul_workspace_bytes(r.context, &canonical, &bytes));
    api_check(strix_matmul_workspace_bytes(r.context, &regional, &regional_bytes));
    insist(bytes == regional_bytes, "parts changed workspace/dispatch");
    GuardedPrepared workspace(r, bytes);
    canonical.workspace = regional.workspace = workspace.p;
    canonical.workspace_bytes = regional.workspace_bytes = bytes;
    auto ordinary = [&](const strix_matmul_desc& d) {
        api_check(operation == STRIX_OPERATION_GEMV ? strix_gemv(r.context, &d, &r.s) : strix_gemm(r.context, &d, &r.s));
    };
    auto reset = [&](GuardedPrepared& output, float beta) {
        if (beta == 0) hip_check(hipMemsetAsync(output.p, 255, output.bytes, r.stream));
        else upload(output.p, f.parts.at("old").data(), output.bytes, r);
    };
    for (auto scalars : std::vector<std::array<float, 2>>{{1, 0}, {.75f, 0}, {1, .5f}, {-.5f, -.25f}}) {
        canonical.alpha = regional.alpha = scalars[0]; canonical.beta = regional.beta = scalars[1];
        reset(y_blob, scalars[1]); ordinary(canonical); const auto expected_bytes = y_blob.download();
        reset(y_parts, scalars[1]); ordinary(regional); insist(y_parts.download() == expected_bytes, "parts/blob output mismatch");
        PreparedPlan plan(r, regional, operation);
        insist(strix_weights_destroy(parts) == STRIX_ERROR_INVALID_ARGUMENT, "plan failed to retain weights");
        reset(y_parts, scalars[1]); api_check(strix_linear_plan_submit(plan.plan));
        const auto actual = y_parts.download(); insist(actual == expected_bytes, "frozen plan output mismatch");
        check_cpu("prepared rows=" + std::to_string(rows) + " alpha=" + std::to_string(scalars[0]) + " beta=" + std::to_string(scalars[1]), actual, f, rows, scalars[0], scalars[1]);
        plan.close();
    }
    canonical.alpha = regional.alpha = 1; canonical.beta = regional.beta = 0;
    reset(y_blob, 0); ordinary(canonical); const auto expected_bytes = y_blob.download();
    PreparedPlan plan(r, regional, operation);
    {
        PreparedGraph normal_graph, plan_graph;
        const size_t ordinary_nodes = normal_graph.capture(r, [&] { ordinary(regional); });
        const size_t plan_nodes = plan_graph.capture(r, [&] { api_check(strix_linear_plan_submit(plan.plan)); });
        insist(ordinary_nodes == plan_nodes, "plan changed captured kernels");
        reset(y_parts, 0); normal_graph.replay(r); insist(y_parts.download() == expected_bytes, "ordinary graph output");
        reset(y_parts, 0); plan_graph.replay(r); insist(y_parts.download() == expected_bytes, "prepared graph output");
        printf("PREPARED_GRAPH PASS rows=%u kernels=%zu frozen_descriptors=1\n", rows, plan_nodes);
    }
    // Measure observed submission cost at warm, fixed addresses. Host time can
    // include backpressure; event time includes gaps caused by host submission.
    // This is not a DRAM test or a pure validation microbenchmark.
    hipEvent_t begin{}, end{}; hip_check(hipEventCreate(&begin)); hip_check(hipEventCreate(&end));
    const int repeat = rows <= 8 ? 64 : 8;
    std::array<std::vector<double>, 2> host_samples, gpu_samples, completion_samples;
    for (int sweep = -1; sweep < 11; ++sweep) for (int order = 0; order < 2; ++order) {
        const int mode = (sweep + order + 2) % 2;
        hip_check(hipStreamSynchronize(r.stream)); hip_check(hipEventRecord(begin, r.stream));
        const auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < repeat; ++k) {
            if (mode) api_check(strix_linear_plan_submit(plan.plan)); else ordinary(regional);
        }
        const auto submitted = std::chrono::steady_clock::now(); hip_check(hipEventRecord(end, r.stream)); hip_check(hipEventSynchronize(end));
        const auto completed = std::chrono::steady_clock::now(); float elapsed = 0; hip_check(hipEventElapsedTime(&elapsed, begin, end));
        const double host_us = std::chrono::duration<double, std::micro>(submitted - t0).count() / repeat;
        const double completion_us = std::chrono::duration<double, std::micro>(completed - t0).count() / repeat;
        const double gpu_us = elapsed * 1000. / repeat;
        insist(std::isfinite(host_us) && std::isfinite(gpu_us) && host_us > 0 && gpu_us > 0, "prepared timing invalid");
        if (sweep >= 0) {
            host_samples[mode].push_back(host_us); gpu_samples[mode].push_back(gpu_us); completion_samples[mode].push_back(completion_us);
            printf("PREPARED_SAMPLE rows=%u mode=%s sweep=%d repeats=%d host_submit_us=%.9g gpu_event_us=%.9g completion_us=%.9g\n", rows, mode ? "plan" : "validated", sweep, repeat, host_us, gpu_us, completion_us);
        }
        insist(y_parts.download() == expected_bytes, "prepared timed output changed");
    }
    hip_check(hipEventDestroy(begin)); hip_check(hipEventDestroy(end));
    for (int mode = 0; mode < 2; ++mode) {
        auto host = host_samples[mode], gpu = gpu_samples[mode], completion = completion_samples[mode];
        std::sort(host.begin(), host.end()); std::sort(gpu.begin(), gpu.end()); std::sort(completion.begin(), completion.end());
        printf("PREPARED_RESULT I=%u N=%u bits=%u codebook=%u rows=%u mode=%s protocol=hot_fixed_addresses samples=11 repeats=%d host_submit_median_us=%.9g gpu_event_median_us=%.9g completion_median_us=%.9g\n", f.i, f.n, f.bits, f.cb, rows, mode ? "plan" : "validated", repeat, host[5], gpu[5], completion[5]);
    }
    plan.close();
    insist(x.download() == std::vector<unsigned char>(f.parts.at("x").begin(), f.parts.at("x").begin() + x.bytes), "prepared input mutated");
    for (GuardedPrepared* buffer : {&x, &y_blob, &y_parts, &workspace}) buffer->check();
    printf("PREPARED_CASE PASS rows=%u bias=%u parts_vs_blob=1 full_cpu=1 frozen_plan=1 retention=1 graph=1 canaries=1\n", rows, f.bias);
}
void prepared_fixture(Runtime& runtime, const std::string& path) {
    Fixture fixture(path); Weight canonical(runtime, fixture); PartsWeight regional(runtime, fixture);
    printf("PREPARED_FIXTURE path=%s I=%u N=%u bits=%u codebook=%u rows=%u bias=%u\n", path.c_str(), fixture.i, fixture.n, fixture.bits, fixture.cb, fixture.rows, fixture.bias);
    for (uint32_t rows : {1u, 8u, 32u, 128u, 256u}) if (rows <= fixture.rows)
        prepared_case(runtime, fixture, canonical.w, regional.w, rows);
    regional.close(); canonical.close();
}
} // namespace
int main(int argc, char** argv) {
    try {
        setvbuf(stdout, nullptr, _IOLBF, 0);
        insist(argc >= 2, "usage: strix_trellis_prepared_api_probe fixture.stgpu [...]");
        Runtime runtime;
        for (int i = 1; i < argc; ++i) prepared_fixture(runtime, argv[i]);
        runtime.close(); puts("PASS prepared_api_probe"); return 0;
    } catch (const std::exception& error) { fprintf(stderr, "FAIL %s\n", error.what()); return 1; }
}
