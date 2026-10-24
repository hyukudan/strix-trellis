// Bitwise QA of prepared linear plans against strix_gemv on the same inputs, for every alpha/beta/bias case of the
// one-launch R1 completion (STRIX_TRELLIS_PLAN_FUSED_COMPLETION): poisoned and finite old outputs, plan reuse, and a
// graph captured from a plan submission replayed on another stream after the plan is destroyed. Reuses this project's
// fixture/lifecycle helpers; the comparison is bit-exact, not a tolerance.
#define main strix_correctness_harness_main
#include "../tests/gpu_runtime.hip"
#undef main
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>

namespace {
std::vector<uint32_t> bits_of(Runtime& r, const void* y, size_t n) {
    std::vector<uint32_t> host(n);
    hip_check(hipMemcpyAsync(host.data(), y, n * 4, hipMemcpyDeviceToHost, r.stream));
    hip_check(hipStreamSynchronize(r.stream));
    return host;
}
// Old output contents: NaN payloads (beta == +-0 must never read them) or finite values (beta != 0 reads them).
std::vector<float> old_values(size_t n, bool poison) {
    std::vector<float> v(n);
    for (size_t k = 0; k < n; ++k) {
        if (poison) { uint32_t b = 0x7fc00000u | uint32_t(k & 0x3fffff); std::memcpy(&v[k], &b, 4); }
        else v[k] = float(int(k % 97) - 48) * 0.0625f;
    }
    return v;
}
int failures = 0;
void check(const std::string& label, const std::vector<uint32_t>& want, const std::vector<uint32_t>& got) {
    size_t diff = 0, first = 0;
    for (size_t k = 0; k < want.size(); ++k) if (want[k] != got[k]) { if (!diff++) first = k; }
    std::cout << (diff ? "QA_FAIL " : "QA_PASS ") << label << " values=" << want.size() << " diff=" << diff;
    if (diff) std::cout << " first=" << first << std::hex << " want=" << want[first] << " got=" << got[first] << std::dec;
    std::cout << "\n";
    failures += diff != 0;
}
void run(Runtime& r, const std::string& path) {
    for (bool with_bias : {false, true}) {
        Fixture f(path);
        for (const char* key : {"q", "core", "scaled", "inh", "a", "old"}) f.parts.erase(key);
        if (with_bias) {
            // Deterministic synthetic bias (the fixtures carry none); same weights otherwise.
            std::vector<unsigned char> bias(size_t(f.n) * 4);
            for (uint32_t k = 0; k < f.n; ++k) { float b = float(int(k % 13) - 6) * 0.125f; std::memcpy(&bias[size_t(k) * 4], &b, 4); }
            f.parts["bias"] = bias;
        }
        Weight weight(r, f);
        for (uint32_t rows : {1u, 8u}) {
            if (rows > f.rows) continue;
            Device x(size_t(rows) * f.i * 4), y(size_t(rows) * f.n * 4), yref(size_t(rows) * f.n * 4);
            upload(x.p, f.parts.at("x").data(), size_t(rows) * f.i * 4, r);
            for (float alpha : {1.0f, -0.75f})
            for (float beta : {0.0f, -0.0f, 0.5f}) {
                auto d = desc(weight.w, x.p, y.p, rows, f, false, alpha, beta);
                uint64_t bytes = 0; api_check(strix_matmul_workspace_bytes(r.context, &d, &bytes));
                Device workspace(bytes); d.workspace = workspace.p; d.workspace_bytes = bytes;
                auto dref = d; dref.output.data = yref.p;
                const auto old = old_values(size_t(rows) * f.n, beta == 0.0f);
                auto reset = [&](void* p) { upload(p, old.data(), old.size() * 4, r); };
                std::ostringstream tag;
                tag << "fixture=" << path.substr(path.find_last_of('/') + 1) << " bias=" << with_bias << " rows=" << rows
                    << " alpha=" << alpha << " beta=" << (std::signbit(beta) ? "-" : "") << std::fabs(beta);
                // Reference: strix_gemv (input transform + core + output transform) on its own output buffer.
                reset(yref.p); api_check(strix_gemv(r.context, &dref, &r.s));
                const auto want = bits_of(r, yref.p, old.size());
                strix_linear_plan plan = nullptr;
                api_check(strix_linear_plan_create(r.context, &d, &r.s, STRIX_OPERATION_GEMV, &plan));
                for (int use = 0; use < 3; ++use) {  // plan reuse: counters must be zero again after every launch
                    reset(y.p); api_check(strix_linear_plan_submit(plan));
                    check(tag.str() + " path=plan use=" + std::to_string(use), want, bits_of(r, y.p, old.size()));
                }
                // Graph of a plan submission; replay after plan destruction on a second stream.
                hipGraph_t graph = nullptr; hipGraphExec_t exec = nullptr;
                hip_check(hipStreamSynchronize(r.stream));
                hip_check(hipStreamBeginCapture(r.stream, hipStreamCaptureModeRelaxed));
                const auto status = strix_linear_plan_submit(plan);
                hip_check(hipStreamEndCapture(r.stream, &graph));
                api_check(status);
                hip_check(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
                api_check(strix_linear_plan_destroy(plan));
                hipStream_t other = nullptr; hip_check(hipStreamCreateWithFlags(&other, hipStreamNonBlocking));
                for (int replay = 0; replay < 2; ++replay) {
                    reset(y.p); hip_check(hipStreamSynchronize(r.stream));
                    hip_check(hipGraphLaunch(exec, other)); hip_check(hipStreamSynchronize(other));
                    check(tag.str() + " path=plan_graph replay=" + std::to_string(replay), want, bits_of(r, y.p, old.size()));
                }
                hip_check(hipStreamDestroy(other));
                hip_check(hipGraphExecDestroy(exec)); hip_check(hipGraphDestroy(graph));
            }
        }
    }
}
}
int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("Usage: plan_completion_qa FIXTURE...");
        std::cout << std::setprecision(10) << std::unitbuf;
        Runtime r;
        for (int k = 1; k < argc; ++k) run(r, argv[k]);
        r.close();
        std::cout << (failures ? "QA_RESULT FAIL failures=" : "QA_RESULT PASS failures=") << failures << "\n";
        return failures ? 1 : 0;
    } catch (const std::exception& e) { std::cerr << "FAIL " << e.what() << "\n"; return 2; }
}
