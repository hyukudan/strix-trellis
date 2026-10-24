// Grouped strided GEMV (wo_a 4096 -> 1024): exactness against strix_gemv row by row, and a cold schedule sweep.
//   check                 K4/K5, G 4/8, rows 1..8, strided views with poison padding, public plan API
//   tune BITS ROWS...     G 4, 8 weight copies (> MALL), every schedule: bitwise vs the reference, then timing;
//                         baselines: per-group strix_gemv and per-group linear plans over the same copies
#include "strix_trellis.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "../src/dense_r1.hpp"
#include "../src/dense_r1_policy.hpp"

namespace {
void need(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
void hip_ok(hipError_t s) { if (s != hipSuccess) throw std::runtime_error(hipGetErrorString(s)); }
void api_ok(strix_status s, const char* what = "api") { if (s != STRIX_SUCCESS) throw std::runtime_error(std::string(what) + ": " + strix_status_string(s)); }
constexpr int I = 4096, N = 1024;
struct Buffer {
    void* data = nullptr;
    size_t bytes;
    explicit Buffer(size_t n) : bytes(n) { hip_ok(hipMalloc(&data, n)); hip_ok(hipMemset(data, 0xa5, n)); hip_ok(hipDeviceSynchronize()); }
    ~Buffer() { if (data) (void)hipFree(data); }
    Buffer(const Buffer&) = delete;
    float* f() const { return static_cast<float*>(data); }
    void upload(const void* p, size_t n) { need(n <= bytes, "upload"); hip_ok(hipMemcpy(data, p, n, hipMemcpyHostToDevice)); }
    std::vector<unsigned char> download() const {
        std::vector<unsigned char> v(bytes); hip_ok(hipDeviceSynchronize()); hip_ok(hipMemcpy(v.data(), data, bytes, hipMemcpyDeviceToHost)); return v;
    }
};
struct Runtime {
    hipStream_t stream = nullptr;
    strix_context context = nullptr;
    strix_stream descriptor{};
    Runtime() {
        hip_ok(hipSetDevice(0));
        hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        descriptor.struct_size = sizeof(descriptor); descriptor.abi_version = 1; descriptor.hip_stream = stream;
        strix_context_desc c{}; c.struct_size = sizeof(c); c.abi_version = 1; api_ok(strix_context_create(&c, &context));
    }
    ~Runtime() { (void)hipStreamSynchronize(stream); (void)strix_context_destroy(context); (void)hipStreamDestroy(stream); }
};
uint32_t random32(uint32_t& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
struct Weight {
    strix_weights handle = nullptr;
    Buffer codes, suh, svh;
    strix_trellis::DeviceWeights raw{};
    Weight(Runtime& rt, int bits, uint32_t seed, int I = ::I, int N = ::N)
        : codes(size_t(I) * N * bits / 8), suh(size_t(I) * 2), svh(size_t(N) * 2) {
        std::vector<uint32_t> c(size_t(I) * N * bits / 32);
        std::vector<uint16_t> u(I), v(N);
        uint32_t s = 0x67dca521u + seed * 7919u + uint32_t(bits);
        for (auto& x : c) x = random32(s);
        for (auto& x : u) { const auto m = random32(s) & 0x3ffu, g = random32(s) & 1u; x = uint16_t(0x2c00u + m + (g << 15)); }
        for (auto& x : v) { const auto m = random32(s) & 0x3ffu, g = random32(s) & 1u; x = uint16_t(0x3000u + m + (g << 15)); }
        codes.upload(c.data(), codes.bytes); suh.upload(u.data(), suh.bytes); svh.upload(v.data(), svh.bytes);
        raw = {static_cast<const uint32_t*>(codes.data), static_cast<const uint16_t*>(suh.data), static_cast<const uint16_t*>(svh.data),
               nullptr, I, N, bits, 2};
        strix_exl3_metadata_v1 m{}; m.struct_size = sizeof(m); m.transport_version = 1;
        m.bits_x2 = bits * 2; m.rows = N; m.columns = I; m.codebook = STRIX_CODEBOOK_MUL1; m.marker = STRIX_EXL3_MARKER_MUL1;
        m.trellis_bytes = c.size() * 4; m.suh_bytes = u.size() * 2; m.svh_bytes = v.size() * 2; m.bias_bytes = 0;
        m.suh_offset = m.trellis_bytes; m.svh_offset = m.suh_offset + m.suh_bytes; m.bias_offset = 0;
        std::vector<unsigned char> blob(m.svh_offset + m.svh_bytes);
        std::memcpy(blob.data(), c.data(), m.trellis_bytes);
        std::memcpy(blob.data() + m.suh_offset, u.data(), m.suh_bytes);
        std::memcpy(blob.data() + m.svh_offset, v.data(), m.svh_bytes);
        strix_exl3_transport_desc p{}; p.struct_size = sizeof(p); p.abi_version = 1; p.format = STRIX_TRANSPORT_EXL3_CANONICAL;
        p.transport_version = 1; p.data = blob.data(); p.data_bytes = blob.size(); p.metadata = &m; p.metadata_bytes = sizeof(m);
        p.rows = N; p.columns = I;
        api_ok(strix_weights_prepare(rt.context, &p, &rt.descriptor, &handle), "weights prepare");
    }
    ~Weight() { if (handle) (void)strix_weights_destroy(handle); }
};
strix_tensor_desc tensor(const void* p, size_t bytes, int r, int w, size_t stride) {
    strix_tensor_desc t{}; t.struct_size = sizeof(t); t.abi_version = 1; t.data = const_cast<void*>(p); t.buffer_bytes = bytes;
    t.dtype = STRIX_DTYPE_F32; t.rank = 2; t.shape[0] = uint64_t(r); t.shape[1] = uint64_t(w); t.strides[0] = stride; t.strides[1] = 1; return t;
}
uint64_t generic_bytes(Runtime& rt, strix_weights w) {
    uint64_t best = 256;
    for (int r = 1; r <= 8; ++r) {
        strix_matmul_desc d{}; strix_matmul_desc_init(&d); d.weights = w;
        d.input = tensor(reinterpret_cast<void*>(uintptr_t{1} << 40), size_t(r) * I * 4, r, I, I);
        d.output = tensor(reinterpret_cast<void*>(uintptr_t{1} << 41), size_t(r) * N * 4, r, N, N);
        uint64_t b = 0; api_ok(strix_matmul_workspace_bytes(rt.context, &d, &b), "generic workspace query"); best = std::max(best, (b + 255) & ~uint64_t{255});
    }
    return best;
}
// Workspace and strix_gemv for any projection shape (I1 -> N1), rows contiguous.
uint64_t generic_bytes_shape(Runtime& rt, strix_weights w, int I1, int N1) {
    uint64_t best = 256;
    for (int r = 1; r <= 8; ++r) {
        strix_matmul_desc d{}; strix_matmul_desc_init(&d); d.weights = w;
        d.input = tensor(reinterpret_cast<void*>(uintptr_t{1} << 40), size_t(r) * I1 * 4, r, I1, size_t(I1));
        d.output = tensor(reinterpret_cast<void*>(uintptr_t{1} << 41), size_t(r) * N1 * 4, r, N1, size_t(N1));
        uint64_t b = 0; api_ok(strix_matmul_workspace_bytes(rt.context, &d, &b), "generic workspace query");
        best = std::max(best, (b + 255) & ~uint64_t{255});
    }
    return best;
}
void gemv_shape(Runtime& rt, strix_weights w, const float* x, float* y, int rows, int I1, int N1, void* ws, uint64_t wsb) {
    strix_matmul_desc d{}; strix_matmul_desc_init(&d); d.weights = w;
    d.input = tensor(x, size_t(rows) * I1 * 4, rows, I1, size_t(I1));
    d.output = tensor(y, size_t(rows) * N1 * 4, rows, N1, size_t(N1));
    d.workspace = ws; d.workspace_bytes = wsb;
    api_ok(strix_gemv(rt.context, &d, &rt.descriptor), "strix_gemv");
}
// y (contiguous [rows][N]) = strix_gemv over the rows of x (stride xs)
void gemv(Runtime& rt, strix_weights w, const float* x, size_t xs, float* y, int rows, void* ws, uint64_t wsb) {
    strix_matmul_desc d{}; strix_matmul_desc_init(&d); d.weights = w;
    if (rows == 1) xs = I;
    d.input = tensor(x, ((size_t(rows) - 1) * xs + I) * 4, rows, I, xs);
    d.output = tensor(y, size_t(rows) * N * 4, rows, N, N);
    d.workspace = ws; d.workspace_bytes = wsb;
    api_ok(strix_gemv(rt.context, &d, &rt.descriptor), "strix_gemv");
}
strix_grouped_gemv_desc group_desc(const std::vector<strix_weights>& w, int rows, const float* x, size_t xrs, size_t xbytes,
                                   float* y, size_t yrs, size_t ybytes) {
    strix_grouped_gemv_desc d{}; d.struct_size = sizeof(d); d.abi_version = STRIX_TRELLIS_ABI_VERSION;
    d.weights = w.data(); d.group_count = w.size(); d.rows = uint32_t(rows); d.flags = STRIX_GEMV_ROW_INVARIANT_R1;
    d.input = x; d.input_row_stride = xrs; d.input_group_stride = I; d.input_buffer_bytes = xbytes;
    d.output = y; d.output_row_stride = yrs; d.output_group_stride = N; d.output_buffer_bytes = ybytes;
    return d;
}
std::vector<float> inputs(int rows, size_t stride, uint32_t seed) {
    std::vector<float> x(size_t(rows) * stride);
    uint32_t s = 0x1234567u + seed;
    for (auto& v : x) v = float(int(random32(s) % 4097) - 2048) * 0x1p-10f;
    return x;
}

int check() {
    // Every (bits, G): plans for rows 1..8 created up front on ONE shared workspace, then submitted in a shuffled order,
    // twice (plans reused after other row counts ran on the same workspace), each output checked bitwise against
    // strix_gemv of every row alone; padding between rows untouched.
    Runtime rt;
    int failures = 0, cases = 0;
    for (int bits : {4, 5}) {
        for (int G : {4, 8}) {
            std::vector<std::unique_ptr<Weight>> ws;
            std::vector<strix_weights> h;
            for (int g = 0; g < G; ++g) { ws.push_back(std::make_unique<Weight>(rt, bits, uint32_t(g + 1))); h.push_back(ws.back()->handle); }
            const uint64_t gb = generic_bytes(rt, h[0]);
            Buffer gws(gb);
            const size_t xrs = size_t(G) * I + 64, yrs = size_t(G) * N + 32;
            std::vector<std::unique_ptr<Buffer>> xs, ys;
            std::vector<std::vector<unsigned char>> expect(9);
            std::vector<bool> whole_ok(9, true);
            Buffer alone(size_t(N) * 4), whole(size_t(8) * N * 4);
            uint64_t need_bytes = 0;
            for (int rows = 1; rows <= 8; ++rows) {
                xs.push_back(std::make_unique<Buffer>(size_t(rows) * xrs * 4));
                ys.push_back(std::make_unique<Buffer>(size_t(rows) * yrs * 4));
                const auto hx = inputs(rows, xrs, uint32_t(rows * 31 + G + bits));
                xs.back()->upload(hx.data(), xs.back()->bytes);
                auto d = group_desc(h, rows, xs.back()->f(), xrs, xs.back()->bytes, ys.back()->f(), yrs, ys.back()->bytes);
                uint64_t b = 0; api_ok(strix_grouped_gemv_workspace_bytes(rt.context, &d, &b), "grouped workspace query"); need_bytes = std::max(need_bytes, b);
                // expected output: poison everywhere, rows of each group = strix_gemv of that row alone
                std::vector<unsigned char> e(size_t(rows) * yrs * 4, 0xa5);
                Buffer contiguous(size_t(rows) * I * 4);
                for (int g = 0; g < G; ++g) {
                    for (int r = 0; r < rows; ++r)
                        hip_ok(hipMemcpy(contiguous.f() + size_t(r) * I, xs.back()->f() + size_t(r) * xrs + size_t(g) * I, size_t(I) * 4,
                                         hipMemcpyDeviceToDevice));
                    gemv(rt, h[g], contiguous.f(), I, whole.f(), rows, gws.data, gb);
                    hip_ok(hipStreamSynchronize(rt.stream));
                    const auto wv = whole.download();
                    for (int r = 0; r < rows; ++r) {
                        gemv(rt, h[g], xs.back()->f() + size_t(r) * xrs + size_t(g) * I, xrs, alone.f(), 1, gws.data, gb);
                        hip_ok(hipStreamSynchronize(rt.stream));
                        const auto av = alone.download();
                        std::memcpy(e.data() + (size_t(r) * yrs + size_t(g) * N) * 4, av.data(), size_t(N) * 4);
                        if (std::memcmp(wv.data() + size_t(r) * N * 4, av.data(), size_t(N) * 4) != 0) whole_ok[size_t(rows)] = false;
                    }
                }
                expect[size_t(rows)] = std::move(e);
            }
            Buffer work(need_bytes);
            hip_ok(hipMemset(work.data, 0xa5, work.bytes)); hip_ok(hipDeviceSynchronize());  // creation must zero the counters
            std::vector<strix_grouped_gemv_plan> plans(9, nullptr);
            for (int rows = 1; rows <= 8; ++rows) {
                auto& x = *xs[size_t(rows - 1)]; auto& y = *ys[size_t(rows - 1)];
                auto d = group_desc(h, rows, x.f(), xrs, x.bytes, y.f(), yrs, y.bytes);
                d.workspace = work.data; d.workspace_bytes = work.bytes;
                api_ok(strix_grouped_gemv_plan_create(rt.context, &d, &rt.descriptor, &plans[size_t(rows)]), "grouped plan create");
            }
            for (int pass = 0; pass < 2; ++pass)
                for (int rows : {8, 1, 5, 2, 7, 3, 6, 4}) {
                    auto& y = *ys[size_t(rows - 1)];
                    hip_ok(hipMemsetAsync(y.data, 0xa5, y.bytes, rt.stream));
                    api_ok(strix_grouped_gemv_plan_submit(plans[size_t(rows)]));
                    hip_ok(hipStreamSynchronize(rt.stream));
                    const auto out = y.download();
                    const bool ok = out == expect[size_t(rows)];
                    if (!ok) {
                        size_t k = 0; while (out[k] == expect[size_t(rows)][k]) ++k;
                        float av = 0, bv = 0; std::memcpy(&av, out.data() + (k & ~size_t(3)), 4); std::memcpy(&bv, expect[size_t(rows)].data() + (k & ~size_t(3)), 4);
                        std::printf("GROUP_MISMATCH bits=%d G=%d rows=%d pass=%d float=%zu got=%a expected=%a\n", bits, G, rows, pass, k / 4, double(av), double(bv));
                    }
                    ++cases; if (!ok) ++failures;
                    std::printf("GROUP_CHECK bits=%d G=%d rows=%d pass=%d %s (generic R%d rows %s R1)\n", bits, G, rows, pass, ok ? "PASS" : "FAIL", rows,
                                whole_ok[size_t(rows)] ? "==" : "!=");
                }
            for (auto p : plans) if (p) api_ok(strix_grouped_gemv_plan_destroy(p));
        }
    }
    std::printf("GROUP_CHECK_SUMMARY cases=%d failures=%d\n", cases, failures);
    return failures ? 1 : 0;
}

double timed(Runtime& rt, const std::function<void(int)>& call, int copies, int repeats) {
    hipEvent_t b{}, e{}; hip_ok(hipEventCreate(&b)); hip_ok(hipEventCreate(&e));
    std::vector<double> v;
    const char* sw = std::getenv("PROBE_SWEEPS");  // timed sweeps after one discarded sweep (default 7)
    const int sweeps = sw ? std::max(1, std::atoi(sw)) : 7;
    // Whole rotations: every copy is visited the same number of times in each sweep.
    repeats = copies * std::max(1, (repeats + copies - 1) / copies);
    for (int s = -1; s < sweeps; ++s) {
        hip_ok(hipEventRecord(b, rt.stream));
        for (int k = 0; k < repeats; ++k) call(k % copies);
        hip_ok(hipEventRecord(e, rt.stream)); hip_ok(hipEventSynchronize(e));
        float ms = 0; hip_ok(hipEventElapsedTime(&ms, b, e));
        if (s >= 0) v.push_back(ms * 1000.0 / repeats);
    }
    (void)hipEventDestroy(b); (void)hipEventDestroy(e);
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int tune(int bits, const std::vector<int>& row_list) {
    using namespace strix_trellis;
    Runtime rt;
    constexpr int G = 4;
    // Weight copies rotate over PROBE_ROTATE_MB when set (default 8 copies of the group).
    const char* rot = std::getenv("PROBE_ROTATE_MB");
    const size_t group_bytes = size_t(G) * I * N * bits / 8;
    const int copies = rot ? int(std::max<size_t>(8, (size_t(std::atoi(rot)) << 20) / group_bytes + 1)) : 8;
    const int calls = copies * std::max(1, (64 + copies - 1) / copies);
    std::printf("GROUP_ROTATION bits=%d copies=%d bytes=%zu calls_per_sweep=%d visited_bytes_per_sweep=%zu\n", bits, copies,
                size_t(copies) * group_bytes, calls, size_t(calls) * group_bytes);
    std::vector<std::unique_ptr<Weight>> ws;
    for (int c = 0; c < copies * G; ++c) ws.push_back(std::make_unique<Weight>(rt, bits, uint32_t(c + 1)));
    const uint64_t gb = generic_bytes(rt, ws[0]->handle);
    Buffer gws(gb);
    const int splits = dense_r1_group_splits(I, N, bits);
    int failures = 0;
    for (int rows : row_list) {
        const size_t xrs = size_t(G) * I, yrs = size_t(G) * N;
        Buffer x(size_t(rows) * xrs * 4), y(size_t(rows) * yrs * 4), ref(size_t(rows) * yrs * 4), alone(size_t(N) * 4);
        const auto hx = inputs(rows, xrs, uint32_t(rows));
        x.upload(hx.data(), x.bytes);
        // reference rows for copy 0: each row alone through strix_gemv
        std::vector<unsigned char> expect(size_t(rows) * yrs * 4);
        for (int g = 0; g < G; ++g)
            for (int r = 0; r < rows; ++r) {
                gemv(rt, ws[size_t(g)]->handle, x.f() + size_t(r) * xrs + size_t(g) * I, xrs, alone.f(), 1, gws.data, gb);
                hip_ok(hipStreamSynchronize(rt.stream));
                const auto av = alone.download();
                std::memcpy(expect.data() + (size_t(r) * yrs + size_t(g) * N) * 4, av.data(), size_t(N) * 4);
            }
        // baselines
        Buffer xc(size_t(G) * rows * I * 4);
        {
            std::vector<float> staged(size_t(G) * rows * I);
            for (int g = 0; g < G; ++g)
                for (int r = 0; r < rows; ++r)
                    std::memcpy(staged.data() + (size_t(g) * rows + r) * I, hx.data() + size_t(r) * xrs + size_t(g) * I, size_t(I) * 4);
            xc.upload(staged.data(), xc.bytes);
        }
        const double t_gemv = timed(rt, [&](int c) {
            for (int g = 0; g < G; ++g) gemv(rt, ws[size_t(c * G + g)]->handle, xc.f() + size_t(g) * rows * I, I, ref.f() + size_t(g) * rows * N, rows, gws.data, gb);
        }, copies, 64);
        std::vector<strix_linear_plan> lp(size_t(copies) * G);
        for (int c = 0; c < copies; ++c)
            for (int g = 0; g < G; ++g) {
                strix_matmul_desc d{}; strix_matmul_desc_init(&d); d.weights = ws[size_t(c * G + g)]->handle;
                // contiguous staged rows as the engine's staged path uses them: [g][rows][I] -> [g][rows][N]
                d.input = tensor(xc.f() + size_t(g) * rows * I, size_t(rows) * I * 4, rows, I, I);
                d.output = tensor(ref.f() + size_t(g) * rows * N, size_t(rows) * N * 4, rows, N, N);
                d.workspace = gws.data; d.workspace_bytes = gb;
                api_ok(strix_linear_plan_create(rt.context, &d, &rt.descriptor, STRIX_OPERATION_GEMV, &lp[size_t(c * G + g)]));
            }
        const double t_plan = timed(rt, [&](int c) { for (int g = 0; g < G; ++g) api_ok(strix_linear_plan_submit(lp[size_t(c * G + g)])); }, copies, 64);
        for (auto p : lp) (void)strix_linear_plan_destroy(p);
        std::printf("GROUP_BASE bits=%d rows=%d gemv4_us=%.2f linplan4_us=%.2f\n", bits, rows, t_gemv, t_plan);
        // dense-r1 grouped, every schedule
        const size_t member = (size_t(splits) * rows * N * 4 + 255) & ~size_t(255);
        Buffer work(256 * size_t(copies) + member * G * copies);
        hip_ok(hipMemset(work.data, 0, work.bytes)); hip_ok(hipDeviceSynchronize());
        double best = 1e30; std::string best_s;
        // The schedule the public grouped plan selects (dense_r1_policy.hpp).
        const auto pol = dense_r1_group_schedule(I, N, bits, rows, splits);
        const bool pol_pipe = dense_r1_group_pipelined(I, N, bits, rows, splits);
        std::printf("GROUP_POLICY bits=%d rows=%d pipe=%d cols=%d outs=%d u=%d\n", bits, rows, pol_pipe ? 1 : 0, pol.columns,
                    pol.outputs_per_wave, pol.prefetch);
        const bool policy_only = std::getenv("PROBE_POLICY_ONLY") != nullptr;
        std::vector<unsigned char> policy_bytes;  // output of the shipped schedule on copy 0
        int measured = 0, unsupported = 0, exact_count = 0;
        std::printf("GROUP_SCOPE bits=%d rows=%d exact=copy0_before_and_after_timing_vs_rowwise_strix_gemv_and_shipped_schedule_bytes\n", bits, rows);
        for (int pass = 0; pass < 2; ++pass)  // pass 0: the shipped schedule only (records its bytes); pass 1: the sweep
        for (int pipe : {0, 1})
        for (int cols : {16, 32, 64, 128})
            for (int outs : {16, 8})
                for (int u : {1, 2, 4}) {
                    const bool is_policy = pipe == (pol_pipe ? 1 : 0) && cols == pol.columns && outs == pol.outputs_per_wave && u == pol.prefetch;
                    if (pass == 0 && !is_policy) continue;
                    if (pass == 1 && policy_only && !is_policy) continue;
                    std::vector<DenseR1Group> grp(copies);
                    for (int c = 0; c < copies; ++c)
                        for (int g = 0; g < G; ++g) {
                            auto& a = grp[size_t(c)].member[g];
                            a.projection = {ws[size_t(c * G + g)]->raw, x.f() + size_t(g) * I, y.f() + size_t(g) * N, rows > 1 ? xrs : 0,
                                            rows > 1 ? yrs : 0, rows, 1.0f, 0.0f, rows};
                            char* base = static_cast<char*>(work.data);
                            a.counters = reinterpret_cast<uint32_t*>(base + 256 * size_t(c) + 32 * size_t(g));
                            a.partial = reinterpret_cast<float*>(base + 256 * size_t(copies) + member * size_t(c * G + g));
                            a.splits = splits; a.columns_per_block = cols; a.prefetch = u; a.outputs_per_wave = outs; a.pipelined = pipe != 0;
                        }
                    bool supported = false;
                    hip_ok(query_dense_r1_group_resources(grp[0].member[0], 65536, &supported));
                    if (!supported) {
                        if (pass == 1) { ++unsupported; std::printf("GROUP_SCHED bits=%d rows=%d pipe=%d cols=%d outs=%d u=%d unsupported\n", bits, rows, pipe, cols, outs, u); }
                        continue;
                    }
                    if (pass == 0) {
                        hip_ok(hipMemset(y.data, 0xa5, y.bytes)); hip_ok(hipDeviceSynchronize());
                        hip_ok(launch_dense_r1_group_trusted(grp[0], G, rt.stream)); hip_ok(hipStreamSynchronize(rt.stream));
                        policy_bytes = y.download();
                        continue;
                    }
                    hip_ok(hipMemset(y.data, 0xa5, y.bytes)); hip_ok(hipDeviceSynchronize());
                    hip_ok(launch_dense_r1_group_trusted(grp[0], G, rt.stream)); hip_ok(hipStreamSynchronize(rt.stream));
                    const bool exact = y.download() == expect;
                    const double t = timed(rt, [&](int c) { hip_ok(launch_dense_r1_group_trusted(grp[size_t(c)], G, rt.stream)); }, copies, 64);
                    hip_ok(launch_dense_r1_group_trusted(grp[0], G, rt.stream)); hip_ok(hipStreamSynchronize(rt.stream));
                    const auto after = y.download();
                    const bool exact2 = after == expect, same_policy = after == policy_bytes;
                    if (!exact || !exact2 || !same_policy) ++failures;
                    ++measured; exact_count += exact && exact2 && same_policy;
                    // Scope: copy 0, before and after timing, vs row-by-row strix_gemv and vs the shipped schedule's bytes.
                    std::printf("GROUP_SCHED bits=%d rows=%d pipe=%d cols=%d outs=%d u=%d us=%.2f GBps=%.1f %s policy=%d%s\n", bits, rows, pipe, cols, outs, u, t,
                                double(G) * I * N * bits / 8 / (t * 1e3), exact && exact2 && same_policy ? "EXACT" : "DIFF", same_policy ? 1 : 0,
                                is_policy ? " shipped" : "");
                    if (exact && exact2 && same_policy && t < best) { best = t; best_s = std::to_string(cols) + "," + std::to_string(u) + "," + std::to_string(outs) + (pipe ? ",pipe" : ""); }
                }
        std::printf("GROUP_SWEEP bits=%d rows=%d measured=%d unsupported=%d exact=%d policy_recorded=%d\n", bits, rows, measured, unsupported,
                    exact_count, policy_bytes.empty() ? 0 : 1);
        std::printf("GROUP_BEST bits=%d rows=%d sched={%s} us=%.2f vs linplan4 %.2f gemv4 %.2f\n", bits, rows, best_s.c_str(), best, t_plan, t_gemv);
        std::fflush(stdout);
    }
    return failures ? 1 : 0;
}

// Public grouped plan (the shipped schedule policy) on the o_a geometry, cold: one plan per weight copy, whole
// rotations over PROBE_ROTATE_MB, PROBE_SWEEPS sweeps. Copy 0's output is checked byte for byte against row-by-row
// strix_gemv before and after timing, and its FNV-1a bit hash is printed for cross-build comparison.
int tuneg(int bits, const std::vector<int>& row_list) {
    using namespace strix_trellis;
    Runtime rt;
    constexpr int G = 4;
    const char* rot = std::getenv("PROBE_ROTATE_MB");
    const size_t group_bytes = size_t(G) * I * N * bits / 8;
    const int copies = rot ? int(std::max<size_t>(8, (size_t(std::atoi(rot)) << 20) / group_bytes + 1)) : 8;
    const int calls = copies * std::max(1, (64 + copies - 1) / copies);
    std::printf("GPLAN_ROTATION bits=%d copies=%d bytes=%zu calls_per_sweep=%d visited_bytes_per_sweep=%zu\n", bits, copies,
                size_t(copies) * group_bytes, calls, size_t(calls) * group_bytes);
    std::vector<std::unique_ptr<Weight>> ws;
    for (int c = 0; c < copies * G; ++c) ws.push_back(std::make_unique<Weight>(rt, bits, uint32_t(c + 1)));
    const uint64_t gb = generic_bytes(rt, ws[0]->handle);
    Buffer gws(gb), alone(size_t(N) * 4);
    int failures = 0;
    for (int rows : row_list) {
        const size_t xrs = size_t(G) * I, yrs = size_t(G) * N;
        Buffer x(size_t(rows) * xrs * 4), y(size_t(rows) * yrs * 4);
        const auto hx = inputs(rows, xrs, uint32_t(rows));
        x.upload(hx.data(), x.bytes);
        std::vector<unsigned char> expect(size_t(rows) * yrs * 4);
        for (int g = 0; g < G; ++g)
            for (int r = 0; r < rows; ++r) {
                gemv(rt, ws[size_t(g)]->handle, x.f() + size_t(r) * xrs + size_t(g) * I, xrs, alone.f(), 1, gws.data, gb);
                hip_ok(hipStreamSynchronize(rt.stream));
                const auto av = alone.download();
                std::memcpy(expect.data() + (size_t(r) * yrs + size_t(g) * N) * 4, av.data(), size_t(N) * 4);
            }
        std::vector<std::vector<strix_weights>> handles(static_cast<size_t>(copies));
        uint64_t need = 0;
        for (int c = 0; c < copies; ++c) {
            for (int g = 0; g < G; ++g) handles[size_t(c)].push_back(ws[size_t(c * G + g)]->handle);
            auto d = group_desc(handles[size_t(c)], rows, x.f(), xrs, x.bytes, y.f(), yrs, y.bytes);
            uint64_t b = 0; api_ok(strix_grouped_gemv_workspace_bytes(rt.context, &d, &b), "grouped workspace query"); need = std::max(need, b);
        }
        Buffer work(need);
        std::vector<strix_grouped_gemv_plan> plans(size_t(copies), nullptr);
        for (int c = 0; c < copies; ++c) {
            auto d = group_desc(handles[size_t(c)], rows, x.f(), xrs, x.bytes, y.f(), yrs, y.bytes);
            d.workspace = work.data; d.workspace_bytes = work.bytes;
            api_ok(strix_grouped_gemv_plan_create(rt.context, &d, &rt.descriptor, &plans[size_t(c)]), "grouped plan create");
        }
        auto run0 = [&]() {
            hip_ok(hipMemset(y.data, 0xa5, y.bytes)); hip_ok(hipDeviceSynchronize());
            api_ok(strix_grouped_gemv_plan_submit(plans[0])); hip_ok(hipStreamSynchronize(rt.stream));
            return y.download();
        };
        const auto before = run0();
        const double t = timed(rt, [&](int c) { api_ok(strix_grouped_gemv_plan_submit(plans[size_t(c)])); }, copies, 64);
        const auto after = run0();
        const bool exact = before == expect && after == expect;
        if (!exact) ++failures;
        uint64_t h = 1469598103934665603ull;
        for (unsigned char b : after) h = (h ^ b) * 1099511628211ull;
        const auto pol = dense_r1_group_schedule(I, N, bits, rows, dense_r1_group_splits(I, N, bits));
        std::printf("GPLAN bits=%d rows=%d copies=%d sched={%d,%d,%d,%s} us=%.2f GBps=%.1f %s output_fnv=%llx\n", bits, rows, copies,
                    pol.columns, pol.prefetch, pol.outputs_per_wave,
                    dense_r1_group_pipelined(I, N, bits, rows, dense_r1_group_splits(I, N, bits)) ? "pipe" : "nopipe", t,
                    double(group_bytes) / (t * 1e3), exact ? "EXACT" : "DIFF", (unsigned long long)h);
        for (auto p : plans) api_ok(strix_grouped_gemv_plan_destroy(p));
        std::fflush(stdout);
    }
    return failures ? 1 : 0;
}

// Cold schedule sweep of one dense-r1 LEGACY projection (I -> N, bits): copies of the weights rotate over > 96 MB.
// Reference = the public plan (current policy) on copy 0; every schedule must reproduce its bytes.
int tune1(int I1, int N1, int bits, const std::vector<int>& row_list) {
    using namespace strix_trellis;
    Runtime rt;
    const size_t wbytes = size_t(I1) * N1 * bits / 8;
    const char* rot = std::getenv("PROBE_ROTATE_MB");  // weight rotation (default 96 MB; the engine benches use 640)
    const size_t rotate = size_t(rot ? std::atoi(rot) : 96) << 20;
    const int copies = int(std::max<size_t>(4, rotate / wbytes + 1));
    std::vector<std::unique_ptr<Weight>> ws;
    for (int c = 0; c < copies; ++c) ws.push_back(std::make_unique<Weight>(rt, bits, uint32_t(c + 101), I1, N1));
    const int splits = dense_r1_splits(I1, N1, true);
    int failures = 0;
    for (int rows : row_list) {
        Buffer x(size_t(rows) * I1 * 4), y(size_t(rows) * N1 * 4);
        const auto hx = inputs(rows, size_t(I1), uint32_t(rows + 7));
        x.upload(hx.data(), x.bytes);
        // public plans (the shipped policy) for every copy
        std::vector<strix_tp_dense_plan> plans(size_t(copies), nullptr);
        std::vector<std::unique_ptr<Buffer>> pws;
        for (int c = 0; c < copies; ++c) {
            strix_matmul_desc d{}; strix_matmul_desc_init(&d); d.weights = ws[size_t(c)]->handle;
            d.input = tensor(x.f(), x.bytes, rows, I1, size_t(I1));
            d.output = tensor(y.f(), y.bytes, rows, N1, size_t(N1));
            uint64_t b = 0; api_ok(strix_tp_dense_r1_workspace_bytes(rt.context, &d, 0, STRIX_DENSE_R1_LEGACY, &b), "dense-r1 workspace");
            pws.push_back(std::make_unique<Buffer>(std::max<uint64_t>(b, 256)));
            d.workspace = pws.back()->data; d.workspace_bytes = pws.back()->bytes;
            api_ok(strix_tp_dense_plan_create_r1(rt.context, &d, &rt.descriptor, 0, STRIX_DENSE_R1_LEGACY, &plans[size_t(c)]), "dense-r1 plan");
        }
        hip_ok(hipMemset(y.data, 0xa5, y.bytes)); hip_ok(hipDeviceSynchronize());
        api_ok(strix_tp_dense_plan_submit(plans[0]), "submit"); hip_ok(hipStreamSynchronize(rt.stream));
        const auto expect = y.download();
        const double t_policy = timed(rt, [&](int c) { api_ok(strix_tp_dense_plan_submit(plans[size_t(c)]), "submit"); }, copies, 64);
        const auto pol = dense_r1_schedule(I1, N1, bits, rows, splits);
        std::printf("TUNE1_POLICY I=%d N=%d bits=%d rows=%d copies=%d rotation_bytes=%zu sched={%d,%d,%d,%s} us=%.2f GBps=%.1f\n", I1, N1, bits,
                    rows, copies, size_t(copies) * wbytes, pol.columns, pol.prefetch, pol.outputs_per_wave,
                    pol.use_splitk04 ? "splitk04" : "r1", t_policy, double(wbytes) / (t_policy * 1e3));
        for (auto p : plans) (void)strix_tp_dense_plan_destroy(p);
        {
            // Generic strix_gemv over the same copies; the LEGACY plan must reproduce its bytes.
            const uint64_t gb = generic_bytes_shape(rt, ws[0]->handle, I1, N1);
            Buffer gws(std::max<uint64_t>(gb, 256)), yg(y.bytes);
            hip_ok(hipMemset(yg.data, 0xa5, yg.bytes)); hip_ok(hipDeviceSynchronize());
            gemv_shape(rt, ws[0]->handle, x.f(), yg.f(), rows, I1, N1, gws.data, gb);
            hip_ok(hipStreamSynchronize(rt.stream));
            const bool same = yg.download() == expect;
            const double t_gemv = timed(rt, [&](int c) { gemv_shape(rt, ws[size_t(c)]->handle, x.f(), yg.f(), rows, I1, N1, gws.data, gb); },
                                        copies, 64);
            if (!same) ++failures;
            std::printf("TUNE1_BASE I=%d N=%d bits=%d rows=%d gemv_us=%.2f GBps=%.1f plan_equals_gemv=%d\n", I1, N1, bits, rows, t_gemv,
                        double(wbytes) / (t_gemv * 1e3), same ? 1 : 0);
        }
        if (std::getenv("PROBE_POLICY_ONLY")) { std::fflush(stdout); continue; }
        const size_t partial = (size_t(splits) * rows * N1 * 4 + 255) & ~size_t(255), member = partial + ((size_t(N1 / 128) * 4 + 255) & ~size_t(255));
        Buffer work(member * size_t(copies));
        hip_ok(hipMemset(work.data, 0, work.bytes)); hip_ok(hipDeviceSynchronize());
        double best = 1e30; std::string best_s;
        for (int cols : {32, 64, 128})
            for (int outs : {16, 8})
                for (int u : {1, 2, 4}) {
                    if (N1 % cols) continue;
                    std::vector<DenseR1Launch> L(static_cast<size_t>(copies));
                    for (int c = 0; c < copies; ++c) {
                        auto& a = L[size_t(c)];
                        a.projection = {ws[size_t(c)]->raw, x.f(), y.f(), size_t(I1), size_t(N1), rows, 1.0f, 0.0f, rows};
                        char* base = static_cast<char*>(work.data) + member * size_t(c);
                        a.partial = reinterpret_cast<float*>(base); a.counters = reinterpret_cast<uint32_t*>(base + partial);
                        a.splits = splits; a.columns_per_block = cols; a.prefetch = u; a.outputs_per_wave = outs;
                    }
                    bool supported = false;
                    if (query_dense_r1_resources(L[0], 65536, &supported) != hipSuccess || !supported) continue;
                    hip_ok(hipMemset(y.data, 0xa5, y.bytes)); hip_ok(hipDeviceSynchronize());
                    hip_ok(launch_dense_r1_trusted(L[0], rt.stream)); hip_ok(hipStreamSynchronize(rt.stream));
                    const bool exact = y.download() == expect;
                    const double t = timed(rt, [&](int c) { hip_ok(launch_dense_r1_trusted(L[size_t(c)], rt.stream)); }, copies, 64);
                    if (!exact) ++failures;
                    std::printf("TUNE1_SCHED I=%d N=%d bits=%d rows=%d cols=%d outs=%d u=%d us=%.2f GBps=%.1f %s\n", I1, N1, bits, rows, cols,
                                outs, u, t, double(wbytes) / (t * 1e3), exact ? "EXACT" : "DIFF");
                    if (exact && t < best) { best = t; best_s = std::to_string(cols) + "," + std::to_string(u) + "," + std::to_string(outs); }
                }
        std::printf("TUNE1_BEST I=%d N=%d bits=%d rows=%d sched={%s} us=%.2f GBps=%.1f vs policy %.2f\n", I1, N1, bits, rows, best_s.c_str(),
                    best, double(wbytes) / (best * 1e3), t_policy);
        std::fflush(stdout);
    }
    return failures ? 1 : 0;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc >= 2 && std::strcmp(argv[1], "check") == 0) return check();
        if (argc >= 4 && std::strcmp(argv[1], "tuneg") == 0) {
            std::vector<int> rows;
            for (int k = 3; k < argc; ++k) rows.push_back(std::atoi(argv[k]));
            return tuneg(std::atoi(argv[2]), rows);
        }
        if (argc >= 4 && std::strcmp(argv[1], "tune") == 0) {
            std::vector<int> rows;
            for (int k = 3; k < argc; ++k) rows.push_back(std::atoi(argv[k]));
            return tune(std::atoi(argv[2]), rows);
        }
        if (argc >= 6 && std::strcmp(argv[1], "tune1") == 0) {
            std::vector<int> rows;
            for (int k = 5; k < argc; ++k) rows.push_back(std::atoi(argv[k]));
            return tune1(std::atoi(argv[2]), std::atoi(argv[3]), std::atoi(argv[4]), rows);
        }
        std::fprintf(stderr, "usage: %s check | tune BITS ROWS... | tune1 I N BITS ROWS...\n", argv[0]);
        return 2;
    } catch (const std::exception& e) {
        std::printf("GROUP_ERROR %s\n", e.what());
        return 3;
    }
}
