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
    for (int s = -1; s < 7; ++s) {
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
    constexpr int G = 4, copies = 8;
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
        for (int pipe : {0, 1})
        for (int cols : {16, 32, 64, 128})
            for (int outs : {16, 8})
                for (int u : {1, 2, 4}) {
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
                    if (!supported) { std::printf("GROUP_SCHED bits=%d rows=%d pipe=%d cols=%d outs=%d u=%d unsupported\n", bits, rows, pipe, cols, outs, u); continue; }
                    hip_ok(hipMemset(y.data, 0xa5, y.bytes)); hip_ok(hipDeviceSynchronize());
                    hip_ok(launch_dense_r1_group_trusted(grp[0], G, rt.stream)); hip_ok(hipStreamSynchronize(rt.stream));
                    const bool exact = y.download() == expect;
                    const double t = timed(rt, [&](int c) { hip_ok(launch_dense_r1_group_trusted(grp[size_t(c)], G, rt.stream)); }, copies, 64);
                    hip_ok(launch_dense_r1_group_trusted(grp[0], G, rt.stream)); hip_ok(hipStreamSynchronize(rt.stream));
                    const bool exact2 = y.download() == expect;
                    if (!exact || !exact2) ++failures;
                    std::printf("GROUP_SCHED bits=%d rows=%d pipe=%d cols=%d outs=%d u=%d us=%.2f GBps=%.1f %s\n", bits, rows, pipe, cols, outs, u, t,
                                double(G) * I * N * bits / 8 / (t * 1e3), exact && exact2 ? "EXACT" : "DIFF");
                    if (exact && exact2 && t < best) { best = t; best_s = std::to_string(cols) + "," + std::to_string(u) + "," + std::to_string(outs) + (pipe ? ",pipe" : ""); }
                }
        std::printf("GROUP_BEST bits=%d rows=%d sched={%s} us=%.2f vs linplan4 %.2f gemv4 %.2f\n", bits, rows, best_s.c_str(), best, t_plan, t_gemv);
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
        std::printf("TUNE1_POLICY I=%d N=%d bits=%d rows=%d sched={%d,%d,%d,%s} us=%.2f GBps=%.1f\n", I1, N1, bits, rows, pol.columns,
                    pol.prefetch, pol.outputs_per_wave, pol.use_splitk04 ? "splitk04" : "r1", t_policy, double(wbytes) / (t_policy * 1e3));
        for (auto p : plans) (void)strix_tp_dense_plan_destroy(p);
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
