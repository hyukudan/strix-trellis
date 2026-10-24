// Prefill WMMA core: tiled vs current, bitwise and timing.
//   prefill_tiled_bench check                 decoder identity (all 65,536 states) + synthetic bitwise grid
//   prefill_tiled_bench sweep [M...] -- FIX... real fixtures (STGPU001), per M: current core, every tile,
//                                             probes (decode / MMA decomposition), full pipeline with and
//                                             without the fused output transform. Every tiled output is
//                                             compared bit for bit with the current one.
// Numerics: current = launch_wmma_core with launch_linear's split count.
#include "../src/backend.hpp"
#include "../src/gemm.hpp"
#include "../src/wmma_tiled.hpp"
#include "../src/trellis_device.hpp"
#include "../src/packed_decoder.hpp"
#define input_transform bench_input_transform
#define output_transform bench_output_transform
#include "../src/transforms.hpp"
#undef input_transform
#undef output_transform
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix_trellis {
hipError_t launch_wmma_tiled_probe(const DeviceWeights&, const _Float16*, float*, int, int, hipStream_t, int, int);
}
using namespace strix_trellis;
#define HIP(x) do { hipError_t s_ = (x); if (s_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(s_)); std::exit(1); } } while (0)

static uint64_t fnv(const void* p, size_t n) {
    const unsigned char* b = static_cast<const unsigned char*>(p); uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
struct Buf {
    void* p = nullptr; size_t n = 0;
    explicit Buf(size_t bytes) : n(bytes) { if (bytes) HIP(hipMalloc(&p, bytes)); }
    ~Buf() { if (p) (void)hipFree(p); }
    template<class T> T* as() const { return static_cast<T*>(p); }
};
static uint32_t rng = 0x9e3779b9u;
static uint32_t next() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

// ---- decoder identity: codebook_value(.,2) rounded to FP16 vs the packed mul1 decoder
__global__ void decoder_identity(uint32_t* mismatches) {
    uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= 65536u) return;
    _Float16 a = _Float16(codebook_value(s, 2));
    __half2 p = packed_decode::mul1_half2(s, s ^ 0x5a5au);
    _Float16 b = __builtin_bit_cast(_Float16, uint16_t(__builtin_bit_cast(uint32_t, p) & 65535u));
    _Float16 c = _Float16(codebook_value(s ^ 0x5a5au, 2));
    _Float16 d = __builtin_bit_cast(_Float16, uint16_t(__builtin_bit_cast(uint32_t, p) >> 16));
    if (__builtin_bit_cast(uint16_t, a) != __builtin_bit_cast(uint16_t, b) ||
        __builtin_bit_cast(uint16_t, c) != __builtin_bit_cast(uint16_t, d)) atomicAdd(mismatches, 1u);
}

struct Weights {
    Buf packed, suh, svh, bias; DeviceWeights w{};
    Weights(int i, int n, int bits, int cb, const unsigned char* trellis, const unsigned char* u,
            const unsigned char* v, const unsigned char* b)
        : packed(size_t(i) * n * bits / 8), suh(2 * size_t(i)), svh(2 * size_t(n)), bias(b ? 4 * size_t(n) : 0) {
        HIP(hipMemcpy(packed.p, trellis, packed.n, hipMemcpyHostToDevice));
        HIP(hipMemcpy(suh.p, u, suh.n, hipMemcpyHostToDevice));
        HIP(hipMemcpy(svh.p, v, svh.n, hipMemcpyHostToDevice));
        if (b) HIP(hipMemcpy(bias.p, b, bias.n, hipMemcpyHostToDevice));
        w = DeviceWeights{packed.as<uint32_t>(), suh.as<uint16_t>(), svh.as<uint16_t>(), b ? bias.as<float>() : nullptr, i, n, bits, cb};
    }
};
static std::vector<unsigned char> random_bytes(size_t n) { std::vector<unsigned char> v(n); for (auto& x : v) x = (unsigned char)next(); return v; }
static std::vector<unsigned char> random_scales(size_t n) {
    std::vector<unsigned char> v(2 * n);
    for (size_t k = 0; k < n; ++k) { uint16_t h = (next() & 1) ? 0x3c00 : 0xbc00; std::memcpy(&v[2 * k], &h, 2); }
    return v;
}
static void random_input(Buf& x, size_t count) {   // FP32 activations, a few large outliers
    std::vector<float> h(count);
    for (auto& v : h) { float u = (int(next() % 20001) - 10000) / 4000.0f; if (next() % 997 == 0) u *= 40; v = u; }
    HIP(hipMemcpy(x.p, h.data(), count * 4, hipMemcpyHostToDevice));
}
static std::vector<unsigned char> download(const Buf& b, size_t bytes) {
    std::vector<unsigned char> h(bytes); HIP(hipMemcpy(h.data(), b.p, bytes, hipMemcpyDeviceToHost)); return h;
}

template<class F> static double time_us(F launch, int reps = 9) {
    hipEvent_t a, b; HIP(hipEventCreate(&a)); HIP(hipEventCreate(&b));
    launch(); launch(); HIP(hipDeviceSynchronize());
    std::vector<float> t;
    for (int r = 0; r < reps; ++r) {
        HIP(hipEventRecord(a)); launch(); HIP(hipEventRecord(b)); HIP(hipEventSynchronize(b));
        float ms; HIP(hipEventElapsedTime(&ms, a, b)); t.push_back(ms * 1000);
    }
    HIP(hipEventDestroy(a)); HIP(hipEventDestroy(b));
    std::sort(t.begin(), t.end()); return t[t.size() / 2];
}


static int failures = 0;
// TRPF_TILE_LIST="0,4" restricts the timed tiles (bitwise checks always cover every tile).
static bool tile_selected(int tile) {
    const char* list = std::getenv("TRPF_TILE_LIST");
    if (!list || !*list) return true;
    std::string l = std::string(",") + list + ",";
    return l.find("," + std::to_string(tile) + ",") != std::string::npos;
}
struct Pipeline {
    // Reproduces launch_linear(prefer_wmma) stage by stage so each stage can be timed.
    const DeviceWeights* w; int rows, splits; const float* x; Buf transformed, partial, out;
    float alpha = 1.0f, beta = 0.0f;
    Pipeline(const DeviceWeights& w_, int rows_, const float* x_, int splits_ = -1)
        : w(&w_), rows(rows_), splits(splits_ > 0 ? splits_ : linear_split_count(w_, rows_)), x(x_),
          transformed(size_t(rows_) * w_.input * 2), partial(size_t(splits) * rows_ * w_.output * 4), out(size_t(rows_) * w_.output * 4) {}
    void in() { bench_input_transform<<<(size_t(rows) * (w->input / 128) + 3) / 4, 128>>>(x, w->suh, transformed.as<_Float16>(), w->input, rows); HIP(hipGetLastError()); }
    void core_old() { HIP(launch_wmma_core(*w, transformed.as<_Float16>(), partial.as<float>(), rows, splits, nullptr)); }
    hipError_t core_new(int tile) { return launch_wmma_tiled(*w, transformed.as<_Float16>(), partial.as<float>(), rows, splits, nullptr, nullptr, tile); }
    void outx() { bench_output_transform<<<(size_t(rows) * (w->output / 128) + 3) / 4, 128>>>(partial.as<float>(), w->svh, w->bias, out.as<float>(), w->output, rows, splits, alpha, beta); HIP(hipGetLastError()); }
    hipError_t fused(int tile) { WmmaFusedOutput f{out.as<float>(), alpha, beta}; return launch_wmma_tiled(*w, transformed.as<_Float16>(), nullptr, rows, splits, nullptr, &f, tile); }
};

static void compare(const char* what, const std::vector<unsigned char>& a, const std::vector<unsigned char>& b, const std::string& cell) {
    if (a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size()) == 0) return;
    size_t diff = a.size() == b.size() ? 0 : 1, first = SIZE_MAX;
    for (size_t k = 0; k + 4 <= std::min(a.size(), b.size()); k += 4) if (std::memcmp(&a[k], &b[k], 4)) { ++diff; if (first == SIZE_MAX) first = k / 4; }
    if (diff) { ++failures; std::printf("{\"FAIL\":\"%s\",%s,\"diff_words\":%zu,\"first\":%zu}\n", what, cell.c_str(), diff, first); }
}
// Old-output pattern for beta != 0: signed zeros, subnormals, huge (overflow), inf, nan, ordinary.
static std::vector<unsigned char> old_pattern(size_t count) {
    static const uint32_t special[] = {0x00000000u, 0x80000000u, 0x00000001u, 0x807fffffu, 0x7f7fffffu, 0xff7fffffu,
                                       0x7f800000u, 0xff800000u, 0x7fc00000u, 0x3f800000u, 0xc0490fdbu};
    std::vector<unsigned char> v(count * 4);
    for (size_t k = 0; k < count; ++k) {
        uint32_t u;
        if (next() % 4 == 0) u = special[next() % 11];          // includes an explicit +0
        else { float f = (int(next() % 20001) - 10000) / 333.0f; std::memcpy(&u, &f, 4); }
        std::memcpy(&v[4 * k], &u, 4);
    }
    return v;
}
static void exceptional_input(Buf& x, size_t count) {
    std::vector<float> h(count);
    const float special[] = {0.0f, -0.0f, 1e-40f, -1e-40f, 65504.0f, 70000.0f, -1e30f, INFINITY, -INFINITY, NAN};
    for (auto& v : h) v = (next() % 50 == 0) ? special[next() % 10] : (int(next() % 20001) - 10000) / 4000.0f;
    HIP(hipMemcpy(x.p, h.data(), count * 4, hipMemcpyHostToDevice));
}

// Bit-for-bit: every tile's partial vs the current core; fused output vs core + output_transform,
// for (alpha, beta) = (1, 0) and (0.75, -1.5) with an exceptional old output.
static void bitwise_cell(const DeviceWeights& w, int rows, const float* x, const std::string& tag, int tiles, int splits = -1) {
    Pipeline p(w, rows, x, splits);
    p.in(); p.core_old(); HIP(hipDeviceSynchronize());
    auto ref_partial = download(p.partial, p.partial.n);
    char cell[256]; std::snprintf(cell, sizeof cell, "\"cell\":\"%s\",\"M\":%d,\"I\":%d,\"N\":%d,\"bits\":%d,\"cb\":%d,\"splits\":%d", tag.c_str(), rows, w.input, w.output, w.bits, w.codebook, p.splits);
    int checked = 0, fused_checked = 0;
    auto old = old_pattern(size_t(rows) * w.output);
    struct AB { float a, b; } abs[] = {{1.0f, 0.0f}, {0.75f, -1.5f}};
    std::vector<std::vector<unsigned char>> ref_out;
    for (auto ab : abs) {
        p.alpha = ab.a; p.beta = ab.b;
        HIP(hipMemcpy(p.out.p, old.data(), old.size(), hipMemcpyHostToDevice));
        p.outx(); HIP(hipDeviceSynchronize()); ref_out.push_back(download(p.out, p.out.n));
    }
    for (int tile = 0; tile < tiles; ++tile) {
        HIP(hipMemset(p.partial.p, 0x7f, p.partial.n));
        hipError_t s = p.core_new(tile);
        if (s == hipErrorNotSupported) continue;
        HIP(s); HIP(hipDeviceSynchronize());
        compare("partial", download(p.partial, p.partial.n), ref_partial, std::string(cell) + ",\"tile\":" + std::to_string(tile)); ++checked;
        if (wmma_tiled_fuses_output(w, rows, p.splits, tile)) {
            for (size_t k = 0; k < 2; ++k) {
                p.alpha = abs[k].a; p.beta = abs[k].b;
                HIP(hipMemcpy(p.out.p, old.data(), old.size(), hipMemcpyHostToDevice));
                HIP(p.fused(tile)); HIP(hipDeviceSynchronize());
                compare("fused_output", download(p.out, p.out.n), ref_out[k], std::string(cell) + ",\"tile\":" + std::to_string(tile) + ",\"beta\":" + std::to_string(abs[k].b)); ++fused_checked;
            }
        }
    }
    std::printf("{\"check\":\"bitwise\",%s,\"tiles_compared\":%d,\"fused_compared\":%d,\"ref_fnv\":\"%016llx\"}\n", cell, checked, fused_checked, (unsigned long long)fnv(ref_out[0].data(), ref_out[0].size()));
}

static std::unique_ptr<Weights> synthetic(int i, int n, int bits, int cb, bool with_bias) {
    auto t = random_bytes(size_t(i) * n * bits / 8), u = random_scales(i), v = random_scales(n), b = random_bytes(4 * size_t(n));
    for (size_t k = 0; k < b.size(); k += 4) { float f = (int(next() % 2001) - 1000) / 1000.0f; std::memcpy(&b[k], &f, 4); }
    return std::make_unique<Weights>(i, n, bits, cb, t.data(), u.data(), v.data(), with_bias ? b.data() : nullptr);
}
static void check_mode() {
    Buf mism(4); HIP(hipMemset(mism.p, 0, 4));
    decoder_identity<<<256, 256>>>(mism.as<uint32_t>()); HIP(hipDeviceSynchronize());
    uint32_t m = 0; HIP(hipMemcpy(&m, mism.p, 4, hipMemcpyDeviceToHost));
    std::printf("{\"check\":\"decoder_identity_mul1\",\"states\":65536,\"mismatches\":%u}\n", m);
    if (m) ++failures;
    // All bits/codebooks, odd row counts, policy splits; then forced odd / empty splits (tiles_k = 24 < 29).
    for (int bits = 1; bits <= 8; ++bits) for (int cb = 0; cb < 3; ++cb) {
        auto W = synthetic(384, 256, bits, cb, true);
        for (int rows : {32, 77, 160}) {
            Buf x(size_t(rows) * 384 * 4); random_input(x, size_t(rows) * 384);
            bitwise_cell(W->w, rows, x.as<float>(), "synthetic", 8);
        }
        for (int splits : {1, 3, 7, 29}) {
            Buf x(size_t(77) * 384 * 4); exceptional_input(x, size_t(77) * 384);
            bitwise_cell(W->w, 77, x.as<float>(), "synthetic_forced_splits_exceptional_x", 8, splits);
        }
    }
    // One split (fused epilogue reachable), partial last M block, with and without bias, exceptional inputs.
    for (int bits : {4, 5, 6}) for (int with_bias : {0, 1}) {
        auto W = synthetic(256, 8192, bits, 2, with_bias);
        Buf x(size_t(320) * 256 * 4); exceptional_input(x, size_t(320) * 256);
        bitwise_cell(W->w, 320, x.as<float>(), "synthetic_split1", 8);
    }
    // Geometries where the current core dispatches wmma_packed / the medium M64 padded variant.
    struct Geo { int i, n, bits; std::vector<int> rows; } geos[] = {
        {1280, 32768, 5, {64, 128, 256}}, {8192, 5120, 5, {64, 256}}, {5120, 2304, 5, {64, 128}},
        {2304, 5120, 5, {128, 256}}, {4096, 1024, 5, {256, 300}}, {5120, 1280, 6, {256, 96}}};
    for (auto& g : geos) {
        auto W = synthetic(g.i, g.n, g.bits, 2, true);
        for (int rows : g.rows) {
            Buf x(size_t(rows) * g.i * 4); random_input(x, size_t(rows) * g.i);
            bitwise_cell(W->w, rows, x.as<float>(), "dispatch_geometry", 8);
        }
    }
}

struct Fixture {
    std::map<std::string, std::vector<unsigned char>> parts; uint32_t i, n, bits, cb, bias;
    explicit Fixture(const std::string& path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) throw std::runtime_error("fixture " + path);
        std::vector<unsigned char> bytes(size_t(f.tellg())); f.seekg(0); f.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
        if (bytes.size() < 16 || std::memcmp(bytes.data(), "STGPU001", 8)) throw std::runtime_error("header " + path);
        auto le = [&](size_t pos, int nb) { uint64_t v = 0; for (int k = 0; k < nb; ++k) v |= uint64_t(bytes[pos + k]) << (8 * k); return v; };
        size_t pos = 16; uint64_t count = le(12, 4);
        for (uint64_t k = 0; k < count; ++k) {
            std::string tag(reinterpret_cast<char*>(bytes.data() + pos), 8); tag.resize(tag.find('\0') == std::string::npos ? 8 : tag.find('\0'));
            uint64_t len = le(pos + 8, 8); pos += 16;
            parts.emplace(tag, std::vector<unsigned char>(bytes.begin() + pos, bytes.begin() + pos + len)); pos += len;
        }
        const auto& m = parts.at("meta");
        auto rd = [&](int off) { uint32_t v; std::memcpy(&v, m.data() + off, 4); return v; };
        i = rd(0); n = rd(4); bits = rd(8); cb = rd(12); bias = rd(24);
    }
};

static void sweep_fixture(const std::string& path, const std::vector<int>& ms, int tiles, bool cold) {
    Fixture f(path);
    std::string name = path.substr(path.find_last_of('/') + 1);
    auto make = [&] { return std::make_unique<Weights>(f.i, f.n, f.bits, f.cb, f.parts.at("trellis").data(), f.parts.at("suh").data(),
                                                       f.parts.at("svh").data(), f.bias ? f.parts.at("bias").data() : nullptr); };
    std::vector<std::unique_ptr<Weights>> copies; copies.push_back(make());
    // Cold: rotate over disjoint copies of >= 256 MiB of packed weights (README method).
    if (cold) { size_t bytes = size_t(f.i) * f.n * f.bits / 8; while (copies.size() * bytes < (256ull << 20)) copies.push_back(make()); }
    const DeviceWeights& w = copies[0]->w;
    for (int rows : ms) {
        Buf x(size_t(rows) * w.input * 4); random_input(x, size_t(rows) * w.input);
        bitwise_cell(w, rows, x.as<float>(), name, tiles);
        // One workspace for all copies (as an engine would); only the weights rotate.
        Pipeline p(w, rows, x.as<float>());
        size_t turn = 0;
        auto rot = [&]() -> Pipeline& { p.w = &copies[turn++ % copies.size()]->w; return p; };
        auto home = [&] { p.w = &copies[0]->w; };
        const double flop = 2.0 * rows * w.input * w.output;
        // Every timing below goes through rot(): warm = one copy, cold = disjoint copies rotated per call.
        double t_in = time_us([&] { rot().in(); }); home();
        double t_old = time_us([&] { rot().core_old(); }); home();
        double t_out = time_us([&] { rot().outx(); }); home();
        // Consecutive in -> core -> out in one timed unit; cold rotates the weight copy per call.
        double t_pipe_old = time_us([&] { Pipeline& q = rot(); q.in(); q.core_old(); q.outx(); }); home();
        std::printf("{\"time\":\"current\",\"mode\":\"%s\",\"fixture\":\"%s\",\"M\":%d,\"I\":%d,\"N\":%d,\"bits\":%d,\"splits\":%d,\"cold_copies\":%zu,\"input_us\":%.2f,\"core_us\":%.2f,\"output_us\":%.2f,\"pipeline_us\":%.2f,\"core_TOPS\":%.3f,\"pipeline_TOPS\":%.3f}\n",
                    cold ? "cold" : "warm", name.c_str(), rows, w.input, w.output, w.bits, p.splits, copies.size(), t_in, t_old, t_out, t_pipe_old, flop / t_old / 1e6, flop / t_pipe_old / 1e6);
        for (int tile = 0; tile < tiles; ++tile) {
            if (!tile_selected(tile)) continue;
            if (p.core_new(tile) == hipErrorNotSupported) continue;
            double t_new = time_us([&] { HIP(rot().core_new(tile)); }); home();
            const bool can_fuse = wmma_tiled_fuses_output(w, rows, p.splits, tile);
            double t_fused = can_fuse ? time_us([&] { HIP(rot().fused(tile)); }) : -1; home();
            double t_pipe = time_us([&] { Pipeline& q = rot(); q.in(); if (can_fuse) HIP(q.fused(tile)); else { HIP(q.core_new(tile)); q.outx(); } }); home();
            double probe[4] = {t_new, -1, -1, -1};
#ifdef STRIX_TRELLIS_WMMA_PROBE
            for (int pr = 1; pr <= 3; ++pr)
                if (launch_wmma_tiled_probe(w, p.transformed.as<_Float16>(), p.partial.as<float>(), rows, p.splits, nullptr, tile, pr) == hipSuccess)
                    probe[pr] = time_us([&] { Pipeline& q = rot(); HIP(launch_wmma_tiled_probe(*q.w, q.transformed.as<_Float16>(), q.partial.as<float>(), rows, q.splits, nullptr, tile, pr)); }), home();
#endif
            std::printf("{\"time\":\"tiled\",\"mode\":\"%s\",\"fixture\":\"%s\",\"M\":%d,\"tile\":%d,\"core_us\":%.2f,\"fused_core_out_us\":%.2f,\"var_no_decode_us\":%.2f,\"var_no_wmma_us\":%.2f,\"var_wmma_frag_only_us\":%.2f,\"core_TOPS\":%.3f,\"pipeline_us\":%.2f,\"pipeline_TOPS\":%.3f,\"speedup_pipeline\":%.3f}\n",
                        cold ? "cold" : "warm", name.c_str(), rows, tile, t_new, t_fused, probe[1], probe[2], probe[3], flop / t_new / 1e6, t_pipe, flop / t_pipe / 1e6, t_pipe_old / t_pipe);
        }
        std::fflush(stdout);
    }
}

// Full operator (launch_linear, prefer_wmma) as an engine calls strix_gemm. The tiled path is chosen
// by the library policy and the STRIX_TRELLIS_WMMA_TILED / _FUSED flags of THIS process, so an A/B is
// two processes; the output hash per cell must match across them. Cold: rotating weight copies.
static void linear_cell(const std::string& name, std::vector<std::unique_ptr<Weights>>& copies, int rows, uint32_t seed) {
    const DeviceWeights& w = copies[0]->w;
    rng = seed;
    Buf x(size_t(rows) * w.input * 4); random_input(x, size_t(rows) * w.input);
    Buf y(size_t(rows) * w.output * 4), ws(linear_workspace_bytes(w, rows) + 256);
    void* wsa = reinterpret_cast<void*>((uintptr_t(ws.p) + 255) & ~uintptr_t(255));
    auto old = old_pattern(size_t(rows) * w.output);           // beta != 0 exercised too
    HIP(hipMemcpy(y.p, old.data(), old.size(), hipMemcpyHostToDevice));
    HIP(launch_linear(w, x.as<float>(), y.as<float>(), rows, wsa, nullptr, 0.75f, -1.5f, true)); HIP(hipDeviceSynchronize());
    auto out = download(y, y.n);
    HIP(hipMemcpy(y.p, old.data(), old.size(), hipMemcpyHostToDevice));
    HIP(launch_linear(w, x.as<float>(), y.as<float>(), rows, wsa, nullptr, 1.0f, 0.0f, true)); HIP(hipDeviceSynchronize());
    auto out1 = download(y, y.n);
    size_t turn = 0;
    double t = time_us([&] { HIP(launch_linear(copies[turn++ % copies.size()]->w, x.as<float>(), y.as<float>(), rows, wsa, nullptr, 1.0f, 0.0f, true)); }, 11);
    std::printf("{\"linear\":\"%s\",\"M\":%d,\"I\":%d,\"N\":%d,\"bits\":%d,\"splits\":%d,\"tiled\":%d,\"fused\":%d,\"copies\":%zu,\"us\":%.2f,\"TOPS\":%.3f,\"fnv_ab\":\"%016llx\",\"fnv_10\":\"%016llx\"}\n",
                name.c_str(), rows, w.input, w.output, w.bits, linear_split_count(w, rows), int(wmma_tiled_enabled()), int(wmma_tiled_fused_output_enabled()),
                copies.size(), t, 2.0 * rows * w.input * w.output / t / 1e6, (unsigned long long)fnv(out.data(), out.size()), (unsigned long long)fnv(out1.data(), out1.size()));
    std::fflush(stdout);
}
static void linear_mode(const std::vector<int>& ms, const std::vector<std::string>& fx, bool synth) {
    // Warm by default (compute-bound: warm and cold sweeps agree); TRPF_COLD=1 rotates >= 256 MiB of copies.
    const bool cold = std::getenv("TRPF_COLD") && std::strcmp(std::getenv("TRPF_COLD"), "1") == 0;
    auto rotate = [cold](std::vector<std::unique_ptr<Weights>>& c, auto make, size_t bytes) {
        c.push_back(make()); while (cold && c.size() * bytes < (256ull << 20)) c.push_back(make()); };
    for (auto& path : fx) {
        Fixture f(path); std::string name = path.substr(path.find_last_of('/') + 1);
        auto make = [&] { return std::make_unique<Weights>(f.i, f.n, f.bits, f.cb, f.parts.at("trellis").data(), f.parts.at("suh").data(),
                                                           f.parts.at("svh").data(), f.bias ? f.parts.at("bias").data() : nullptr); };
        std::vector<std::unique_ptr<Weights>> copies; rotate(copies, make, size_t(f.i) * f.n * f.bits / 8);
        for (int m : ms) linear_cell(name, copies, m, 0x1234u + m);
    }
    if (synth) {   // README per-rank (TP2) shapes; deterministic synthetic weights (same bits in every process)
        struct S { const char* name; int i, n; } shapes[] = {{"rank_q_b", 1280, 16384}, {"rank_o_b", 4096, 5120},
            {"rank_shared_w13", 5120, 1152}, {"rank_shared_w2", 1152, 5120}, {"rank_q_a", 5120, 1280}, {"rank_o_a_slice", 4096, 1024}};
        for (int bits : {4, 5}) for (auto& sh : shapes) {
            rng = 0xabcdef01u + bits * 977 + sh.i;
            auto t = random_bytes(size_t(sh.i) * sh.n * bits / 8), u = random_scales(sh.i), v = random_scales(sh.n);
            auto make = [&] { return std::make_unique<Weights>(sh.i, sh.n, bits, 2, t.data(), u.data(), v.data(), nullptr); };
            std::vector<std::unique_ptr<Weights>> copies; rotate(copies, make, size_t(sh.i) * sh.n * bits / 8);
            for (int m : ms) linear_cell(std::string(sh.name) + "_k" + std::to_string(bits), copies, m, 0x777u + m);
        }
    }
}

int main(int argc, char** argv) {
    hipDeviceProp_t prop; HIP(hipGetDeviceProperties(&prop, 0));
    std::printf("{\"device\":\"%s\",\"arch\":\"%s\",\"cus\":%d}\n", prop.name, prop.gcnArchName, prop.multiProcessorCount);
    std::string mode = argc > 1 ? argv[1] : "check";
    if (mode == "check") check_mode();
    else if (mode == "linear" || mode == "linear-synth") {
        std::vector<int> ms; std::vector<std::string> fx; bool files = false;
        for (int k = 2; k < argc; ++k) { std::string a = argv[k]; if (a == "--") files = true; else if (files) fx.push_back(a); else ms.push_back(std::atoi(a.c_str())); }
        if (ms.empty()) ms = {128, 256, 512, 1024, 2048, 4096};
        linear_mode(ms, fx, mode == "linear-synth");
    }
    else if (mode == "sweep" || mode == "cold") {
        std::vector<int> ms; std::vector<std::string> fx; bool files = false;
        for (int k = 2; k < argc; ++k) { std::string a = argv[k]; if (a == "--") files = true; else if (files) fx.push_back(a); else ms.push_back(std::atoi(a.c_str())); }
        if (ms.empty()) ms = {128, 256, 512, 1024, 2048, 4096};
        const char* t = std::getenv("TRPF_TILES"); int tiles = t ? std::atoi(t) : 8;
        for (auto& p : fx) sweep_fixture(p, ms, tiles, mode == "cold");
    }
    std::printf("{\"result\":\"%s\",\"failures\":%d}\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 2 : 0;
}
