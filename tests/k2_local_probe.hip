// Opt-in qualification of the experimental intra-block K2 reduction. Reuse this
// project's independent fixture parser, guarded storage and CPU trellis oracle.
#define main strix_k2_original_probe_main
#include "k2_core_probe.hip"
#undef main
#include "../src/gemv_k2_local.hpp"
#include "../src/k2_local_decode.hpp"
#include <chrono>

namespace {
constexpr std::array<K2LocalSchedule, 5> schedules = {
    K2LocalSchedule::U1, K2LocalSchedule::U2, K2LocalSchedule::U4,
    K2LocalSchedule::Pipeline2, K2LocalSchedule::Pipeline4};
__global__ void local_books_probe(uint32_t* out) {
    const uint32_t s = blockIdx.x * 256 + threadIdx.x;
    uint32_t states[8], hashes[8];
    for (int j = 0; j < 8; ++j)
        states[j] = ((j & 1) ? 65535 - s : s) | (uint32_t(0xa55a + j) << 16);
    k2_local_kernel::staged(states, hashes);
    for (int j = 0; j < 8; ++j) out[s * 12 + j] = hashes[j];
    for (int j = 0; j < 4; ++j)
        out[s * 12 + 8 + j] = __builtin_bit_cast(uint32_t,
            k2_local_kernel::sad_pair(hashes[2 * j], hashes[2 * j + 1]));
}
__global__ void local_reader_probe(const uint32_t* packed, const _Float16* a, uint16_t* out) {
    const int tile = blockIdx.x, lane = threadIdx.x;
    auto raw = k2_local_kernel::read_raw(packed, a, 0, gridDim.x * 16, tile);
    uint32_t states[8]; k2_local_kernel::states(raw, states);
    for (int j = 0; j < 8; ++j) out[tile * 256 + lane * 8 + j] = uint16_t(states[j]);
}
float get_float(const std::vector<unsigned char>& bytes, size_t index) {
    float out; std::memcpy(&out, bytes.data() + index * 4, 4); return out;
}
uint32_t float_bits(float value) { uint32_t out; std::memcpy(&out, &value, 4); return out; }
void local_exhaustive() {
    Buffer hashes(65536 * 12 * 4);
    local_books_probe<<<256, 256>>>((uint32_t*)hashes.p); ck(hipGetLastError());
    const auto h = hashes.get();
    for (uint32_t s = 0; s < 65536; ++s) {
        for (int j = 0; j < 8; ++j) {
            uint32_t got; std::memcpy(&got, h.data() + (s * 12 + j) * 4, 4);
            require(got == ((j & 1) ? 65535 - s : s) * 0x83dcd12du, "local hash high-bit contamination");
        }
        for (int j = 0; j < 4; ++j) {
            uint32_t got; std::memcpy(&got, h.data() + (s * 12 + 8 + j) * 4, 4);
            require(got == (uint32_t(book(s)) | (uint32_t(book(65535 - s)) << 16)), "local exact half bits");
        }
    }
    std::vector<uint32_t> words(769 * 16);
    for (int b = 0; b < 512; ++b) words[b * 16 + b / 32] = 1u << (b % 32);
    for (size_t j = 512 * 16; j < words.size(); ++j) words[j] = random_word();
    Buffer packed(words.size() * 4), a(32), states(769 * 256 * 2);
    packed.upload(words.data(), packed.size); ck(hipMemset(a.p, 0, a.size));
    local_reader_probe<<<769, 32>>>((uint32_t*)packed.p, (_Float16*)a.p, (uint16_t*)states.p);
    ck(hipGetLastError()); const auto raw = states.get();
    for (int tile = 0; tile < 769; ++tile) for (int j = 0; j < 256; ++j) {
        uint16_t got; std::memcpy(&got, raw.data() + (tile * 256 + j) * 2, 2);
        require(got == state(words.data() + tile * 16, j), "local reader basis/random");
    }
    for (Buffer* b : {&hashes, &packed, &a, &states}) b->guards();
    puts("LOCAL_EXHAUSTIVE PASS hashes=524288 halves=524288 reader_states=196864");
}
void local_small() {
    for (int I : {16, 32, 48, 80, 128, 176, 272}) for (int N : {16, 48, 128}) {
        std::vector<uint32_t> packed(size_t(I) * N / 16);
        for (auto& word : packed) word = random_word();
        const uint16_t special[] = {0, 0x8000, 1, 0x8001, 0x3ff, 0x83ff, 0x7bff, 0xfbff, 0x3800, 0xb400};
        std::vector<uint16_t> activation(I + 1);
        for (int j = 0; j < I + 1; ++j) activation[j] = special[j % 10];
        Buffer b(packed.size() * 4), a(activation.size() * 2), control(4 * 3 * size_t(N) * 4), result((size_t(N) + 32) * 4);
        b.upload(packed.data(), b.size); a.upload(activation.data(), a.size);
        DeviceWeights w{(uint32_t*)b.p, nullptr, nullptr, nullptr, I, N, 2, 2};
        ck(launch_gemv_k2_core(w, (_Float16*)a.p, (float*)control.p + N, 3, 4, nullptr));
        const auto reference = control.get();
        std::vector<float> sum(N, 0.0f);
        for (int split = 0; split < 4; ++split) for (int c = 0; c < N; ++c)
            sum[c] += get_float(reference, (size_t(split) * 3 + 1) * N + c);
        for (auto schedule : schedules) {
            ck(hipMemset(result.p, 0xa5, result.size));
            ck(launch_gemv_k2_local_core(w, (_Float16*)a.p, (float*)result.p + 16, schedule, nullptr));
            const auto actual = result.get();
            for (size_t j = 0; j < actual.size(); ++j)
                if (j < 64 || j >= size_t(N + 16) * 4) require(actual[j] == 0xa5, "local prefix/suffix canary");
            for (int c = 0; c < N; ++c) {
                const float got = get_float(actual, size_t(c) + 16);
                require(float_bits(got) == float_bits(sum[c]), "local ordered split reduction bytes");
                double exact = 0, absolute = 0;
                for (int i = 0; i < I; ++i) {
                    const double v = host_half(activation[i]) * weight(packed.data(), I, N, i, c);
                    exact += v; absolute += std::abs(v);
                }
                require(std::isfinite(got) && std::abs(double(got) - exact) <= 3e-6 * std::max(1., absolute), "local full FP64 envelope");
            }
            result.guards();
        }
        require(launch_gemv_k2_local_core(w, (_Float16*)a.p + 1, (float*)result.p, schedules[0], nullptr) == hipErrorInvalidValue, "local offset2 rejection");
        require(launch_gemv_k2_local_core(w, (_Float16*)a.p, (float*)(result.p + 2), schedules[0], nullptr) == hipErrorInvalidValue, "local output alignment");
        require(launch_gemv_k2_local_core(w, (_Float16*)a.p, (float*)a.p, schedules[0], nullptr) == hipErrorInvalidValue, "local input/output alias");
        require(launch_gemv_k2_local_core(w, (_Float16*)a.p, (float*)result.p, static_cast<K2LocalSchedule>(99), nullptr) == hipErrorInvalidValue, "local invalid schedule");
        for (Buffer* buffer : {&b, &a, &control, &result}) buffer->guards();
        printf("LOCAL_SMALL PASS I=%d N=%d schedules=5 internal_splits=4 partial_planes=1 full_cpu=1\n", I, N);
    }
}
struct EventPair {
    hipEvent_t begin{}, end{};
    EventPair() { ck(hipEventCreate(&begin)); ck(hipEventCreate(&end)); }
    ~EventPair() { (void)hipEventDestroy(begin); (void)hipEventDestroy(end); }
};
void local_large(const char* path) {
    Fixture fixture(path); const int I = fixture.I, N = fixture.N;
    require((I == 8192 && N == 5120) || (I == 1280 && N == 32768), "local performance geometry");
    const size_t packed_bytes = size_t(I) * N / 4;
    const size_t clones = ((size_t(256) << 20) + packed_bytes - 1) / packed_bytes;
    require(fixture.v.at("trellis").size() == packed_bytes, "local packed fixture size");
    require(fixture.v.at("a").size() >= size_t(I) * 2 && fixture.v.at("x").size() >= size_t(I) * 4 &&
        fixture.v.at("suh").size() == size_t(I) * 2 && fixture.v.at("svh").size() == size_t(N) * 2 &&
        fixture.v.at("old").size() >= size_t(N) * 4 && fixture.v.at("ref").size() >= size_t(N) * 4 &&
        (fixture.v.at("bias").empty() || fixture.v.at("bias").size() == size_t(N) * 4), "local fixture extents");
    require(fixture.v.at("p64").size() == size_t(4) * N * 8 && fixture.v.at("pabs").size() == size_t(4) * N * 8, "local partial oracle size");
    Buffer packed(clones * packed_bytes), a(I * 2), x(I * 4), su(I * 2), sv(N * 2),
        bias(fixture.v.at("bias").size()), baseline_p(4 * size_t(N) * 4), local_p(size_t(N) * 4), y(size_t(N) * 4);
    DeviceWeights w{(uint32_t*)packed.p, (uint16_t*)su.p, (uint16_t*)sv.p,
        bias.size ? (float*)bias.p : nullptr, I, N, 2, 2};
    const size_t local_bytes = gemv_k2_local_workspace_bytes(w);
    const size_t baseline_bytes = linear_workspace_bytes(w, 1);
    require(baseline_bytes == ((size_t(I) * 2 + 255) & ~size_t(255)) + 4 * size_t(N) * 4, "V16 control must use four split planes");
    require(local_bytes < baseline_bytes, "local workspace reduction");
    Buffer local_workspace(local_bytes), baseline_workspace(baseline_bytes);
    size_t allocation_bound = 0;
    for (Buffer* b : {&packed, &a, &x, &su, &sv, &bias, &baseline_p, &local_p, &y, &local_workspace, &baseline_workspace}) allocation_bound += b->size + 512;
    require(allocation_bound < (size_t(1) << 30), "local GPU allocation budget");
    for (size_t c = 0; c < clones; ++c) ck(hipMemcpy(packed.p + c * packed_bytes, fixture.v.at("trellis").data(), packed_bytes, hipMemcpyHostToDevice));
    for (auto pair : {std::make_pair(&a, "a"), {&x, "x"}, {&su, "suh"}, {&sv, "svh"}, {&bias, "bias"}})
        if (pair.first->size) { require(fixture.v.at(pair.second).size() >= pair.first->size, "local fixture region"); pair.first->upload(fixture.v.at(pair.second).data(), pair.first->size); }
    auto submit = [&](int mode, size_t clone, bool full, float alpha = 1, float beta = 0, hipStream_t stream = nullptr) {
        w.trellis = (uint32_t*)(packed.p + clone * packed_bytes);
        if (full) {
            if (mode == 0) ck(launch_linear(w, (float*)x.p, (float*)y.p, 1, baseline_workspace.p, stream, alpha, beta, false));
            else ck(launch_gemv_k2_local_linear(w, (float*)x.p, (float*)y.p, local_workspace.p, local_bytes, schedules[mode - 1], stream, alpha, beta));
        } else if (mode == 0) ck(launch_gemv_k2_core(w, (_Float16*)a.p, (float*)baseline_p.p, 1, 4, stream));
        else ck(launch_gemv_k2_local_core(w, (_Float16*)a.p, (float*)local_p.p, schedules[mode - 1], stream));
    };
    submit(0, 0, false); const auto baseline = baseline_p.get();
    std::vector<float> core_sum(N, 0.0f);
    for (int c = 0; c < N; ++c) {
        double exact = 0, absolute = 0;
        for (int split = 0; split < 4; ++split) {
            const size_t j = size_t(split) * N + c; const float value = get_float(baseline, j);
            double p64, pabs; std::memcpy(&p64, fixture.v.at("p64").data() + j * 8, 8); std::memcpy(&pabs, fixture.v.at("pabs").data() + j * 8, 8);
            require(std::isfinite(value) && std::abs(double(value) - p64) <= 2e-6 * std::max(1., pabs), "local control FP64 partials");
            core_sum[c] += value; exact += p64; absolute += pabs;
        }
        require(std::abs(double(core_sum[c]) - exact) <= 3e-6 * std::max(1., absolute), "local combined FP64 core");
    }
    auto check_core = [&](int mode) {
        if (mode == 0) { require(baseline_p.get() == baseline, "local baseline core changed"); return; }
        const auto values = local_p.get();
        for (int c = 0; c < N; ++c) require(float_bits(get_float(values, c)) == float_bits(core_sum[c]), "local complete core differs");
    };
    for (auto scalars : std::vector<std::array<float, 2>>{{1, 0}, {.75f, 0}, {1, .5f}, {-.5f, -.25f}}) {
        std::vector<unsigned char> reference;
        for (int mode = 0; mode < 6; ++mode) {
            if (scalars[1] == 0) ck(hipMemset(y.p, 255, y.size)); else y.upload(fixture.v.at("old").data(), y.size);
            submit(mode, 0, true, scalars[0], scalars[1]); const auto actual = y.get();
            if (!mode) reference = actual; else require(actual == reference, "local complete operator bytes alpha/beta");
            double max_error = 0, max_reference = 1;
            for (int c = 0; c < N; ++c) {
                const float got = get_float(actual, c);
                float expected = get_float(fixture.v.at("ref"), c) * scalars[0];
                if (scalars[1] != 0) expected += get_float(fixture.v.at("old"), c) * scalars[1];
                require(std::isfinite(got), "local beta-zero poison");
                max_error = std::max(max_error, std::abs(double(got) - expected)); max_reference = std::max(max_reference, std::abs(double(expected)));
            }
            require(max_error <= 5e-4 * max_reference, "local full CPU operator envelope");
        }
    }
    // Invalid storage and schedule must enqueue no kernel, including in capture.
    hipStream_t stream{}; ck(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    hipGraph_t graph{}; ck(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    require(launch_gemv_k2_local_linear(w, (float*)x.p, (float*)y.p, local_workspace.p, local_bytes - 1, schedules[0], stream) == hipErrorInvalidValue, "local short workspace");
    require(launch_gemv_k2_local_linear(w, (float*)x.p, (float*)y.p, local_workspace.p, local_bytes, static_cast<K2LocalSchedule>(99), stream) == hipErrorInvalidValue, "local full invalid schedule");
    ck(hipStreamEndCapture(stream, &graph)); size_t nodes = 0; ck(hipGraphGetNodes(graph, nullptr, &nodes)); require(nodes == 0, "local invalid capture enqueued work"); ck(hipGraphDestroy(graph));
    for (int mode = 1; mode < 6; ++mode) {
        ck(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal)); submit(mode, 0, true, 1, 0, stream); ck(hipStreamEndCapture(stream, &graph));
        ck(hipGraphGetNodes(graph, nullptr, &nodes)); require(nodes == 3, "local full graph must contain three kernels");
        hipGraphExec_t executable{}; ck(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
        ck(hipGraphLaunch(executable, stream)); ck(hipStreamSynchronize(stream)); const auto actual = y.get();
        submit(0, 0, true); require(actual == y.get(), "local graph replay output");
        ck(hipGraphExecDestroy(executable)); ck(hipGraphDestroy(graph));
    }
    ck(hipStreamDestroy(stream));
    for (bool full : {false, true}) {
        submit(0, 0, true); const auto expected_y = y.get();
        auto check = [&](int mode) { if (full) require(y.get() == expected_y, "local complete timed output"); else check_core(mode); };
        for (int phase = 0; phase < 2; ++phase) {
            // Every independently addressed clone is checked before and after timing.
            for (size_t clone = 0; clone < clones; ++clone) for (int mode = 0; mode < 6; ++mode) { submit(mode, clone, full); check(mode); }
            if (phase) break;
            EventPair events; std::array<std::vector<double>, 6> samples, wall;
            for (int sweep = -1; sweep < 11; ++sweep) for (int order = 0; order < 6; ++order) {
                const int mode = (order + sweep + 6) % 6;
                ck(hipEventRecord(events.begin)); const auto started = std::chrono::steady_clock::now();
                for (size_t clone = 0; clone < clones; ++clone) submit(mode, clone, full);
                ck(hipEventRecord(events.end)); ck(hipEventSynchronize(events.end));
                const double host_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - started).count() / clones;
                float ms = 0; ck(hipEventElapsedTime(&ms, events.begin, events.end));
                if (sweep >= 0) { const double us = ms * 1000. / clones; samples[mode].push_back(us); wall[mode].push_back(host_us);
                    printf("LOCAL_SAMPLE I=%d N=%d protocol=%s mode=%d sweep=%d us=%.9g wall_us=%.9g\n", I, N, full ? "operator" : "core", mode, sweep, us, host_us); }
                if (sweep == 10) check(mode);
            }
            for (int mode = 0; mode < 6; ++mode) {
                auto times = samples[mode], host = wall[mode]; std::sort(times.begin(), times.end()); std::sort(host.begin(), host.end());
                printf("LOCAL_RESULT I=%d N=%d R=1 internal_splits=4 partial_planes=%d protocol=%s mode=%d schedule=%s clones=%zu resident_bytes=%zu workspace_bytes=%zu min_us=%.9g median_us=%.9g p90_us=%.9g wall_median_us=%.9g compressed_GBs=%.9g useful_TOPS=%.9g\n", I, N, mode ? 1 : 4, full ? "operator" : "core", mode, mode ? k2_local_schedule_name(schedules[mode - 1]) : "v16", clones, clones * packed_bytes, mode ? local_bytes : baseline_bytes, times[0], times[5], times[9], host[5], packed_bytes / times[5] / 1000., 2. * I * N / times[5] / 1e6);
            }
        }
    }
    for (Buffer* b : {&packed, &a, &x, &su, &sv, &bias, &baseline_p, &local_p, &y, &local_workspace, &baseline_workspace}) b->guards();
    printf("LOCAL_LARGE PASS I=%d N=%d schedules=5 full_cpu=1 all_outputs=1 graph=1 canaries=1 all_clones_before_after_last=1\n", I, N);
}
} // namespace
int main(int argc, char** argv) {
    try {
        setvbuf(stdout, nullptr, _IOLBF, 0);
        require(argc == 1 || argc == 3, "usage: strix_trellis_k2_local_probe [wo_b.stgpu q_b.stgpu]");
        int device = -1; ck(hipGetDevice(&device)); hipDeviceProp_t properties{}; ck(hipGetDeviceProperties(&properties, device));
        require(properties.warpSize == 32 && !std::strncmp(properties.gcnArchName, "gfx1151", 7) &&
            (!properties.gcnArchName[7] || properties.gcnArchName[7] == ':'), "requires gfx1151 wave32");
        printf("DEVICE name=%s arch=%s wave=%d device=%d experiment=k2-local opt_in=1\n", properties.name, properties.gcnArchName, properties.warpSize, device);
        local_exhaustive(); local_small();
        if (argc == 3) { local_large(argv[1]); local_large(argv[2]); }
        else puts("LOCAL_LARGE NOT_RUN reason=no_fixtures; operator/performance/graphs remain unqualified");
        puts("PASS k2_local_probe"); return 0;
    } catch (const std::exception& e) { fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
}
