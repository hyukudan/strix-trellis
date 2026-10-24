#include "dense_r1_device.hpp"
// q_a (5120 -> 1280, S16) and kv (5120 -> 512, S32) of one input in one dispatch, the LEGACY partitions: each member
// runs its own projection arithmetic (columns 128, 16 outputs per wave, all rows in one row block), so its bits equal
// its own dense-r1 plan. CTAs are linearized: member 0 first, then member 1; no idle CTA.
namespace strix_trellis::dense_r1 {
template<int Rows> union PairStorage {
    Storage<5120, 16, Rows> a;
    Storage<5120, 32, Rows> b;
};
template<int Bits, int Rows, int UA, int UB>
__global__ __launch_bounds__(256) void pair(DenseR1Pair p) {
    __shared__ PairStorage<Rows> shared;
    constexpr int ax = 1280 / 128, ay = 16, bx = 512 / 128;
    const int i = int(blockIdx.x);
    if (i < ax * ay) projection_body<5120, 16, Bits, Rows, 128, UA, false>(p.member[0], shared.a, i / ax, i % ax, 0);
    else projection_body<5120, 32, Bits, Rows, 128, UB, false>(p.member[1], shared.b, (i - ax * ay) / bx, (i - ax * ay) % bx, 0);
}
template<int Bits, int Rows, int UA>
const void* pair_ub(int ub) {
    if (ub == 1) return reinterpret_cast<const void*>(pair<Bits, Rows, UA, 1>);
    if (ub == 2) return reinterpret_cast<const void*>(pair<Bits, Rows, UA, 2>);
    if (ub == 4) return reinterpret_cast<const void*>(pair<Bits, Rows, UA, 4>);
    return nullptr;
}
template<int Bits, int Rows>
const void* pair_ua(int ua, int ub) {
    if (ua == 1) return pair_ub<Bits, Rows, 1>(ub);
    if (ua == 2) return pair_ub<Bits, Rows, 2>(ub);
    if (ua == 4) return pair_ub<Bits, Rows, 4>(ub);
    return nullptr;
}
template<int Bits>
const void* pair_rows(int rows, int ua, int ub) {
    switch (rows) {
        case 1: return pair_ua<Bits, 1>(ua, ub); case 2: return pair_ua<Bits, 2>(ua, ub);
        case 3: return pair_ua<Bits, 3>(ua, ub); case 4: return pair_ua<Bits, 4>(ua, ub);
        case 5: return pair_ua<Bits, 5>(ua, ub); case 6: return pair_ua<Bits, 6>(ua, ub);
        case 7: return pair_ua<Bits, 7>(ua, ub); case 8: return pair_ua<Bits, 8>(ua, ub);
    }
    return nullptr;
}
const void* pair_function(const DenseR1Pair& p) {
    const auto& a = p.member[0];
    const auto& b = p.member[1];
    const auto& wa = a.projection.weights;
    const auto& wb = b.projection.weights;
    if (wa.input != 5120 || wa.output != 1280 || wb.input != 5120 || wb.output != 512 || a.splits != 16 || b.splits != 32 ||
        a.columns_per_block != 128 || b.columns_per_block != 128 || a.outputs_per_wave != 16 || b.outputs_per_wave != 16 ||
        wa.bits != wb.bits || a.projection.rows != b.projection.rows || a.projection.rows_per_block != a.projection.rows ||
        b.projection.rows_per_block != b.projection.rows) return nullptr;
    if (wa.bits == 4) return pair_rows<4>(a.projection.rows, a.prefetch, b.prefetch);
    if (wa.bits == 5) return pair_rows<5>(a.projection.rows, a.prefetch, b.prefetch);
    return nullptr;
}
} // namespace strix_trellis::dense_r1

namespace strix_trellis {
hipError_t query_dense_r1_pair_resources(const DenseR1Pair& p, size_t shared, bool* supported) {
    if (!supported) return hipErrorInvalidValue;
    *supported = false;
    const void* f = dense_r1::pair_function(p);
    if (!f) return hipErrorInvalidValue;
    hipFuncAttributes at{};
    const auto e = hipFuncGetAttributes(&at, f);
    if (e != hipSuccess) return e;
    *supported = at.maxThreadsPerBlock >= 256 && at.sharedSizeBytes <= shared && !at.localSizeBytes;
    return hipSuccess;
}
hipError_t launch_dense_r1_pair_trusted(const DenseR1Pair& p, hipStream_t stream) {
    const void* f = dense_r1::pair_function(p);
    if (!f || !p.member[0].partial || !p.member[0].counters || !p.member[1].partial || !p.member[1].counters)
        return hipErrorInvalidValue;
    void* args[] = {const_cast<DenseR1Pair*>(&p)};
    return hipLaunchKernel(f, dim3(1280 / 128 * 16 + 512 / 128 * 32), dim3(256), args, 0, stream);
}
} // namespace strix_trellis
