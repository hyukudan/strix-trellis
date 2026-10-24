#include "dense_r1.hpp"
#include <cstdio>
namespace strix_trellis::dense_r1 {
const void* function_i1280_s4(const DenseR1Launch&);
const void* function_i4096_s4(const DenseR1Launch&);
const void* function_i1152_s4(const DenseR1Launch&);
const void* function_i5120_s4(const DenseR1Launch&);
const void* function_i5120_s16(const DenseR1Launch&);
const void* function_i5120_s32(const DenseR1Launch&);
const void* group_i4096_s16(const DenseR1Launch&);
const void* group_i4096_s32(const DenseR1Launch&);
const void* group_function(const DenseR1Launch& a) {
    if (a.projection.weights.input != 4096) return nullptr;
    if (a.splits == 16) return group_i4096_s16(a);
    if (a.splits == 32) return group_i4096_s32(a);
    return nullptr;
}
const void* function(const DenseR1Launch& a) {
    switch (a.projection.weights.input) {
        case 1280: return a.splits == 4 ? function_i1280_s4(a) : nullptr;
        case 4096: return a.splits == 4 ? function_i4096_s4(a) : nullptr;
        case 1152: return a.splits == 4 ? function_i1152_s4(a) : nullptr;
        case 5120:
            if (a.splits == 4) return function_i5120_s4(a);
            if (a.splits == 16) return function_i5120_s16(a);
            if (a.splits == 32) return function_i5120_s32(a);
    }
    return nullptr;
}
} // namespace strix_trellis::dense_r1

namespace strix_trellis {
hipError_t query_dense_r1_resources(const DenseR1Launch& a, size_t shared, bool* supported) {
    if (!supported) return hipErrorInvalidValue;
    *supported = false;
    const void* f = dense_r1::function(a);
    if (!f) return hipErrorInvalidValue;
    hipFuncAttributes at{};
    const auto e = hipFuncGetAttributes(&at, f);
    if (e != hipSuccess) return e;
    std::printf("DENSE_R1_RESOURCE input=%d bits=%d tile=%d columns=%d prefetch=%d splits=%d outputs_per_wave=%d vgpr=%d lds=%zu scratch=%zu\n",
        a.projection.weights.input,a.projection.weights.bits,a.projection.rows_per_block,a.columns_per_block,a.prefetch,a.splits,
        a.columns_per_block==8?8:a.outputs_per_wave,at.numRegs,at.sharedSizeBytes,at.localSizeBytes);
    const int threads=a.columns_per_block*(a.columns_per_block==8 || a.outputs_per_wave==8?4:2);
    *supported = at.maxThreadsPerBlock >= threads && at.sharedSizeBytes <= shared && !at.localSizeBytes;
    return hipSuccess;
}
hipError_t query_dense_r1_group_resources(const DenseR1Launch& a, size_t shared, bool* supported) {
    if (!supported) return hipErrorInvalidValue;
    *supported = false;
    const void* f = dense_r1::group_function(a);
    if (!f) return hipErrorInvalidValue;
    hipFuncAttributes at{};
    const auto e = hipFuncGetAttributes(&at, f);
    if (e != hipSuccess) return e;
    const int threads=a.columns_per_block*(a.columns_per_block==8 || a.outputs_per_wave==8?4:2);
    *supported = at.maxThreadsPerBlock >= threads && at.sharedSizeBytes <= shared && !at.localSizeBytes;
    return hipSuccess;
}
hipError_t launch_dense_r1_group_trusted(const DenseR1Group& g, int count, hipStream_t stream) {
    if (count < 1 || count > dense_r1_group_max) return hipErrorInvalidValue;
    const auto& a = g.member[0];
    const auto& d = a.projection;
    const void* f = dense_r1::group_function(a);
    if (!f || d.rows < 1 || d.rows > 8 || d.rows_per_block != d.rows || !dense_r1_group_geometry(d.weights.input,d.weights.output))
        return hipErrorInvalidValue;
    for (int m = 0; m < count; ++m) {
        const auto& b = g.member[m];
        if (!b.partial || !b.counters || b.splits != a.splits || b.columns_per_block != a.columns_per_block ||
            b.prefetch != a.prefetch || b.outputs_per_wave != a.outputs_per_wave ||
            b.projection.rows != d.rows || b.projection.rows_per_block != d.rows_per_block ||
            b.projection.weights.input != d.weights.input || b.projection.weights.output != d.weights.output ||
            b.projection.weights.bits != d.weights.bits) return hipErrorInvalidValue;
    }
    void* args[] = {const_cast<DenseR1Group*>(&g)};
    const int threads=a.columns_per_block*(a.columns_per_block==8 || a.outputs_per_wave==8?4:2);
    return hipLaunchKernel(f, dim3(d.weights.output/a.columns_per_block,a.splits,count), dim3(threads), args, 0, stream);
}
hipError_t launch_dense_r1_trusted(const DenseR1Launch& a, hipStream_t stream) {
    const auto& d = a.projection;
    const void* f = dense_r1::function(a);
    if (!f || !a.partial || !a.counters || d.rows < 1 || d.rows > 8 ||
        d.rows_per_block < 1 || d.rows_per_block > d.rows ||
        !dense_r1_geometry(d.weights.input,d.weights.output)) return hipErrorInvalidValue;
    void* args[] = {const_cast<DenseR1Launch*>(&a)};
    const int threads=a.columns_per_block*(a.columns_per_block==8 || a.outputs_per_wave==8?4:2);
    return hipLaunchKernel(f, dim3(d.weights.output/a.columns_per_block,a.splits,
        (d.rows+d.rows_per_block-1)/d.rows_per_block), dim3(threads), args, 0, stream);
}
} // namespace strix_trellis
