#pragma once
#include "backend.hpp"
#include <array>

namespace strix_trellis::fused_groups {
// Private experiment: no ABI, policy, allocation, or fallback promises.
enum class Mode { GlobalPartial, Direct, Diagnostic };
struct Region { void* data=nullptr; size_t bytes=0; };
struct Request {
    DeviceWeights weights{};
    const _Float16* a=nullptr;
    float* y=nullptr;
    float* partial=nullptr;
    int rows=8, groups=4;
    float alpha=1, beta=0;
    Mode mode=Mode::Direct;
    hipStream_t stream=nullptr;
    // Exact used subranges: packed, A, svh, optional bias, Y, optional P.
    std::array<Region,6> regions{};
};
class Token {
    Request request_{};
    bool ready_=false;
    friend hipError_t prepare(const Request&,Token&);
    friend hipError_t submit(const Token&);
};
// Host-only shape/capacity/overflow/alignment/noalias guards; no HIP calls.
hipError_t guard(const Request&);
// Must run inside a lease. Checks each new image and actual device limits.
hipError_t resources();
// One-time device/range/stream checks, outside all timed intervals.
hipError_t prepare(const Request&,Token&);
// Immutable validated owned storage and stream must outlive all submissions.
hipError_t submit(const Token&);
}
