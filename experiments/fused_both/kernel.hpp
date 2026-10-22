#pragma once
#include "backend.hpp"
#include <array>

namespace strix_trellis::fused_both {
enum class Mode { GlobalPartial, Direct, Diagnostic };
struct Region { const void* data=nullptr; size_t bytes=0; };
struct Request {
    DeviceWeights weights{};
    const float* x=nullptr;
    float* y=nullptr;
    float* partial=nullptr;
    _Float16* debug_a=nullptr;
    float* debug_h=nullptr;
    int rows=1, groups=4;
    float alpha=1, beta=0;
    Mode mode=Mode::Direct;
    hipStream_t stream=nullptr;
    // Declared capacities: packed, X, suh, svh, bias, Y, P, A, H.
    std::array<Region,9> regions{};
};
class Token {
    Request request_{};
    size_t dynamic_=0;
    bool ready_=false;
    friend hipError_t prepare(const Request&,Token&);
    friend hipError_t submit(const Token&);
};
hipError_t guard(const Request&);
hipError_t resources();
hipError_t prepare(const Request&,Token&);
// No queries, allocation, copies or synchronization. Lifetimes remain caller-owned.
hipError_t submit(const Token&);
}
