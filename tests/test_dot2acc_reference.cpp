// CPU checks of the native v_dot2acc_f32_f16 model (src/dot2acc_reference.hpp).
// Expected bits are gfx1151 silicon results from the dot2cal dump (7,340,032
// cases, 0 mismatches) and the original S4 calibration tuple.
#include "../src/dot2acc_reference.hpp"
#include "../src/tp_moe_activation.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
namespace d = cpu_dot2acc;
static uint32_t run(uint16_t a0, uint16_t a1, uint16_t b0, uint16_t b1, float acc) {
    float r = d::dot2acc(a0, a1, b0, b1, acc); uint32_t b; std::memcpy(&b, &r, 4); return b;
}
int main() {
    // S4 calibration tuple: 7*6 + (-6)(-5) - 128 = -56 exactly, silicon -0x1.bffffep+5.
    assert(run(0x4700, 0xc600, 0x4600, 0xc500, -128.f) == 0xc25fffffu);
    // Frame fixed by the accumulator: +1 - 64 + 64 = 0.999996185 on silicon.
    assert(run(0xd400, 0x5400, 0x3c00, 0x3c00, 1.f) == 0x3f7fffc0u);
    // Kernel-like samples from the gfx1151 dump (class 0).
    assert(run(0x3c6f, 0xbb6e, 0x342e, 0xab81, -0x1.d343fp-2f) == 0xbde608c0u);
    assert(run(0x3e16, 0x411d, 0xaa9f, 0xb81e, 0x1.9e3d8ep+7f) == 0x434db9cbu);
    assert(run(0x3c00, 0x407e, 0x24be, 0xaf2e, 0x1.a3008p-10f) == 0xbe6d6a20u);
    // Same-sign addends are exact; no -0 result; subnormal accumulator flushed.
    assert(run(0x3c00, 0x3c00, 0x3c00, 0x3c00, 1.f) == 0x40400000u);
    assert(run(0x8000, 0x8000, 0x3c00, 0x3c00, -0.f) == 0u);
    assert(run(0, 0, 0, 0, 0x1p-140f) == 0u);
    // Declared SwiGLU: clamp at 10 on both factors' documented sides.
    assert(strix_trellis::tp_moe_declared_swiglu(20.f, 1.f) == strix_trellis::tp_moe_declared_swiglu(10.f, 1.f));
    assert(strix_trellis::tp_moe_declared_swiglu(1.f, -30.f) == strix_trellis::tp_moe_declared_swiglu(1.f, -10.f));
    std::cout << "PASS dot2acc_reference\n";
}
