#include "../src/tp_moe_a8r_reference.hpp"
#include <cassert>
#include <iostream>
namespace a = strix_trellis::a8r;
int main() {
    // Independent direct affine numerator, including all hash states and
    // endpoint code groups which exercise wrapping intermediate products.
    for (uint32_t state=0;state<65536;++state) {
        const uint32_t h=state*0x83dcd12du;
        const unsigned b=(h&255)+((h>>8)&255)+((h>>16)&255)+(h>>24);
        for (int q:{-127,-1,0,1,127}) {
            const int64_t expected=(int64_t(887)*b-452608)*q*32;
            assert(expected>=INT32_MIN&&expected<=INT32_MAX);
            assert(a::numerator(int(b)*q*32,q*32)==expected);
        }
    }
    for (int lower=-127;lower<127;++lower) {
        const int expected=(lower&1)?lower+1:lower;
        assert(a::quantize(float(lower)+.5f,127.f,1.f)==expected);
    }
    assert(a::scale(0.f)==1.f&&a::quantize(-0.f,0.f,1.f)==0);
    assert(a::scale(0x1p-120f)==1.f&&a::quantize(0x1p-120f,0x1p-120f,1.f)==0);
    // Folding 2^-17 into delta cannot round when all intermediates stay
    // normal. RN16/H128 input groups bound finite nonzero delta to this range.
    for (int exponent=-35;exponent<=13;++exponent) for (unsigned mantissa=0;mantissa<128;++mantissa) {
        const float delta=std::ldexp(1.f+float(mantissa)/128.f,exponent);
        for (int32_t n:{-1839398912,-16777217,-1,0,1,16777217,1839398912}) {
            const float original=a::mul(a::mul(float(n),0x1p-17f),delta);
            const float folded=a::mul(float(n),a::mul(delta,0x1p-17f));
            assert(a::bits(original)==a::bits(folded));
        }
    }
    // A zero optional fixture offset must preserve an existing negative-zero
    // bias. Output scale -1 makes the projected zero negative as well.
    std::vector<uint16_t> svh(128,0xbc00);
    std::vector<float> bias(128,a::from_bits(0x80000000u)),p(4*128,0.f);
    strix_trellis::a8r_reference::Weights w{nullptr,nullptr,svh.data(),bias.data(),128,128};
    const auto y=strix_trellis::a8r_reference::finish(w,p);
    for (float value:y) assert(a::bits(value)==0x80000000u);
    std::cout<<"PASS A8R integer bounds, independent affine oracle, RN-even ties, signed-zero bias\n";
}
