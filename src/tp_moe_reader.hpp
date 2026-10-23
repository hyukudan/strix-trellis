#pragma once
// Independent cyclic bit-window contract for the experimental cooperative
// reader. This header is shared with the CPU bit-by-bit coverage proof.
#include <cstdint>
#if defined(__HIPCC__) || defined(__CUDACC__)
#define STRIX_MOE_READER_HD __host__ __device__
#else
#define STRIX_MOE_READER_HD
#endif
namespace strix_trellis {
struct TpMoeReaderWindow { unsigned a,b,c,shift; };
template<unsigned Bits>
STRIX_MOE_READER_HD constexpr TpMoeReaderWindow tp_moe_reader_window(unsigned lane) {
    static_assert(Bits==2 || Bits==3);
    constexpr unsigned words=Bits*8;
    // Add one full cyclic payload before subtracting the 16-bit state width.
    const unsigned start=((lane*8+1)*Bits+words*32-16)%(words*32);
    const unsigned word=start/32;
    return {word,(word+1)%words,(word+2)%words,start%32};
}
template<unsigned Bits,unsigned J>
STRIX_MOE_READER_HD constexpr uint32_t tp_moe_reader_state(uint32_t high,uint32_t low,uint32_t tail) {
    constexpr unsigned offset=J*Bits;
    if constexpr(offset<=16)return(high>>(16-offset))&65535u;
    else if constexpr(offset<32)return((high<<(offset-16))|(low>>(48-offset)))&65535u;
    else if constexpr(offset<=48)return(low>>(48-offset))&65535u;
    else return((low<<(offset-48))|(tail>>(80-offset)))&65535u;
}
} // namespace strix_trellis
#undef STRIX_MOE_READER_HD
