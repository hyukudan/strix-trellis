#pragma once
// Project-owned trellis pair adapter for the split S4 schedule.
#include "backend.hpp"
#include "packed_decoder.hpp"
namespace strix_trellis {
template<int Bits> struct TpMoeWeightTile {
    packed_decode::SaddrLane<Bits> reader;
    __device__ TpMoeWeightTile(const DeviceWeights& w, int k, int n)
        : reader(w.trellis + (size_t(k)*(w.output/16)+n)*(Bits*8)) {}
    template<int P> __device__ __forceinline__ __half2 pair() const {
        return packed_decode::mul1_half2(reader.template state<2*P>(), reader.template state<2*P+1>());
    }
};
} // namespace strix_trellis
