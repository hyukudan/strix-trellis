#pragma once
#include "backend.hpp"

namespace strix_trellis {
// Prefill policy is deliberately restricted to measured dense geometries.
inline bool medium_prefill_candidate(const DeviceWeights& w, int rows) {
    if(rows<32 || w.codebook!=2) return false;
    if(rows>=256) return (w.bits==6 && w.input==5120 && w.output==1280) ||
        (w.bits==5 && w.input==4096 && w.output==1024);
    if(w.bits==6) return w.input==5120 && w.output==1280;
    if(w.bits!=5) return false;
    return (w.input==1280 && w.output==32768) ||
        (w.input==8192 && w.output==5120) || (w.input==5120 && w.output==2304) ||
        (w.input==2304 && w.output==5120) || (w.input==4096 && w.output==1024);
}
inline int wmma_tile_rows(const DeviceWeights& w, int rows) {
    return medium_prefill_candidate(w,rows) || (rows>=256 && w.output>=2304) ||
        (rows>=64 && w.output>=32768 && w.bits==5 && w.codebook==2)?64:16;
}
} // namespace strix_trellis
