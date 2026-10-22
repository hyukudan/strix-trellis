// Isolated experiment. This interface is not part of the production backend.
#pragma once
#include "backend.hpp"
#include <memory>

namespace strix_trellis::blocked {
enum class Decoder : int { slow = 0, fast = 1 };
// Explicit non-default stream on the current gfx1151 device; device allocations
// only (not managed/host), A aligned to 2 bytes, packed/partial to 4 bytes.
// No aliasing; caller owns complete buffers, no allocations or synchronization.
// mul1 bits5/6, 1<=rows<=1024, I/N positive multiples128, splits1..32.
// tile_m is exactly16/64; block_k exactly16/32/64. Unknown values are rejected.
hipError_t launch(const DeviceWeights&, const _Float16*, float*, int rows,
    int splits, hipStream_t, int tile_m, int block_k, Decoder);
// Small diagnostic of the SAME staging/fragment accessors as launch(). Dumps
// one CTA at first_row, first_n (N16 index), base (K16 index), valid subtiles.
// A: [C,TileM,16], including masked/invalid +0; Q: [C,4,16,16]
// in logical coordinates, invalid subtiles/waves +0. Never used for timing.
hipError_t checkpoint(const DeviceWeights&, const _Float16*, uint16_t* a,
    uint16_t* q, int rows, hipStream_t, int tile_m, int block_k, Decoder,
    int first_row, int first_n, int base, int valid);
}

namespace strix_trellis::blocked {
// EXPERIMENTAL trusted internal descriptor, never a public/backend API.
// The host must keep device, stream and every allocation alive and unchanged
// for its entire lifetime. No concurrent mutation/free/device switching.
// Construction runs the checked launch's complete validator without launching.
class Prevalidated final {
    DeviceWeights w_;
    const _Float16* x_;
    float* y_;
    int rows_, splits_, m_, bk_;
    hipStream_t stream_;
    Decoder decoder_;
    bool baseline_;
    Prevalidated(const DeviceWeights& w,const _Float16* x,float* y,int rows,
        int splits,hipStream_t s,int m,int bk,Decoder d,bool baseline):
        w_(w),x_(x),y_(y),rows_(rows),splits_(splits),m_(m),bk_(bk),
        stream_(s),decoder_(d),baseline_(baseline) {}
    friend hipError_t prevalidate(const DeviceWeights&,const _Float16*,float*,
        int,int,hipStream_t,int,int,Decoder,bool,std::unique_ptr<const Prevalidated>&);
public:
    Prevalidated(const Prevalidated&)=delete;
    Prevalidated& operator=(const Prevalidated&)=delete;
    // Only launches the captured tuple: no alloc/copy/sync/storage queries.
    hipError_t submit() const;
};
hipError_t prevalidate(const DeviceWeights&,const _Float16*,float*,int,int,
    hipStream_t,int,int,Decoder,bool,std::unique_ptr<const Prevalidated>&);
}
