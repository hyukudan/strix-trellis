#pragma once
#include "backend.hpp"
namespace strix_trellis {
// Register-blocked WMMA core for prefill. Same numerical contract as
// launch_wmma_core: per output, K16 tiles of the split accumulate in ascending
// order into an FP32 WMMA accumulator that starts at +0. Only the work
// decomposition differs (block tile, LDS layout, pipelining).
struct WmmaFusedOutput {
    float* output = nullptr;   // non-null: apply the output transform in the epilogue
    float alpha = 1, beta = 0; // same meaning as output_transform
};
// Returns hipErrorNotSupported when the geometry is outside the tiled path;
// the caller then uses launch_wmma_core. tile < 0 selects the policy.
hipError_t launch_wmma_tiled(const DeviceWeights& weights, const _Float16* input,
    float* partial, int rows, int splits, hipStream_t stream,
    const WmmaFusedOutput* fused = nullptr, int tile = -1);
// True when launch_wmma_tiled would accept this call with a fused epilogue.
bool wmma_tiled_fuses_output(const DeviceWeights& weights, int rows, int splits, int tile = -1);
bool wmma_tiled_enabled();             // STRIX_TRELLIS_WMMA_TILED=0 disables
bool wmma_tiled_fused_output_enabled(); // false only with STRIX_TRELLIS_WMMA_TILED_FUSED=0
int wmma_tiled_fused_mode();            // 0 off (=0), 1 always (=1), 2 auto (default)
// Fused epilogue for this call (policy tile): mode 1 whenever possible; mode 2 only when the grid has
// at least STRIX_TRELLIS_WMMA_TILED_FUSED_MIN_BLOCKS blocks (default: CU count + 1). Same bits either way.
bool wmma_tiled_fuse_selected(const DeviceWeights& weights, int rows, int splits);
}
