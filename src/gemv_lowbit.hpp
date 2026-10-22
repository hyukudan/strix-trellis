#pragma once
#include "backend.hpp"
namespace strix_trellis_lowbit {
// Private experiment only. 0: K3 U1; 1: K3 U4; 2: K3 cooperative U4;
// 3: K4 U4; 4: K4 cooperative U4. Ineligible calls use immutable v9.
hipError_t launch(const strix_trellis::DeviceWeights&, const _Float16*, float*,
    int rows, int total_rows, int splits, int candidate, hipStream_t = nullptr);
hipError_t probe(int bits, bool cooperative, const uint32_t* rings,
    uint32_t* states, int count, hipStream_t = nullptr);
hipError_t probe_hash(uint32_t* hashes, uint32_t* pairs, hipStream_t = nullptr);
}
