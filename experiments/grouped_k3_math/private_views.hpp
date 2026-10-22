// Literal current pinned V13 private views; not public ABI.
#pragma once
#include "../../include/strix_trellis.h"
#include "../../src/backend.hpp"
struct strix_context_impl {
    int device;
    size_t weights_count = 0;
    bool fused_output_capable = false;
};
struct strix_weights_impl {
    strix_context owner;
    strix_trellis::DeviceWeights device{};
    void* allocation = nullptr;
    size_t allocation_bytes = 0;
};

static_assert(sizeof(strix_trellis::DeviceWeights)==56);
static_assert(offsetof(strix_trellis::DeviceWeights,fused_output_capable)==48);
static_assert(offsetof(strix_trellis::DeviceWeights,trellis)==0);
static_assert(offsetof(strix_trellis::DeviceWeights,suh)==8);
static_assert(offsetof(strix_trellis::DeviceWeights,svh)==16);
static_assert(offsetof(strix_trellis::DeviceWeights,bias)==24);
static_assert(offsetof(strix_trellis::DeviceWeights,input)==32);
static_assert(offsetof(strix_trellis::DeviceWeights,output)==36);
static_assert(offsetof(strix_trellis::DeviceWeights,bits)==40);
static_assert(offsetof(strix_trellis::DeviceWeights,codebook)==44);
