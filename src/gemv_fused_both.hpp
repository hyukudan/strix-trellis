#pragma once
#include "backend.hpp"
namespace strix_trellis::fused_both {
// Preserve the literal prototype template's enum tokens and values.
enum class Mode { GlobalPartial, Direct, Diagnostic };
const void* direct_function();
}
namespace strix_trellis {
// Called only during context creation on the active, viable ordinal.
hipError_t query_fused_both_resources(size_t shared_bytes,bool* supported);
hipError_t launch_fused_both_trusted(const DeviceWeights&,const float*,float*,
    int rows,bool prefer_wmma,hipStream_t,float alpha,float beta);
}
