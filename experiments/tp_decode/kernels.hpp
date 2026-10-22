// Independent private TP schedule; reuses this repository's own H128 and hash math.
#pragma once
#include "../../src/backend.hpp"
#include "contract.hpp"
namespace strix_trellis {
hipError_t baseline_core(const DeviceWeights&,const _Float16*,float*,int,int,int,hipStream_t);
}
namespace tp_decode {
struct Arguments {
    strix_trellis::DeviceWeights w;
    const float* x; float* y; _Float16* a; float* p;
    uint64_t xs,ys;
    int rows;
    float alpha,beta;
};
struct Function { const void* pointer; const char* name; int rows,threads; Variant variant; };
const Function* functions(size_t& count);
size_t shared_bytes(const Function&,int input);
hipError_t enqueue(const Arguments&,Variant,hipStream_t);
}
