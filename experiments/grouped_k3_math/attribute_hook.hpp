// Include after real HIP declarations, or the CPU test's ABI stand-ins.
#pragma once
#include "attribute_trace.hpp"
namespace grouped_k3_math {
inline AttributeTrace<hipFuncAttributes> attribute_trace;
using AttrRegion=AttributeTrace<hipFuncAttributes>::Region;
}
extern "C" hipError_t __real_hipFuncGetAttributes(hipFuncAttributes*,const void*);
extern "C" hipError_t __wrap_hipFuncGetAttributes(hipFuncAttributes* a,const void* f) {
    const auto result=__real_hipFuncGetAttributes(a,f);
    grouped_k3_math::attribute_trace.record(f,static_cast<int>(result),result==hipSuccess?a:nullptr);
    return result;
}
