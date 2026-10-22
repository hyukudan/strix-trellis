#pragma once
#include "fused_output_policy.hpp"
namespace strix_trellis {
// Shared by the real HIP adapter and CPU API mocks. Query every image even
// after a successful but insufficient certificate; later HIP errors prevail.
template<class Error,class Attributes,class GetDevice,class GetAttributes>
Error probe_fused_output_images(int expected,size_t available,bool* supported,
    Error success,Error invalid,const void* const (&functions)[4],
    GetDevice get_device,GetAttributes get_attributes) {
    if(!supported) return invalid;
    *supported=false;
    int active=-1;
    Error e=get_device(&active);
    if(e!=success) return e;
    if(active!=expected) return success;
    const int required[4]={1024,1024,256,512};
    bool fits=true;
    for(int i=0;i<4;++i) {
        Attributes a{};
        e=get_attributes(&a,functions[i]);
        if(e!=success) return e;
        fits=fused_output_image_limits(a.maxThreadsPerBlock,a.sharedSizeBytes,
            a.localSizeBytes,available,required[i]) && fits;
    }
    *supported=fits;
    return success;
}
}
