#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
namespace strix_trellis {
// Preserve an observable RN32 boundary before a later RN16 conversion. Without
// this register boundary LLVM selects mixed FP16-output arithmetic, which differs
// on double-rounding ties even with -ffp-contract=off. No memory/scratch access.
__device__ __forceinline__ float materialize_fp32(float value) {
#if defined(__HIP_DEVICE_COMPILE__)
    asm volatile("" : "+v"(value));
#endif
    return value;
}
// Four contiguous wave-sized slices give each lane indices lane+32*j.
// Strides 1..16 communicate across lanes; strides 32,64 remain in registers.
// The operation order matches the ascending-stride FP32 reference exactly.
__device__ __forceinline__ void hadamard128(float (&values)[4]) {
    const int lane = threadIdx.x & 31;
    #pragma unroll
    for(int stride=1;stride<32;stride*=2) {
        #pragma unroll
        for(int j=0;j<4;++j) {
            float peer=__shfl_xor(values[j],stride,32);
            values[j]=(lane&stride)?peer-values[j]:values[j]+peer;
        }
    }
    #pragma unroll
    for(int stride=1;stride<4;stride*=2) {
        #pragma unroll
        for(int j=0;j<4;++j) if(!(j&stride)) {
            float left=values[j],right=values[j+stride];
            values[j]=left+right; values[j+stride]=left-right;
        }
    }
    #pragma unroll
    for(int j=0;j<4;++j) values[j]=materialize_fp32(values[j]*0x1.6a09e6p-4f);
}

}
