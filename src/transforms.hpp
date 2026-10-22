#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hadamard.hpp"

namespace strix_trellis {

__global__ void input_transform(const float* input,const uint16_t* scales,
    _Float16* transformed,int input_size,int rows) {
    size_t group=size_t(blockIdx.x)*4+(threadIdx.x/32);
    int groups_per_row=input_size/128;
    if(group>=size_t(rows)*groups_per_row) return;
    int row=int(group/groups_per_row);
    int begin=(group%groups_per_row)*128;
    int lane=threadIdx.x&31;
    float values[4];
    #pragma unroll
    for(int j=0;j<4;++j) {
        int channel=begin+lane+32*j;
        float scaled=materialize_fp32(input[size_t(row)*input_size+channel]*__half2float(__ushort_as_half(scales[channel])));
        values[j]=float(_Float16(scaled));
    }
    hadamard128(values);
    #pragma unroll
    for(int j=0;j<4;++j) transformed[size_t(row)*input_size+begin+lane+32*j]=_Float16(values[j]);
}

__global__ void output_transform(const float* partial,const uint16_t* scales,
    const float* bias,float* output,int output_size,int rows,int splits,float alpha,float beta) {
    size_t group=size_t(blockIdx.x)*4+(threadIdx.x/32);
    int groups_per_row=output_size/128;
    if(group>=size_t(rows)*groups_per_row) return;
    int row=int(group/groups_per_row);
    int begin=(group%groups_per_row)*128;
    int lane=threadIdx.x&31;
    float values[4]={};
    for(int split=0;split<splits;++split) {
        #pragma unroll
        for(int j=0;j<4;++j) values[j]+=partial[(size_t(split)*rows+row)*output_size+begin+lane+32*j];
    }
    hadamard128(values);
    #pragma unroll
    for(int j=0;j<4;++j) {
        int channel=begin+lane+32*j;
        size_t index=size_t(row)*output_size+channel;
        float value=values[j]*__half2float(__ushort_as_half(scales[channel]));
        if(bias) value+=bias[channel];
        value*=alpha;
        if(beta!=0) value+=beta*output[index];
        output[index]=value;
    }
}
}
