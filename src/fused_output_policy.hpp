#pragma once
#include <cstddef>

namespace strix_trellis {
inline bool fused_output_device_limits(int threads,int dimension_x,size_t shared) {
    return threads>=1024 && dimension_x>=1024 && shared>=24576;
}
inline bool fused_output_image_limits(int threads,size_t shared,size_t local,size_t available,
    int required=1024) {
    return threads>=required && shared<=available && local==0;
}
inline int fused_output_groups(int input,int output,int bits,int codebook,
    int rows,int splits,bool prefer_wmma,bool capable,bool aligned) {
    if(!capable || !aligned || prefer_wmma || rows!=8 || splits!=4 || codebook!=2)
        return 0;
    if(bits==5 && input==1280 && output==32768) return 1;
    if(bits==6 && input==5120 && output==129280) return 2;
    if(bits==5 && output==5120 && (input==8192 || input==2304)) return 4;
    return 0;
}
inline bool fused_output_candidate(int input,int output,int bits,int codebook,
    int rows,int splits,bool prefer_wmma,bool capable,bool aligned) {
    return fused_output_groups(input,output,bits,codebook,rows,splits,
        prefer_wmma,capable,aligned)!=0;
}
} // namespace strix_trellis
