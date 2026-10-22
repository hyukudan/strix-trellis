#pragma once
#include <cstdint>
namespace strix_trellis {
inline bool k2_multirow_candidate(int i,int n,int bits,int cb,int rows,int total,int s,
    bool a4,bool p4,bool b4) {
    return bits==2 && cb==2 && s==4 && total==rows &&
        (rows==2 || rows==4 || rows==8) && a4 && p4 && b4 &&
        ((i==8192 && n==5120) || (i==1280 && n==32768));
}
inline bool wmma_packed_candidate(int i,int n,int bits,int cb,int rows,int s,
    int tile_rows,bool pad18,bool a4,bool p4,bool b4) {
    if(cb!=2 || tile_rows!=64 || !pad18 || !a4 || !p4 || !b4 ||
       (rows!=64 && rows!=128 && rows!=256)) return false;
    if(bits==6) return i==5120 && n==1280 && rows==256 && s==4;
    if(bits!=5) return false;
    if(i==1280 && n==32768) return s==1;
    const int scale=256/rows;
    if((i==8192 && n==5120) || (i==2304 && n==5120)) return s==scale;
    if(i==5120 && n==2304) return rows!=256 && s==2*scale;
    return i==4096 && n==1024 && rows==256 && s==4*scale;
}
}
