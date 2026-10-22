#pragma once

namespace strix_trellis {
inline bool pipeline_k2_candidate(int input,int output,int bits,int codebook,
                                  int rows,int splits,bool aligned) {
    return aligned && bits==2 && codebook==2 && rows==1 && splits==4 &&
        ((input==8192 && output==5120) || (input==1280 && output==32768));
}
// Zero retains v9. Modes 1/2/3 are the frozen exact-row experiments;
// mode 4 retains v9's LDS body at a separately measured split count.
inline int exact_gemv_mode(int input,int output,int bits,int codebook,
                           int rows,int splits,bool aligned) {
    if(!aligned || codebook!=2 || rows<1 || rows>8) return 0;
    const int* choices=nullptr;
    static constexpr int qb[8]={2,2,2,2,1,1,3,0};
    static constexpr int wob[8]={2,2,2,3,3,3,3,0};
    static constexpr int shared[8]={2,2,2,2,2,3,3,0};
    static constexpr int inverse[8]={2,2,2,2,1,1,3,0};
    static constexpr int qa[8]={0,0,2,2,2,2,2,2};
    static constexpr int woa[8]={2,2,2,0,3,3,3,0};
    static constexpr int kv[8]={0,1,2,1,2,1,2,2};
    static constexpr int head[8]={0,0,2,2,2,3,3,0};
    if(bits==5) {
        if(splits==4 && input==1280 && output==32768) choices=qb;
        if(splits==4 && input==8192 && output==5120) choices=wob;
        if(splits==4 && input==2304 && output==5120) choices=inverse;
        if(input==5120 && output==2304) {
            if(splits==8 && rows==8) return 4;
            if(splits==16) choices=shared;
        }
        if(input==4096 && output==1024) {
            if(splits==16 && (rows==4 || rows==8)) return 4;
            if(splits==32) choices=woa;
        }
    } else if(bits==6) {
        if(splits==16 && input==5120 && output==1280) choices=qa;
        if(splits==32 && input==5120 && output==512) choices=kv;
        if(splits==4 && input==5120 && output==129280) choices=head;
    }
    return choices?choices[rows-1]:0;
}

// Do not combine the shared R4 exact-global S16 experiment with unmeasured S8.
// R8 retains precisely the static LDS body used in the split experiment.
inline int measured_vector_splits(int input,int output,int bits,int codebook,int rows) {
    if(bits==5 && codebook==2) {
        if(input==4096 && output==1024 && (rows==1 || rows==4 || rows==8)) return 16;
        if(input==5120 && output==2304 && rows==8) return 8;
    }
    return output<=1024?32:(output<=4096 && input>=4096?16:4);
}
} // namespace strix_trellis
