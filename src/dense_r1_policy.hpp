#pragma once
namespace strix_trellis {
struct DenseR1Schedule { int columns, prefetch, outputs_per_wave; bool use_splitk04; };
// Empirical HOT schedule; numerical identity is fixed separately by splits.
// K4 4096 -> 5120 rows 5-6: COLD sweep (weights rotating over > MALL), U2 instead of the HOT U1 (-11 % at R6).
// Odd row counts borrow the next even-row schedule; correctness is qualified independently.
inline DenseR1Schedule dense_r1_schedule(int input,int output,int bits,int rows,int splits) {
    if((bits!=4 && bits!=5) || rows<1 || rows>8) return {64,2,16,false};
    if(input==5120 && output==64640 && splits==4) {
        static constexpr DenseR1Schedule table[2][8] = {
            {{128,4,16,false},{128,4,16,false},{128,4,16,false},{128,4,16,false},{128,4,16,false},{128,4,16,false},{128,4,16,false},{128,4,16,false}},
            {{128,4,16,false},{64,4,8,false},{128,4,16,false},{128,4,16,false},{128,4,16,false},{128,4,16,false},{128,4,16,false},{128,4,16,false}},
        };
        return table[bits-4][rows-1];
    }
    if(input==5120 && output==512 && splits==4) {
        static constexpr DenseR1Schedule table[2][8] = {
            {{128,4,8,false},{128,4,8,false},{128,4,8,false},{128,4,8,false},{128,4,8,false},{128,4,8,false},{128,4,8,false},{128,4,8,false}},
            {{128,4,8,false},{128,4,8,false},{128,2,8,false},{128,2,8,false},{128,4,8,false},{128,4,8,false},{128,4,8,false},{128,4,8,false}},
        };
        return table[bits-4][rows-1];
    }
    if(input==5120 && output==512 && splits==32) {
        static constexpr DenseR1Schedule table[2][8] = {
            {{128,2,16,false},{128,2,16,false},{128,2,16,false},{128,2,16,false},{128,4,16,false},{128,4,16,false},{128,2,16,false},{128,2,16,false}},
            {{128,1,16,false},{128,1,16,false},{128,2,16,false},{128,2,16,false},{128,4,16,false},{128,4,16,false},{128,4,16,false},{128,4,16,false}},
        };
        return table[bits-4][rows-1];
    }
    if(input==4096 && output==5120 && splits==4) {
        static constexpr DenseR1Schedule table[2][8] = {
            {{128,2,16,false},{128,2,16,true},{128,2,16,true},{128,2,16,true},{128,2,16,false},{128,2,16,false},{128,4,16,true},{128,4,16,true}},
            {{128,1,16,false},{128,4,16,true},{128,2,16,true},{128,2,16,true},{128,1,16,true},{128,1,16,true},{128,1,16,false},{128,1,16,false}},
        };
        return table[bits-4][rows-1];
    }
    if(input==5120 && output==1280 && splits==4) {
        static constexpr DenseR1Schedule table[2][8] = {
            {{128,4,8,false},{128,4,8,false},{128,4,8,false},{128,4,8,false},{128,4,8,false},{128,4,8,false},{128,4,8,false},{128,4,8,false}},
            {{128,4,16,false},{128,4,8,false},{128,2,8,false},{128,2,8,false},{128,4,8,false},{128,4,8,false},{128,4,8,false},{128,4,8,false}},
        };
        return table[bits-4][rows-1];
    }
    if(input==5120 && output==1280 && splits==16) {
        static constexpr DenseR1Schedule table[2][8] = {
            {{128,2,16,false},{128,2,16,false},{128,2,16,false},{128,2,16,false},{128,4,16,false},{128,4,16,false},{128,4,16,false},{128,4,16,false}},
            {{128,1,16,false},{128,1,16,false},{128,1,16,false},{128,1,16,false},{128,2,16,false},{128,2,16,false},{128,1,16,false},{128,1,16,false}},
        };
        return table[bits-4][rows-1];
    }
    if(input==1280 && output==16384 && splits==4) {
        static constexpr DenseR1Schedule table[2][8] = {
            {{128,2,16,false},{128,2,16,true},{128,2,16,true},{128,2,16,true},{128,4,16,false},{128,4,16,false},{128,4,16,false},{128,4,16,false}},
            {{128,4,16,false},{128,4,16,true},{128,1,16,true},{128,1,16,true},{128,2,16,false},{128,2,16,false},{128,2,16,false},{128,2,16,false}},
        };
        return table[bits-4][rows-1];
    }
    if(input==1152 && output==5120 && splits==4) {
        static constexpr DenseR1Schedule table[2][8] = {
            {{128,2,16,false},{128,4,16,false},{128,2,16,false},{128,2,16,false},{128,4,16,false},{128,4,16,false},{128,4,16,false},{128,4,16,false}},
            {{128,1,16,false},{128,4,16,false},{128,4,16,false},{128,4,16,false},{128,4,16,false},{128,4,16,false},{128,4,16,false},{128,4,16,false}},
        };
        return table[bits-4][rows-1];
    }
    if(input==5120 && output==1152 && splits==4) {
        static constexpr DenseR1Schedule table[2][8] = {
            {{128,4,8,false},{128,4,8,false},{128,4,8,false},{128,4,8,false},{128,4,8,false},{128,4,8,false},{128,4,8,false},{128,4,8,false}},
            {{128,4,16,false},{128,4,8,false},{128,2,8,false},{128,2,8,false},{128,4,8,false},{128,4,8,false},{128,4,8,false},{128,4,8,false}},
        };
        return table[bits-4][rows-1];
    }
    if(input==5120 && output==1152 && splits==16) {
        static constexpr DenseR1Schedule table[2][8] = {
            {{64,2,16,false},{128,2,16,false},{128,2,16,false},{128,2,16,false},{128,4,16,false},{128,4,16,false},{128,4,16,false},{128,4,16,false}},
            {{64,1,16,false},{128,1,16,false},{128,1,16,false},{128,1,16,false},{128,2,16,false},{128,2,16,false},{128,1,16,false},{128,1,16,false}},
        };
        return table[bits-4][rows-1];
    }
    return {output<=512?32:64,2,16,false};
}
// Grouped wo_a schedule (4096 -> 1024), K4 S32 / K5 S16, rows 1..8. K4: COLD sweep (weights rotating over > MALL),
// 128 columns, four tiles in flight, pipelined weight batches (dense_r1_group_pipelined) at every row count.
// K5 with STRIX_TRELLIS_GROUP_K5_PIPELINE: the same schedule (full cold sweep of all 48 schedules, every one bit-identical:
// best or within 0.74 % of the best at R1/R2/R6/R8); otherwise the earlier {64,2,16} without pipelining.
inline DenseR1Schedule dense_r1_group_schedule(int input,int output,int bits,int rows,int splits) {
    (void)input;(void)output;(void)rows;(void)splits;
#ifdef STRIX_TRELLIS_GROUP_K5_PIPELINE
    return DenseR1Schedule{128,4,16,false};
#else
    return bits==4 ? DenseR1Schedule{128,4,16,false} : DenseR1Schedule{64,2,16,false};
#endif
}
inline bool dense_r1_group_pipelined(int input,int output,int bits,int rows,int splits) {
    (void)input;(void)output;(void)rows;(void)splits;
#ifdef STRIX_TRELLIS_GROUP_K5_PIPELINE
    return bits==4 || bits==5;
#else
    return bits==4;
#endif
}
} // namespace strix_trellis
