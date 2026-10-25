#pragma once
#include <cstdlib>
namespace strix_trellis {
// Diagnostic switches while the in-situ cost is root-caused: =0 disables the selection (same bits either way).
inline bool dense_r1_switch(const char* name) { const char* e = std::getenv(name); return !e || *e != '0'; }
struct DenseR1Schedule { int columns, prefetch, outputs_per_wave; bool use_splitk04; };
// Empirical HOT schedule; numerical identity is fixed separately by splits.
// K4 4096 -> 5120: COLD sweep (weights rotating over > MALL), U2 dense-r1 at every row count (-2..-4 % at R2-R4/R7/R8
// against splitk04 / U4; U2 instead of the HOT U1 -11 % at R6).
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
            {{128,2,16,false},{128,2,16,false},{128,2,16,false},{128,2,16,false},{128,2,16,false},{128,2,16,false},{128,2,16,false},{128,2,16,false}},
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
    static const bool on = dense_r1_switch("STRIX_TRELLIS_DENSE_PIPE");
#ifdef STRIX_TRELLIS_GROUP_K5_PIPELINE
    return on && (bits==4 || bits==5);
#else
    return on && bits==4;
#endif
}
// Single plans with pipelined weight batches (requested before the current batch is consumed, combined when consumed;
// same arithmetic). COLD sweep, K4 4096 -> 5120 S4 U2: R1/R2/R4 50.5/51.8/55.2 us against 52.0/53.2/56.1; R6/R8 even.
inline bool dense_r1_pipelined(int input,int output,int bits,int rows,int splits) {
    static const bool on = dense_r1_switch("STRIX_TRELLIS_DENSE_PIPE");
    return on && bits==4 && input==4096 && output==5120 && splits==4 && rows>=1 && rows<=4;
}
// Owner-fold kernel (partials folded by the CTA owning each output H128; same bits as the split plan), with its weight
// prefetch. COLD sweep, K4 wo_a 4096 -> 1024 S32 R1/R2/R4/R6/R8 43.6/44.5/49.2/60.7/67.0 us against the best grouped
// split plan 48.5/51.6/58.6/65.6/72.1. q_b 1280 -> 16384 S4 is even with its split plan once the staging loads overlap
// (R1/R6/R8 56.2/66.2/71.0 against 56.2/66.2/70.5): split plan kept. The o_b K-half (16 waves over 40 CTAs) loses.
inline bool dense_r1_owner(int input,int output,int bits,int rows,int splits) {
    static const bool on = dense_r1_switch("STRIX_TRELLIS_DENSE_OWNER");
#ifdef STRIX_TRELLIS_GROUP_K5_OWNER
    // K5 wo_a S16 (Pass 4) with STRIX_TRELLIS_GROUP_K5_OWNER, rows 1-4 only: same chains and fold order as its S16
    // split plan. Cold public-plan sweep: faster at R1-R4, slower from R6 on (32 owner CTAs whose LDS/VGPR grow with rows).
    if (on && bits==5 && rows>=1 && rows<=4 && input==4096 && output==1024 && splits==16) return true;
#endif
    return on && bits==4 && rows>=1 && rows<=8 && input==4096 && output==1024 && splits==32;
}
// Previous activation staging order (each slice's loads after the previous RN32 boundary; same values and bits) for
// the K5 output head at two rows in one block ({64,4,8} schedule): measured in the same session, two blocks of three
// paired runs, the early staging was 0.9 % slower there and faster or even at every other measured cell.
inline bool dense_r1_late_staging(int input,int output,int bits,int rows,int rows_per_block,int splits) {
    return input==5120 && output==64640 && bits==5 && rows==2 && rows_per_block==2 && splits==4;
}
} // namespace strix_trellis
