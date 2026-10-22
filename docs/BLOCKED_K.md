# Proposal: WMMA with K32/K64 blocks

> **Type:** design note — engineering record; may describe plans, candidates or experiments that are not part of the supported API.

Status: experimental proposal after v7; it does not change the qualified runtime.
The goal is to reduce barriers and decode cost in R4/R8 and prefill. Performance
remains unmeasured. Independent review is required before integrating large kernels.

## Rationale and controls

The current core processes one K16 tile per iteration, writes A/Q to LDS and
crosses two barriers. It reuses Q across M16/M64 but repeats the barriers for
each K16. GEMV R8 remains well below 187–198 GB/s on several shapes. WMMA with
M16 and eight useful rows wastes half the computation, but lower scalar
instruction pressure might compensate if barriers are amortized. This is a
hypothesis, not a measured roofline.

The experiment compares (a) current GEMV/LDS, (b) current WMMA, (c) new BK16 with
the same decode, (d) BK16 with optimized mul1 decode and (e) BK32/BK64.
Start with M16 and M64; use M32 only if it adds information. Bits5/6 and mul1
are enough for an initial bounded prototype, retaining the production fallback
for everything else. Do not replace the dispatcher during the search.

## Algorithm and indices

Each CTA retains four waves and N64 (one N16 per wave). `C=BK/16` is 1, 2 or 4.
The split partition remains `[floor((I/16)*s/S), floor((I/16)*(s+1)/S))` in K16
tiles. For each base, `valid=min(C,end-base)` is uniform across the CTA.
Neither the partition nor ascending WMMA order changes.

Keep subtiles physically separate to preserve the tested layout:

- A: `[C][TileM][18]`, where the two extra columns are padding, not data.
  The linear copy computes sub=j/(TileM*16), m=(j/16)%TileM, k=j%16, and reads
  X[(first_row+m)*I+(base+sub)*16+k] if the row and sub are valid; otherwise it
  writes zero.
- Q: `[C][4][16][16]`. Each wave decodes its `(base+sub,tile_n)` tile at offset
  `(tile_k*(N/16)+tile_n)*(Bits*8)` uint32. Keep the qualified kernel's state-to-r/c
  permutation and column XOR by `(r&7)*2`.
- Consumption: for sub=0..valid-1, load the same Half16 fragments as today, using
  lane%16 and replication in lanes16..31. For each m16, update the same Float8
  acc[m] with one WMMA. Do not execute subtiles outside the split, to avoid
  unnecessary reads and signed-zero changes.
- Barrier after all writes and after the K block's last WMMA. No wave may exit
  before a barrier; inactive rows or columns use predicates. Empty splits skip
  the loop uniformly and write zero partials.
- Final store remains identical: `[split,total_rows,N]`, masked by row<rows,
  with accumulator mapping row=first_row+m16*16+2*j+lane/16,
  col=tile_n*16+lane%16.

The first prototype does not fuse H128 or use double buffering. This keeps A
exact and separates the barrier hypothesis from other optimizations. Input/output
transforms and alpha/beta follow [NUMERICS](NUMERICS.md). Evaluate the complete
operator after the core; do not describe the complete API as a core measurement.

## Optimized decoder

The initial control may retain the current WaveTile/codebook_value. A separate
control uses the project's AlignedLane<Bits,true>, constant state<j> and
mul1_half2. State pairs (0,1), (2,3), (4,5), (6,7) are stored at the same r/c
coordinates for j. Do not assume that they form a contiguous column pair.
Bitcast half2 to the two F16 values to avoid additional rounding. All 65,536
states and a logical Q dump by coordinate must match CPU bits before accepting
any matmul.

## Expected resources, not measured

LDS = `2*C*(TileM*18+4*256)` bytes:

| Configuration | LDS, bytes |
|---|---:|
| M16/BK16 | 2,624 |
| M16/BK32 | 5,248 |
| M16/BK64 | 10,496 |
| M64/BK16 | 4,352 |
| M64/BK32 | 8,704 |
| M64/BK64 | 17,408 |

The unpadded layout is another control, with an adjusted formula. Do not infer
active occupancy from these sizes. Record LDS, VGPR, SGPR and scratch from the
emitted ISA. Do not accept a candidate with significant spills based on an
optimistic estimate. Per-sub temporaries must die before the next WMMA so that
not all K fragments remain live in registers.

Dimensions and address products use size_t/int64 where appropriate. Wrappers
validate bits/codebook, R, dimensions, splits and required alignment. They do
not silently expand the buffer contract. The prototype's grid limits cover the
measured shapes (M<=1024), without promising a new generic range.

## Correctness and performance gates

1. Exact Q dump: asymmetric synthetic data and real tiles, word/ring boundaries
   and all mul1 states. Test each sub in BK32/64.
2. A/B impulses in each K sub to check offsets and wave32 replication. Keep
   the documented native WMMA contract, without requiring IEEE exactness for
   arbitrary mixed sums.
3. Partials against WMMA BK16 at the same split: the order is unchanged, so
   bitwise equality is expected when only staging changes. If the compiler
   changes instructions, explain and compare against CPU. Do not hide changes
   with new tolerances.
4. Complete matmul against existing references: R1/2/3/4/5/6/7/8,
   M16/17/32/64/65/128/129/256; splits1/3/4/16/32 and empty splits, final chunks
   of 1/2/3 tiles, tail rows, canaries and no nonfinite values. Subnormal/high-range
   FP16 inputs remain part of the gate.
5. Cold performance: >=256 MiB compressed weights rotating through disjoint
   addresses, eleven sweeps and alternating paths in the same GPU slot. Real
   shapes q_b, wo_b, shared, q_a, wo_a and wkv; a synthetic head only if
   <=1 GiB GPU and identified as synthetic. Prefill may also have an identified
   hot control. Report compressed bytes and useful TOPS without calling them
   physical traffic.
6. Do not change defaults based on the minimum from one search. A winning
   candidate must pass the complete API and engine example with immutable
   source/binary before being published as qualified.

All GPU work uses the allocated queue slot and respects maintenance and
site-specific quiet hours. Compile with nice, use a dedicated scratch directory
and leave other processes alone. Keep the prototype and its reports in a
dedicated branch/worktree. Runtime v7 remains a usable baseline during research.
