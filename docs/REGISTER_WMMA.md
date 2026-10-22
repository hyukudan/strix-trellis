# Proposal: direct decode into WMMA register fragments

> **Type:** design note — engineering record; may describe plans, candidates or experiments that are not part of the supported API.

Status: **CPU proposal, with no GPU kernel or performance figures**. This is
another hypothesis after v7, independent of the BK/LDS experiment. It requires
independent review before implementing the large kernel. Its initial domain
would be mul1 K5/K6, wave32, R1–16, I/N multiples of 128 and FP16 A aligned to 32.
It does not change the usable runtime.

## Goal and ownership change

Current WMMA decodes Q into LDS and reloads fragments, with barriers for each K.
For R8, an M16 WMMA wastes half the matrix but replaces many dot2 instructions.
This proposal avoids LDS and **core** barriers by changing which eight states
each lane decodes. A loads directly from the already transformed input. H128
remains in the existing kernels. No speedup is assumed: word requests per lane
and A requests increase, although duplication/coalescing and caching may help.

In the project's current permutation, state `8*t+j` maps to
r=2*(t%4)+(j%2)+8*((j//2)%2), c=t//4+8*(j//4). The WMMA B fragment requires the
complete column c=lane%16, replicated in both half-waves.

New lane l owns that column c, but only eight k rows: h=l//16 selects k0–7 or
k8–15. For pair p0–3, the rows are 8*h+2*p and +1. Their state indices are:

```
s0 = 32*(l&7) + 2*(l>>4) + 4*((l>>3)&1) + 8*p
s1 = s0+1
```

The 32 lanes still decode 256 distinct states; the codebook is not duplicated.
Each lane retains four half2 values P0..P3, with the first state in the low half.
Four XOR16 shuffles obtain the peer's P values. Lower lanes form B from their
four pairs followed by the peer's four; upper lanes use the reverse order.
Both half-waves therefore have identical B[k0..15,c]. Choose the order **after**
the shuffle: a shuffle retrieves the register selected by the source lane,
not the receiving lane. This scheme avoids confusing the two.

A uses row=lane%16, with the required duplication in lanes16–31, and sixteen
contiguous K values from the transformed FP16 matrix. For row>=R, construct a
zero vector without reading memory, but no lane exits before shuffles/WMMA.
A 32-byte copy with proven alignment could emit two vector loads. Use
memcpy/builtin rather than incompatible object aliasing. Alignment: base32,
row*I*2 is a multiple of 256 and tile_k*32 is a multiple of 32. The wrapper
rejects lower alignments. The production caller would have a fallback, and its
workspace256 already satisfies the requirement.

## MSB-first windows and loaded words

EXL3 states are 16-bit windows starting at `(s+1)*K-16` inside their 256*K-bit
ring, not at s*K. For c_low=l&7, preload K+1 uint32 words from
`base=c_low*K-1`. Only the initial word can be -1: then read the last word of
the **same** tile. The following K words stay within the tile. K5 requires
6 words/lane and K6 seven, versus three for current AlignedLane. The cost of
additional requests must be measured.

Relative to that first word, the pair starts at:

```
class_states = 2*(l>>4) + 4*((l>>3)&1)
start = 16 + (class_states+1+8*p)*K
minimum_word = floor((16+(1+8*p)*K)/32)
```

For K5/K6, start//32 is only minimum_word or minimum_word+1. Select two words
with **constant** indices and a predicate, avoiding a divergently indexed
register array. The high word beyond the last register may be replaced by 0
only when the pair's 16+K bits fit entirely within the last word. The basis
test verifies this case; do not read outside the ring.

`shift=start&31` is never 0: it is odd for K5 and 2 mod4 for K6. Align the
MSB-first window as `(low<<shift)|(high>>(32-shift))`, truncated to 32 bits.
State0=window>>16; state1=(window>>(16-K))&0xffff. This specific formula is
validated for K5/K6; do not extend it to other K by analogy.

Then apply the exhaustively checked mul1_half2 helper. The uint32 bitcast keeps
low/high, and a final bitcast from eight uint32 values to Half16 preserves K
order. HIP code generation and this packing still require GPU tests; the CPU
state test does not establish them.

## Loop and stores

One wave owns one N16. Four waves per CTA may retain N64 geometry. They share
no memory, so there are no CTA barriers. The N predicate is wave-uniform and
may skip an entire wave. Inactive A-row lanes use zeros and always participate
in collectives. Splits keep I16 boundaries and ascending K order; promote global
products before multiplication. An empty split stores zero.

Float8 acc and the store keep the tested map: row=2*j+lane//16,
column=tile_n*16+lane%16, in `[split,total_rows,N]`. Accumulators/outputs are
**not** duplicated. A caller could provide a chunk offset as in GEMV; the first
prototype does not extend to R>16 or change the C ABI.

## Required tests and comparisons

`tools/check_register_wmma_map.py` uses an independent scalar bit reader to
construct logical Q with the permutation above. It checks the new extraction
and complete/duplicated fragments against each basis bit of the K5/K6 ring and
eight random tiles per K. Result (internal record):
724,992 states and 1,449,984 fragment values, plus the store bijection, CPU PASS.
Selections depend only on lane/K/p, and extraction is bit-linear; the basis
vectors cover every tile position. This does not replace ISA, shuffle or
physical WMMA validation.

GPU gates: all mul1 states and both half2 halves; logical Q and per-lane B dumps;
exact A including zero rows; impulses and bitwise partial comparisons against
current M16 WMMA at the same splits; all R1..16 cases, empty splits, tails,
canaries and rejected alignment; complete operator with existing references and
tolerances, subnormal/high-range values, alpha/beta and the engine example.
Do not require IEEE equality for arbitrary mixtures. Locate any difference
between two pipelines with identical operands/WMMA.

## Clarifications from independent CPU review

The review and its complete gates are in
the independent review. The additional
`tools/check_register_wmma_review.py` test traces every bit to its physical word,
verifies bijective state ownership, order after XOR16, R1..16 and partitions
with empty splits. Its evidence is CPU-only; it does not establish compiler
vector representation or HIP code generation.

When implementing selectors with constant registers, any candidate beyond W[K]
must be literal zero: never form W[K+1], even as a candidate later discarded.
Pairs using that zero satisfy `shift+16+K<=32` (16 pairs/wave for K5, eight for
K6). Both shift operands are uint32; signed promotion and shift32 are forbidden.
P order is selected after four shuffles of each lane's own P values, with
explicit width32 and participation by all 32 lanes of each active wave.

The first prototype has `1<=rows<=16`, without chunks: `total_rows=rows` and
zero row offset. A reads exactly 32 bytes per valid row and tile, and forms no
address for an invalid row. An empty split keeps acc=+0 and writes all valid
outputs, even when the loop is skipped. Do not copy a per-row return from GEMV.
In the domain where N is a multiple of 128, N64 blocks have no excess N waves.
A synthetic inactive-wave test needs a separate harness or future domain extension.

The wrapper must validate K5/K6, mul1, shapes, pointers, lengths, A alignment32,
packed/partial alignment4, non-overlap and grid limits. Non-null pointers alone
are insufficient. Promote products before multiplication for split bounds and
offsets, and check byte sizes and address ends without wrap. Workspace is
`align256(2*rows*I)+4*splits*rows*N`, with every operation checked. Arbitrary
positive splits can overflow it even with valid int32 dimensions. Current policy
bounds splits, and a harness overriding them must validate them. This review
changes neither that policy nor the API.

Performance: compare in the same GPU slot against current GEMV/LDS, new
scheduling and BK/LDS once qualified. Use identical splits first, then a bounded
search and independent confirmation of one selection. Resident rotation
>=256 MiB, eleven alternating sweeps, separate core and complete API timing.
No global expanded Q in the measured path. Inspect words/loads, emitted shuffles,
VGPR/SGPR/scratch and cache behavior without inferring active occupancy or DRAM
traffic from static sizes. A losing candidate remains experimental.
