# Dense TP decode: contract and bounded experiment
> **Type:** design note — engineering record; may describe plans, candidates or experiments that are not part of the supported API.

2026-10-21; HIGH review of `636b398`, branch `design/tp-decode`.
**Design ready for private implementation; no new policy qualified.**
V16 (`573567f`) remains the default: archive `8eb47b095b6ffdc3444e61da5d52dc2cf922d87da8aaec3ef4b4db8c7fed1db8`,
C ABI1, private DeviceWeights TEN/56 B, bools at offsets 48/49. NINE also
measured 56 B. Recompile internal consumers; never infer compatibility from
size. This change adds only documentation and a CPU test, with no engine changes.

## Priority and fixed domain

K5 means **5 integer bits/weight**, CB2/mul1. Each call processes actual
R1..R8 rows, without padding or a WMMA path. Start with q_b and o_b K-half,
then the output head. The o_b N-half alternative is a design control, not a
change to the TP partitioning.

| Site | I → N per rank | Logical S | K16 boundaries | A/B/C core CTAs | Packed weights, B |
|---|---|---:|---|---:|---:|
| q_b N-half | 1280 → 16384 | 4 | 0,20,40,60,80 | 1024 | 13107200 |
| o_b K-half | 4096 → 5120 | 4 | 0,64,128,192,256 | 320 | 13107200 |
| o_b N-half, alternative | 8192 → 2560 | 4 | 0,128,256,384,512 | 160 | 13107200 |
| head N-half | 5120 → 64640 | 4 | 0,80,160,240,320 | 4040 | 206848000 |
| o_a per group, next phase | 4096 → 1024 | 16 | 16*s, s=0..16 | 256 | 2621440 |
| shared gate/up, next phase | 5120 → 2304 | 16 | 20*s, s=0..16 | 576 | 7372800 |
| shared down, next phase | 2304 → 5120 | 4 | 36*s, s=0..4 | 320 | 7372800 |

S is part of the identity, independent of R and the physical schedule. The
last three shapes already need a canonical three-stage path to remove R
classes. Their grouped optimization follows the first TP experiment. Do not
change S while tuning performance for this identity.

**Estimated, unmeasured** targets: 200 GB/s at R1 and 185 GB/s at R6, defined
as unique packed weight bytes / complete linear time, including transforms,
epilogue and required adaptation. For the three 13.1072 MB shapes, these are
65.536/70.850 µs; for the head, 1034.240/1118.097 µs. These are neither promises
nor isolated core times. Also record suh/svh/bias, A/P and copies. Do not
inflate the numerator with repeated traffic. Local o_b time must not attribute
peer wait time to the kernel. q_a/kv remain U6; fusing that pair is outside
this phase.

## Logical producer and two distinct contracts

Inspected sources: [backend](../src/backend.hip), `split_count/launch_linear`;
[core](../src/gemv_lds.hip), `gemv_baseline_kernel/gemv_lds_kernel`;
[fused input](../src/gemv_fused.hip), `beginH/endH`;
[transforms](../src/transforms.hpp) and [H128](../src/hadamard.hpp).
The actual baseline formula uses **floor**, not ceil chunks:
`T=I/16; b_s=T*s/S; e_s=T*(s+1)/S`, with nonnegative integer division.

For split s, transform complete H blocks from `floor(b_s/8)` through
`ceil(e_s/8)` exclusive. Accumulate **only** k in `[b_s,e_s)`, in increasing
order. An H block shared by two splits may be transformed twice; no K tile is
accumulated twice. q_b has 10 H blocks: the S4 baseline boundaries are
0/20/40/60/80; the old H partition boundaries are 0/16/40/56/80. Clipping
transforms H2 and H7 in both neighboring splits, preserving their distinct
intersections. Shared S16 and down S4 also have boundaries inside H blocks.
Do not reuse their old R1 fusion.

The private canonical producer is the three-stage path with the earlier
independent scalar decoder, S from this table and new strides. Freeze source,
compiler and options before comparing. Its per-row operations are:

1. Materialize `RN32(X[r,c]*float(suh[c]))` before `RN16`; convert to F32,
   apply H128 with strides 1,2,4,8,16,32,64, materialize F32 normalization
   `0x1.6a09e6p-4f`, then RN16 A. Use the same producer for every R.
2. For each lane l, use two F32 +0 accumulators; visit k in increasing order
   and p=0,1,2,3 in increasing order. Obtain states 2p/2p+1. Use A at
   `16*k+2*(l&3)` and its pair at a+8 according to p&1;
   `fdot2(weight_pair,A_pair,acc[p/2],clamp=false)`. Never regroup pairs across k.
3. Each accumulator reduces with `v+=shfl_down(v,2,width4)`, then
   `v+=shfl_down(v,1,width4)`. Lanes that are multiples of 4 publish
   `P[s,r,16*tile_n+l/4+8*h]`. The row does not participate in any reduction.
4. Start at F32 +0 and add P in **ascending s=0..S-1** order; apply output
   F32 H128; multiply by svh, add optional bias, multiply by alpha and, only
   if beta!=0, multiply beta by Yold and add. Use separate operations, no fast
   math, `-ffp-contract=off`. There is no alpha0 shortcut. Bias is inside Linear.

The new path requires **A/P/Y to be byte-identical to the canonical producer
for each row**, for every R, row position and supported physical variant.
P may be a local diagnostic in a direct path. Compare it in full before
removing that diagnostic. This is stronger than passing a final tolerance.
The CPU [partition test](../tools/prove_tp_logical_partitions.py) proves
coverage, clipping and ownership. It does not prove native fdot2 semantics
or GPU bits.

The old numerical contract allows reduction differences within its limits;
it does not become bitwise retroactively. The new mode also does not promise
equality with every old fused R1 path: shared and o_a had different reduction
classes. q_b TP, o_b TP and the head must also be compared against complete
V16. Similar dispatch is not proof of equality.

As an independent control, use complete CPU A with both RN16 steps, exact
FP16 Q, an FP64 dot **per split**, then ordered FP32 H/epilogue. Also retain
the algebraic FP64 reference, which has a different contract because of half
activations. For P and Y, freeze `maxabs<=5e-4*max(1,maxabs(ref))` and
`NRMS=sqrt(sum(err²)/max(sum(ref²),1e-30))<=3e-5`, following the earlier
acceptance gates. These are admission thresholds, not IEEE theorems about
fdot2. Add checks based on the sum of absolute products and cancellation/
subnormal cases. Do not emulate fdot2 as two host additions and call it a
native test. Investigate a failure; do not widen tolerances. NaN Yold with
beta±0 must give the same result as finite Yold. NaN payload identity is not
promised for active nonfinite operands.

## Minimal additive ABI, without a new owning plan

Proposal, **with no symbols implemented yet**. Do not use ABI1 reserved/flags
or change `strix_weights_prepare`, `strix_gemv/gemm/grouped_moe` or their
validators. Existing weight handles suffice. A new entry point explicitly
selects the canonical contract and accepts native strides. Neither prepare_ex
nor a plan owner is needed to support these TP projections and the head.

```c
typedef enum strix_decode_numerics {
    STRIX_DECODE_LOGICAL_K16_V1 = 1
} strix_decode_numerics;
typedef struct strix_decode_desc {
    uint32_t struct_size, abi_version;
    strix_matmul_desc linear;
    strix_decode_numerics numerics;
    uint32_t logical_splits;
    uint64_t reserved[4];
} strix_decode_desc;
void strix_decode_desc_init(strix_decode_desc *desc);
strix_status strix_decode_workspace_bytes(strix_context context,
    const strix_decode_desc *desc, uint64_t *out_bytes);
strix_status strix_decode(strix_context context, const strix_decode_desc *desc,
    const strix_stream *stream);
```

Init sets headers, alpha1/beta0, numerics=LOGICAL_K16_V1 and logical_splits=0
(the caller must set it). Only K5/CB2, the seven table shapes, exact S and
R1..R8 are supported. Outside this domain, return UNSUPPORTED_FORMAT without
silent fallback. Incorrect S/unknown enum: INVALID_ARGUMENT. Rank1 retains
its vector representation; rank2 `[R,width]` requires inner stride 1 and row
stride **>=width** for both X and Y. Strides are in F32 elements, including
R1. X/Y ranks must match. F16/BF16, R0/R9 and other layouts are excluded.
Alpha/beta must be finite; headers must have exact sizes and all reserved
fields must be zero. Natural host64 layout: tensor 96 B, matmul 264 B; new
desc 312 B/align8, linear@8, numerics@272, logical_splits@276, reserved@280.
Confirm C11/C++17 with asserts before publishing. The public header remains
HIP-free.

The query validates structure, shapes, S, arithmetic and declared aliasing.
It is not a GPU allocation certificate. It zeroes a valid out before an
error; it neither launches nor inspects contents. Workspace is stable across
all variants: `align256(2*R*I)+4*S*R*N`, A followed by P, each internally
contiguous. Even if a path omits global A/P, reserving that maximum avoids
changing the contract by row/dispatch. At submission, validate declared
capacity and actual physical extent as in the current API, before any enqueue.
Keep the 256-aligned workspace requirement and checks of the entire supplied
capacity.

For each nonempty tensor, the extent from data is
`4*((R-1)*row_stride+width)`, **not** `4*R*row_stride`. Check multiplication,
addition, conversion to size_t, address+bytes and extent<=buffer_bytes before
forming pointers. R1 with a huge stride needs only width; the stride does
not cause an extra access. Rows are disjoint because stride>=width. The
kernel writes only N elements per row, never the gaps. Place a canary in
every gap and at the end of the allocation.

Initial conservative decision: compare complete half-open envelopes of
X/Y/workspace/weights. Reject any overlap within a call, even if it only
affects row gaps and would be safe under exact analysis. Exact adjacency is
allowed; in-place input/output is not. This is not memory certification: it
does not detect later free/reallocation or unrelated operations. Sharing
read-only X between calls is valid. The caller prevents cross-call writers
and protects the bytes each call reads/writes. There is no global call
registry. A future grouped o_a must review **per-row regions between groups**
to allow interleaved views. Do not extend this envelope rejection blindly.

Submission reads host descriptors during the call. It does not own
X/Y/workspace, stream or weights. Prepare, allocate and validate outside
capture where appropriate. Do not introduce new resource queries during
submission. Capture/replay stores pointers: the caller retains context,
weights, allocations and stream until final completion and destroys graphs
that reference them before freeing them. There is no pin, raw-pointer lease
or implicit persistent plan. Sharing workspace across concurrent streams
requires caller exclusion; test two separate workspaces. Verify actual
capture of this entry point before offering support. The new engine identity
must record mode/S/pack/archive. Do not silently replace the K5 pack archive (`trellis-k5-v1`)
or remove its padding classes without an acceptance gate.

## Physical implementations and fixed matrix

gfx1151 wave32. A/B/C: 128 threads, 4 waves, 16 columns per wave, 64 N per CTA;
grid `(N/64,S)`. Each implementation has exact Rows=1,2,3,4,5,6,7,8, without
row predicates or padding. Input/output stages keep four H blocks per CTA128.
All listed shapes are multiples of 128, including 64640=505*128.

| Variant | Launches | Fixed schedule | First experiment domain |
|---|---:|---|---|
| A ExactHash | 3 | Input; hash/SADDR core, direct loads R1..4, chunk8 LDS R5..8; output | Two primary TP projections, R1..8 |
| B Prefetch2 | 3 | Same as A with two **raw** tiles in registers | Two primary TP projections, R1..8 |
| C LogicalInput | 2 | Complete split H blocks in LDS and K16 clipping; core without prefetch2; output | Two primary TP projections, R1..8 |
| D Both | 1 | Both H transforms fused, local P per logical S; no prefetch2 | Both TP projections R1/G4; only o_b K-half R6/G2 |

Then apply A/B/C to the head and the N-half alternative with the same matrix,
without Cartesian tuning. The remaining integration shapes receive canonical
A. Do not expand D or add load/chunk/G/S combinations before deciding on this
set. Templates with runtime I/N and a host whitelist: 8 A + 8 B + 8 C + 2 D =
26 core implementations, plus 2 shared transforms. Private references/
diagnostics are counted separately. Freeze the inventory, ELF size and
compilation time. Do not generate one copy per shape or the product of all
parameters. Compile with nice during any future test-site preparation. No
HIP compilation is performed in this work.

A/B exclusively reuse the [independent decoder/hash](../src/packed_decoder.hpp).
For K5, each tile is 40 u32 words = 160 B; base
`packed+(k*(N/16)+tile_n)*40`, with N **local to the pack**. Half pairs and
state order do not change. The three words per lane wrap their indices
within the same tile. Never read a word outside the ring to complete a state.
Little-endian `low16|(high16<<16)` and MSB-first windows are format properties,
not guesses about host endianness. Hash `uint32(state*0x83dcd12d)` preserves
the full carry32. Do not replace it with two half multiplications without
carry. Revalidate 65,536 states and tile maps against the independent scalar
reader. Fractional CPU rates do not imply GPU support: this ABI accepts K5.
FP8/FP4 hardware is not assumed.

In B, load raw(current) in the prologue. If k+1<e_s, issue raw(next) **before**
decoding current; decode/dot current, then promote next. The last iteration
masks next without forming/dereferencing its address. Word wrapping is within
a tile, not between tiles or splits. Do not compute the second tile's dot
product early. The double buffer stores only the three raw words per lane,
not decoded Q or new accumulators. At chunk8 boundaries, preserve carry or
reload under a guard. Do not access A from the next chunk before publication.
Barriers occur only at uniform chunk boundaries. All threads participate,
including inactive waves. ISA must show next VMEM before current computation
**without an early wait for next**. C++ source alone does not prove overlap.

C uses LDS `2*R*128*max_s(ceil(e_s/8)-floor(b_s/8))`: q_b 768R B,
o_b K-half 2048R, alternative 4096R, head 2560R. At R8:
6144/16384/32768/20480 B. Complete H blocks are computed from X with native
row_stride; temporary A retains the global H index. The dot loop consumes
only the original intersection, including the first/last partial H block.
Publish/reuse the cache with a uniform CTA barrier. No wave may return before
the last barrier.

D: N128 per CTA, eight column waves per physical group. G4/R1 uses 1024
threads; G2/R6 uses 512. Phase t of group g processes s=g+G*t; the cache per
physical group is reused after a uniform barrier. Store **P[s,r,c]**, never a
sum per g. At the end, one wave per row sums s in ascending order from +0 and
runs native H/epilogue. G2/G4 change only when each P is computed.
`(P0+P1)+(P2+P3)` or `(P0+P2)+(P1+P3)` fail: the CPU test produces 1/0/2 for
the same four P values. No atomics, last-CTA counter or global communication
inside D.

**Calculated/expected resources, not observed in these kernels:** A chunk8
uses 256R B LDS (1536 at R6). B adds approximately 3..6 live VGPRs per lane,
depending on lifetimes/transformation, without an allocation guarantee.
Historical R6 reference: fallback 65 VGPR/2048 B versus exact hash-LDS 46
VGPR/1536 B, with no spills. The 2R accumulators and up to 4R words of
activation operands are minimum liveness requirements, not a total register
count. D stores 2048R B of P; cache G*2*R*128*maxH (one split per group/phase):
q_b G4/R1 totals 5120 B; o_b G4/R1 totals 10240 B; o_b G2/R6 totals 36864 B.
Include actual alignment/temporaries when auditing ELF.

Prerequisite gate: actual wave32/thread support, total LDS within queried
capacity, private/spills0. Initial budget: A/B R6<=64 VGPR and C/D<=96 VGPR
(experiment admission limits, not occupancy predictions). If a variant does
not fit, remove it and document that result; retain A. Record SGPR, VGPR,
LDS, maxThreads and occupancy calculated with actual allocation granularity.
Do not claim measured occupancy. tile_n and packed base must be uniform
**per wave**; use readfirstlane only after proving this. Lane offsets remain
VGPR. Audit address instructions, SGPR versus VGPR and waits in linked ELF.
A name does not guarantee SADDR. The HIP report shows 20 multiprocessors;
test records count 40 CUs. Confirm the WGP/CU interpretation when calculating
occupancy. Neither number is measured occupancy. Large grids do not prove
efficiency.

## Implementation prerequisites and fixtures

The independent review sets the P0/P1 gates.
Private kernel implementation and fixture preparation follow this design.
This work performs no GPU requests, SSH access or test-site file creation.
Keep future test-site files in a scratch directory. First prepare both TP
projections and their complete controls. A K3 MoE GPU performance campaign
must wait until the first TP experiment is ready.

Fixture: an **actual independently prepared K5 pack**, both ranks and distinct
inputs per row. Do not rename a whole q_b fixture as a slice. q_b copies 16K
of 32K output columns tile by tile for each K16, with local N=16384; `svh`
and bias use the same 16384 interval, with complete `suh` of 1280. o_b K-half
selects I0/4096 and 4096/4096, their respective suh and X, and retains svh5120.
The N-half alternative retains I8192 and selects svh2560. The head retains
I5120 and selects 64640 of 129280 per rank. All cuts at 128 preserve H blocks.
Check the first/last boundary, especially head H504/505. A q_b pack with
different weights/svh is not validated by a matching shape. Record source,
offsets and Q/suh/svh/bias/X hashes. Synchronize fixtures using existing range
fixtures; do not download full shards.

An N-slice preserves output H blocks and the full operator's K tree. Confirm
that its rows match the full pack. A K-half followed by h0+h1 does change the
tree relative to the full dot product: do not require or promise the same
bits. Require the same rank summation order for every R, including emulated
world1. If the model has bias, apply it only once during recomposition, not
in both halves. Synthetic per-linear bias tests validate the local epilogue
separately from that TP recomposition.
