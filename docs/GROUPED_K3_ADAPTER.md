# Private K3/CB2/R1 mathematical adapter contract

> **Type:** design note — engineering record; may describe plans, candidates or experiments that are not part of the supported API.

2026-10-21. Design and CPU review only. Planned private source/build work is
described in the follow-up review (internal record).
This work does not implement a grouped mathematical kernel. The trace pilot
does not establish qualification, native resource use or speedup.

## Frozen mathematical origin

The current qualified default is V13, archive
`0b1453a3c9c25e3f7e4f12251d310a3a9f2c3b53c76d78c314e9624ba763e3b3`,
private DeviceWeights nine fields/56 bytes, capability offset48, public ABI1.
V14 is a separate pending candidate and does not change K3/R1. Historical V12 wording in
the old design/review is not the control selection for this follow-up.

The real path is `launch_linear(prefer_wmma=false, rows=1)` in `backend.hip`,
`measured_vector_splits`, `launch_gemv_lds_core`, mode0,
`launch_gemv_lds_variant`, `baseline_core`,
`gemv_baseline_kernel<3,true,1,false>` in `src/gemv_lds.hip`.
K3 cannot select the K2 pipeline, K5/K6 exact/fast modes, input fusion or R8
output fusion. `src/gemv.hip::gemv_aligned` was inspected but is not this dispatch
target. The chosen core has no LDS, Chunk8 staging or CTA barrier.

Pin the literal region from its template declaration through the blank line
preceding `gemv_baseline_rows`, UTF-8/LF SHA256:
`a5fbb60a5cde97442455e041e7f10c76397209cf2c335fc16600346c3da59503`.
An internal pin manifest fixes complete source
files, all three old function-body token streams, dispatch and ABI headers.
Changing the pins requires a new source review, not an automatic refresh.

Input/output bodies come from `src/transforms.hpp::input_transform` and
`output_transform`. The active set of project helper dependencies is
`materialize_fp32`, `hadamard128`, `optimized::AlignedLane<3,true>` constructor
and `state<J>`, `optimized::mul1_half2`, and its `mul1_sum`.
`trellis_device.hpp` and `packed_decoder.hpp` are pinned too: preserved dead
branches still name `codebook_value`, WaveTile, SaddrLane and packed hash helpers.
Do not activate them. Keep canonical pinned includes; any private helper copy
must be literal or have every token compared to those headers, including asm
strings, constants, types and preprocessor branches. Do not use shadow helpers/macros.
HIP builtins, half/vector ABI and compiler semantics remain native gates.
For copied helper headers run the verifier with `--helper-dir <directory>`;
it compares the complete token streams of all four project helper headers.

## Mechanical extraction, without algebraic edits

The simplest permitted extraction retains every original body token. Add only
the field-binding prologue specified as `PROLOGUES` in
[the CPU verifier](../tools/verify_grouped_k3_adapter.py); retain the old local
H128 variable named `group`, with a distinct new `batch_group` index.
The future kernel names are `grouped_input_transform`, `grouped_baseline_core`,
`grouped_output_transform`; each takes **only NativeBatch b by value** and uses
`__launch_bounds__(128)`. The core fixes constexpr Bits3, Mul1true, Rows1,
Fastfalse and the literal BaselinePair vector alias. These are not tuning knobs.
Retain the core's three pointer restrict bindings in its new local prologue.

| Stage | Only new group selection / bindings | Old body sees |
|---|---|---|
| Input | `batch_group=blockIdx.y`; input=g.x, scales=g.suh, transformed=g.a | input_size=b.input, rows=1; old x H128 indexing |
| Core | `batch_group=blockIdx.z`; packed=g.packed, x=g.a, partial=g.partial | input/output=b.input/output, rows=total_rows=1, splits=b.splits, codebook=2 |
| Output | `batch_group=blockIdx.y`; partial=g.partial, scales=g.svh, bias=g.bias, output=g.y, alpha/beta=g.alpha/beta | output_size=b.output, rows=1, splits=b.splits |

All bases are already group-local, with row_offset=0; do not add a group offset
to A/P/Y or packed. Batch metadata/group selection must be CTA-uniform from
blockIdx at origin. Require a scalar prologue and scalar metadata loads in the
linked ISA. Do not use readfirstlane on divergent addresses to mask a bug; do
not scalarize the core's per-wave tile_n or change its Fastfalse reader.
If the compiler needs an explicit uniform-index annotation, review that exact
prologue change and add a finite verifier case before building; no broad
identifier substitution or removal of arithmetic predicates is permitted.

Allowed lexical changes are whitespace/comments, function name/signature and
the exact bindings above. The body otherwise stays literal. If extraction of
scalar transforms into device helpers becomes necessary, compare balanced helper
bodies token-for-token and separately review signature, call arguments and
inlining. The current verifier intentionally rejects such an unreviewed change.
Do not modify production source to share the extracted code.

Disallowed changes include simplification of rows/total_rows expressions,
different split endpoints, reader/hash replacement, altered pair/lane order,
sum precision/initialization, reassociation, fusing multiply/add, approximate
H, changing rounding boundaries, alpha0 fast paths, LDS/barriers, atomics,
full-Q global materialization, changed S or a descriptor upload in submit.

Input must still materialize RN32(X*suh), convert to RN16 and back to float,
execute ascending-stride FP32 H128 and materialized normalization, then RN16 A.
Core retains all four pairs, the two sums and fdot2 clamp=false, then shuffles
down2 followed by down1 with width4. Output starts at +0, adds splits in ascending
order, applies the identical H128, svh, optional bias, alpha, then separate
beta*Y and addition. `beta!=0` skips Y for **both signs of zero**, including NaN
Yold. Negative beta must take the read branch. Preserve these operations and
`-ffp-contract=off`, no fast math, even when a numerical sample seems equal.

| I / N / S | Input grid | Core grid | Output grid | Split K endpoints |
|---|---|---|---|---|
| 5120 / 2304 / 16 | (10,G,1) | (36,16,G) | (5,G,1) | floor(320*s/16), 20 tiles each |
| 2304 / 5120 / 4 | (5,G,1) | (80,4,G) | (10,G,1) | floor(144*s/4), 36 tiles each |

All blocks128/wave32. Input/output old local H128 index is 4*x+wave; width2304
has two inactive final waves which return before data access. Core old x tile
and y split are unchanged; neither N has inactive core waves. P address remains
`(size_t(split)*1+0)*N+col` within that group's allocation. No CTA barriers exist
in any chosen body; different algorithms need another review.

## CPU correspondence gate and its limits

Run `python -B tools/verify_grouped_k3_adapter.py`, then later
`python -B tools/verify_grouped_k3_adapter.py --candidate <private-source.hip>`.
Balanced function extraction ignores line positions. It checks the pinned old
files, exact finite binding prologue and each remaining token, without replacing
arbitrary expressions or accepting a matching hash as evidence of behavior.
The selftests exercise the same comparator with accepted whitespace/comments
and 16 rejected mutations, plus explicit endpoint/tail/local-P enumeration.
An out-of-range P counterexample exposes total_rows=G.

There is no candidate yet. String selftests are not a mathematical kernel.
The lexer is not a C++ preprocessor/parser; signature, scopes, overload resolution,
includes, restrict/alias assumptions, macros, launch args and the complete TU
still require source review and build. A body-only PASS cannot prove helper
binding, runtime arithmetic, occupancy or instruction identity. Inspect native
instructions against the real same-body control after building; ABI differences
may change registers/addresses, but math/rounding operations must be preserved.

## Prepared plan and allocation ownership

Keep Group72/align8 and Batch2320/align8 exactly: eight pointers and two floats
per32 slots, four-int header. Group offsets are 0/8/16/24/32/40/48/56/64/68;
Batch header offsets0/4/8/12 and groups16. Native A is `_Float16*`.
Use the pilot's fieldwise opaque-half bridge through void*, never type-pun a
whole struct. Assert every offset/size/alignment with both actual host and
device HIP compilation; zero all inactive slots and any padding explicitly.
There is **no Trace*** argument in the mathematical kernels.

An immutable private prepared plan owns its copied Batch, registered allocation
leases, streams, device and generation, grids and checked resource context.
Authenticate real GPU allocation/device/extent/alignment/stream provenance only
before capture/submit. Checked size_t64 arithmetic must precede pointer formation.
Validate all groups, including the last, before publishing any plan/enqueue.
Allow read/read shares across groups; every writable A/P/Y slice must be disjoint
from all other read/write slices, including bias and reservation guard regions.
Prepare cannot trust caller-fabricated region certificates or a forgeable Plan
constructor. Freeze descriptors across all preparation passes.

The pilot uses a single owning registry with two streams. A future math plan may
pin the existing production weights owner separately from its private workspace
owner; do not claim the current single-owner guard authenticates external weights.
Production API controls must use their existing weights owner and public API.
Private stage controls compile with matching V13 nine-field headers and link the
actual pinned V13 archive; never fabricate opaque public context/weights layouts.

Submit accepts prepared state only and performs three enqueues with first-error
return. No pointer attributes, function queries, alloc/copy/sync, temporary heap,
fallback or mutable plan rebuilding. Graphs and in-flight requests retain leases
until completion; a graph alone does not retain GPU allocations. Test the graph
lease separately after dropping both external Owner and Plan handles. Destruction
syncs only owned streams outside submit/capture. An error after partial enqueue
does not roll back; retain all leases, stop later enqueues, invalidate/discard a
failed capture and perform cleanup outside submit. Public grouping acceptance,
MAX workspace query, API/default/CMake remain unchanged.

## Build and first native mathematical campaign: OPEN gates

Build privately with nice/one job. Inspect the **actually linked** three math
images, not parallel assembly or trace images. Record full ELF/ISA/hash/toolchain,
all explicit/hidden argument offsets, total kernarg/alignment and admission.
Require only Batch2320 explicit bytes, total including hidden/reserved <=4096
under the proposed budget; 4096 is not an architectural fact. Check wave32,
max_flat_workgroup_size128 and block128 admission, static LDS0, scratch/private0
and SGPR/VGPR spills0. Trace
zero-spill does not predict math zero-spill. Record registers and scalar metadata
loads; static resources alone do not prove occupancy. Query exactly the three
actual functions once in the prepared context, before any capture/submit; record
and check maxThreads>=128, static/local storage0 and device/grid limits.

The first native campaign must be <=180s physical, <=145s inner,
first failure STOP. Freeze/verify both test machines before queueing. Explicit GPU <1GiB,
host <2GiB, with an actual allocation ledger and RSS; generate references outside
the lease, stream/host transfers outside timing. Keep all test files under `<scratch dir>`. GPU access must use a site-specific
GPU queue script (not part of this repository), respecting maintenance and
site-specific quiet hours. Native execution remains a separate validation step.

QA matrix: both shapes, G1/2/4/8/16/31/32, all groups and all clones including
group31. Use new deterministic row/group data and frozen seeds; decode the full
Q from existing range fixtures independently, with no downloaded shards, K2
reference or M64 reference reuse. Reusing immutable Q is allowed; recompute
each row's reference with its own X, scales, bias, alpha/beta and Yold.
Use full CPU A (RN32->RN16/H128/RN16), full FP64 Q*A per split, and full CPU Y
with the ordered FP32 H128/epilogue. Do not emulate fdot2 with two host FP32 adds
and call that a native proof. Also retain an independent full FP64 operator Y
reference to report transform/rounding error separately.

Controls: A = G full real V13 `launch_linear`/public operators, B = G original
input/core/output sequences with the same S and accessible A/P/Y, C = three
grouped math kernels. Require every byte of A/P/Y B==C, and Y A==B==C, with
identical initial Yold. Verify each group/split against the new complete CPU
reference as well, with explicit pre-frozen tolerances for P and Y:
`maxabs <= 5e-4*max(1,maxabs(reference))`,
`NRMS=sqrt(sum(error^2)/max(sum(reference^2),1e-30)) <= 3e-5`.
These are acceptance gates, not a claim already established for native fdot2.
Record per-group/per-split maxima, norms and worst coordinates; no aggregate
norm may hide one bad group. A remains exact to its ordered CPU construction.
If extremes violate the frozen tolerance, STOP and investigate native semantics;
do not widen tolerance or silently drop subnormal/cancellation cases.

Cover positive/negative scales and inputs, subnormals and signed zeros, selected
half/RN32 double-rounding ties, H128 lane/register boundaries, split endpoints,
null/non-null and negative bias, alpha0/negative alpha, beta0/-0/positive/negative,
YNaN under beta±0, cancellation, weights and X distinct between neighbors.
Alternate real w1/w3 where geometry permits; identify repeated w2/clones as such.
For QA also prepare deterministic, independently decoded packed patterns that
actually differ by group (including the w2 geometry); disjoint identical clones
alone cannot expose selection of a neighboring packed pointer. Preserve original
fixtures and hash the derived QA patterns separately from performance clones.
Test both disjoint reads and legal shared reads. Preserve all payloads, immutable
inputs, guards and stage outputs before/after and after the LAST replay/sample.

Metadata/buffer negatives must extend the old full guard matrix: G0/33/overflow,
wrong geometry/S/R/bits/CB, nonfinite alpha/beta, malformed null bias, exact-size
and one-byte-short each field, alignment, offsets/capacity/allocation overflow,
adjacent slices and one-element overlaps, all writers vs every read/write kind
within/across groups including bias, stale generation/foreign/dead stream/device,
forged certificates/Trace-free plan. Last-group faults require zero native nodes
and unchanged complete buffers. Unsupported-device cases needing another GPU
remain honestly scoped. Inject first-error paths at all three enqueues in a
private controlled harness; separate simulated API errors from actual HIP errors.
Observe asynchronous completion failures and lifetime cleanup without adding sync
to submit or claiming rollback.

Repeat actual math with two independently writable requests/streams, both queued
before either wait, direct execution, capture and at least two replays. Destroy
and replace temporary host Batch before instantiate, changing every field to
other valid live allocations/scalars/geometry; inspect captured full arguments,
three real math functions/grids and two edges. Drop external Plan/Owner handles
to isolate graph lifetime, then compare complete A/P/Y/guards for both streams.
Initialize/reset Yold outside timing for nonzero beta; otherwise repeat semantics
can legitimately accumulate. Audit submit/capture for zero copies/alloc/host nodes.

## Performance remains two subsequent physical campaigns

Freeze selection before either campaign: both shapes, G4/8/16/32, eleven sweeps
alternating original A and grouped C, cold-address schedule, all loss/tail records.
Keep >=256MiB of disjoint packed weights resident and actually traverse them.
For scale, 61 clones * 4,423,680B = 269,844,480B packed per projection (planned
arithmetic, not a measured allocation). Budget both streams, scale/bias/Y/A/P,
guard/storage, driver RSS and references explicitly. No clone is a new trained
expert. Prepare independent complete references for all clone/group combinations
actually used, and verify all before/after/LAST, not a rotating sample.

Compare full three-stage operators with event AND host timing. Graph A must
contain **3G original kernel nodes**, graph C exactly3; no direct-vs-graph or
graph-shape asymmetry, no omitted transforms/epilogue, same stream semantics,
S and Yold. Register actual nodes/functions. Require >=2% full-operator gain in
both independent physical campaigns for any proposed winning selection; retain
regressions. No performance estimate comes from the trace or this CPU test.
Splitting QA and performance into bounded runs requires a separate review. Routing, SiLU, SwiGLU, MoE throughput, public integration and
default changes are outside these gates and remain unqualified.
