# Proposal: R8 GEMV with in-block reduction and output H128

> **Type:** design note — engineering record; may describe plans, candidates or experiments that are not part of the supported API.

Status: design for independent review. This proposal has no new kernel or
measurements. The production reference remains v9. Figures from v8/v9 are not
attributed to this design.

## Rationale and initial domain

The v9 API reaches 190.90 GB/s on head R8, while q_b/wo_b R8 reach
158.33/173.11 GB/s. The v8 counters show more fixed work per byte and write
requests in q_b R8. The next experiment would remove the global partial tensor
and output launch, keeping the input transform separate. It does not assume
that this is the only bottleneck.

Initial domain: mul1 K5/K6, exactly R8, positive I/N multiples of 128, contiguous
FP16 A aligned to 4 bytes, canonical weights, FP32 Y. The caller supplies A
already transformed under the existing contract. There is no input fusion,
WMMA, atomics, inter-block barrier or new format. First measure q_b, wo_b,
shared, inverse and head. Shapes with few blocks may lose parallelism; do not
enable them based on geometry without confirmation.

## Proposed organization

A block covers 128 columns and eight rows. There are eight column waves per
K partition. G=1/2/4 internal partitions produce 256/512/1024 threads. Global
wave w identifies g=w/8 and n=w%8. The lane remains threadIdx.x&31. Each wave
keeps exactly GEMV v9's reader/decode, pairs0/1/2/3, dot2 and four-lane reduction.
Tile N = blockIdx.x*8+n; make the index and g uniform before any wave branch.

Ktiles=I/16 is a multiple of 8 and divisible by G. All partitions therefore
have the same length, and chunk8 loops, including tails, cross the same barriers.
No wave exits early. The block has no N tail because N%128=0 and the grid is N/128.

Shared activation storage has shape uint32[G][8*8*8], 2048 bytes per g. Within
each group of 256 threads, q=local_thread+256*t traverses the 512 pairs. Write
only if the corresponding K tile exists. After a CTA barrier, consume the eight
valid tiles in order. The second barrier precedes reuse of this memory.
Never read unwritten LDS entries.

At the end of the core, store FP32 partials in P[G][8][128]. For lanes that are
multiples of 4, the local column is n*16+lane/4+h*8, h=0/1. Each element has
one writer. A CTA barrier publishes P. Waves0..7 become the eight output rows;
the others finish after that barrier, and there are no further CTA barriers.

Each output wave loads columns lane+32*j, j=0..3, sums P[g] in ascending g
order and reproduces the exact FP32 output_transform sequence: natural H128,
normalization, svh, bias, alpha/beta and the beta0 case that does not read Y.
The project's helper code may be reused, but it must not call a global kernel
from another kernel or change existing transforms to introduce the experiment.

| G | Threads | LDS activations | LDS partials | Logical total |
|---:|---:|---:|---:|---:|
| 1 | 256 | 2048 B | 4096 B | 6144 B |
| 2 | 512 | 4096 B | 8192 B | 12288 B |
| 4 | 1024 | 8192 B | 16384 B | 24576 B |

These are logical allocations, not occupancy or emitted resources. Verify the
actual image and the 1024-thread kernel's limits without forcing a VGPR limit
that causes spills.

## Identity and causal controls

G4 must produce the same partials as the v9 core at S4 and bitwise-identical Y
if all reduction and output operations are preserved. LDS stores FP32 without
conversion. Preserve summation from +0 and signed-zero cases. Do not use H's
mathematical commutativity to reorder floating-point reduction.
The output_transform sequence is authoritative.

G1/G2 change the partition relative to S4: bitwise equality with S4 is not
promised. They must pass the original operator tolerance and an independent
reference, and be compared against existing cores at S1/S2. Do not widen
thresholds or hide failing cases.

A proposed G4 control uses the same new topology, writes global partials and
uses the old output stage. This distinguishes topology/load changes from output
fusion, although LLVM may still change register allocation. The baseline is
the complete v9 operator. Never compare a core without finalization against an
operator.

## Gates before integration

1. Independent review of this design before writing the large kernel. CPU proof
   of unique P and Y ownership, equal K lengths/barriers and coverage of every
   LDS access. Test all 128 H128 basis vectors and asymmetric patterns against
   the reference helper.
2. Start with G4 only. Complete logical A/P/Y dumps at small I128/256/384/512/640
   and N128/256 for chunk2/4/6 tails. Exact P against v9 S4; exact Y against
   the old sequence. Canaries, alpha/beta, bias, signed zeros, finite range and
   subnormals. Do not scalar-bitcast half-vector elements: earlier probes found
   incorrect code generation. Use complete words/memcpy.
3. Then test G1/G2 if G4 passes. Independent dump/CPU checks and operators at
   the same partition, with all fixture columns and the complete real head
   against the qualified baseline. Previously established exhaustive codebook
   coverage does not replace validation of the new thread layout.
4. Compile with nice and freeze sources/binaries on both test machines before
   each request. All GPU work uses the allocated queue slot and respects
   maintenance/site-specific quiet hours, with at most 180s per trial. Explicit
   bounds <1GiB GPU and <2GiB host. Do not expand Q in the timed path or download
   the model.
5. Two campaigns with fixed code: resident rotations >=256MiB, eleven alternating
   sweeps, complete results before/after/the final sweep, minimum/median/p90.
   Prepare and validate ranges/context outside experimental submit. The kernel
   launcher adds no HIP queries absent from the control. Do not subtract overhead
   from different protocols.
6. Only a confirmed winner may enter the API. Query and execution must then share
   the guard and required workspace. Repeat grouped and capture/replay tests,
   alongside 82 fixtures, the complete head and engine plain/bias. Until then,
   CMake/runtime/defaults remain unchanged.

The design alone does not solve very small matrices: N1024 provides only eight
N128 blocks. Grouping the eight wo_a projections or experts may be a later step,
requiring its own metadata, buffer and concurrency design. This proposal adds
no persistent global counter or new workspace-initialization requirement.

## Requirements added after independent review

Before implementation, adopt all P1-1/P1-2/P1-3 conditions from
the independent review. The first experiment
is limited to G4/R8: sum from +0 in ascending order, preserve the exact H128 graph
and current epilogue, prevalidate buffers and resources outside events, and use
separate policy, topology and fusion controls. P/Y must match byte for byte at
the same S4. This addendum does not enable a public path or reduce workspace.
Physical and API validation gates remain open.
