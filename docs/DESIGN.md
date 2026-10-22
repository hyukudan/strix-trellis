# Independent EXL3/QTIP design for gfx1151

> **Type:** design note — engineering record; may describe plans, candidates or experiments that are not part of the supported API.

Date: 2026-10-20. The design was reviewed, with B1–B5 applied.
The complete M16 operator passes the 49 reference HIP fixtures; see
[NUMERICS.md](NUMERICS.md), and [API.md](API.md).
The M64 prefill variant is integrated, and its validation gate is being extended
with real weights. Resource estimates in the following sections do not replace
measured results in the reports. The design figures below are retained as
analytical hypotheses. Calculated costs and times are analytical estimates,
not results. Sources, revisions and licenses: [SOURCES.md](SOURCES.md).

## 1. Scope and notation

The goal is to consume existing EXL3 weights, reconstruct their exact FP16 values
in the rotated basis and multiply them efficiently on AMD gfx1151. This does not
include implementing Viterbi, converting models or reproducing the Cornell
optimizer. The Python reference is developed separately; this document specifies
contracts and tests. Official sources were read without copying upstream
implementations, code fragments, assembly or tests.

We use X with shape [B,I], Q with shape [I,O] and Y with shape [B,O]. I is the
reduction dimension and O is the output dimension. K is reserved for the format's
bits per weight; it is not the K dimension of a GEMM. The state has L=16 bits,
or 65,536 possible states. This does not mean that each weight occupies 16 bits.
A tile contains 16 inputs by 16 outputs, or 256 weights. “Confirmed” means
observed in the pinned upstream revision. It does not mean validated against
the bytes of a local checkpoint yet.

## 2. From QTIP to the EXL3 contract

QTIP combines an incoherence transform with sequence quantization through a
bitshift trellis. Path search occurs during quantization; inference extracts
local windows from the bitstream. The next state retains overlapping bits and
incorporates new bits. Procedural codebooks avoid a full table of 65,536 values.
Circular tile closure avoids storing an additional initial state. These are
the ideas in §§3.1–3.2 of the paper; its specific storage format must not be
confused with EXL3 [Q1,Q2].

Cornell also uses a vector HYB codebook with V=2 and a small LUT. The EXL3 path
studied here uses scalar procedural codebooks, V=1, and tiles of 256 values.
Neither the HYB LUT nor the kV shift in its kernels should be imported as EXL3
checkpoint requirements. The labels “1MAD”, “3INST” and “mul1” do not guarantee
interchangeable constants, operations or rounding [Q2,E1,E2].

### 2.1 Confirmed container and sizes

For the modern variant, `trellis` is contiguous int16 with shape
[I/16,O/16,16K]. Its int16 values are unsigned bit containers. Interpreting the
sign numerically breaks extraction. I and O must be multiples of 16 for the
tile and of 128 for the complete Hadamard path described here.

Each tile occupies 256K bits = 32K bytes = 16K 16-bit words. K is inferred by
dividing the last dimension by 16. Integer rates 1…8 are present in the packer;
upstream also includes half-integer rates for mul1. The first implementation
must explicitly declare which rates it accepts. Reject unknown rates, markers
or shapes rather than inferring them [E1,E3,E4].

Outer order: input tile first, then output tile, then words within the tile.
The byte offset of tile (a,b) is 32K × (a × (O/16) + b). This favors reading
neighboring output tiles for the same input. Traversing the entire reduction
for a single output has a large stride. Do not change the stored order to
simplify the kernel. Any repacking needs its own descriptor, with its cost
and memory accounted for.

### 2.2 Bit ring and independent extraction

For integer K, let R=256K and S(i)=(i+1)K, with i in 0…255. The state at
position i is the 16-bit ring window ending immediately before S(i).
Mathematically, its most significant bit comes from position (S(i)−16) mod R
and its least significant bit from (S(i)−1) mod R. Ring position zero is bit 31
of the first logical uint32. The uint32 values are stored little-endian: the
first physical uint16 is the low half and the second is the high half. Bit
order within the logical stream is MSB first. Distinguishing these two orders
avoids an incorrect “fix” with a simple byte swap [E3,E5].

For example, K=6 gives 192 bytes per tile and 48 uint32 values. The window at
i=0 takes 10 bits from the end of the tile and the first 6 bits from its start.
Never read bytes from an adjacent tile to resolve this wraparound. A loader can
combine two uint32 values from the same tile and shift/extract the window.
The operation must define the zero-shift case. Shifting a 32-bit integer by
32 positions in C++ is not a valid implementation of this specification.

For rates K=A+0.5, upstream alternates widths A and A+1 with mask 0xAAAA:
even positions advance A bits and odd positions A+1. S(i) is now the inclusive
sum of widths through i; the rest of the circular definition is unchanged.
There are 256K bits per tile, with no extra state. This is confirmed in the
revision consulted. Acceptance of these rates may be deferred until fixtures
are available [E1,E5].

There is no mandatory sequential dependency among the 256 states during
inference. A warp/wave can extract windows simultaneously. The trellis
recurrence is useful for checking overlaps, but does not justify a serial
decoder or device-side Viterbi.

### 2.3 Order of the 256 weights within a tile

The ring index is not the row-major index. A compatible permutation can be
specified without upstream code by decomposing i=8t+j, with t in 0…31 and
j in 0…7. Then:

- input row r = 2(t mod 4) + (j mod 2) + 8(floor(j/2) mod 2);
- output column c = floor(t/4) + 8 floor(j/4).

Q[16a+r,16b+c] receives the value of state i in tile (a,b). As a manual check,
the first eight positions are (0,0),(1,0),(8,0),(9,0),
(0,8),(1,8),(8,8),(9,8); the last eight are
(6,7),(7,7),(14,7),(15,7),(6,15),(7,15),(14,15),(15,15).
This permutation comes from the original CUDA fragment contract, but is fixed
in the format. An AMD backend must adapt its register distribution to it,
without reinterpreting the checkpoint using the WMMA layout [E2,E5].

### 2.4 Codebooks and mul1: exactness before optimization

All integer operations below use uint32, modulo 2^32, starting from a
zero-extended uint16 state. The constants are required format compatibility
data, not free parameters for the new backend [E2].

| Selection | Product/mixing | FP16 conversion |
|---|---|---|
| No marker, 3INST (cb0) | 89226354 × state + 64248484 | Apply mask 0x8FFF8FFF and XOR 0x3B603B60; reinterpret both halves as FP16; add with one FP16 rounding |
| `.mcg` (cb1) | 0xCBAC1FED × state | Same construction and sum of two FP16 values |
| `.mul1` (cb2) | 0x83DCD12D × state | Sum the four **unsigned** bytes of the product and follow the definition below |

For mul1, let s be the byte sum (0…1020). The FP16 pattern 0x6400+s represents
exactly the number 1024+s. The decoded value is an FP16-result FMA of that
number with the FP16 constants whose bit patterns are 0x1EEE and 0xC931,
rounded once to nearest with ties to even. Do not replace it with (s−510)/147.8:
neither the paper's normalization nor a decimal approximation is identical to
this FMA. Do not introduce intermediate FP16 rounding between multiplication
and addition. The host reference can evaluate the exact binary operands with
sufficient precision and apply a single final FP16 rounding.

On gfx1151, investigate `V_DOT4_U32_U8` with four ones for the byte sum.
Do not accidentally use a signed variant for bytes greater than 127. Keep an
independent extraction/addition alternative to isolate errors. Instruction
availability does not guarantee that the compiler selects that form or sustains
one instruction per cycle [A1].

Modern markers identify the codebook. The upstream loader checks markers and
expected multipliers. Do not design an API that allows arbitrary constants to
be rewritten without validation. Reject simultaneous mcg and mul1 markers as
ambiguous. Support for historical packed `su/sv` variants must be an explicit
loader extension, not an implicit conversion of magnitudes to signs [E1,E4].

### 2.5 suh, svh, Hadamard128 and out_scales

Define H_I and H_O as block-diagonal matrices, each block a Sylvester Hadamard
matrix of order 128 normalized by 1/sqrt(128). They are symmetric and square
to the identity in real arithmetic. With D_u=diag(suh) and D_v=diag(svh), the
effective weight is:

W = D_u H_I Q H_O D_v; therefore Y = ((X D_u) H_I) Q H_O D_v + bias.

It is confirmed that `suh` contains I FP16 values and `svh` contains O FP16
values. They include signs **and magnitudes**, and may contain zeros. Apply suh
before the input Hadamard and svh after the output Hadamard. General diagonal
matrices do not commute with H. Changing the orientation of X or W must
transform the entire equation, not swap su and sv based on their names [E1,E5,E6].

Review of `regularize` confirms that the converter can divide by output-channel
scales and incorporate them into sv, which is then serialized as svh. Automatic
selection depends on conversion statistics and sensitivity; subsequent scale
adjustments also exist. Inference must use the complete stored vectors. Do not
reconstruct scales from the RMS of Q or multiply by them a second time.
`apply_out_scales` is a converter decision, not a requirement for an additional
`.out_scales` tensor in the inspected modern contract [E4,E7].

**Unknown for the target model at this design stage:** its complete
configuration, converter history and header manifest had not been read.
Therefore, the chosen out_scales mode could not be established, nor whether it
used historical conventions, padding, external architecture scales or additional
metadata. If a tensor named out_scales appears, record its key, dtype, shape
and producer. Do not assume it is a scale per quantization group. The contract
above is confirmed for revision E, pending comparison with that checkpoint.

## 3. Proposed mapping on gfx1151

The official AMD table identifies gfx1151/Radeon 8060S with 40 CUs and waves
of 32 or 64 lanes [A2]. We choose wave32: it matches the 32 t positions per
tile and reduces communication cost compared with wave64. CUDA masks and its
SM model are not carried over. All mappings below are unmeasured proposals.

### 3.1 Microdecoder and GEMV

Starting point: one wave per tile, with lane t owning the eight positions
8t…8t+7. It extracts windows, decodes and produces eight FP16 values. This is
a correctness mapping; it does not ensure minimal global loads. Coordinating
contiguous uint32 loads per wave and sharing registers through permutations,
or placing a packed tile in LDS, are alternatives to compare with overlapping
loads. The cache may reuse loads, but those still consume instructions and
requests. Specializing by K and codebook allows positions to be precomputed
and general division or modulo operations to be removed from the loop. Packing
guarantees uint32 alignment per tile for the rates described, but not 128-byte
alignment for every tile: with K=6, the tile stride is 192 bytes. Grouped
transfers must respect both that alignment and each ring's wraparound.

For GEMV, each lane accumulates contributions from four input rows into two
output columns. Lanes sharing t/4 cooperate to reduce the 16 inputs of each
column. This can use exchange operations among groups of four lanes. The r,c
specification above must be checked with input impulses. Random matrices alone
are insufficient.

For B=1…8, decode once and reuse each weight for B rows. Keeping eight FP32
accumulators per lane for B=1, one per separate product, would be unnecessary.
Combining the four contributions per column requires two accumulators per row
of X: 2B accumulation VGPRs in this mapping, plus reduction temporaries.
Processing several column pairs or tiles increases that cost. Keeping only
one or two weights live instead of eight reduces registers at a scheduling
cost. Measure both options.

A CTA/workgroup of 4 waves (128 threads) is a starting point for exploration,
not a final launch configuration. Assigning neighboring outputs to neighboring
waves improves load continuity. Each wave traverses reduction tiles and retains
partial outputs. With O=512 there are only 32 output tiles; without splitting I,
parallelism may be insufficient for 40 CUs. Allow split-I with FP32 workspace
and a subsequent deterministic reduction. Avoid atomics in the validation path
and quantify their cost if tested later.

### 3.2 Hadamard and LDS

Hadamard128 can map four values per lane. Five butterfly stages exchange lanes
(distances 1,2,4,8,16); two stages operate on local values, depending on the
chosen mapping. Normalization is applied once with an explicit FP policy.
Initially, use FP32 for sums and multiplication by the scale, with FP16
conversion only at the declared boundary. The upstream-compatible path may
require reproducing its conversion points. Validate them separately before
fusion.

SGPRs: uniform pointers, bounds, strides, constants and variant selection.
VGPRs: packed words, states, codebook products, activations and accumulators.
LDS: shared X tiles, transposition to the WMMA fragment and inter-wave partials
when needed. Choose addresses with banks and alignment in mind. Do not assume
that half access avoids conflicts. Inspect the compiler's register,
scratch/spill and LDS report. This proposal has no measured or guaranteed
occupancy figure.

An expanded Q tile uses 512 bytes; a 16×16 FP16 X tile uses another 512 bytes.
Double-buffering both requires 2048 bytes per tile pair, without padding,
metadata or partials. A 16×16 FP32 accumulation tile contains 1024 bytes,
normally in distributed registers, not necessarily in LDS. Initial targets
are 48…64 VGPRs per lane and less than 16 KiB LDS per workgroup. These are
design budgets, not hardware limits or calculated occupancy figures.

### 3.3 GEMM with WMMA f16

Initially use WMMA 16×16×16 with FP16 inputs and an FP32 accumulator. EXL3 is
not an INT4/INT8 matrix directly usable by integer WMMA: its K bits encode
overlapping states. Decode first. The AMD WMMA fragment layout differs from
EXL3 tile order. Consider a transpose through LDS or register permutations;
verify with identity matrices and impulses [A1,A3].

RDNA3/3.5 documentation requires replicating operands from lanes 0…15 into
16…31 for wave32. The AMD guide describes 8 VGPRs per lane for each FP16 A
and B fragment, and 8 for FP32 C. This totals 24 fragment registers before
decode, addresses and buffering. Do not attribute another generation's
replication-free format to gfx1151. The specific builtin emission and wave32
mode remain to be verified with the compiler version and gfx1151 target [A1,A3].

As B grows, reusing a Q tile across several X rows amortizes decode and traffic.
Compare fused GEMM with temporary reconstruction followed by library GEMM.
The latter is a useful baseline, but expands weights, writes them and reads
them again. Never present its bandwidth as though it had read only EXL3 bytes.
For B<16, WMMA wastes rows. Compare with multirow GEMV; the crossover requires
measurement.

## 4. ALU budget for 220 GB/s

220 decimal GB/s is an **assumed target**, not measured bandwidth. With 40 CUs,
this corresponds to 5.5 GB/s per CU. At an assumed frequency f GHz, each CU must
consume 5.5/f bytes/cycle. In the conservative model of two SIMD32 units per CU,
with one basic vector instruction per SIMD per cycle, there are 64 lane
operations per cycle. The ideal budget is:

lane operations per byte = (40 × 64 × f × 10^9) / (220 × 10^9)
= 11.636 f.

| Assumed frequency | Bytes/CU/cycle | Lane operations/byte | Wave32 instructions/CU per byte |
|---|---:|---:|---:|
| 2.0 GHz | 2.750 | 23.27 | 0.727 |
| 2.5 GHz | 2.200 | 29.09 | 0.909 |
| 2.9 GHz | 1.897 | 33.75 | 1.055 |

The last column divides lane operations by 32. Do not confuse one wave
instruction with 32 scalar instructions. The model does not count FP dual issue
as a free doubling of integer capacity: integer multiplication, extraction,
permutations and WMMA have their own constraints and dependencies. It also does
not count one FMA as two instructions. At 2.5 GHz and a hypothetical efficiency
of 50%, 14.55 lane operations/byte would remain. That efficiency is a sensitivity
scenario, not a result [A1,A2].

| K bits/weight | Weights/byte | Ideal lane operations/weight budget at 2.5 GHz |
|---|---:|---:|
| 2 | 4.000 | 7.27 |
| 3 | 2.667 | 10.91 |
| 4 | 2.000 | 14.55 |
| 6 | 1.333 | 21.82 |
| 8 | 1.000 | 29.09 |

An approximate scenario of 8 lane instructions per weight to assemble a window,
extract, multiply, sum bytes, form FP16 and execute the mul1 FMA consumes
32 instructions/byte at 2 bpw, already above 29.09 before the dot product.
At 12 instructions, it consumes 48/byte at 2 bpw and 16/byte at 6 bpw. These
are not compiled ISA counts: window reuse, half2 packing, addresses and
permutations may change them. The dot product adds at least B FMAs per weight
in the simple vector model, or 8B/K lane operations per byte, plus X loads
and reductions. The formula allows actual disassembly to be evaluated without
promising 220 GB/s.

Plausible limits include integer issue/latency at low bit rates, redundant load
instructions, VGPR pressure and spills, redistribution to the WMMA layout,
LDS/barriers, shared CPU/GPU bandwidth, expert size and locality, and launch
overhead for small matrices. More CUs do not solve narrow outputs without
sufficient parallelism. Hadamard and routing stages must count toward
end-to-end latency even if the microkernel reaches its ceiling.

## 5. Numerical identity and oracles

Separating three contracts avoids diagnosing rounding as format corruption:

1. **Bits and codebook:** tile uint16 states and FP16 values must match the
   host oracle bit for bit. Test all 65,536 states of each supported codebook
   exhaustively. For mul1, check unsigned summation and a single FP16 FMA.
2. **Real-valued operator:** use decoded Q promoted to FP64, normalized H and
   stored diagonals promoted to FP64. This is the mathematical oracle for
   orientation, indices and scales. An x@W with W materialized in FP16 is not
   the same oracle: it introduces additional rounding in W.
3. **FP path:** specify dtypes and rounding in X D_u, H, the Q product,
   split-I reduction, output H, D_v, bias and output. Sequential FP32 FMA,
   reduction trees and WMMA need not produce identical bits. Reconstructing W
   with Hadamard followed by a dot product also need not match transforming X
   and multiplying by Q.

The initial goal is exact decode and FP32 accumulation, comparing against FP64
with absolute error, relative error, normalized RMS, maximum error, NaN/Inf and
zero output channels. Avoid using relative error alone near zero. Do not set a
universal tolerance without examining dimensions and scale ranges. To review a
threshold, relate it to sums of absolute product values and the reduction order.
The classical analysis gamma_n=n*u/(1−n*u) guides FP32 summation under its
assumptions; it does not guarantee WMMA or the complete Hadamard chain. Record
denormal mode, fast-math flags and compiler version. Agreement with an upstream
GPU implementation does not replace the host oracle.

Minimum cases: all-zero and all-one tile bit patterns, alternating patterns,
one changed bit, windows crossing word boundaries and ring wraparound,
impulses at all 256 coordinates, non-square matrices, mixed-sign scales with
different magnitudes, zero in svh, bias and B=1…8. One ring bit affects several
windows; do not expect it to change only one weight. Verify that the permutation
is bijective and extraction preserves circular overlap. Real fixtures check
compatibility; synthetic patterns check boundaries a fixture may not cover.

## 6. Stages and API boundaries

| Stage | Proposed contract | Exit criterion |
|---|---|---|
| 0: loader/reference | Immutable descriptor with I,O,K,codebook, strides, version, trellis/suh/svh/bias pointers and provenance | Reject ambiguous metadata; document fixture and hashes |
| 1: micro decode | Packed tile/range → uint16 states or Q FP16, selectable for diagnostics | Exhaustive codebook exactness and all ring positions |
| 2: core GEMV1…8 | Already transformed X [B,I] × Q → FP32 partials/output [B,O] | Correctness without hidden scales; deterministic split-I if used |
| 3: complete linear | pretransform(X,suh), core, posttransform(svh), bias | Compare with FP64 and FP-path oracles; report errors |
| 4: GEMM | Configurable B, WMMA tiles, explicit scratch | Measured GEMV/GEMM crossover; tested layout and tails |
| 5: grouped MoE | Per-expert descriptors, per-group offsets, token ids and routing weights | Stable routing, correct mixing and per-phase latency |

These are semantic boundaries, not implemented HIP signatures. The caller
provides a stream and workspace with queryable size/alignment. The hot path has
no allocations, global synchronization or implicit host transfers. A transformed
output can be reused only if its scales and operator match. Caching Q in FP16
must be a visible decision with a memory footprint and invalidation policy,
never a silent benchmark optimization.

For MoE, group tokens by expert with offsets/prefix sums and retain indices for
the final scatter. Each expert may have different suh/svh. Do not share an input
Hadamard among experts merely because X is common. The weighted sum occurs after
the expert's output transform and scaling, and after bias according to model
semantics. Evaluate gate/up, activation and down separately. Do not reorder SiLU
and Hadamard. Use stable FP32 top-k reduction as the baseline; atomics and later
fusion require explicit comparison.

## 7. Benchmarks with real dimensions and a cold working set

### 7.1 Observed local fixture without inferred headers

Only sizes, hashes and the marker of the four `layers.0.attn.wkv.*.bin` files
were read from a local fixture directory (set `STRIX_TRELLIS_FIXTURES`).
No shards were downloaded and no fixtures were modified. Observed sizes:
suh=10,240 bytes, svh=1024, trellis=1,966,080 and mul1=4. The little-endian
marker contains 0x83DCD12D. **Conditional inference**, assuming complete,
contiguous FP16/int16 tensors of the modern variant: I=5120, O=512 and K=6,
with trellis shape [320,32,96]. This inference matches the sizes exactly, but
the manifest with dtype, shape and shard range was still needed to confirm it.
Do not extrapolate the rest of the model's size from the directory name.

For this case, EXL3 weight bytes=1,966,080; scales=11,264 bytes. The ideal
read limit at 220 GB/s is 8.94 microseconds for trellis alone. It excludes
launches, activations, results, scales, decode and reduction. Expanded FP16 Q
occupies 5,242,880 bytes. A microbenchmark repeating this small fixture may be
measuring cache, so it does not validate 220 GB/s of DRAM bandwidth.

### 7.2 Pending experiment matrix

- Conditional wkv fixture: (I,O)=(5120,512), K=6, mul1; B=1,2,4,8 and
  B=16,32,64,128 for the GEMM crossover. Measure the penalty from narrow O
  and split-I.
- Published real DeepSeek-V3.1 shapes: hidden=7168, MoE intermediate=2048,
  256 experts, top-k=8, dense intermediate=18432 [M1]. Test gate/up
  7168→2048, down 2048→7168, dense 7168→18432 and its reverse. These are
  official geometries from **another model**, not verified shapes of the target
  model. Use compatible synthetic data to study performance and label it as such.
- For grouped MoE: uniform, concentrated and long-tail loads; 0,1,2,4,8,16,32,
  64 tokens per active expert and the actual top-k from the manifest once
  available. Distinguish unique tokens from token-expert assignments. Record
  unique-expert bytes and rereads per group. Blindly multiplying by top-k
  exaggerates traffic when there is reuse.
- Add each real checkpoint shape once its manifest is available. Do not make
  full-model claims from a single wkv matrix or square geometries.

For each shape, run micro decode, core, complete linear and complete MoE.
For decode, also count the FP16 writes: they outweigh compressed reads. For
GEMV, include at least weights, scales, inputs, outputs and workspace. Publish
both the algorithm's minimum bytes and observed traffic if counters are
available. Effective weight bandwidth is not automatically physical DRAM
bandwidth.

**Cold-cache protocol:** warm up compilation, page loading and clocks outside
the timed interval. Then rotate among disjoint weight buffers whose total
working set comfortably exceeds the hardware's identified last-level cache
(for example 4×, checking its actual size). Vary addresses and access order,
retain seeds and ensure the kernel consumes all weights. Report warm-cache
and cold-cache cases with resident pages separately. The first run with page
faults is another category. If copies of the same small fixture are used,
state that they study synthetic traffic and decode, not value or expert
coverage for a complete model.

Measure with device events and synchronize outside the interval. Include host
and GPU latency separately when launches matter. Publish repetitions, median
and percentiles, sufficient duration without hidden throttling, power/clock/
temperature and concurrent CPU load on unified memory. Every claim must include
the exact command, commit, toolchain, flags, hardware, dimensions, K/codebook,
working set, warmup, cache policy, bytes, times and correctness results. At this
design stage, there were no executable benchmark commands: the harness did not
yet exist here, and no CLI was invented. The future harness would need to expose
those parameters and save the actual command in the report.

## 8. Review gates before larger kernels

Before approving optimized GEMV/GEMM, the reviewer must check I/O orientation
and the complete permutation; circular windows and endianness; exhaustive mul1
with FP16 FMA; suh/svh with magnitudes; Hadamard normalization and FP boundaries;
the fixture manifest; the O=512 strategy; replicated WMMA layout; and that ALU
budgets are not presented as measured throughput. The Python reference can
resolve format uncertainties without a GPU. Resources, occupancy, emitted
instructions and performance crossovers remain hypotheses until authorized
compilation and measurement in a later phase.

## 9. Review implementation

B1: independent staged FP32/RN16 contract and CPU tests in NUMERICS; retains a
separate FP64 oracle. Tolerance is accepted for core reduction; decode is exact.
B2: transport implementation and API validation (in progress).
B3: suite v3 will pin sources/binaries and run exhaustive codebooks, the K6
fixture and a contiguous baseline. Do not extrapolate V1 to GEMV or observed
DRAM traffic.
B4: `src/design_probe.hip` checks 513 WMMA cases and sweeps split-I1/2/4/8/16/32/64
for O512 and O32768 before choosing a production launch configuration.
B5: `bench/micro-turn.sh` now delegates to the central runner, which wraps a
site-specific GPU queue script (not part of this repository). It checks local
time `TZ=:/etc/localtime`, budget and maintenance before and during the turn.

## 10. Implementation status after the proposal

The preceding sections retain the hypotheses from before measurement. B1–B5
have been addressed for the qualified runtime: validated C transport and shared
query/submit logic; exact decode against the CUDA oracle; kernels and benchmarks
with pinned sources/binaries; native WMMA qualified under its own tolerance;
and a runner with maintenance and local-time checks before and inside the queue.
The subsequent review resolved boundary-index and path-isolation issues.

The implementation selects vector/LDS GEMV for R1–8 and WMMA M16 or M64 with
swizzle/padding according to the measured geometry. The split policy is shared
between query and launch. Current results and the difference between the core
and complete API are summarized in the [README](../README.md#measured-performance). The
ALU/occupancy estimates in this proposal are not treated as measurements.
M128 reuse, persistent MoE kernels and external out_scales support remain
pending. Reproducible runners and commands exist in `tools/` and `bench/`.

## Budget observed with counters — October 21, 2026

| Shape / rows | Waves per kernel | VALU wave-inst/B | SALU wave-inst/B | LDS wave-inst/B | Read/B | Write/B |
|---|---:|---:|---:|---:|---:|---:|
| wq_b R1 | 8192 | 0.440937 | 0.126250 | 0.000625 | 1.000234 | 0.017036 |
| wq_b R8 | 8192 | 0.612500 | 0.128125 | 0.058125 | 1.001938 | 0.145171 |
| wo_b R8 | 1280 | 0.571436 | 0.095801 | 0.053906 | 1.005049 | 0.001929 |

These are measurements from actual GEMV v8 kernels, not the isolated decoder.
Wave instructions are issued instructions per compressed byte; reads/writes are
SDK GL2C/EA requests, not physical DRAM traffic. At 220 GB/s, VALU requires
97.01/134.75/125.72G wave instances/s in those three cases. This does not imply
ALU saturation. The profiler changes queues/timings, and MemUnitBusy R1 is
invalid (>100%); it is not used as a ceiling. The raw profiler data is an internal record
and is not published. Partial-result traffic and fixed cost per wave justify
measuring splits and output fusion before choosing another policy.
