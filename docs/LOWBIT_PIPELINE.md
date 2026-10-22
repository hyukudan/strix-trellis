# Proposal: R1 readers and pipeline for K3/K4

> **Type:** design note — engineering record; may describe plans, candidates or experiments that are not part of the supported API.

Status: design for independent review, with no new kernel or K3/K4 measurements
from this proposal. Confirmed D4 K2 is a separate project reference; its gain is
not extrapolated.

## Priority and provenance

Available EAQ shard1 headers contain 1152 K3/mul1 expert groups. Shard38 shows
K3 experts from layer39 and 400 MTP K4 groups, as well as dense K5/K6 groups.
This is a two-shard sample, not a complete census. No experts have been
downloaded for this design. Observed shapes: w1/w3 I5120,N2304;
w2 I2304,N5120. Current splits are 16 and 4 respectively.

The design starts solely from the project's D4/U4 algorithm and readers.
Upstream implementation code will not be translated. Keep the staged V1 hash,
separate from M-1, half2 FMA and each accumulator's order. Integer-bit R1/mul1 only.

## Access derivation

With canonical uint32 words, lane l and K bits, the first state of its eight
weights starts at bit(8*l+1)*K-16 of the 256*K-bit ring. State j takes 16 bits
from that start plus j*K, in the validated reader's same MSB order.

### K3: two words are enough

There are 24 words per tile. The unwrapped start is 24*l-13; adding 768 gives
24*l+755. index=floor(start/32)-24 ranges from -1 to 22; ia wraps -1 to 23,
and ib=index+1 stays in 0..23. Shifts are 3/11/19/27. The window needs
16+7*3=37 bits and 37+shift<=64: no third word is needed.

For a=word[ia], b=word[ib], high=(a<<shift) OR (b>>(32-shift)), low=b<<shift.
J0..5 uses low16(high>>(16-3*j)); J6/J7 uses
low16((high<<(3*j-16)) OR (low>>(48-3*j))). All shifts are defined. Omit masks
only where the consumer is proven u16; the test compares masked states.

Cooperative alternative: lanes0..23 load word[lane]; the others produce zero
without forming/loading an invalid address. After EXEC reconverges, two shuffles
obtain ia/ib. All source lanes0..23 exist and participate. Test this separately
from the two-load variant; do not assume it wins.

### K4: one local load plus circular shuffle

There are 32 words per tile and a constant shift20. b=word[lane],
a=word[(lane+31)&31], obtaining a from the previous lane with wrap0->31.
The global load is coalesced and all lanes participate.

| j | Expression, taking low16 at the end |
|---:|---|
| 0 | (a<<4) OR (b>>28) |
| 1 | (a<<8) OR (b>>24) |
| 2 | (a<<12) OR (b>>20) |
| 3 | b>>16 |
| 4 | b>>12 |
| 5 | b>>8 |
| 6 | b>>4 |
| 7 | b |

OR is bitwise, never addition. There are no shifts by 32. A two-load K4 control
isolates cooperation.

## Bounded trial

Fix the set before timing: K3 two-load U1/U4, K3 cooperative U4, K4 two-load U4
and K4 one-load/shuffle U4. Use v9 controls at the same split. U4 reads raw B/A
from four tiles before the first decode; it does not reload B/A during decoding.
The U1 epilogue stays within the split. Do not change the baseline after seeing
timings.

R1 only; X offset by 2 bytes retains the fallback. A partial N16 disables whole
waves, never shuffle lanes. There are no CTA barriers or actual LDS, although
permutes may count as LDS instructions. Output[split,total_rows,N] supports
backend row offsets.

## Gates

Before the kernel: independent review and a linear CPU test of the full
768/1024-bit basis, all lanes/states, wraps and high-bit contamination. Verify
that hoisted B/A values are consumed and that the K4 shuffle uses its own
iteration's load.

Compile with nice and one job, without spills. Inspect the linked image, early
loads, visible waits and earlyclobber on all temporaries/results in inherited
asm. Do not hide loads/waits in asm. Use the complete half pair; do not
scalar-bitcast FP16 elements.

GPU: exhaustive codebook, basis/random reader tests, I16/80/128/176/272 and
N16/128, S1/3/11, all U4 remainders/empty splits, X0/2, canaries and all small
partials against CPU. Expert shapes I5120/N2304 at S16 and the inverse at S4,
plus S4 controls. Use a real reference from a small range only after confirming
fixture-cache availability. Do not assume that sparse regions of partial shards
are populated.

Project results must be bitwise identical to the v9 core at the same S. The
complete operator uses existing thresholds without widening them. An optional
upstream decoder may serve as an externally linked correctness oracle, never
as copied implementation code.

At most two 180s GPU slots; synchronize/freeze both test machines, with one
pending request, and respect maintenance and site-specific quiet hours. Use
resident rotations >=256MiB, eleven alternating sweeps, complete checks
before/after/the final result, and minimum/median/p90. GPU<1GiB/host<2GiB;
no timed expanded Q, large downloads or additional test systems.

A core gain does not qualify CAPI/MoE. Only after confirmation should an exact
bits/shape/rows/S guard be proposed. Maintainers repeat API, graphs, complete
head and engine validation. If it does not win, keep v9. Format, transforms,
policy and defaults remain unchanged.
