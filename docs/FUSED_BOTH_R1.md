# R1 experiment with both transforms fused

> **Type:** design note — engineering record; may describe plans, candidates or experiments that are not part of the supported API.

Design proposal, with no kernel or results. V11 remains the qualified reference.
The goal is to study dense R1 operator latency. No gain is assumed, and the
public path is not replaced.

A 1024-thread G4 CTA covers N128, with eight N16 waves per K group. Each group g
uses the qualified R1 kernel's H128 boundaries:
`beginH=floor((I/128)*g/4)`, `endH=floor((I/128)*(g+1)/4)`.
Its waves load original FP32 X and suh, materialize RN32, convert RN16, apply
the same FP32 H128 and store FP16 A in a group-private LDS cache. A CTA barrier
publishes all A before decoding. This experiment does not use global A workspace.

The cache keeps the entire partition, as the current R1 kernel does. The four
groups have a uniform stride of `ceil((I/128)/4)*128` half; I<=8192 bounds total
cache storage at 16384 bytes. The number of H and K16 tiles may differ between
groups, but there are no barriers inside load or decode loops. Each group
finishes its interval, keeps two R1 accumulators and publishes FP32 P[G4][128].
A second CTA barrier precedes wave zero, which sums g in ascending order from
+0, applies output H128 and svh/bias/alpha/beta, and does not read Y when beta
is zero. The other waves exit with no later barriers. P requires 2048 bytes LDS.

The candidate reader and decoder are the project's already tested SADDR/M−1
helpers. H128 and epilogue operations are preserved. A loads require an LDS
declaration with sufficient alignment and pair memcpy, without type aliasing
or scalar bitcasts of half components. There is no expanded Q, atomics, global
barriers, new format or precision change.

Required controls: the v11 operator with its actual policy; v11 fused input
forced to S4 with old output; new topology with global P and old output; direct
new topology. At the same H128 S4 partition, all P/Y must be identical to the
previous control, and complete A/H must be checked against CPU. Do not compare
P against the generic K16 split when boundaries differ. Shared v11 uses S16,
q_b uses K16 partitioning and other shapes may use different S. Check those
differences against the complete reference and existing tolerance.

Independent review before implementation must cover LDS ownership and bounds,
groups with empty or unequal H ranges, both barriers, RN32/RN16, N128/P/Y maps,
FP32 sums, alignment, function attributes and dynamic capacity. Initial domain:
R1, CB2/K5–K6, I/N multiples of 128 and I<=8192. Performance is initially limited
to q_b, wo_b, shared, inverse and head, with explicit negative controls. Static
resources do not prove occupancy or latency. Only after CPU tests, actual ISA,
complete numerical validation and two campaigns would a guard be proposed for
maintainer review. API, CMake, workspace and defaults remain unchanged.
