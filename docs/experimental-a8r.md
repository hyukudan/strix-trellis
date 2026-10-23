# Explicit K2 A8R arithmetic

This experimental identity is selected only by
`strix_tp_moe_plan_create_a8r` or `strix_tp_moe_plan_create_device_a8r`.
Build with `STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP=ON`,
`STRIX_TRELLIS_TP_MOE_PIPELINED_K2=ON` and
`STRIX_TRELLIS_TP_MOE_A8R=ON`. A8R is OFF by default. Its creators return
`STRIX_ERROR_NOT_IMPLEMENTED` when it is compiled out. Existing creators
continue to select their existing arithmetic in an A8R-enabled build.

Packed storage remains K2 EXL3 mul1. Format and paper attribution are in
[the project README](../README.md). Algorithms here are independently
implemented. This identity changes activation quantization, weight arithmetic
and SiLU; it does not claim bitwise equality with the original S4/DOT2ACC path.
Full-model quality and engine integration require separate qualification.

For each projection, scale the input in FP32, round to binary16, then apply
the ascending-stride normalized H128 transform in FP32. Its normalization is
the binary32 constant `0x1.6a09e6p-4`. In every consecutive group of32, let
`m=max(abs(x))`. If `m < 2^-119`, use codes0 and scale1. Otherwise set
`d=RN32(m/127)` and `q=clamp(RN-even(RN32(x/d)),-127,127)`. A nonfinite
group uses codes0 and canonical quiet-NaN scale `0x7fc00000`.

Let `b` be the sum of the four unsigned bytes of the modulo32 mul1 state
hash. The unrounded affine weight is `(887*b-452608)/131072`. For one
group, form the exact integer

```
N = sum_k ((887*b_k - 452608) * q_k)
```

The GPU uses mixed signed/unsigned integer dot4 and the equivalent
`887*sum(b*q)-452608*sum(q)`. The final integer fits signed32
(`abs(N) <= 1839398912`), although its intermediate products can exceed it.
Unsigned modulo32 arithmetic followed by a bit-preserving signed conversion
avoids signed overflow. Convert with `RN32(N)`, multiply by `2^-17`, then
multiply by `d`, with separate FP32 rounding boundaries.

Accumulate groups in ascending input order within each of the four original
S4 partitions, starting from positive zero. Output projection finish adds
the four partitions in order, applies H128, multiplies by the binary16 output
scale converted to FP32, then adds optional FP32 bias. No FP contraction or
fast math is permitted. The scalar reference uses a circular bit reader and
direct int64 affine accumulation, independent of the GPU lane reader and
dot4 decomposition.

SiLU uses the fixed operations in `src/tp_moe_a8r_math.hpp`, rather than a
platform exponential function: range reduction with split ln2 constants,
degree-seven Horner polynomial, and an exact power-of-two scale. At
`g>=80` it returns `g`; at `g<=-80` it returns signed zero from `g*0`.
Nonfinite gate values return canonical quiet NaN. The hidden value is a
separately rounded product of SiLU(gate) and up; nonfinite hidden values are
canonicalized. Down applies the same input quantization and projection rule.

Finish materializes each route score times its down value, then adds the
nonzero-score slots in original order from positive zero. Zero scores,
including negative zero, do not read hidden or down partials. Apply alpha
and, only when beta is nonzero, beta times old Y, using separate FP32 products
and addition. Invalid device routing or a nonfinite final value produces
canonical quiet NaN; valid neighboring tokens remain independent.

Query workspace using `strix_tp_moe_split_workspace_bytes` from the loaded
library. H and P retain their existing locations. Each projection's dead
activation region stores A8 codes, group scales and integer code sums;
down uses its own smaller region stride. No allocation, copy, host routing
inspection or synchronization occurs during submission. Fixed full submit
has five ordered kernel nodes. Staged gate, down-core and finish calls have
three, two and one respectively. `down_reference_core` rejects an A8R plan
because that diagnostic exports the original S4 identity.

The scalar reference assumes round-to-nearest-even host FP arithmetic and
FP contraction disabled. It is intended for bounded qualification fixtures,
not full-model weight expansion. Performance and whole-model correctness
must be established separately for a specific build and workload.

Optional schedules retain this identity. `A8R_WIDE` assigns complete outputs
to lanes; `A8R_OUTPUTS_PER_LANE=1|2` selects a single channel or a channel pair.
`A8R_SCALAR_INPUT` declares the shared activation values uniform within each
wave. `A8R_CLUSTER_UNROLL=1|2|4` and `A8R_HASH_FENCE=0|1|2|4` control hash
instruction lifetimes. A hash fence is an empty compiler dependency, not a
hardware synchronization primitive. The boolean options are OFF by default.
These option names have the full `STRIX_TRELLIS_TP_MOE_` prefix in CMake.

`A8R_PREPARE_AFFINE` optionally stores `-452608*sum(q)` and `d*2^-17` during
preparation. The correction fits signed32. Finite nonzero RN16/H128 inputs
bound `d` between `2^-35` and `2^13`: the smallest nonzero pre-normalization
value is at least `2^-24`, while the largest is at most `128*65504`.
Multiplication by `2^-17` remains normal and exact on both sides of the
final multiplication. Consequently `RN32(RN32(N)*2^-17*d)` equals
`RN32(RN32(N)*(d*2^-17))` in this domain. The scalar reference continues to
use the unfactored formula; native qualification compares H, P and Y against
that reference. Zero groups use scale1, and invalid groups retain the
canonical-NaN policy.
