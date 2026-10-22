# CPU numerical contracts
> **Type:** user documentation.

Date: 2026-10-20. Independent implementation in `python/strix_trellis/linear.py`.
This complements [FORMAT.md](FORMAT.md). Precision claims in this document
apply only to the stated contract. GPU qualification results for the
current runtime are summarized in the [README](../README.md#measured-performance).
Full parity with exllamav3 is not claimed. Decoding was cross-checked against
an external CUDA reference decoder without copying its implementation into the project.

## Preserved mathematical identity

For `X[..., I]`, FP16-decoded `Q[I,O]`, `u=suh[I]` and `v=svh[O]`,
H is a Sylvester Hadamard matrix in natural order, block diagonal with
128-element blocks, normalized by `1/sqrt(128)`.
I and O must be positive multiples of 128.

```text
W = diag(u) H_I Q H_O diag(v)
Y = ((X * u) H_I Q) H_O * v + bias
```

`matmul`, `hadamard` and `reconstruct_weight` retain their earlier FP64
implementations. They are the algebraic reference; they do not model
production activation conversions. Comparing an FP16/FP32 layer with this
reference must not assume that all differences come from matrix-product
reduction order. Intermediate rounding means that the production path has
no single effective matrix W independent of X.

## The `matmul_fp32` contract

All operations use IEEE round to nearest, ties to even.
Subnormals are preserved, without FTZ. Multiplication and addition are not fused.
`RN16` and `RN32` denote binary16 and binary32 conversion/rounding, respectively.
Real inputs are converted to FP32; scale vectors are converted to FP16
(existing FP16 values are preserved exactly). Scale signs, magnitudes and
zeros are meaningful. `out_scales` is not applied again.

| Checkpoint | Operation and stored dtype |
|---|---|
| `x_fp32` | `RN32(X)` |
| `scaled_fp32` | `RN32(x_fp32 * float32(RN16(u)))` |
| `scaled_fp16` | `RN16(scaled_fp32)` **before** Hadamard |
| `input_hadamard_fp32` | `H32(float32(scaled_fp16))` |
| `core_input_fp16` | `RN16(input_hadamard_fp32)` |
| `rotated_fp16` | Q decoded exactly as specified in FORMAT |
| `core_fp32` | Product of FP16 operands with FP32 accumulation, as described below |
| `output_hadamard_fp32` | `H32(core_fp32)` |
| `scaled_output_fp32` | `RN32(output_hadamard_fp32 * float32(RN16(v)))` |
| `output_fp32` | Copy of the preceding value, or `RN32(scaled_output_fp32 + RN32(bias))` |

`H32` applies Sylvester butterflies with strides 1,2,4,8,16,32,64.
Each addition and subtraction rounds to FP32. At the end, it multiplies once
by the FP32 constant `RN32(1/sqrt(128))`, computed from binary64.
It does not divide at each level. The output Hadamard transform and its
result are not converted to FP16.

The core product initializes each accumulator to FP32 +0 and iterates over i=0,…,I−1:
`p=RN32(float32(A_i)*float32(Q_io)); acc=RN32(acc+p)`.
These are separate operations, without FMA or BLAS, to fix the CPU order.
Products of two finite FP16 values are exactly representable in FP32;
summation order still matters. A kernel using WMMA, FMA or split-I must
state its order and measure its deviation. It need not reproduce the bits
of this scalar reduction to qualify as a valid approximation; equivalence
is not assumed either.

`bias` is optional, has shape `[O]`, and is added after `svh` in FP32.
Vector input and arbitrary batch dimensions are supported.
Scales or bias with other shapes are not broadcast. The functions reject
complex numbers, NaN/Inf, overflowing conversions and nonfinite intermediate
results with `ValueError`. Finite underflow rounds normally and is not rejected.
This CPU contract does not define a NaN/Inf propagation policy for production kernels.

## Stage comparison helpers

`matmul_fp32_stages` takes the same arguments as `matmul_fp32` and returns
the full checkpoint dictionary from the table.
`matmul_fp32` returns only `output_fp32`. The exported helpers are:

- `input_stages_fp32(x, suh)`: the first five checkpoints.
- `core_dot_fp32(core_input, rotated)`: the core result; converts both operands
  to FP16 before computing and validates their shapes `[..., I]` and `[I,O]`.
- `output_stages_fp32(core, svh, bias=None)`: the last three checkpoints.
- `hadamard_fp32(values, block_size=128, axis=-1)`: the isolated FP32 transform;
  allows other power-of-two sizes for small tests.

These helpers let each GPU stage consume exactly the preceding CPU output
and locate the first mismatch. A final match alone can hide compensating
sign, scale or layout errors.

## Tolerances and interpretation

Codebooks, permutations, signs and isolated FP16 conversions with identical
input must match bit for bit. Byte comparisons must also distinguish signed
zeros. The same FP32 operation order and IEEE mode allow bitwise equality
of FP32 checkpoints to be required. A Hadamard transform with a different
summation order needs a tolerance: a small difference before RN16 can cross
a rounding tie and produce an adjacent FP16 value. Count and locate such
mismatches rather than hiding them behind a broad final tolerance.

For the core, with identical operands and no overflow/underflow,
a conservative per-output bound against the real product is:

```text
u32 = 2^-24
gamma_n = n*u32 / (1 - n*u32)
abs(core_fp32 - A_real @ Q_real) <= gamma_(2I) * (abs(A_real) @ abs(Q_real))
```

A comparison between two reduction orders can add their bounds.
This bound does not cover input, layout or decoding errors.
It must not serve as a universal tolerance for the entire layer.
To propagate a core error `E`, the exact output Hadamard transform bounds
it componentwise by `abs(E) @ abs(H_O)`, followed by scaling by `abs(v)`
and the rounding error of each stage. For H32, seven additions/subtractions
and one multiplication give a conservative bound against exact H of
`gamma_9 * (abs(z) @ abs(H))` in the normal range, including the rounded
constant. Near zero, absolute bounds are needed: relative error alone is
not informative under cancellation. Underflow requires additional absolute
rounding terms; the relative bounds above do not cover it.

Small synthetic tests also use `rtol=2e-5, atol=1e-5` for the output Hadamard
transform against an explicit FP64 matrix, and gamma bounds for the core.
These are thresholds for those tests, not approved tolerances for an entire
model or evidence of logit accuracy. The involution `H H = identity` is exact
over the reals; H32 and the RN16 path only approximate it.

## Exhaustive codebook SHA-256 hashes

Order: integer states 0…65535, one little-endian binary16 value per state,
131,072 bytes. The hashes were fixed by computing the dyadic mathematics
with Python integers and binary16 rounding through `struct.pack`,
independently of the codec's NumPy LUT.
The test compares all bytes as well as both hashes.

| Codebook | SHA-256 |
|---|---|
| `3inst` | `ad88de2dcff5bb05fff2faec37cceceb43aa1c2e8203f9d81a6fbd6dd3393eea` |
| `mcg` | `95383563929baa9d1ae5de0be22490284382673620bdd85929feea7bdbef04a9` |
| `mul1` | `bc48d02cb1c14939dc47b90f870dd689d63e3b1ab69a157e65538db68677a6c8` |

For `3inst/mcg`, half operands are expressed in integer units of `2^-24`
and added before a single rounding step. For `mul1`, the exact numerator
`(1024+suma_bytes)*1774 - 1329*2048` is divided by `2^18`.
These are not hashes of upstream-generated outputs and do not establish
parity with upstream kernels or a complete layer.
They fix the published mathematics, constants and rounding.

## Upstream evidence and historical signs

The local MIT checkout of exllamav3 at revision
`a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a` was read only as a reference.
Files consulted: `modules/quant/exl3.py`, `exllamav3_ext/quant/hadamard.cu`,
`hadamard_inner.cuh` and `codebook.cuh`. The half paths round prescaling
to half; the file also contains a float variant whose prescaling remains
float. There are reconstructed paths and other dispatch choices.
The independent pipeline above is therefore not claimed to reproduce every
upstream path or to match even one complete path bit for bit.
The RN16 boundary before H is an explicit decision in the requested contract,
not a deduction of global equivalence from the FP64 equation.

Through its power-of-two masks, per-word expansion and flattening, the
upstream `unpack_bf` function documents that sign index `16*j+b` comes
from bit b of word j, starting with the least significant bit;
0 means +1 and 1 means −1. `decode_legacy_signs` implements this relation
independently with scalar integers. It accepts int16/uint16 vectors,
including big-endian NumPy arrays by numerical value, and returns FP16.
Tests cover all 16 bit impulses, word boundaries, signed storage and a
round trip through all 65,536 words. Disk bytes must first be interpreted
as `<i2` or `<u2`; the helper does not infer endianness or layer lengths.

This only handles historical sign unpacking. It does not provide a general
loader for old models, external scales or format variants.
The modern loader still requires its `suh/svh` tensors and current validation.

## Native gfx1151 evidence: WMMA FP16/FP32

The small test run on 2026-10-20 verifies all 512 A/B impulses exactly
to establish the fragment mapping. In a nonsymmetric matrix of small
integers, 107 native results differ from the exact integer, with an observed
maximum error of approximately 9.54e-6. The host integer/double references
agree; the builtin and direct assembly produce the same differences,
while the scalar FP32 control and diagnostic sign separation across four
WMMA operations are exact in this case.
No compiler or driver cause is asserted without evidence.

The native WMMA path is evaluated using absolute error and error normalized
by the sum of absolute products, as well as exact impulses.
The small acceptance test uses a conservative empirical envelope of
16*2^-24 for that ratio. This is not a formal proof of an IEEE instruction
bound or identity with scalar sums. Full-layer qualification uses the
documented per-fixture limits. Sign separation is a control,
not the default performance path.
