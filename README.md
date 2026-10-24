# strix-trellis

An independent inference library for EXL3 / QTIP trellis-quantized weights on
AMD Strix Halo (`gfx1151`, RDNA 3.5, wave32). It provides:

- a NumPy CPU reference that decodes EXL3 tensors exactly and runs reference
  matmuls with explicit rounding stages;
- a safetensors loader that reads EXL3 groups by memory map without expanding
  the weights;
- a HIP runtime with a C ABI: trellis decode, GEMV for 1–8 rows, WMMA GEMM for
  prefill, Hadamard (H128) transforms, and tensor-parallel (TP) rank paths.

The decode, GEMV, GEMM and transform kernels are written from scratch. See
[Background and credits](#background-and-credits).

> **Status: in use, with experimental parts.** The dense paths (the mix-B
> dense layers and every attention projection) run inside a DeepSeek V4.1
> Flash inference engine on two Strix Halo nodes, gated bit for bit end to end
> against that engine's reference. The K2 expert paths are measured on the GPU
> at operator level, bit-exact, but are not yet used in a full model. A few
> paths are CPU-tested only. The C ABI is at version 1 and the library is at
> 0.3.0. The [status table](#status) gives the level and the evidence for each
> component.

## Target model

The library is developed and tested against **DeepSeek V4.1 Flash**. The
quantization formats, the tensor shapes and all measured results in this
README come from that model:

- dense layers (attention `q_a`, `kv`, `q_b`, `o_a`, `o_b`; shared expert;
  output head) in EXL3 `mul1` format: 4-bit (K4) dense mix with a 5-bit (K5)
  head, called **mix-B** below;
- routed experts in K2 / K3 formats (kernels only; see [Limitations](#limitations));
- per-rank shapes for two-way tensor parallelism (table below).

Nothing in the C API is specific to this model. Any EXL3 tensor that meets the
[runtime constraints](#runtime-constraints) can be used.

### Per-rank shapes (two ranks)

All shapes are DeepSeek V4.1 Flash projections split for two ranks.
`I` is the input size, `N` the output size of one rank.

| Projection | Local shape `(I, N)` | Rows | Bit rates | Launches per call |
|---|---|---|---|---|
| `q_b` | `(1280, 16384)` | R1–R8 | K4 / K5 `mul1` | 1 |
| `o_b` | `(4096, 5120)` | R1–R8 | K4 / K5 `mul1` | 1 |
| shared gate / up | `(5120, 1152)` | R1–R8 | K4 / K5 `mul1` | 1 per projection |
| shared down | `(1152, 5120)` | R1–R8 | K4 / K5 `mul1` | 1 |
| `q_a` / `kv` | `(5120, 1280)` / `(5120, 512)` | R1–R8 | K4 / K5 `mul1` | 1 per projection |
| Experts gate/up + down | `(5120, 1152)` + `(1152, 5120)` | R1–R8 | K2 / K3 per weight, top-6 | 2 per MoE block |

### Quality of the mix-B dense format

Measured offline with a teacher-forced harness (not part of this repository).
The quantized model runs in PyTorch with the engine's numerics emulated
(8-bit activations, FP32 residual stream). It does **not** run through the
strix-trellis GPU kernels.

- **Reference:** the original DeepSeek V4.1 Flash weights (FP8 dense, FP4
  experts), unquantized activations, FP32.
- **Data:** 103,908 scored assistant tokens: 100 third-party chat
  conversations in 10 strata (agentic, code, tools, chat, multilingual, with
  and without thinking) and long tool sessions of 8K and 24–31K tokens.
- **Metric:** KL(reference ‖ quantized) over the full 129,280-token
  vocabulary, mean per token, 95 % confidence interval from a stratified
  bootstrap over conversations (5,000 resamples).
- **Fixed parts:** routed experts stay in a 2-bit GGUF format (not produced by
  this library) in all rows. Dense weights were quantized with Hessians
  collected from the original model.

| Dense format | Dense bytes per token per rank | KL, all tokens [95 % CI] | Top-1 agreement | Paired ΔKL vs 6.5-bit format [95 % CI] |
|---|---:|---:|---:|---:|
| Uniform 6.5-bit format (previous baseline) | 3.63 GB | 0.0447 [0.0417, 0.0480] | 94.00 % | — |
| EXL3 K5, all dense | 2.79 GB | 0.0444 [0.0415, 0.0475] | 93.99 % | −0.0003 [−0.0006, −0.0001] |
| **mix-B: EXL3 K4 dense, K5 head** | **2.27 GB** | **0.0446 [0.0417, 0.0476]** | **94.01 %** | **−0.0001 [−0.0004, +0.0002]** |

Reading: the 4-bit mix-B dense format has the same full-model KL as the
6.5-bit format within the confidence interval, and reads 1.36 GB less per
token on each rank. Its worst category relative to the 6.5-bit format is long
English documents (+0.0045, 274 tokens); all others are within +0.0015.
Almost all of the remaining distance to the original model comes from the
2-bit routed experts. On a 12,759-token subset, original dense weights with
the same 2-bit experts gave KL 0.0476, and 8-bit dense weights with original
experts gave 0.0024, the same as the original model with 8-bit activations.

Bytes per token count the quantized dense weights read by one of two TP ranks
for one decoded token.

## Status

Three levels are used below. **In use**: runs inside a full DeepSeek V4.1
Flash engine on two Strix Halo nodes and is gated end to end (bit-identical
tokens and logits against the engine's reference) and timed there.
**Measured on GPU**: bit-exact against the CPU reference or the previous
operator on hardware, with operator timings, but not yet used in a full
model. **CPU-tested**: implemented and covered by CPU contract tests only.

| Component | Level | Evidence |
|---|---|---|
| CPU reference (`python/strix_trellis`) | Reference | Exact decode for the `3inst`, `mcg` and `mul1` codebooks; integer rates 1–8 bits, plus 1.5 / 2.5 / 3.5 bits for `mul1`. Matches the EXL3 reference CUDA kernels bit for bit on 2,621,440 real weights and on all 65,536 states of each codebook. |
| Loader | In use | safetensors, mmap per group: `trellis`, `suh`, `svh`, codebook marker, optional bias. Exports borrowed regions or a reusable buffer. Rejects external `out_scales` and older formats without an explicit scale contract. |
| HIP runtime, dense GEMV R1–R8 and WMMA GEMM | **In use** | mix-B dense layers (shared expert, output head) of the DeepSeek V4.1 Flash engine. Operator qualification: campaign V16, see [Measured performance](#measured-performance). |
| Region upload (`strix_weights_prepare_parts`) | **In use** | Weight upload path of the engine loaders. |
| Prepared plans (`strix_linear_plan_*`) | CPU-tested | Implemented in 0.2; no separate GPU qualification yet. |
| Dense TP rank API (`strix_tp_dense_*`), dense R1 TP plans and grouped GEMV (`strix_tp_dense_plan_create_r1`, `strix_tp_dense_pair_plan_create_r1`, `strix_grouped_gemv_plan_*`) | **In use** (build option `STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP`) | Every attention projection (`q_a`‖`kv`, `q_b`, grouped `o_a`, `o_b`) of the DeepSeek V4.1 Flash engine runs through these plans. 40-layer bit-identical gate and A/B on hardware, 2026-10-24: greedy decode 34.7 → 30.6 ms/token, 6-token verify window 63.3 → 58.9 ms, 8 streams 85.4 → 83.6 ms/step. Exact-order split-K with S4 or LEGACY identities; grouped plans match `strix_gemv` R1 bits per row. See [DENSE_R1.md](docs/DENSE_R1.md). |
| K2 expert split plans, lean core, pipeline (`strix_tp_moe_plan_create_split`, `STRIX_TRELLIS_TP_MOE_PIPELINED_K2`) | **Measured on GPU** (build option `STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP`) | Bitwise equal to the 0.3 expert operator on GPU fixtures (24/24 cases, K2 and K2/K3) and to a lane-exact CPU reference; the pipeline with its measured configuration is on by default and was never slower than the non-pipelined operator in any measured cell. Not yet used in a full model (the engine's routed experts use another format). See [experimental-k2-lean.md](docs/experimental-k2-lean.md). |
| MoE TP rank API (`strix_tp_moe_*`) | Measured on GPU | Through the K2 split plans above. See [TP_RANK.md](docs/TP_RANK.md). |
| K2 R1 local-reduction kernel | CPU-tested | Off by default (`STRIX_TRELLIS_BUILD_EXPERIMENTAL_K2_LOCAL`). See [K2_LOCAL.md](docs/K2_LOCAL.md). |
| K2 A8R expert arithmetic (`STRIX_TRELLIS_TP_MOE_A8R`) | Experimental | Compiled out by default. A distinct arithmetic identity, **not** bitwise equal to the S4 path; not measured here. See [experimental-a8r.md](docs/experimental-a8r.md). |
| Grouped MoE (`strix_grouped_moe`) | Works | One stream over groups; no performance work. |
| Python HIP bindings | None | Python provides the reference and weight transport; C/C++ drives the GPU. |
| Full model inference | Not provided | The library supplies linear operators to an inference engine. |

"Implemented" does not mean "faster". Speed is claimed only for measured paths.

## Measured performance

All tables in this section were measured on **2026-10-24** on this revision
(0.3.0, commit `892b26e`). Where a row existed in the previous README, the
V16 qualification build (2026-10-21) was measured again in the same session,
alternating with this revision, and the previously published value is shown
for comparison.

**Hardware and software:** AMD Ryzen AI Max+ 395, Radeon 8060S GPU
(`gfx1151`, 40 CU, wave32), 128 GB unified memory (AMD lists 256 GB/s
peak bandwidth), ROCm 10.1, Linux.

**Method (same as V16):** one fixture per process, three repetitions with
alternating build order; each repetition reports the median of 11 sweeps of
HIP events, and the table shows the median of the three. Packed weights
rotate over disjoint copies of at least 256 MiB (640 MB for the plan tables)
so that they are not cache resident. Times include both H128 transforms,
decode and C API validation. Weight preparation, allocations and host copies
are excluded. Dense outputs are checked against the CPU reference before and
after timing; plan outputs are compared with `strix_gemv` bit for bit. The
timed head benchmark checks three of 1,010 H128 output blocks and finiteness;
a separate gate checks every head value.

**Units:** GB/s = packed weight bytes / operator time. It is not measured
DRAM traffic. Useful TOPS = 2·M·I·N / time of the full API call, including
decompression; it is not FP16 GEMM throughput.

**Read ceiling:** a standalone HIP streaming-read kernel (load and sum of
32-bit words, not part of this repository) was run in the same session. It
swept 392 launch shapes (grid, block, loads in flight, unroll) over a 1 GiB
buffer read cold. The best shape sustained **246.7 GB/s** (96.4 % of the
256 GB/s peak); half of the shapes reached 236 GB/s or more. *% of read
ceiling* = GB/s / 246.7.

**Call floor:** the same kernel, timed as one launch that reads S bytes
after 512 MB of unrelated reads, takes t ≈ 6.0 µs + S / 248 GB/s for S from
0.7 to 64 MB (for example 15.0 µs at 2.1 MB, 48.2 µs at 10.0 MB, 141 µs at
32 MB). *% of call floor* = that time / measured time: how close one operator
call is to a launch that does nothing but read its weights once. It is shown
for weights up to 64 MB.

Weights marked "EAQ" are real layer-0 tensors and the output head of a public
DeepSeek V4.1 Flash EXL3 checkpoint (EAQ v1). "Synthetic" rows use DeepSeek
V4.1 Flash shapes with synthetic weights and activations.

### Decode (GEMV)

| Tensor | K (bits) | Rows | Previous README (V16) µs | V16 build, same session µs | This revision µs | GB/s | % of read ceiling | % of call floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EAQ `q_b` | 5 | 1 | 130.752 | 130.723 | 130.720 | 200.5 | 81.3 % | 85.3 % |
| EAQ `q_b` | 5 | 8 | 150.422 | 151.364 | 154.506 | 169.7 | 68.8 % | 72.2 % |
| EAQ `o_b` | 5 | 1 | 124.304 | 124.617 | 124.705 | 210.2 | 85.2 % | 89.5 % |
| EAQ `o_b` | 5 | 8 | 146.171 | 145.270 | 146.259 | 179.2 | 72.7 % | 76.3 % |
| EAQ shared w1 | 5 | 8 | 52.137 | 51.685 | 51.833 | 142.2 | 57.7 % | 68.9 % |
| EAQ shared w2 | 5 | 1 | 39.138 | 39.090 | 39.036 | 188.9 | 76.6 % | 91.5 % |
| EAQ shared w2 | 5 | 8 | 46.660 | 46.665 | 46.811 | 157.5 | 63.8 % | 76.3 % |
| EAQ `q_a` | 5 | 1 | — | 34.391 | 34.411 | 142.8 | 57.9 % | 75.0 % |
| EAQ `q_a` | 5 | 8 | — | 38.414 | 38.046 | 129.2 | 52.4 % | 67.9 % |
| EAQ `o_a`, one slice | 5 | 1 | — | 22.495 | 22.490 | 116.6 | 47.2 % | 73.7 % |
| EAQ output head | 6 | 1 | 2258.708 | 2262.116 | 2262.357 | 219.4 | 88.9 % | — |
| EAQ output head | 6 | 8 | 2389.518 | 2392.366 | 2398.606 | 207.0 | 83.9 % | — |
| Synthetic `o_b` | 2 | 2 | 65.011 | 65.182 | 65.710 | 159.6 | 64.7 % | 73.4 % |
| Synthetic `o_b` | 2 | 8 | 91.766 | 90.963 | 92.379 | 113.5 | 46.0 % | 52.2 % |
| Synthetic `q_b` | 2 | 2 | 68.627 | 69.048 | 69.181 | 151.6 | 61.4 % | 69.7 % |
| Synthetic `q_b` | 2 | 8 | 99.923 | 100.295 | 101.241 | 103.6 | 42.0 % | 47.7 % |

- The generic GEMV kernels are the V16 kernels. This revision is within
  ±1.6 % of the V16 build in the same session on every row except EAQ `q_b`
  R8, which is 2.1 % slower (154.5 µs against 151.4 µs, the same in all
  three repetitions).
- The large single-row reads reach 81–89 % of the read ceiling and 85–90 %
  of the call floor. Small tensors (`q_a`, `o_a`) are limited by the fixed
  per-launch cost rather than by bandwidth.

### Prefill (WMMA GEMM)

| EAQ tensor | M | Previous README (V16) µs | V16 build, same session µs | This revision µs | Useful TOPS |
|---|---:|---:|---:|---:|---:|
| `q_b` | 256 | 1820.727 | 1833.235 | 1811.863 | 11.85 |
| `o_b` | 256 | 1481.859 | 1465.656 | 1436.243 | 14.95 |
| shared w1 | 128 | 246.790 | 240.717 | 243.143 | 12.42 |
| shared w2 | 256 | 415.835 | 412.373 | 410.307 | 14.72 |
| `q_a` | 256 | 263.889 | 257.699 | 257.312 | 13.04 |
| `o_a`, one slice | 256 | 212.899 | 208.064 | 208.044 | 10.32 |

Prefill is not limited by weight reads: at M = 256 the weights stream at 10–
15 % of the read ceiling. No compute ceiling was measured for this README, so
no percentage of peak is claimed for prefill.

### Dense R1 TP plans (decode, one rank)

`strix_tp_dense_plan_create_r1` with the LEGACY identity and the shipped
schedule policy, on the per-rank DeepSeek V4.1 Flash shapes, against
`strix_gemv` on the same weight copies. Synthetic weights; at least 640 MB of
rotating copies; one process per site, median of 11 sweeps (one run per
cell, not three repetitions). Plan and `strix_gemv` outputs are equal bit for
bit in every cell.

| Site | Local shape `(I, N)` | K (bits) | Rows | Plan µs | `strix_gemv` µs | Speed-up | GB/s | % of read ceiling | % of call floor |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `q_a` | `(5120, 1280)` | 4 | 1 | 21.06 | 29.73 | 1.41× | 155.6 | 63.1 % | 91.3 % |
| `q_a` | `(5120, 1280)` | 4 | 2 | 21.96 | 31.43 | 1.43× | 149.2 | 60.5 % | 87.5 % |
| `q_a` | `(5120, 1280)` | 4 | 6 | 26.75 | 41.69 | 1.56× | 122.5 | 49.7 % | 71.9 % |
| `q_a` | `(5120, 1280)` | 4 | 8 | 28.77 | 43.01 | 1.49× | 113.9 | 46.2 % | 66.8 % |
| `kv` | `(5120, 512)` | 4 | 1 | 12.47 | 20.53 | 1.65× | 105.1 | 42.6 % | 90.7 % |
| `kv` | `(5120, 512)` | 4 | 2 | 12.75 | 21.34 | 1.67× | 102.8 | 41.7 % | 88.7 % |
| `kv` | `(5120, 512)` | 4 | 6 | 16.00 | 24.70 | 1.54× | 81.9 | 33.2 % | 70.7 % |
| `kv` | `(5120, 512)` | 4 | 8 | 17.56 | 25.54 | 1.45× | 74.7 | 30.3 % | 64.4 % |
| `q_b` | `(1280, 16384)` | 4 | 1 | 61.45 | 74.20 | 1.21× | 170.7 | 69.2 % | 78.5 % |
| `q_b` | `(1280, 16384)` | 4 | 2 | 63.44 | 79.64 | 1.26× | 165.3 | 67.0 % | 76.1 % |
| `q_b` | `(1280, 16384)` | 4 | 6 | 78.70 | 110.28 | 1.40× | 133.2 | 54.0 % | 61.3 % |
| `q_b` | `(1280, 16384)` | 4 | 8 | 84.27 | 114.94 | 1.36× | 124.4 | 50.4 % | 57.2 % |
| `o_b` | `(4096, 5120)` | 4 | 1 | 52.48 | 67.32 | 1.28× | 199.8 | 81.0 % | 91.9 % |
| `o_b` | `(4096, 5120)` | 4 | 2 | 55.10 | 72.82 | 1.32× | 190.3 | 77.1 % | 87.6 % |
| `o_b` | `(4096, 5120)` | 4 | 6 | 62.91 | 106.71 | 1.70× | 166.7 | 67.6 % | 76.7 % |
| `o_b` | `(4096, 5120)` | 4 | 8 | 69.59 | 109.68 | 1.58× | 150.7 | 61.1 % | 69.3 % |
| shared gate / up | `(5120, 1152)` | 4 | 1 | 20.35 | 28.60 | 1.41× | 144.9 | 58.7 % | 88.0 % |
| shared gate / up | `(5120, 1152)` | 4 | 2 | 21.11 | 30.17 | 1.43× | 139.7 | 56.6 % | 84.8 % |
| shared gate / up | `(5120, 1152)` | 4 | 6 | 25.75 | 37.58 | 1.46× | 114.5 | 46.4 % | 69.5 % |
| shared gate / up | `(5120, 1152)` | 4 | 8 | 27.82 | 38.83 | 1.40× | 106.0 | 43.0 % | 64.3 % |
| shared down | `(1152, 5120)` | 4 | 1 | 19.21 | 26.25 | 1.37× | 153.5 | 62.2 % | 93.2 % |
| shared down | `(1152, 5120)` | 4 | 2 | 20.70 | 28.04 | 1.35× | 142.5 | 57.8 % | 86.5 % |
| shared down | `(1152, 5120)` | 4 | 6 | 24.95 | 36.88 | 1.48× | 118.2 | 47.9 % | 71.8 % |
| shared down | `(1152, 5120)` | 4 | 8 | 27.73 | 38.84 | 1.40× | 106.4 | 43.1 % | 64.6 % |
| output head | `(5120, 64640)` | 5 | 1 | 947.18 | 1011.14 | 1.07× | 218.4 | 88.5 % | — |
| output head | `(5120, 64640)` | 5 | 2 | 975.04 | 1064.06 | 1.09× | 212.1 | 86.0 % | — |
| output head | `(5120, 64640)` | 5 | 6 | 1064.69 | 1342.51 | 1.26× | 194.3 | 78.8 % | — |
| output head | `(5120, 64640)` | 5 | 8 | 1112.50 | 1202.01 | 1.08× | 185.9 | 75.4 % | — |

- At R1 every site except `q_b` reaches 88–93 % of the call floor (1.3–10.5
  MB of weights per call); for these small reads most of the distance to the
  read ceiling is the fixed cost of one launch. `q_b` (16,384 outputs)
  reaches 78.5 %.
- The output head uses the same plan at 88.5 % of the read ceiling at R1.

### Grouped GEMV: `o_a` (decode, one rank)

`strix_grouped_gemv_plan_*` runs the four `o_a` groups of one rank
(4 × `(4096, 1024)`) in one launch. Synthetic weights; 640 MB rotation; the
grouped output equals four `strix_gemv` calls bit for bit in every cell.

| K (bits) | Rows | Grouped plan µs | 4 × `strix_gemv` µs | Speed-up | GB/s | % of read ceiling | % of call floor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 52.14 | 107.06 | 2.05× | 160.9 | 65.2 % | 76.3 % |
| 4 | 2 | 55.21 | 112.11 | 2.03× | 151.9 | 61.6 % | 72.1 % |
| 4 | 6 | 68.15 | 143.19 | 2.10× | 123.1 | 49.9 % | 58.4 % |
| 4 | 8 | 73.93 | 150.07 | 2.03× | 113.5 | 46.0 % | 53.8 % |
| 5 | 1 | 58.55 | 90.18 | 1.54× | 179.1 | 72.6 % | 82.4 % |
| 5 | 2 | 60.41 | 109.03 | 1.80× | 173.6 | 70.4 % | 79.9 % |
| 5 | 6 | 77.53 | 120.59 | 1.56× | 135.2 | 54.8 % | 62.2 % |
| 5 | 8 | 84.57 | 107.73 | 1.27× | 124.0 | 50.3 % | 57.0 % |

### Experimental: K2 expert block (decode, one rank)

Operator timings of the TP expert block for one rank: K2 `mul1` gate/up
`(5120, 1152)` and down `(1152, 5120)`, top-6 routes per row, submitted as a
split plan. Synthetic weights (48 experts). "Pipeline off" is the same
revision built with `STRIX_TRELLIS_TP_MOE_PIPELINED_K2=OFF`; "Default" is the
default configuration (pipeline, lean core, adaptive rows, device route
maps). The outputs of the two builds are identical byte for byte, and both
pass the bitwise QA against the 0.3 operator (24 K2 and 24 mixed K2/K3
fixtures) on this revision.

**Method:** one process per run, four runs in A/B/B/A order; each run has
three pairs of 15 samples. The table shows the median over all pairs.
*Cold*: 256 MiB read before every submission. *Hot*: device-routed plan,
fixed weight addresses, 11 samples. GB/s counts the distinct expert bytes
read (gate/up and down of the experts selected by any row).

| Routes | Rows | Distinct experts | Weight MB | Pipeline off, cold µs | Default, cold µs | GB/s | % of read ceiling | Pipeline off, hot µs | Default, hot µs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Partly shared | 1 | 6 | 26.5 | 311.8 | 150.9 | 175.9 | 71.3 % | 235.7 | 112.8 |
| Partly shared | 2 | 9 | 38.3 | 516.0 | 213.9 | 179.3 | 72.7 % | 479.1 | 201.8 |
| Partly shared | 4 | 15 | 66.4 | 734.1 | 360.6 | 184.0 | 74.6 % | 744.7 | 379.3 |
| Partly shared | 6 | 21 | 92.9 | 1080.9 | 501.4 | 185.3 | 75.1 % | 1155.2 | 557.3 |
| Partly shared | 8 | 27 | 118.0 | 1349.2 | 638.2 | 184.8 | 74.9 % | 1541.6 | 718.3 |
| Disjoint | 6 | 36 | 156.3 | 1720.1 | 762.8 | 204.9 | 83.1 % | 1870.9 | 792.0 |
| All rows same experts | 6 | 6 | 26.5 | 392.8 | 279.8 | 94.9 | 38.5 % | 377.0 | 317.6 |

- The default configuration halves the cold time of the non-pipelined split
  path at every row count, and reads distinct expert weights at 71–83 % of
  the read ceiling.
- When all rows select the same six experts the block is not bound by weight
  reads (26.5 MB for six rows), so the ceiling percentage is low by
  construction.
- Earlier campaign (2026-10-23), lean core against the base S4 core of the
  same pipeline: 153.6–155.6 µs against 158.3–159.3 µs at R1 cold. Other
  defaults and their measured effects are listed in
  [experimental-k2-lean.md](docs/experimental-k2-lean.md);
  `STRIX_TRELLIS_TP_MOE_FOLD_R1` is exact but slower and stays off.
- Correctness: a lane-exact CPU model of the S4 core reproduces the GPU P and
  Y bit for bit; its `v_dot2acc_f32_f16` model has 0 mismatches on 7,340,032
  native gfx1151 cases.
- These are operator measurements only; no model throughput claim follows.

### Correctness

- V16 campaign: 84 GPU fixtures, 156 API cases and 20 graph-capture cases
  pass against the CPU reference. Tolerances: max abs ≤ 5e-4 × max(1,
  max |ref|), NRMS ≤ 3e-5 (see [NUMERICS.md](docs/NUMERICS.md)). Output head
  R1–R8 checked over all values against the CPU reference (FP64-ordered sums)
  and an independent CUDA reference; worst NRMS 2.39e-6.
- 2026-10-24: every dense API run (24 per build) and head run (3 per build)
  above passed its reference checks on both builds; every plan cell matched
  `strix_gemv` bit for bit; K2 QA 24/24 on both fixture sets.
- WMMA results are not bit-identical to scalar sums; the declared tolerances
  apply.

### Reproducing

Build with benchmarks enabled (see [Build with HIP](#build-with-hip) and add
`-DSTRIX_TRELLIS_BUILD_BENCHMARKS=ON`), create fixtures, then run the API
benchmark on an idle GPU:

```sh
python3 tools/fetch_dense_fixtures.py --directory "$FIXTURES/ranges"   # byte ranges only, no full shards
python3 tools/make_dense_gpu_fixtures.py \
  --real-directory "$FIXTURES/ranges" --out "$FIXTURES/gpu-dense"
build-hip/strix_trellis_api_linear_bench \
  "$FIXTURES/gpu-dense/layers.0.attn.wq_b.stgpu" \
  "$FIXTURES/gpu-dense/layers.0.attn.wo_b.stgpu"
```

The plan tables come from `strix_trellis_dense_r1_group_probe` (built with
`-DSTRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP=ON`) on synthetic weights:

```sh
export PROBE_ROTATE_MB=640 PROBE_SWEEPS=11 PROBE_POLICY_ONLY=1
build-hip/strix_trellis_dense_r1_group_probe tune1 1280 16384 4 1 2 6 8   # dense R1 plan, q_b shape
build-hip/strix_trellis_dense_r1_group_probe tune 4 1 2 6 8               # grouped o_a
```

Record the commit, compiler version, binary and fixture hashes, clocks and
temperature with every result. The qualification scripts used for V16 live in
`tools/` and assume the authors' test machines; see
[GPU_TESTS.md](docs/GPU_TESTS.md).

## Quick start (CPU)

Requirements: Python ≥ 3.10, NumPy ≥ 1.24. For the C/C++ checks: CMake ≥ 3.24,
a C11/C++17 compiler, Bash.

```sh
python -m pip install -e ./python
bash tools/ci_cpu.sh
```

```python
import numpy as np
from strix_trellis import load_exl3

weights = load_exl3("model-00001-of-00039.safetensors", "layers.0.attn.wkv")
x = np.zeros((1, weights.input_size), dtype=np.float32)
y = weights.matmul(x, production=True)          # reference with explicit rounding stages
blob, metadata = weights.transport_blob()        # input for strix_weights_prepare
regions, metadata = weights.transport_regions()  # input for strix_weights_prepare_parts
# The memoryviews keep the mmap alive until preparation finishes.
```

The reference expands the quantized weights, so use it on single layers. The
transport functions keep weights compressed. To split an output head by
vocabulary, `weights.slice_outputs(begin, end)` requires H128-aligned bounds.

## Build with HIP

Requires ROCm with `gfx1151` support. Use separate build directories for CPU
and HIP builds.

```sh
ROCM=/opt/rocm   # adjust to your ROCm install
cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release \
  -DSTRIX_TRELLIS_ENABLE_HIP=ON \
  -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/clang++" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM" \
  -DCMAKE_PREFIX_PATH="$ROCM" -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build-hip --parallel
```

This builds `libstrix_trellis.a` (CMake target `strix_trellis::runtime`) and
the test harness `strix_trellis_gpu_runtime`. `strix_trellis::api` is a
header-only interface target. GPU programs are not registered with CTest.

| CMake option | Default | Effect |
|---|---|---|
| `STRIX_TRELLIS_ENABLE_HIP` | OFF | Build the HIP runtime and GPU programs. |
| `STRIX_TRELLIS_BUILD_TESTING` | ON | Build CPU contract checks. |
| `STRIX_TRELLIS_BUILD_BENCHMARKS` | OFF | Build HIP benchmark programs. |
| `STRIX_TRELLIS_BUILD_EXPERIMENTAL_K2_LOCAL` | OFF | Build the K2 local-reduction candidate and its probe. |
| `STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP` | OFF | Build the TP dense / MoE rank APIs and their probe. |

More detail: [BUILD.md](docs/BUILD.md), [GPU_TESTS.md](docs/GPU_TESTS.md).

## API overview

The public header is [`include/strix_trellis.h`](include/strix_trellis.h)
(C ABI version 1). Full contract, memory ownership and stream rules:
[API.md](docs/API.md).

| Function family | Purpose |
|---|---|
| `strix_context_create` / `_destroy` | Device context; queries device resources once. |
| `strix_weights_prepare`, `strix_weights_prepare_parts`, `strix_weights_destroy` | Upload compressed weights from one blob or from borrowed regions. |
| `strix_matmul_desc_init`, `strix_matmul_workspace_bytes` | Describe a linear call (rows, alpha/beta, bias). |
| `strix_gemv`, `strix_gemm` | Decode (1–8 rows) and prefill (WMMA) linear operators. |
| `strix_linear_plan_create` / `_submit` / `_destroy` | Validate buffers and stream once, then submit repeatedly. |
| `strix_grouped_moe`, `strix_grouped_moe_workspace_bytes` | Pre-routed expert groups. |
| `strix_tp_dense_plan_*` | Per-rank dense projections on strided views (behind `STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP`). |
| `strix_tp_moe_*` | Per-rank fused expert block: gate/up + SiLU, down + weighted top-6 sum; host or device routing (behind `STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP`). |

Submit paths do not allocate, copy or synchronize. Callers own all buffers
and streams.

### Runtime constraints

- Integer bit rates 1–8 on the GPU.
- Input and output channel counts are multiples of 128.
- Contiguous F32 input and output (TP APIs also accept strided F32 views).
- F16 operands inside the kernels, F32 accumulation. No FP8 / FP4 paths.

## Format and numerics

- [FORMAT.md](docs/FORMAT.md): EXL3 tensor layout, trellis bitstream,
  codebooks, sign vectors and H128 transforms as implemented here.
- [LOADER.md](docs/LOADER.md): safetensors keys, markers and scale contract.
- [NUMERICS.md](docs/NUMERICS.md): rounding stages, tolerances and how GPU
  results are compared with the CPU reference.
- [DESIGN.md](docs/DESIGN.md): kernel design and the gfx1151 cost model.
- [SOURCES.md](docs/SOURCES.md): pinned sources, papers and licenses.

## Limitations

- Paths that the [status table](#status) does not list as in use or
  measured on GPU are CPU-tested only. The TP rank APIs still need
  `STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP=ON`, and their signatures may change.
- One GPU target: `gfx1151`. Other RDNA targets are untested.
- No inter-rank communication: TP rank outputs of `o_b` and expert down must
  be summed by the caller. No complete generation engine.
- Expert kernels: the private grouped K3 path passes 14 native
  configurations with up to 32 groups. A fast public grouped path and its
  performance are pending. Quality results above use 2-bit GGUF
  experts, not EXL3 experts.
- R8, small-matrix and K2 paths reach 104–179 GB/s in the decode table
  (42–73 % of the read ceiling), against 189–219 GB/s (77–89 %) for the best
  R1 paths.
- The quality numbers come from an emulation harness that is not included
  here, not from these kernels end to end.
- Model weights and binary fixtures are not stored in Git. Fixture scripts
  fetch byte ranges only.

## Background and credits

- **EXL3 format** by turboderp, defined by
  [exllamav3](https://github.com/turboderp-org/exllamav3) (MIT License). This
  library reads that format and was checked against it for correctness.
- **QTIP** — Albert Tseng, Qingyao Sun, David Hou, Christopher De Sa,
  *QTIP: Quantization with Trellises and Incoherence Processing*, NeurIPS 2024
  ([paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/6de2e84b8da47bb2eb5e2ac96c63d2b0-Paper-Conference.pdf),
  [arXiv 2406.11235](https://arxiv.org/abs/2406.11235),
  [code](https://github.com/Cornell-RelaxML/qtip), GPL-3.0). EXL3 builds on
  QTIP's trellis and incoherence-processing ideas.
- **AMD** RDNA 3.5 Instruction Set Architecture Reference Guide and ROCm
  documentation for `gfx1151`.
- **DeepSeek V4.1 Flash** by DeepSeek; range fixtures come from the public
  EAQ v1 EXL3 checkpoint
  ([Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw), MIT).

This is an independent reimplementation. Upstream sources and the paper were
read to establish the format and its arithmetic; no upstream implementation
code is included. Exact
revisions, files consulted and license notes: [SOURCES.md](docs/SOURCES.md).

## Contributing

Contributions are welcome. The development rules (also in
[AGENTS.md](AGENTS.md)):

- Reimplement algorithms independently. Do not copy upstream implementation
  code.
- Code, commits and documentation in English.
- Every performance claim needs the exact command, hardware, correctness
  result and measured numbers. Mark estimates as unmeasured.
- Keep GPU runs bounded and do not commit model weights or large fixtures.
- Run `bash tools/ci_cpu.sh` before sending a change; it must pass without GPU.

Release history: [CHANGELOG.md](CHANGELOG.md).

## License

Licensed under the [Apache License, Version 2.0](LICENSE). See [NOTICE](NOTICE)
for attribution.
