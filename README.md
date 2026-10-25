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
> 0.3.0; the current release checkpoint is **RC6** (`7a33c4b`,
> 2026-10-25), following RC5 (`5e86cb6`) on the same date. The
> [status table](#status) gives the level and the evidence for each component.

## At a glance

**Strix Trellis is a low-level kernel library, not a standalone LLM server.** It loads and executes trellis-quantized EXL3/QTIP tensors on AMD Strix Halo. Use it if you are integrating compressed linear layers into an inference engine or reproducing its numerical and performance measurements.

| Question | Answer |
|---|---|
| What runs today? | Dense GEMV/GEMM kernels used in DeepSeek V4.1 Flash, with end-to-end verification in the [Strix Halo engine](https://github.com/hyukudan/strix-halo-engine-lab). |
| What remains experimental? | K2 routed-expert operators and some APIs are operator-qualified or CPU-tested only. Check [Status](#status) before using a path. |
| Hardware and software | AMD Strix Halo (`gfx1151`) with a supported ROCm/HIP toolchain for GPU execution; Python 3.10+ and NumPy for the CPU reference. |
| Start without a GPU | [Quick start (CPU)](#quick-start-cpu) and `bash tools/ci_cpu.sh`. |
| Compile GPU kernels | [Build with HIP](#build-with-hip); enable experimental options only when needed. |
| Integrate into an engine | [C ABI](#api-overview), [API contract](docs/API.md), [build guide](docs/BUILD.md) and [GPU tests](docs/GPU_TESTS.md). |
| Reproduce benchmarks | [Measurements and commands](#measured-performance), [GPU test protocol](docs/GPU_TESTS.md). |

### RC6 defaults and engine adoption

| Path | Shipped policy | Measured evidence |
|---|---|---|
| Grouped `o_a` | Owner-fold for K4 R1–R8 and K5 R1–R4; K5 R5–R8 keep the pipelined split plan. | RC6 K5 R1: **51.27 µs**, **82.9 %** of the read ceiling; bit-identical to RC5 and independent R1 calls. [Grouped measurements](#grouped-gemv-o_a-decode-one-rank). |
| Dense TP plans | Issue staging loads before the first FP32 rounding boundary; pipeline K4 `o_b` at R1–R4. The K5 head at R2 retains its previous staging order. | [Cold plan measurements](#dense-r1-tp-plans-decode-one-rank); each row retains its declared S4 or LEGACY identity. |
| Direct-output R8 | Dual-issue dot order and next-row packed-weight prefetch enabled. | [Decode measurements](#decode-gemv); outputs unchanged bit for bit. |
| WMMA prefill | Register-blocked `mul1` K4–K6 core from 128 rows; fuse the output epilogue for single-split grids of at least 21 blocks on gfx1151. | Auto-fusion reduced summed operator time by **13–14 %**, with identical output bits. [Prefill measurements](#prefill-wmma-gemm). |

These are operator measurements on DeepSeek V4.1 Flash tensors or synthetic
weights of its shapes, not prompt-throughput results. Conditions and commands
are given below. Prepared-R1 one-launch completion remains opt-in and OFF.

The [Strix Halo engine](https://github.com/hyukudan/strix-halo-engine-lab)
uses this library for attention, shared-expert projections and the head on
two Strix Halo nodes over USB4. Attention uses the dense TP plans. Its
measured attention integration uses the
dense-plan series at `1e8011b`, incorporated into RC5; later engine builds
also carry a separate activation extension outside RC6. A full-model relink
against RC6 is pending qualification. The engine owns routing, buffers,
streams and inter-rank reductions; its routed experts use a different format
and do not call this library's K2 pipeline. The register-blocked WMMA core,
R8 direct-output changes and K5 owner-fold policy have operator qualification;
their full-model adoption is not established by those timings.

## Install and try it

Clone the repository and run its **CPU reference** first. The CPU tests do not require a Strix Halo GPU or model weights:

```sh
git clone https://github.com/hyukudan/strix-trellis.git
cd strix-trellis
python -m pip install -e ./python
bash tools/ci_cpu.sh
```

For **native GPU use** on a supported Strix Halo Linux machine, configure and build the HIP runtime:

```sh
ROCM=/opt/rocm   # change to match your installation
cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release \
  -DSTRIX_TRELLIS_ENABLE_HIP=ON \
  -DCMAKE_HIP_COMPILER="$ROCM/lib/llvm/bin/clang++" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$ROCM" \
  -DCMAKE_PREFIX_PATH="$ROCM" -DCMAKE_HIP_ARCHITECTURES=gfx1151
cmake --build build-hip --parallel
```

This produces the `strix_trellis` static runtime and GPU test harness; it does **not** download model weights or start an inference endpoint. To load a real EXL3 safetensors layer with the Python reference, follow [the example](#quick-start-cpu). For C/C++ integration, consult [API.md](docs/API.md). For full-model serving, see [Strix Halo Engine Lab](https://github.com/hyukudan/strix-halo-engine-lab); for routing/placement, [TensorMeld](https://github.com/hyukudan/tensormeld); for cluster deployment/control, [Hyperloom](https://github.com/hyukudan/hyperloom).

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
| Register-blocked WMMA GEMM core (prefill, `mul1` K4–K6, ≥ 128 rows) and its fused output epilogue (`auto`: grids of at least 21 blocks on gfx1151) | **Measured on GPU** | Bitwise equal to the previous WMMA core on GPU (qualification grid with partial tiles, odd and empty splits, alpha/beta with ±0, subnormal, infinite and NaN outputs; 7 EAQ tensors and the per-rank shapes at M = 128–4096); every fused-epilogue mode gives the same bits. Not yet used in a full model (`STRIX_TRELLIS_WMMA_TILED=0` selects the previous core). See [Prefill](#prefill-wmma-gemm). |
| Dual-issue order and next-row prefetch in the direct-output R8 GEMV (`STRIX_TRELLIS_DIRECT_DUAL_ISSUE`, `STRIX_TRELLIS_DIRECT_PREFETCH`) | **Measured on GPU** | Outputs bit-identical to the previous kernels in all 50 fixture/row cells; see [Decode](#decode-gemv). Not yet used in a full model. |
| Region upload (`strix_weights_prepare_parts`) | **In use** | Weight upload path of the engine loaders. |
| Prepared plans (`strix_linear_plan_*`) | CPU-tested | Implemented in 0.2; no separate GPU qualification yet. |
| Dense TP rank API (`strix_tp_dense_*`), dense R1 TP plans and grouped GEMV (`strix_tp_dense_plan_create_r1`, `strix_tp_dense_pair_plan_create_r1`, `strix_grouped_gemv_plan_*`) | **In use** (build option `STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP`) | Every attention projection (`q_a`‖`kv`, `q_b`, grouped `o_a`, `o_b`) of the DeepSeek V4.1 Flash engine runs through these plans. The attention integration and the later K4 owner-fold / staging changes have full-model gates; this does not qualify every RC6 policy in the engine. See [engine adoption](#rc6-defaults-and-engine-adoption). Exact-order split-K with S4 or LEGACY identities; grouped plans match `strix_gemv` R1 bits per row. The grouped K4 `o_a` (R1–R8) and K5 `o_a` (R1–R4) run an owner-fold kernel with the same bits as the split plan (GPU: outputs identical to the split plan and to row-wise `strix_gemv` at R1–R8, dispatch checked with a kernel trace). See [DENSE_R1.md](docs/DENSE_R1.md). |
| K2 expert split plans, lean core, pipeline (`strix_tp_moe_plan_create_split`, `STRIX_TRELLIS_TP_MOE_PIPELINED_K2`) | **Measured on GPU** (build option `STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP`) | Bitwise equal to the 0.3 expert operator on GPU fixtures (24/24 cases, K2 and K2/K3) and to a lane-exact CPU reference; the pipeline with its measured configuration is on by default and was never slower than the non-pipelined operator in any measured cell. Not yet used in a full model (the engine's routed experts use another format). See [experimental-k2-lean.md](docs/experimental-k2-lean.md). |
| MoE TP rank API (`strix_tp_moe_*`) | Measured on GPU | Through the K2 split plans above. See [TP_RANK.md](docs/TP_RANK.md). |
| K2 R1 local-reduction kernel | CPU-tested | Off by default (`STRIX_TRELLIS_BUILD_EXPERIMENTAL_K2_LOCAL`). See [K2_LOCAL.md](docs/K2_LOCAL.md). |
| K2 A8R expert arithmetic (`STRIX_TRELLIS_TP_MOE_A8R`) | Experimental | Compiled out by default. A distinct arithmetic identity, **not** bitwise equal to the S4 path; not measured here. See [experimental-a8r.md](docs/experimental-a8r.md). |
| Grouped MoE (`strix_grouped_moe`) | Works | One stream over groups; no performance work. |
| Python HIP bindings | None | Python provides the reference and weight transport; C/C++ drives the GPU. |
| Full model inference | Not provided | The library supplies linear operators to an inference engine. |

"Implemented" does not mean "faster". Speed is claimed only for measured paths.

## Measured performance

Where a row existed in an earlier README, the V16 qualification build
(2026-10-21) was measured again in the same session, alternating with the
measured revision, and the previously published value is shown for
comparison.

- **RC5 (`5e86cb6`), 2026-10-25:** `q_b`, `o_b`, shared down and
  output-head decode rows, and dense R1 plan table, alternating with RC4
  (`62e0123`) and with V16 for the decode rows. RC6 retains these kernels.
- **RC6 (`7a33c4b`), 2026-10-25:** grouped `o_a` table, six runs in two
  blocks alternating with RC5 and a byte-identical RC5 control. K5 rows
  1–4 select the new owner-fold kernel; other grouped kernels are unchanged.
- **`892b26e`, 2026-10-24:** other decode rows (`q_a`, `o_a` slice,
  shared w1, synthetic K2) and the K2 expert table; their compiled kernel
  instructions are unchanged in RC6.
- **RC4, 2026-10-25:** prefill tables, alternating the previous core, tiled
  core with separate output transform, and tiled core with forced fusion,
  one process per run. RC6 keeps those cores. RC5 introduced the default
  auto-fusion policy, measured separately below.

**Hardware and software:** AMD Ryzen AI Max+ 395, Radeon 8060S GPU
(`gfx1151`, 40 CU, wave32), 128 GB unified memory (AMD lists 256 GB/s
peak bandwidth), ROCm 10.1, Linux.

**Method for the decode, plan and head tables (same as V16):** one fixture per process, three repetitions with
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

**WMMA compute ceiling:** a standalone HIP kernel (not part of this
repository) that issues only independent `v_wmma` 16×16×16 instructions
from registers, swept over chains per wave, waves per SIMD and grid size,
was run on the same machine. Best burst rates with a cool GPU: FP16 → FP32
**53.9 TOPS** (90.8 % of the 59.4 TOPS that 40 CU × 512 FLOP/clock give at
2.9 GHz), BF16 → FP32 53.7, INT8 → INT32 54.7, INT4 → INT32 107.4. Run for
60 s, the same kernel holds 49.9 TOPS for about 10 s and then settles at
**47.6–48.3 TOPS** as the GPU reaches its thermal limit (98 °C, 2.53–2.58
GHz); per clock the rate is the same as in the burst. Prefill is reported
as *% of sustained WMMA* = useful TOPS / 47.6. Two more results from the
same kernel bound what a trellis GEMM can reach: VALU instructions do not
overlap with WMMA on this GPU (4 FP32 FMAs per WMMA cost 12 %, 16 cost
39 %), so every decode instruction in the inner loop costs WMMA issue time;
and feeding both operands from LDS reaches 46.0–46.6 TOPS.

Weights marked "EAQ" are real layer-0 tensors and the output head of a public
DeepSeek V4.1 Flash EXL3 checkpoint (EAQ v1). "Synthetic" rows use DeepSeek
V4.1 Flash shapes with synthetic weights and activations.

### Distance to the ceilings

The goal is 100 % of the measured ceiling for every class. Where this
revision stands (details and causes in the sections below):

| Class | Ceiling | Reached |
|---|---|---|
| Decode, large single-row reads (`q_b`, `o_b`, output head, R1) | read, 246.7 GB/s | 81–88 % |
| Decode, R8 direct-output (`q_b`, `o_b`, shared down, head) | read | 66–85 % |
| Decode, small R1 sites through `strix_gemv` (`q_a`, `o_a` slice) | read / call floor | 47–58 % / 74–75 % |
| Decode, dense R1 TP plans, R1 | read / call floor | 43–87 % / 78–94 % |
| Decode, dense R1 TP plans and grouped `o_a`, R8 | read | 32–77 % |
| K2 expert block, cold, partly shared or disjoint routes | read | 71–83 % |
| Prefill, M ≥ 512 | sustained WMMA, 47.6 TOPS | 32–70 % (median 48 %) |
| Prefill, M 128–256 | sustained WMMA | 14–71 % |

The prefill rows were measured with the separate output transform. The
default fused epilogue (`auto`, see [Prefill](#prefill-wmma-gemm)) reduced
the summed time of the measured prefill cells by a further 13–14 %.

### Decode (GEMV)

| Tensor | K (bits) | Rows | Previous README (V16) µs | V16 build, same session µs | RC5 / RC6 µs | GB/s | % of read ceiling | % of call floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EAQ `q_b` | 5 | 1 | 130.752 | 133.131 | 131.302 | 199.7 | 80.9 % | 85.1 % |
| EAQ `q_b` | 5 | 8 | 150.422 | 155.197 | 142.458 | 184.0 | 74.6 % | 78.4 % |
| EAQ `o_b` | 5 | 1 | 124.304 | 127.232 | 126.643 | 207.0 | 83.9 % | 88.2 % |
| EAQ `o_b` | 5 | 8 | 146.171 | 147.579 | 138.335 | 189.5 | 76.8 % | 80.7 % |
| EAQ shared w1 | 5 | 8 | 52.137 | 51.685 | 51.833 | 142.2 | 57.7 % | 68.9 % |
| EAQ shared w2 | 5 | 1 | 39.138 | 39.555 | 39.972 | 184.4 | 74.8 % | 89.4 % |
| EAQ shared w2 | 5 | 8 | 46.660 | 47.457 | 44.979 | 163.9 | 66.4 % | 79.4 % |
| EAQ `q_a` | 6 | 1 | — | 34.391 | 34.411 | 142.8 | 57.9 % | 75.0 % |
| EAQ `q_a` | 6 | 8 | — | 38.414 | 38.046 | 129.2 | 52.4 % | 67.9 % |
| EAQ `o_a`, one slice | 5 | 1 | — | 22.495 | 22.490 | 116.6 | 47.2 % | 73.7 % |
| EAQ output head | 6 | 1 | 2258.708 | 2298.479 | 2299.641 | 215.9 | 87.5 % | — |
| EAQ output head | 6 | 8 | 2389.518 | 2454.089 | 2367.406 | 209.7 | 85.0 % | — |
| Synthetic `o_b` | 2 | 2 | 65.011 | 65.182 | 65.710 | 159.6 | 64.7 % | 73.4 % |
| Synthetic `o_b` | 2 | 8 | 91.766 | 90.963 | 92.379 | 113.5 | 46.0 % | 52.2 % |
| Synthetic `q_b` | 2 | 2 | 68.627 | 69.048 | 69.181 | 151.6 | 61.4 % | 69.7 % |
| Synthetic `q_b` | 2 | 8 | 99.923 | 100.295 | 101.241 | 103.6 | 42.0 % | 47.7 % |

- The R8 rows of `q_b`, `o_b`, shared down and the output head use the
  direct-output kernels, which in this revision request the next K16 row
  before consuming the current one (`STRIX_TRELLIS_DIRECT_PREFETCH`, same
  bits). Against RC4 in the same session (three
  repetitions, alternating): `q_b` R8 153.37 → 142.46 µs (−7.1 %), `o_b`
  R8 146.61 → 138.34 µs (−5.6 %), shared down R8 46.79 → 44.98 µs
  (−3.9 %), head R8 2436.5 → 2367.4 µs (−2.8 %). All four are now faster
  than the V16 build. R1 rows are unchanged within ±1 %.
- In the same session, shared down rows 6, 32, 128 and 255 were slower than
  RC4 in all three repetitions (+0.8 to +7 % median). All
  four rows run device code that is instruction-identical to the previous
  release (row 255 uses two K splits and is not fused); row 256, which uses
  one split and the fused prefill epilogue (see
  [Prefill](#prefill-wmma-gemm)), is 7–10 % faster. A second session (two
  blocks of three paired runs) did not reproduce the row-255 difference:
  −0.12 % (95 % interval −1.40 to +1.16 %).
- The other rows were measured on `892b26e`, within ±1.6 % of the V16
  build in that session.
- The large single-row reads reach 81–88 % of the read ceiling and 85–88 %
  of the call floor. Small tensors (`q_a`, `o_a`) are limited by the fixed
  per-launch cost rather than by bandwidth.

### Prefill (WMMA GEMM)

The register-blocked WMMA core is the default for codebook
`mul1` at 4–6 bits and at least 128 rows (`STRIX_TRELLIS_WMMA_TILED=0`
selects the previous core). It keeps the split count, the split bounds and
the ascending K16 accumulation order of every output, so its outputs are
bit-identical to the previous core; only the block tiling, the LDS layout
and the block order change. Tile per row count: 128 rows → 128×256 (128×128
for `N` ≥ 16,384 or not a multiple of 256), 129–1024 rows → 256×128,
more → 128×128; the 256×128 and 128×128 tiles visit the blocks in groups
of two row blocks per column so that activation rows are reused while hot.
With one split the core can also apply the output transform in its
epilogue, same bits. By default it does so only when the grid has
more blocks than HIP reports multiprocessors for the device (20 on gfx1151, where HIP counts WGPs: at least 21 blocks) (`STRIX_TRELLIS_WMMA_TILED_FUSED=auto`): with at most one block
per WGP the epilogue's serial tail is exposed and fusing does not pay.
`=0` keeps the separate output transform everywhere, `=1` fuses whenever
possible. The tables below were measured with the separate transform on
RC4 (same core code).

**Method:** full operator (input H128 transform, WMMA core, output
transform) through the internal entry point that `strix_gemm` uses, so
without the C API argument validation. One process per run; the previous
core (A), the tiled core with the separate output transform (B) and the
tiled core with forced fusion (F, `STRIX_TRELLIS_WMMA_TILED_FUSED=1`) were
run three times each in the same session, in the orders A F B, F B A,
B A F. Each run reports the median of 11 calls; the tables show the
median of the three runs. The weights stay in place between calls (earlier
sweeps with weights rotated over ≥ 256 MiB gave the same core times within
a few percent: prefill is compute-bound). Bitwise gate: the previous and
the new core give the same output bits on a synthetic grid (partial tiles,
odd and empty splits, alpha/beta with ±0, subnormal, infinite and NaN old
outputs) and on every cell below; every cell's output hash is identical in
all nine runs and in an earlier session of ten runs. *% of sustained WMMA* =
useful TOPS / 47.6.

| EAQ tensor | `(I, N)` | K | M | Previous core µs | Tiled core µs | Speed-up | Useful TOPS | % of sustained WMMA (previous → tiled) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `q_b` (both ranks) | `(1280, 32768)` | 5 | 128 | 942.8 | 424.8 | 2.22× | 25.3 | 24 % → **53 %** |
| `q_b` (both ranks) | `(1280, 32768)` | 5 | 256 | 1800.2 | 993.9 | 1.81× | 21.6 | 25 % → **45 %** |
| `q_b` (both ranks) | `(1280, 32768)` | 5 | 512 | 3659.7 | 1940.0 | 1.89× | 22.1 | 25 % → **47 %** |
| `q_b` (both ranks) | `(1280, 32768)` | 5 | 1024 | 7365.5 | 3807.1 | 1.93× | 22.6 | 25 % → **47 %** |
| `q_b` (both ranks) | `(1280, 32768)` | 5 | 2048 | 14851.8 | 7441.3 | 2.00× | 23.1 | 24 % → **49 %** |
| `q_b` (both ranks) | `(1280, 32768)` | 5 | 4096 | 29251.6 | 15232.7 | 1.92× | 22.6 | 25 % → **47 %** |
| `o_b` (both ranks) | `(8192, 5120)` | 5 | 128 | 717.4 | 350.6 | 2.05× | 30.6 | 31 % → **64 %** |
| `o_b` (both ranks) | `(8192, 5120)` | 5 | 256 | 1329.3 | 634.7 | 2.09× | 33.8 | 34 % → **71 %** |
| `o_b` (both ranks) | `(8192, 5120)` | 5 | 512 | 3091.1 | 1321.5 | 2.34× | 32.5 | 29 % → **68 %** |
| `o_b` (both ranks) | `(8192, 5120)` | 5 | 1024 | 6308.3 | 3067.8 | 2.06× | 28.0 | 29 % → **59 %** |
| `o_b` (both ranks) | `(8192, 5120)` | 5 | 2048 | 12815.6 | 6161.2 | 2.08× | 27.9 | 28 % → **59 %** |
| `o_b` (both ranks) | `(8192, 5120)` | 5 | 4096 | 24817.4 | 11994.8 | 2.07× | 28.6 | 29 % → **60 %** |
| shared w1 | `(5120, 2304)` | 5 | 128 | 191.2 | 130.1 | 1.47× | 23.2 | 33 % → **49 %** |
| shared w1 | `(5120, 2304)` | 5 | 256 | 429.1 | 207.8 | 2.07× | 29.1 | 30 % → **61 %** |
| shared w1 | `(5120, 2304)` | 5 | 512 | 851.4 | 408.2 | 2.09× | 29.6 | 30 % → **62 %** |
| shared w1 | `(5120, 2304)` | 5 | 1024 | 1874.7 | 934.0 | 2.01× | 25.9 | 27 % → **54 %** |
| shared w1 | `(5120, 2304)` | 5 | 2048 | 3894.1 | 1885.2 | 2.07× | 25.6 | 26 % → **54 %** |
| shared w1 | `(5120, 2304)` | 5 | 4096 | 7769.7 | 3834.2 | 2.03× | 25.2 | 26 % → **53 %** |
| shared w3 | `(5120, 2304)` | 5 | 128 | 187.8 | 128.2 | 1.47× | 23.6 | 34 % → **49 %** |
| shared w3 | `(5120, 2304)` | 5 | 256 | 437.2 | 211.7 | 2.07× | 28.5 | 29 % → **60 %** |
| shared w3 | `(5120, 2304)` | 5 | 512 | 879.4 | 395.3 | 2.22× | 30.6 | 29 % → **64 %** |
| shared w3 | `(5120, 2304)` | 5 | 1024 | 1898.8 | 918.6 | 2.07× | 26.3 | 27 % → **55 %** |
| shared w3 | `(5120, 2304)` | 5 | 2048 | 3946.7 | 1903.6 | 2.07× | 25.4 | 26 % → **53 %** |
| shared w3 | `(5120, 2304)` | 5 | 4096 | 7858.2 | 3896.3 | 2.02× | 24.8 | 26 % → **52 %** |
| shared w2 | `(2304, 5120)` | 5 | 128 | 178.5 | 121.2 | 1.47× | 24.9 | 36 % → **52 %** |
| shared w2 | `(2304, 5120)` | 5 | 256 | 330.9 | 195.6 | 1.69× | 30.9 | 38 % → **65 %** |
| shared w2 | `(2304, 5120)` | 5 | 512 | 830.6 | 393.4 | 2.11× | 30.7 | 31 % → **65 %** |
| shared w2 | `(2304, 5120)` | 5 | 1024 | 1818.6 | 951.0 | 1.91× | 25.4 | 28 % → **53 %** |
| shared w2 | `(2304, 5120)` | 5 | 2048 | 3752.8 | 1928.7 | 1.95× | 25.1 | 27 % → **53 %** |
| shared w2 | `(2304, 5120)` | 5 | 4096 | 7632.9 | 3841.4 | 1.99× | 25.2 | 27 % → **53 %** |
| `q_a` | `(5120, 1280)` | 6 | 128 | 129.2 | 90.4 | 1.43× | 18.6 | 27 % → **39 %** |
| `q_a` | `(5120, 1280)` | 6 | 256 | 211.3 | 129.2 | 1.63× | 26.0 | 33 % → **55 %** |
| `q_a` | `(5120, 1280)` | 6 | 512 | 557.1 | 229.9 | 2.42× | 29.2 | 25 % → **61 %** |
| `q_a` | `(5120, 1280)` | 6 | 1024 | 1256.5 | 508.3 | 2.47× | 26.4 | 22 % → **55 %** |
| `q_a` | `(5120, 1280)` | 6 | 2048 | 2586.9 | 1263.3 | 2.05× | 21.2 | 22 % → **45 %** |
| `q_a` | `(5120, 1280)` | 6 | 4096 | 4874.7 | 2431.8 | 2.00× | 22.1 | 23 % → **46 %** |
| `o_a`, one slice | `(4096, 1024)` | 5 | 128 | 108.3 | 67.3 | 1.61× | 15.9 | 21 % → **34 %** |
| `o_a`, one slice | `(4096, 1024)` | 5 | 256 | 188.5 | 109.7 | 1.72× | 19.6 | 24 % → **41 %** |
| `o_a`, one slice | `(4096, 1024)` | 5 | 512 | 415.0 | 187.6 | 2.21× | 22.9 | 22 % → **48 %** |
| `o_a`, one slice | `(4096, 1024)` | 5 | 1024 | 783.8 | 393.7 | 1.99× | 21.8 | 23 % → **46 %** |
| `o_a`, one slice | `(4096, 1024)` | 5 | 2048 | 2034.7 | 969.0 | 2.10× | 17.7 | 18 % → **37 %** |
| `o_a`, one slice | `(4096, 1024)` | 5 | 4096 | 4026.3 | 2255.0 | 1.79× | 15.2 | 18 % → **32 %** |

Per-rank DeepSeek V4.1 Flash shapes (synthetic weights, K4 and K5), % of
sustained WMMA, previous core → tiled core (separate transform):

| Site | `(I, N)` | K | M 128 | M 256 | M 512 | M 1024 | M 2048 | M 4096 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `q_b` | `(1280, 16384)` | 4 | 16 → **49** | 24 → **50** | 24 → **43** | 25 → **45** | 25 → **48** | 25 → **48** |
| `q_b` | `(1280, 16384)` | 5 | 12 → **49** | 25 → **47** | 24 → **44** | 25 → **45** | 26 → **48** | 26 → **48** |
| `o_b` | `(4096, 5120)` | 4 | 18 → **56** | 34 → **71** | 31 → **70** | 28 → **60** | 27 → **53** | 27 → **54** |
| `o_b` | `(4096, 5120)` | 5 | 13 → **55** | 33 → **69** | 30 → **68** | 28 → **58** | 27 → **53** | 28 → **53** |
| shared gate / up | `(5120, 1152)` | 4 | 16 → **38** | 17 → **27** | 16 → **51** | 16 → **52** | 15 → **42** | 15 → **45** |
| shared gate / up | `(5120, 1152)` | 5 | 12 → **38** | 13 → **28** | 12 → **52** | 12 → **51** | 12 → **41** | 12 → **45** |
| shared down | `(1152, 5120)` | 4 | 17 → **44** | 31 → **57** | 31 → **59** | 26 → **46** | 24 → **42** | 24 → **43** |
| shared down | `(1152, 5120)` | 5 | 12 → **44** | 31 → **55** | 30 → **59** | 26 → **46** | 24 → **42** | 24 → **43** |
| `q_a` | `(5120, 1280)` | 4 | 14 → **15** | 18 → **31** | 17 → **57** | 16 → **56** | 15 → **46** | 15 → **48** |
| `q_a` | `(5120, 1280)` | 5 | 12 → **14** | 13 → **30** | 12 → **54** | 12 → **55** | 12 → **45** | 12 → **47** |
| `o_a`, one slice | `(4096, 1024)` | 4 | 16 → **20** | 17 → **24** | 16 → **45** | 15 → **47** | 15 → **37** | 15 → **33** |
| `o_a`, one slice | `(4096, 1024)` | 5 | 21 → **34** | 24 → **41** | 22 → **50** | 23 → **45** | 17 → **36** | 18 → **33** |

- The tiled core is faster than the previous core in all 114 cells: median
  2.04×, from 1.04× (`q_a` K4, M 128) to 4.58×. It reaches a median of 48 %
  of the sustained WMMA rate and at most 71 % (EAQ `o_b`, M 256); the
  previous core reached 12–38 %.
- The fused epilogue (F) takes up to 30 % less time than the separate
  transform in most single-split cells, for example the EAQ `q_b` at every
  M (45–53 % → 61–64 % of sustained) and shared down at M ≥ 1024. It was
  slower in three cells in all three paired runs (more than 0.5 % in the
  median): shared down K4 at M 128 (6–13 %), `q_a` K4 at M 128 (2–4 %) and
  the `o_a` slice K4 at M 256. All of them have at most one block per WGP,
  where the epilogue's serial tail is exposed. The default (`auto`)
  therefore fuses only on grids of at least 21 blocks. In a separate paired
  campaign (12 process pairs on two nodes, 19 shapes × 12 row counts) no
  fused cell was significantly (95 %) and materially (> 1 %) slower than
  the separate transform, and the summed time of the measured cells fell
  13–14 %. One validation run of this revision's default gave the same
  output bits as that campaign in all 228 cells and a summed time within
  1.2 % of it.
- Rows below 128 keep the previous core. Other codebooks and bit rates keep
  it at every row count.

**Distance to the ceiling.** At M ≥ 512 the default reaches a median of
48 % of the sustained rate (32–70 % by tensor and M; `q_b` 43–49 %, `o_b`
53–70 %); with the fused epilogue the `q_b` cells at M ≥ 512 reach
58–64 %. Controlled variants of the core on the EAQ `q_b` at M 4096 (core
only, earlier session, with an exploratory 128×256 tile; the shipped policy uses 128×128
at this size) give an indication of where the rest goes: full core 11.77
ms, without the trellis decode 9.96 ms, WMMA and fragment loads only 8.23
ms (88 % of the sustained rate). The decode costs time because VALU
instructions do not overlap WMMA on this GPU. We attribute most of the
remaining difference to activation staging through LDS and its barriers;
this is a hypothesis, because the fragment-only variant removes several
operations at once. The separate output transform adds further time on
shapes with a wide output. Small shapes are further away: at M 128–256,
the per-rank `q_a`, `o_a` and shared gate / up shapes give the 40 CU only
4–10 output tiles per K split (14–41 %; the EAQ gate / up tensors, `N` =
2,304 with up to 18 tiles, reach 49–61 % there), and the split count is part
of the numerical contract (more splits change the summation order and the
bits). Prefill streams the weights at a small fraction of the read ceiling,
so it is compute-bound, not bandwidth-bound.

### Dense R1 TP plans (decode, one rank)

`strix_tp_dense_plan_create_r1` with the LEGACY identity and the shipped
schedule policy, on the per-rank DeepSeek V4.1 Flash shapes, against
`strix_gemv` on the same weight copies. Measured on 2026-10-25 on RC5,
alternating with RC4 in the same session (three runs each). Synthetic weights; cold rotation over at least 640 MiB of
weight copies; one process per site and run, median of 11 sweeps per run,
median of the three runs. Plan and `strix_gemv` outputs are equal bit for
bit in every cell.

| Site | Local shape `(I, N)` | K (bits) | Rows | Plan µs | `strix_gemv` µs | Speed-up | GB/s | % of read ceiling | % of call floor |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `q_a` | `(5120, 1280)` | 4 | 1 | 21.09 | 30.19 | 1.43× | 155.4 | 63.0 % | 91.1 % |
| `q_a` | `(5120, 1280)` | 4 | 2 | 22.35 | 31.85 | 1.43× | 146.6 | 59.4 % | 86.0 % |
| `q_a` | `(5120, 1280)` | 4 | 6 | 26.90 | 45.36 | 1.69× | 121.8 | 49.4 % | 71.4 % |
| `q_a` | `(5120, 1280)` | 4 | 8 | 29.08 | 47.43 | 1.63× | 112.7 | 45.7 % | 66.1 % |
| `kv` | `(5120, 512)` | 4 | 1 | 12.28 | 20.62 | 1.68× | 106.7 | 43.3 % | 91.9 % |
| `kv` | `(5120, 512)` | 4 | 2 | 12.55 | 21.56 | 1.72× | 104.4 | 42.3 % | 89.9 % |
| `kv` | `(5120, 512)` | 4 | 6 | 15.58 | 24.97 | 1.60× | 84.1 | 34.1 % | 72.4 % |
| `kv` | `(5120, 512)` | 4 | 8 | 16.79 | 25.83 | 1.54× | 78.1 | 31.6 % | 67.2 % |
| `q_b` | `(1280, 16384)` | 4 | 1 | 62.20 | 75.58 | 1.22× | 168.6 | 68.3 % | 77.6 % |
| `q_b` | `(1280, 16384)` | 4 | 2 | 64.87 | 82.11 | 1.27× | 161.6 | 65.5 % | 74.4 % |
| `q_b` | `(1280, 16384)` | 4 | 6 | 77.66 | 115.30 | 1.48× | 135.0 | 54.7 % | 62.2 % |
| `q_b` | `(1280, 16384)` | 4 | 8 | 82.91 | 121.94 | 1.47× | 126.5 | 51.3 % | 58.2 % |
| `o_b` | `(4096, 5120)` | 4 | 1 | 51.68 | 69.19 | 1.34× | 202.9 | 82.2 % | 93.4 % |
| `o_b` | `(4096, 5120)` | 4 | 2 | 53.29 | 75.96 | 1.43× | 196.8 | 79.8 % | 90.6 % |
| `o_b` | `(4096, 5120)` | 4 | 6 | 63.45 | 111.13 | 1.75× | 165.3 | 67.0 % | 76.1 % |
| `o_b` | `(4096, 5120)` | 4 | 8 | 68.16 | 115.17 | 1.69× | 153.8 | 62.4 % | 70.8 % |
| shared gate / up | `(5120, 1152)` | 4 | 1 | 20.38 | 28.75 | 1.41× | 144.7 | 58.7 % | 87.8 % |
| shared gate / up | `(5120, 1152)` | 4 | 2 | 21.32 | 30.55 | 1.43× | 138.3 | 56.1 % | 83.9 % |
| shared gate / up | `(5120, 1152)` | 4 | 6 | 25.74 | 38.68 | 1.50× | 114.6 | 46.4 % | 69.5 % |
| shared gate / up | `(5120, 1152)` | 4 | 8 | 27.14 | 39.70 | 1.46× | 108.7 | 44.0 % | 65.9 % |
| shared down | `(1152, 5120)` | 4 | 1 | 19.04 | 26.47 | 1.39× | 154.9 | 62.8 % | 94.0 % |
| shared down | `(1152, 5120)` | 4 | 2 | 20.69 | 28.21 | 1.36× | 142.5 | 57.8 % | 86.5 % |
| shared down | `(1152, 5120)` | 4 | 6 | 24.58 | 39.06 | 1.59× | 120.0 | 48.6 % | 72.8 % |
| shared down | `(1152, 5120)` | 4 | 8 | 27.19 | 39.55 | 1.45× | 108.5 | 44.0 % | 65.8 % |
| output head | `(5120, 64640)` | 5 | 1 | 965.48 | 1054.64 | 1.09× | 214.2 | 86.8 % | — |
| output head | `(5120, 64640)` | 5 | 2 | 1014.63 | 1098.38 | 1.08× | 203.9 | 82.6 % | — |
| output head | `(5120, 64640)` | 5 | 6 | 1083.16 | 1413.43 | 1.30× | 191.0 | 77.4 % | — |
| output head | `(5120, 64640)` | 5 | 8 | 1092.54 | 1233.82 | 1.13× | 189.3 | 76.7 % | — |

- At R1 every site except `q_b` reaches 88–94 % of the call floor (1.3–10.5
  MB of weights per call); for these small reads most of the distance to the
  read ceiling is the fixed cost of one launch. `q_b` (16,384 outputs)
  reaches 77.6 %.
- The output head uses the same plan at 86.8 % of the read ceiling at R1.
- Against RC4 in the same session the plans are faster in
  27 of 28 cells (up to 7 %; the staging loads of each 128-value block are
  now issued before the first FP32 boundary). The output head at R2 was
  slower in all three runs (median +0.6 %; the R2 row of the table is that
  measurement). That one cell (two rows in one block) now keeps the previous
  staging order, with the same instructions as RC4's
  kernel and the same bits: in a second session (two blocks of three paired
  runs) it measured +0.11 % against RC4 (95 % interval
  −0.05 to +0.27 %), where the new order measured +0.88 %.
- The `o_b` K-half (4096 → 5120, 4 splits) uses pipelined weight batches at
  rows 1–4: R1–R4 51.68 / 53.29 / 54.85 / 56.83 µs against 53.76 / 54.92 /
  56.75 / 59.02 µs with `STRIX_TRELLIS_DENSE_PIPE=0` in the same session.

### Grouped GEMV: `o_a` (decode, one rank)

`strix_grouped_gemv_plan_*` runs the four `o_a` groups of one rank
(4 × `(4096, 1024)`) in one launch with the shipped policy: K4 uses the
owner-fold kernel (see [DENSE_R1.md](docs/DENSE_R1.md)) at every row
count, K5 the owner-fold kernel at rows 1–4
(`STRIX_TRELLIS_GROUP_K5_OWNER`) and the pipelined split schedule from row
5 (`STRIX_TRELLIS_GROUP_K5_PIPELINE`). Measured on 2026-10-25 on RC6.
Synthetic weights; cold rotation over 648 MiB (K4, 81 copies)
and 650 MiB (K5, 65 copies), 11 sweeps per run, median of six runs in two
blocks, alternating with RC5 and a byte-identical copy
of it in the same session. "4 × `strix_gemv`" is the same four projections
as four calls over the same rotation. GB/s, read ceiling and call floor as
defined above. The grouped output equals row-by-row `strix_gemv` bit for
bit in every cell, and the outputs are identical to the previous
release's.

| K (bits) | Rows | Grouped plan µs | 4 × `strix_gemv` µs | Speed-up | GB/s | % of read ceiling | % of call floor |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1 | 43.70 | 106.94 | 2.45× | 192.0 | 77.8 % | 91.1 % |
| 4 | 2 | 44.67 | 111.97 | 2.51× | 187.8 | 76.1 % | 89.1 % |
| 4 | 6 | 60.46 | 143.00 | 2.37× | 138.7 | 56.2 % | 65.9 % |
| 4 | 8 | 65.20 | 150.57 | 2.31× | 128.6 | 52.1 % | 61.1 % |
| 5 | 1 | 51.27 | 90.01 | 1.76× | 204.5 | 82.9 % | 94.2 % |
| 5 | 2 | 51.52 | 108.94 | 2.11× | 203.5 | 82.5 % | 93.7 % |
| 5 | 6 | 59.42 | 120.75 | 2.03× | 176.5 | 71.5 % | 81.2 % |
| 5 | 8 | 60.70 | 107.64 | 1.77× | 172.8 | 70.0 % | 79.5 % |

- K5 against RC5 in the same session (median of six):
  R1 53.81 → 51.27 µs (−4.7 %), R2 54.93 → 51.52 µs (−6.2 %), R3 55.33 →
  54.40 µs, R4 55.83 → 54.59 µs; rows 5–8 and K4 run the same kernels as
  RC5 and were within ±0.6 %. With the owner-fold kernel
  at every K5 row (an earlier session), rows 6–8 were 5–10 % slower than
  the split plan, so rows 5–8 keep it.
- K4, owner-fold against the split plan in an earlier session (median of
  three paired runs): R1 44.52 / 46.48 µs, R2 44.86 / 47.96, R6 58.92 /
  62.08, R8 67.61 / 68.83; faster at every row count from 1 to 8. A
  measurement of the split plan in an earlier session gave R8 63.96 µs;
  session-to-session differences of this size are why only same-session
  comparisons decide defaults here.

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
- Prefill core (this revision): decoder identity on all 65,536 states, the
  synthetic bitwise grid and all 114 timed cells give the same output bits
  with the previous and the new core, in every run.
- Build of this revision, on the GPU: every prefill cell above has the same
  output hash as the build without the dual-issue change, and the decode
  API outputs of `q_b`, `o_b` and shared down (rows 1–256, checked against
  the CPU reference) and of the output head (R1–R8) are bit-identical to
  those of the build used for the decode table.
- WMMA results are not bit-identical to scalar sums; the declared tolerances
  apply.

### Reproducing

Build with benchmarks enabled (see [Build with HIP](#build-with-hip) and add
`-DSTRIX_TRELLIS_BUILD_BENCHMARKS=ON`), create fixtures, then run the API
benchmark through the bounded queue wrapper described in
[BUILD.md](docs/BUILD.md#gpu-turns). On shared machines, every GPU command
below must enter through `tools/run_gpu_turn.sh`; do not run it directly:

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

The prefill tables come from `strix_trellis_prefill_tiled_bench`: `check`
runs the bitwise gate; `linear-synth` times the full operator at the given
row counts for the per-rank shapes and for the fixtures after `--`. Run it
with `STRIX_TRELLIS_WMMA_TILED=0` for the previous core:

```sh
build-hip/strix_trellis_prefill_tiled_bench check
build-hip/strix_trellis_prefill_tiled_bench linear-synth 128 256 512 1024 2048 4096 -- \
  "$FIXTURES/gpu-dense/layers.0.attn.wq_b.stgpu"
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
| `STRIX_TRELLIS_DIRECT_DUAL_ISSUE` | ON | Direct-output R8 GEMV: dot products ordered for dual issue (same bits). OFF keeps the previous order. |
| `STRIX_TRELLIS_DIRECT_PREFETCH` | ON | Direct-output R8 GEMV: request the next K16 row before consuming the current one (same bits). |
| `STRIX_TRELLIS_GROUP_K5_PIPELINE` | ON | Grouped K5 `wo_a` GEMV: pipelined 128-column schedule of the K4 path (same bits). OFF restores {64,2,16} where the split plan runs (it does not disable the owner-fold kernel). |
| `STRIX_TRELLIS_GROUP_K5_OWNER` | ON | Grouped K5 `wo_a` GEMV, rows 1–4: owner-fold kernel (same bits; see [DENSE_R1.md](docs/DENSE_R1.md)). OFF keeps the split plan at every row. |
| `STRIX_TRELLIS_PLAN_FUSED_COMPLETION` | OFF | Prepared R1 plans of the fused-input shapes fold the output transform into the core launch (same bits; see [API.md](docs/API.md)). |

More detail: [BUILD.md](docs/BUILD.md), [GPU_TESTS.md](docs/GPU_TESTS.md).

## API overview

The public header is [`include/strix_trellis.h`](include/strix_trellis.h)
(C ABI version 1). Full contract, memory ownership and stream rules:
[API.md](docs/API.md).

| Function family | Purpose |
|---|---|
| `strix_context_create` / `_destroy` | Device context; queries device resources once. |
| `strix_weights_prepare`, `strix_weights_prepare_parts`, `strix_weights_destroy` | Upload compressed weights from one blob or from borrowed regions. |
| `strix_weights_device_span` | Read-only base and size of the prepared device allocation, for callers that warm caches ahead of use. |
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
- Apart from the output head (210 GB/s, 85.0 % at R8), R8, small-matrix and
  K2 paths reach 104–190 GB/s in the decode table (42–77 % of the read
  ceiling), against 184–216 GB/s (75–88 %) for the best R1 paths.
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
