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

> **Status: experimental.** The C ABI is at version 1 and the library is at
> 0.3.0. Some paths are qualified on hardware; newer paths are implemented and
> pass CPU contract tests but have not yet been built or measured on the GPU.
> The [status table](#status) separates the two.

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

| Component | State |
|---|---|
| CPU reference (`python/strix_trellis`) | Exact decode for the `3inst`, `mcg` and `mul1` codebooks; integer rates 1–8 bits, plus 1.5 / 2.5 / 3.5 bits for `mul1`. Matches the EXL3 reference CUDA kernels bit for bit on 2,621,440 real weights and on all 65,536 states of each codebook. |
| Loader | safetensors, mmap per group: `trellis`, `suh`, `svh`, codebook marker, optional bias. Exports borrowed regions or a reusable buffer. Rejects external `out_scales` and older formats without an explicit scale contract. |
| HIP runtime, dense GEMV R1–R8 and WMMA GEMM | **Qualified on hardware** (campaign V16, see [Measured performance](#measured-performance)). |
| Prepared plans (`strix_linear_plan_*`) and region upload (`strix_weights_prepare_parts`) | Implemented in 0.2. CPU contract tests pass; GPU run pending. |
| K2 R1 local-reduction kernel | Implemented, off by default (`STRIX_TRELLIS_BUILD_EXPERIMENTAL_K2_LOCAL`). GPU qualification pending. See [K2_LOCAL.md](docs/K2_LOCAL.md). |
| TP rank APIs (`strix_tp_dense_*`, `strix_tp_moe_*`) | Implemented in 0.3, off by default (`STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP`). CPU contract tests pass; HIP build and GPU runs pending. See [TP_RANK.md](docs/TP_RANK.md). |
| Grouped MoE (`strix_grouped_moe`) | Works, one stream over groups; no performance work. |
| Python HIP bindings | None. Python provides the reference and weight transport; C/C++ drives the GPU. |
| Full model inference | Not provided. The library supplies linear operators to an inference engine. |

"Implemented" does not mean "faster". Speed is claimed only for measured paths.

## Measured performance

All numbers below come from qualification campaign **V16** (library built from
commit `0e60746`, run on 2026-10-21). Later sources have not been re-measured.

**Hardware and software:** AMD Ryzen AI Max+ 395, Radeon 8060S GPU
(`gfx1151`, 40 CU, wave32), 128 GB unified memory (AMD lists 256 GB/s
peak bandwidth), ROCm 10.1, Linux.

**Method:** median of 11 sweeps of HIP events in one process, three
alternating build pairs. Packed weights rotate over disjoint buffers of at
least 256 MiB so they are not cache resident. Times include both H128
transforms, decode and C API validation. Weight preparation, allocations and
host copies are excluded. Dense outputs are checked against the CPU reference
before and after timing. The timed head benchmark checks three of 1,010 H128
output blocks and finiteness; a separate gate checks every head value.

**Units:** GB/s = compressed weight bytes / operator time. It is not measured
DRAM traffic. Useful TOPS = 2·M·I·N / time of the full API call, including
decompression; it is not FP16 GEMM throughput.

Weights marked "EAQ" are real layer-0 tensors and the output head of a public
DeepSeek V4.1 Flash EXL3 checkpoint (EAQ v1). "Synthetic" rows use DeepSeek
V4.1 Flash shapes with synthetic K2 weights and activations.

### Decode (GEMV)

| Tensor | K (bits) | Rows | Full API µs | GB/s (compressed) |
|---|---:|---:|---:|---:|
| EAQ `q_b` | 5 | 1 | 130.752 | 200.49 |
| EAQ `q_b` | 5 | 8 | 150.422 | 174.27 |
| EAQ `o_b` | 5 | 1 | 124.304 | 210.89 |
| EAQ `o_b` | 5 | 8 | 146.171 | 179.34 |
| EAQ shared w1 | 5 | 8 | 52.137 | 141.41 |
| EAQ shared w2 | 5 | 1 | 39.138 | 188.38 |
| EAQ shared w2 | 5 | 8 | 46.660 | 158.01 |
| EAQ output head | 6 | 1 | 2258.708 | 219.79 |
| EAQ output head | 6 | 8 | 2389.518 | 207.76 |
| Synthetic `o_b` | 2 | 2 | 65.011 | 161.29 |
| Synthetic `o_b` | 2 | 8 | 91.766 | 114.27 |
| Synthetic `q_b` | 2 | 2 | 68.627 | 152.79 |
| Synthetic `q_b` | 2 | 8 | 99.923 | 104.94 |

### Prefill (WMMA GEMM)

| EAQ tensor | M | Full API µs | Useful TOPS |
|---|---:|---:|---:|
| `q_b` | 256 | 1820.727 | 11.79 |
| `o_b` | 256 | 1481.859 | 14.49 |
| shared w1 | 128 | 246.790 | 12.24 |
| shared w2 | 256 | 415.835 | 14.52 |
| `q_a` | 256 | 263.889 | 12.72 |
| `o_a`, one slice | 256 | 212.899 | 10.09 |

### Correctness in the same campaign

- 84 GPU fixtures, 156 API cases and 20 graph-capture cases pass against the
  CPU reference. Tolerances: max abs ≤ 5e-4 × max(1, max |ref|), NRMS ≤ 3e-5
  (see [NUMERICS.md](docs/NUMERICS.md)).
- Output head R1–R8 checked over all values against the CPU reference
  (FP64-ordered sums) and an independent CUDA reference; worst NRMS 2.39e-6.
- 9 CTest programs and 33 Python tests pass with no skips.
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
| `strix_tp_dense_plan_*` | Per-rank dense projections on strided views (experimental). |
| `strix_tp_moe_*` | Per-rank fused expert block: gate/up + SiLU, down + weighted top-6 sum; host or device routing (experimental). |

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

- Experimental. Only the V16 paths are measured on hardware; 0.2 and 0.3
  additions are CPU-tested only.
- One GPU target: `gfx1151`. Other RDNA targets are untested.
- No inter-rank communication: TP rank outputs of `o_b` and expert down must
  be summed by the caller. No complete generation engine.
- Expert kernels: the private grouped K3 path passes 14 native
  configurations with up to 32 groups. A fast public grouped path and its
  performance are pending. Quality results above use 2-bit GGUF
  experts, not EXL3 experts.
- R8, small-matrix and K2 paths reach 104–179 GB/s in the decode table,
  against 188–220 GB/s for the best R1 paths.
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
