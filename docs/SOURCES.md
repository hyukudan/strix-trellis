# Design sources and provenance
> **Type:** reference — sources, provenance and licenses.

Accessed: 2026-10-20. Official sources were read only.
No upstream implementation files have been incorporated into this repository,
and no upstream kernels have been translated.
Compatibility formulas and constants are described in original wording.
The identifiers below make the research reproducible without relying on a
moving branch. The current upstream revision is not claimed to be the one
that produced the local checkpoint.

## Exact revisions and licenses

| ID | Source | Pinned revision | Observed license |
|---|---|---|---|
| Q | [Cornell-RelaxML/qtip](https://github.com/Cornell-RelaxML/qtip/tree/e90c6688c8dfae326a3a81b5eb032db7c6680ec0) | `e90c6688c8dfae326a3a81b5eb032db7c6680ec0` | [LICENSE](https://github.com/Cornell-RelaxML/qtip/blob/e90c6688c8dfae326a3a81b5eb032db7c6680ec0/LICENSE): GNU General Public License, version 3, 29 June 2007; README states GPL v3. Do not infer a permissive exception. |
| E | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3/tree/a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a) | `a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a` | [LICENSE](https://github.com/turboderp-org/exllamav3/blob/a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a/LICENSE): MIT, copyright 2025 Turboderp |

The SHAs were obtained from the official GitHub endpoint
`repos/{owner}/{repo}/commits?per_page=1` and the tree paths at those SHAs.
Repository E's license must not automatically be applied to vendored
components; none of their code was used.
Independent research is not presented as isolation from source reading:
the listed files were read. Their implementation bodies were not copied
into this project.

## Cornell paper and algorithm

- **Q1:** Albert Tseng, Qingyao Sun, David Hou and Christopher De Sa,
  *QTIP: Quantization with Trellises and Incoherence Processing*, NeurIPS 2024.
  [Official conference version](https://proceedings.neurips.cc/paper_files/paper/2024/file/6de2e84b8da47bb2eb5e2ac96c63d2b0-Paper-Conference.pdf),
  [OpenReview](https://openreview.net/forum?id=7sdkLVuYCU),
  [arXiv 2406.11235v2](https://arxiv.org/html/2406.11235v2).
  Sections 3.1–3.2 and 4 were consulted: bitshift windows, computed/HYB
  codebooks, tail-biting and 16×16 tiles. The design document does not
  reproduce the paper's printed algorithms or figures. No separate license
  for the paper's text has been established here; do not confuse it with
  the code's GPL.
- **Q2:** [README](https://github.com/Cornell-RelaxML/qtip/blob/e90c6688c8dfae326a3a81b5eb032db7c6680ec0/README.md)
  and [lib/codebook/bitshift.py](https://github.com/Cornell-RelaxML/qtip/blob/e90c6688c8dfae326a3a81b5eb032db7c6680ec0/lib/codebook/bitshift.py).
  L/K/V terminology, codebook selection, Cornell-method constants and
  differences from EXL3 were cross-checked.

## EXL3 format: pinned paths and evidence

- **E1:** [modules/quant/exl3.py](https://github.com/turboderp-org/exllamav3/blob/a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a/exllamav3/modules/quant/exl3.py),
  constructor, `get_inner_weight_tensor`, `get_weight_tensor`, reconstruction path:
  dtype/shape, K derived from width, mul1/mcg selection, sign and Hadamard
  application, modern support and historical variants.
- **E2:** [quant/codebook.cuh](https://github.com/turboderp-org/exllamav3/blob/a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a/exllamav3/exllamav3_ext/quant/codebook.cuh),
  `decode_3inst`, `decode_mul1_product_2` and variants: constants, modular
  arithmetic, unsigned byte sums and FP16 FMA/addition.
- **E3:** [quant/pack.cu](https://github.com/turboderp-org/exllamav3/blob/a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a/exllamav3/exllamav3_ext/quant/pack.cu),
  pack/unpack: size, int16 as bits, circular stream, half swapping and windows.
  Its packing procedure is not reproduced.
- **E4:** [modules/linear.py](https://github.com/turboderp-org/exllamav3/blob/a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a/exllamav3/modules/linear.py),
  `load_exl3`: serialized keys and marker lookup.
- **E5:** [tests/testlib/trellis.py](https://github.com/turboderp-org/exllamav3/blob/a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a/tests/testlib/trellis.py)
  and [tests/testlib/exl3.py](https://github.com/turboderp-org/exllamav3/blob/a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a/tests/testlib/exl3.py):
  explicit corroboration of the ring, endianness, permutation and operator form.
  [test_trellis_pack.py](https://github.com/turboderp-org/exllamav3/blob/a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a/tests/kernels/quant/test_trellis_pack.py)
  was also consulted. Its reference implementation and test suite are not adopted.
- **E6:** [quant/hadamard.cu](https://github.com/turboderp-org/exllamav3/blob/a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a/exllamav3/exllamav3_ext/quant/hadamard.cu),
  `had_r_128`: normalization factors and pre/post scales.
  Equivalence of all internal rounding steps remains pending validation.
- **E7:** [exl3_lib/quantize.py](https://github.com/turboderp-org/exllamav3/blob/a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a/exllamav3/modules/quant/exl3_lib/quantize.py),
  `tensor_core_perm` (lines 43–65), `pack_trellis` (1113–1128),
  `regularize` (around 1400–1503), `refit_scales` and suh/svh serialization
  (around 1849–1869). These are the primary evidence that channel magnitudes
  are incorporated into the serialized vectors, and for tile order.

## AMD architecture

- **A1:** AMD, [RDNA 3.5 Instruction Set Architecture Reference Guide](https://gpuopen.com/download/rdna35_instruction_set_architecture.pdf),
  PDF served on the access date, SHA256
  `bc42e0f4a1e183521bd419914a0f28d99cdb6363d8c0384e16bd3e44f48e50b8`.
  [Official announcement, 16 September 2024](https://gpuopen.com/news/amd-rdna-3-5-isa/).
  The contents, sections 2.1 and 3.3, and packed/WMMA operation tables
  (section 7, printed pages 67–75) were consulted: wave32, resources, bit
  operations and unsigned dot products, WMMA formats and replication.
  The document is subject to AMD's own terms; it is not redistributed,
  and its tables/diagrams are not copied. It does not grant an MIT license.
- **A2:** [AMD ROCm GPU specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html),
  page accessed as ROCm 10.1.0: Ryzen AI Max+ PRO 395 row, Radeon 8060S,
  RDNA3.5, gfx1151 and 40 CU. This is a changing page: record the machine's
  exact runtime version when taking measurements.
  The budget model uses 2 SIMD32/CU and explicitly hypothetical frequencies.
- **A3:** AMD GPUOpen, [How to accelerate AI applications on RDNA 3 using WMMA](https://gpuopen.com/learn/wmma_on_rdna3/),
  updated 6 February 2024: FP16 operands, FP32 accumulators, 16×16×16 size,
  fragment registers and replication between half-waves.
  Cross-checked against A1 for RDNA3.5; the HIP example is not copied.
- **A4:** [AMD Ryzen AI Halo User Guide](https://developer.amd.com/playbooks/user-guide/),
  platform reference publishing 256 GB/s for AI Max+ 395.
  It does not measure the project's machine or demonstrate sustained
  220 GB/s for EXL3 decoding.

## Geometries and local fixture

- **M1:** [Official DeepSeek-V3.1 config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3.1/blob/c0781d039fb7a1ba2abc4add0bdc293e92d2b8db/config.json),
  accessed 2026-10-20, revision `c0781d039fb7a1ba2abc4add0bdc293e92d2b8db`
  obtained from the model's official API: hidden 7168, intermediate 18432,
  MoE intermediate 2048, 256 experts and 8 experts/token.
  Used only for published real geometries, not to assign a configuration
  to the target model.
- **F1:** shared EAQ v1 fixture in a local fixture directory
  (set `STRIX_TRELLIS_FIXTURES`), revision
  `45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b`.
  Provenance, HTTP206 ranges and complete hashes: `tests/fixtures/eaq_v1.json`;
  additional dense groups: `tests/fixtures/eaq_dense.json`;
  confirmed head: `tests/fixtures/eaq_head_metadata.json`.
  Payloads remain outside Git.

| Suffix of `layers.0.attn.wkv` | Bytes | SHA256 |
|---|---:|---|
| `.mul1.bin` | 4 | `20feb8ff6dbb9b28746f70893914cd343f8a92629ad812923fb29772030dd082` |
| `.suh.bin` | 10,240 | `867a854cd8bbe64c0cc01863ead55ede80917a5ee55a285acafb77719abf1a2b` |
| `.svh.bin` | 1,024 | `81ed45b558c0cc750e3a8beea4f86a3257a23af341d2a625dbb7de6376288742` |
| `.trellis.bin` | 1,966,080 | `d371339f4992ef6ba31f914e0284dad17f3f37ba8bb19e52c137bd349f62e7d7` |

Local inspection commands: `Get-ChildItem $env:STRIX_TRELLIS_FIXTURES -Recurse -File`
and `Get-FileHash "$env:STRIX_TRELLIS_FIXTURES/layers.0.attn.wkv.*.bin" -Algorithm SHA256`.
The marker was read with `System.IO.File.ReadAllBytes` and was `2D-D1-DC-83`.
These commands are not benchmarks.

## Verification added after review

Pinned revision E already contains ROCm/RDNA paths (`USE_ROCM`, `ext.py`
and `rocm/`). This corrects the initial assumption that support was absent.
Neither performance nor compatibility of its complete layer with this backend
is assumed.

Fixture provenance is pinned by `tests/fixtures/eaq_v1.json`:
[EAQ v1, revision 45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw/tree/45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b).
A total of 2,470,950 bytes were received in six HTTP206 responses,
with hashes and offsets. The model license reported by the user is MIT;
weight fragments remain outside Git.
The additional header for shapes is kept in the shared fixture directory;
no full shards were downloaded.

A one-off correctness check (not part of this repository) built the pinned
upstream decode kernels from a separate checkout and compared their output
with ours; no upstream code was copied. The run confirmed all codebooks and reconstruction of the real
fixture, but not Hadamard or full inference.
