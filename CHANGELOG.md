# Changelog

Source versions follow `STRIX_TRELLIS_VERSION_*` in
[`include/strix_trellis.h`](include/strix_trellis.h). Runtime campaigns
(V7 … V16) are hardware qualification runs of a fixed library build; speed-ups
below compare one of our builds with the previous one, measured in the same
session on the same machine.

## Unreleased — 2026-10-24

- Direct-output R8 GEMV (`q_b`, `o_b`, shared down and output head): the
  packed words of the next K16 row are requested before the current row
  is consumed (`STRIX_TRELLIS_DIRECT_PREFETCH`, ON). `SaddrLane` gains a
  two-phase form, `issue()` plus a constructor from the raw words, so the
  loads overlap the current row's decode and dot products instead of
  being waited for at once. Same addresses and the same dot product order;
  outputs bit-identical in every measured cell. Confirmation turn (six
  repetitions in two blocks against the previous build and a
  byte-identical copy of it, cold rotation, 50 cells over all fixture
  rows): EAQ `q_b` R8 152.47 -> 143.40 µs (-5.9 %), `o_b` R8 -4.5 %,
  shared down R8 -3.9 %, output head R8 -4.1 %; every target check
  passed and no cell met the regression condition in both blocks (the
  turn was noisy: single-block excursions up to +2.8 % also appeared
  between the two identical builds). An earlier turn of the same build
  measured `q_b` R8 151.28 -> 138.21 µs (-8.6 %) and was inconclusive
  under a stricter per-block rule. OFF keeps the previous load order.
- Grouped K5 `wo_a` GEMV (4096 -> 1024, rows 1-8) uses the pipelined
  {128,4,16} schedule of the K4 path by default
  (`STRIX_TRELLIS_GROUP_K5_PIPELINE`, ON). Cold public-plan confirmation,
  three repetitions, outputs bit-identical: paired-median reductions of
  4.02 % at R1, 1.17 % at R2, 9.19 % at R6 and 11.55 % at R8 (R8
  74.82 -> 66.19 µs). The K4 policy is unchanged and no K4 cell met the
  registered regression condition (paired medians -0.22 to +0.17 %).
  Disabling the option restores the earlier {64,2,16} schedule without
  pipelining. `dense_r1_group_probe`
  gains a `tuneg` mode that times the public grouped plan.
- Opt-in one-launch completion for single-row prepared plans of the
  fused-input shapes (`STRIX_TRELLIS_PLAN_FUSED_COMPLETION`, OFF): the last
  thread block of each 128-column group folds the split partials (all 128
  threads, ascending split order) and applies the output transform, so the
  output transform launch disappears. Same bits (`plan_completion_qa`:
  600/600 bitwise cases over alpha, beta, bias, rows 1/8, reuse and graph
  replay after plan destruction). Plan R1, cold, three repetitions: `o_a`
  slice 22.50 -> 19.30 µs (-14.2 %), `q_a` 34.43 -> 30.92 µs (-10.2 %),
  shared w1 48.73 -> 46.05 µs (-5.5 %). It stays OFF because a control
  cell that does not run the changed code, `q_a` R8 on the direct API
  path, measured +0.19/+0.61/+0.88 %, which meets the pre-registered
  regression condition; the cause is not established. A pre-registered
  confirmation, six repetitions in two blocks, met every target against
  both OFF arms (an OFF build and a byte-identical rebuild of it); no
  tested cell met the registered regression condition in either candidate
  comparison. However, two control-cell block medians differed by about
  +0.69 % between the byte-identical OFF builds, triggering the registered
  stability flags, so the confirmation was inconclusive and the option
  remains OFF. Completion counters
  belong to the weights handle; plan creation zeroes them and synchronizes.
- README: EAQ `q_a` is K6 (was labelled K5).
- README: grouped `o_a` table re-measured with the K5 pipelined schedule
  (K5 R8 84.57 -> 66.47 µs, 50.3 -> 63.9 % of the read ceiling); the
  rotation is stated as measured (648 MiB K4, 650 MiB K5).
  `dense_r1_group_probe tuneg` also times the four projections as four
  `strix_gemv` calls over the same rotation (`gemv4_us`).
- Prefill: register-blocked WMMA GEMM core, default for codebook `mul1` at
  4–6 bits and at least 128 rows (`STRIX_TRELLIS_WMMA_TILED=0` selects the
  previous core; fewer rows, other codebooks and other bit rates keep it).
  Same split count, split bounds and K16 accumulation order per output, so
  the same bits; tiles of 128×256, 256×128 or 128×128 by row count, with
  grouped block order. `STRIX_TRELLIS_WMMA_RASTER` changes the block order
  for measurement only. Full operator, previous core / new default / new
  core with the fused epilogue rotated three times in one session: output
  hashes identical in all 114 cells (7 EAQ tensors and the per-rank shapes
  at K4/K5, M 128–4096) and all runs; the default is faster in every cell,
  median 2.04× (1.04–4.58×), 14–71 % of the measured 47.6 TOPS sustained
  WMMA rate (median 48 %), against 12–38 % before. New benchmark
  `strix_trellis_prefill_tiled_bench`.
- Optional fused output epilogue for single-split calls of that core
  (`STRIX_TRELLIS_WMMA_TILED_FUSED=1`, same bits), off by default: up to
  30 % faster in most cells, but slower than the separate transform in
  three small cells (shared down K4 M 128 by 6–13 %, `q_a` K4 M 128 by
  2–4 %, `o_a` slice K4 M 256).
- `api_linear_bench`: `STRIX_BENCH_PATH=api|plan|graph` and `STRIX_BENCH_ROWS`;
  every clone and phase must give the same output bits.
- README: measured `gfx1151` WMMA ceiling (burst 53.9 TOPS FP16, sustained
  47.6–48.3 TOPS under the thermal limit) and prefill as a percentage of
  it; distance to the ceilings by class; R8 direct-output decode rows
  measured again; the EAQ `q_a` tensor is K6 (was labelled K5).
- Direct-output R8 GEMV (`q_b`, `o_b`, shared down and output head):
  each K16 step decodes its four weight pairs first and issues the dot
  products of the two independent chains of every row alternately, with the
  activation as the first operand (`STRIX_TRELLIS_DIRECT_DUAL_ISSUE`, ON).
  Same bits (native DOT2 is symmetric in its multiplicands; every chain keeps
  its order). With the current compiler the hot loop has 15 instead of 10-11
  dual-issue dot instructions and 65 instead of 69-70 VALU instructions.
  Isolated operator, three repetitions, outputs bit-identical in all 50
  cells: EAQ `q_b` R8 152.40 -> 151.58 µs (paired -0.47 to -1.12 %),
  shared down R8 -0.62 %, `o_b` R8 -0.22 %; no cell met the pre-registered
  regression criterion. OFF keeps the previous order.
- Direct-output R8 GEMV kernels advance the weight tile pointer per K16 row
  instead of recomputing it; 11–12 fewer instructions per inner iteration with
  the current compiler, bit-identical outputs. Benchmarks print `output_fnv`,
  a hash of the output bits.
- `strix_weights_device_span` (`STRIX_TRELLIS_HAS_WEIGHTS_SPAN`): read-only
  base and size of the prepared weights' device allocation, for callers that
  read the bytes ahead of use. The library does not depend on it. Contract
  test (NULL arguments, cleared outputs) and a GPU runtime check that the span
  covers the uploaded trellis, `suh` and `svh`.
- K2 expert pipeline for device-routed plans: owner maps are resolved once per
  replay (`STRIX_TRELLIS_TP_MOE_DEVICE_ROUTE_MAPS`), and one core serves both
  row classes (`STRIX_TRELLIS_TP_MOE_DEVICE_MERGED_CLASSES`, with the down core
  in four-row chunks under `STRIX_TRELLIS_TP_MOE_DEVICE_MERGED_DOWN_CAP4`;
  windows of at most four rows use an image without the eight-row core).
  Before, every core CTA re-resolved its owner map, and an empty row-class grid
  was launched on every replay. Same bits. Host-routed plans are unchanged.
- Defaults: the pipeline (`STRIX_TRELLIS_TP_MOE_PIPELINED_K2`), `ADAPTIVE_ROWS`
  and the device options are now ON. In 24 hot device-routed cases, 24 hot
  host-routed cases and 18 cold cases (one rank, isolated operator, bitwise QA
  against the 0.3 operator first), the default is never slower than the
  non-pipelined split path or the 0.3 creator. Device-routed shared-expert
  windows at R3–R8 take 208–342 µs instead of 289–417 µs. Split plans need the
  larger pipeline workspace; query `strix_tp_moe_split_workspace_bytes`.
- CPU: the API contract and the pipeline layout test also run with the
  library's pipeline configuration.
- Dense R1 TP path (`strix_tp_dense_plan_create_r1`,
  `strix_tp_dense_r1_workspace_bytes`; see [DENSE_R1.md](docs/DENSE_R1.md)):
  - exact-order single-dispatch split-K GEMV with physical output tiling;
  - S4 or LEGACY numerical identities, each bitwise reproducible per row;
  - schedules chosen per shape and row count from a measured profile.
- Grouped strided GEMV plans (`strix_grouped_gemv_plan_*`): up to eight
  independent projections of one geometry in one dispatch, each row with the
  bits of `strix_gemv` R1. Counters sit at geometry-fixed offsets, so plans of
  different row counts can share one workspace.
- `strix_tp_dense_pair_plan_create_r1`: the `q_a` (S16) and `kv` (S32) LEGACY
  projections of one input in one dispatch.
- Cold-swept schedule policy for the grouped `o_a` projections (four
  4096→1024 groups) and for K4 `o_b` at R5/R6. Isolated operator, one gfx1151
  device, weights rotating over more than 96 MB, every schedule checked bitwise
  against the policy before timing. Results:

  | Case | Before | After |
  |---|---:|---:|
  | Grouped `o_a`, R1 | 108.5 µs as four separate calls; 54.8 µs grouped | 51.6 µs |
  | Grouped `o_a`, R6 | 141.8 µs as four separate calls; 77.9 µs grouped | 68.4 µs |
  | `o_b` K4, R5/R6 | 70.3 µs | 62.6 µs |
- README: all performance tables re-measured on `892b26e` with the V16 method,
  next to the V16 build in the same session, with a "% of read ceiling" column
  (246.7 GB/s, a streaming-read kernel measured in the same session) and a
  "% of call floor" column. New tables for the dense R1 plans, the grouped
  `o_a` GEMV and the K2 expert block (pipeline off against default, cold and
  hot).
- `tests/dense_r1_group_probe.hip`: `PROBE_ROTATE_MB`, `PROBE_SWEEPS` and
  `PROBE_POLICY_ONLY` select the rotation size, the sweep count and a
  policy-only run; `tune1` also times `strix_gemv` over the same copies and
  checks that the plan output equals it bit for bit.

## 2026-10-23

- K2 expert split plans (`strix_tp_moe_split_workspace_bytes`,
  `strix_tp_moe_plan_create_split`, `strix_tp_moe_plan_create_device_split`,
  staged `down_core` / `down_finish`): three-stage S4 expert block with an
  optional pretransformed pipeline (`STRIX_TRELLIS_TP_MOE_PIPELINED_K2`).
  The existing creators keep the two-kernel 0.3 implementation.
- `STRIX_TRELLIS_TP_MOE_LEAN_CORE`: single-row K2 S4 core with wave-uniform
  bases, 32-bit lane offsets and a select-free lane window; 45.1 → 40.3 VALU
  per 16×16 tile, same bits. Isolated operator, one rank, against the same
  pipeline without it: R1 hot 121–130 → 109.5–116.2 µs, R1 cold
  158.3–159.3 → 153.6–155.6 µs, R6 disjoint cold 770–773 → 759–765 µs
  (method in the README).
- `STRIX_TRELLIS_TP_MOE_DECLARED_SWIGLU`: clamped SwiGLU with a host-reproducible
  exponential (changes H). `STRIX_TRELLIS_TP_MOE_FOLD_R1`: exact last-CTA
  finishes; slower, off.
- Host references (header only): `src/dot2acc_reference.hpp` models native
  gfx11 `v_dot2(acc)_f32_f16` (0 mismatches on 7,340,032 gfx1151 cases);
  `src/tp_moe_s4_reference.hpp` is a lane-exact CPU model of the S4 core,
  bitwise equal to the GPU for P and Y (and H with the declared SwiGLU).
- Experimental K2 A8R expert arithmetic (`STRIX_TRELLIS_TP_MOE_A8R`, compiled
  out by default): a distinct identity, not bitwise equal to S4. Not measured.
- Defaults: the pretransformed pipeline (`STRIX_TRELLIS_TP_MOE_PIPELINED_K2`)
  is OFF, so default builds select the same kernels and bits as 0.3.0. When it
  is ON, its options default to the configuration measured with host-routed
  windows (`LEAN_CORE`, `ADAPTIVE_ROWS`, `GROUPED_FINISH`, `SPLIT_PREPARE`,
  `SPLIT_GATE_FINISH`, `FUSE_DOWN_PREPARE`, `PREFETCH_ROWS=4`, `ROW_CAP=2`,
  `SINGLE_DEPTH=8`, `DOWN_DEPTH=6`); each keeps its flag. Options that change
  bits (`DECLARED_SWIGLU`, A8R) or were slower somewhere (`FOLD_R1`) stay OFF.
  Device-routed windows were measured afterwards; see 2026-10-24.
- CPU CTest 18/18, Python 61 tests (3 skipped without the external fixture),
  HIP build for `gfx1151` with default and recommended options.
- CPU reference: `unpack_states_vectorized` and `dequantize_rotated_vectorized`,
  array forms of the scalar decoder with identical output (about 6 M weights/s
  on one CPU process instead of a Python loop per tile).
- Loader: opt-in raw BF16 mapping (`tensor(name, raw_bf16=True)`) and
  `tensor_float32(name)` for unquantized BF16/F16/F32 tensors.

## 0.3.0 — 2026-10-21 — TP rank paths (experimental, not yet GPU-qualified)

- Dense TP API (`strix_tp_dense_plan_*`): per-rank K4/K5 `mul1` projections
  for the DeepSeek V4.1 Flash shapes (`q_a`, `kv`, `q_b`, `o_b`, shared expert),
  R1–R8, row tiles, strided F32 views. One kernel per projection, including
  both H128 transforms, decode, scales, bias and alpha/beta. No global
  workspace and no adapter copies.
- MoE TP API (`strix_tp_moe_*`): fused expert block in two kernels
  (gate/up + SiLU + hidden write; down + weighted top-6 sum), K2/K3 per
  weight. Persistent expert bank of up to 256 experts; the union of up to 48
  selected experts per window is resolved on the GPU, so routes can change
  between submits or graph replays without rebuilding the plan. No atomics in
  the reduction. A host-routed plan is kept for reproducible comparisons.
- `rank_partition` in the Python package prepares compressed weight views for
  two ranks and assigns bias to one rank.
- Built only with `STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP=ON`. CPU contract
  tests pass; HIP build and native runs are pending. The callers must sum
  rank outputs; no inter-node communication is included.

## 0.2.0 — 2026-10-21 — Region upload, prepared plans, K2 local reduction

- `strix_weights_prepare_parts`: upload from borrowed loader regions without
  assembling a temporary blob.
- `strix_linear_plan_*`: validate buffers and stream once, then submit
  without repeating HIP queries.
- K2 R1 candidate that replaces four global partial planes with one,
  using a 256-byte LDS reduction; U1/U2/U4 and prefetch variants. Off by
  default (`STRIX_TRELLIS_BUILD_EXPERIMENTAL_K2_LOCAL`).
- ABI stays at 1; existing descriptor layouts are unchanged. CPU contract
  tests and a native probe exist; GPU runs are pending.

## Runtime V16 — 2026-10-21 (last qualified build)

- Fused R1 path that applies both H128 transforms directly, for K5 `o_b` and
  the shared inverse transform. Measured gain over V14b in three alternating
  pairs: 9.64–10.46 % (`o_b`) and 14.57–14.85 % (shared w2). All other kernel
  images unchanged.
- Passed: 84 GPU fixtures, 156 API cases, 20 graph cases, full output head,
  engine plain/bias checks, 9 CTest and 33 Python tests.
- The internal device-weights record has ten fields; consumers must be
  rebuilt against the current header (V14b used nine, V11 eight).

## Runtime V14b — 2026-10-21

- 13 WMMA prefill tile configurations (scalar addressing, M64, 18-register
  A fragments): 4.01–11.64 % faster than V13. Some shapes keep the previous
  path after measured regressions.
- Six K2 multirow decode routes.

## Runtime V13 — 2026-10-21

- New K partition for output-head R8; checked against full CPU and CUDA
  references (NRMS 2.36e-6, worst row 2.39e-6). Head R1–R8 activations and
  R1–R7 outputs byte-identical to V12.

## Runtime V7 – V12 — 2026-10-20 / 2026-10-21

- V7: first qualified baseline — complete F32 linear operator, fused WMMA
  GEMM, real dense fixtures, cold-weight-rotation benchmark of the full C API.
- V8: long prefill and full real-head identity checks.
- V9–V10: GEMV scheduling work (row reuse, scalar addressing), integrated K2
  pipeline across 84 fixtures.
- V11–V12: R8 output fusion (G1/G2/G4), real K3 expert matrices through the
  API and graph paths, full CPU references for two of our own K4/K5 output
  heads.

## 0.1.0 — 2026-10-20 — Initial implementation

- Format study and gfx1151 kernel design.
- Independent wave32 trellis decoder; bit-exact against the EXL3 reference
  CUDA kernels on real weights and all codebook states.
- Staged FP32 numerical reference with explicit rounding points.
- NumPy CPU reference, safetensors loader, C ABI v1.
