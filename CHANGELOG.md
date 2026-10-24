# Changelog

Source versions follow `STRIX_TRELLIS_VERSION_*` in
[`include/strix_trellis.h`](include/strix_trellis.h). Runtime campaigns
(V7 … V16) are hardware qualification runs of a fixed library build; speed-ups
below compare one of our builds with the previous one, measured in the same
session on the same machine.

## Unreleased — 2026-10-24

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
