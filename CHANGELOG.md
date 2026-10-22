# Changelog

Source versions follow `STRIX_TRELLIS_VERSION_*` in
[`include/strix_trellis.h`](include/strix_trellis.h). Runtime campaigns
(V7 … V16) are hardware qualification runs of a fixed library build; speed-ups
below compare one of our builds with the previous one, measured in the same
session on the same machine.

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
