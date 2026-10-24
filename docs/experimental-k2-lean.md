# K2 S4 expert operator: lean core, folded R1 and host references (experimental)

## Defaults

The split creators use the pretransformed expert pipeline
(`STRIX_TRELLIS_TP_MOE_PIPELINED_K2`), ON by default, with these defaults:

- `LEAN_CORE`, `ADAPTIVE_ROWS`, `GROUPED_FINISH`, `SPLIT_PREPARE`,
  `SPLIT_GATE_FINISH` and `FUSE_DOWN_PREPARE`;
- `PREFETCH_ROWS=4`, `ROW_CAP=2`, `SINGLE_DEPTH=8` and `DOWN_DEPTH=6`;
- for device-routed plans:
  - `DEVICE_ROUTE_MAPS`: owner maps are resolved once per replay by one small
    kernel instead of in every core CTA;
  - `DEVICE_MERGED_CLASSES`: one core per projection serves both row classes,
    so no empty class grid is launched;
  - `DEVICE_MERGED_DOWN_CAP4`: the merged down core runs groups of more than
    two rows in four-row chunks, keeping the occupancy of the small-row core.

  Windows of at most four rows use a merged image without the eight-row core,
  and the small-row path shares the chunked core's code.

Each option keeps its flag, and none changes the S4 identity. Host-routed
plans are unchanged: their row classes are fixed at creation and empty classes
are omitted.

Per-case measurement:

- Setup: one rank, 24 cases (shared, overlap and disjoint routes x R1-R8) with
  device routes and with host routes, hot, plus 18 cold cases.
- Method: medians of alternating build pairs.
- Exactness: every build passed bitwise QA against the 0.3 operator on K2 and
  mixed K2/K3 fixtures before timing.

Results:

- Against the 0.3 creator and the non-pipelined split path, the defaults are
  never slower in any measured case. Device-routed shared windows at R3-R8
  take 208/216/287/311/332/342 µs, against 289/306/367/393/409/417 µs without
  the pipeline. R1 takes about 112-120 µs hot, against 225-245 µs.
- Against the same pipeline with `ADAPTIVE_ROWS` off:
  - device-routed overlapping routes at R2-R8 are 2-21 µs faster;
  - shared-expert windows at R3-R8 are 51-281 µs faster (without adaptive
    rows they are slower than the non-pipelined path);
  - device-routed disjoint routes at R3/R4 are about 2.7 µs (0.5 %) slower in
    every repetition. This is an open item: empty owner CTAs are still
    launched.
- Without the device options, device-routed shared windows at R5-R8 were 9-87
  µs slower than the non-pipelined path, and device-routed overlapping or
  disjoint routes paid for an empty row-class grid (up to 375 µs at R8).

Options that change bits (`DECLARED_SWIGLU`, A8R) or were slower in a
measured case (`FOLD_R1`) stay OFF.

Evidence for the lean core, per case (medians of alternating build pairs,
two separate GPU sessions, 24 cases: shared/overlap/disjoint routes x R1-R8,
hot and cold):

- R1, every route pattern: 3-16 µs faster, hot and cold.
- Overlap and disjoint routes, R2-R8: faster or within noise in every case.
- Shared routes, R2-R8: within noise (the sign of the difference changes
  between sessions). The lean core changes only the single-row K2 core; with
  `ROW_CAP=2` shared multi-row tiles do not execute it.


- `STRIX_TRELLIS_TP_MOE_LEAN_CORE`: single-row K2 S4 core (gate and down) with
  wave-uniform global bases plus 32-bit lane offsets for weight and activation
  loads, and a select-free lane window (one `v_alignbit` by a per-lane constant).
  Same arithmetic, same bits: 45 -> 40 VALU per 16x16 tile in the main loop.
- `STRIX_TRELLIS_TP_MOE_FOLD_R1` (+ `_FOLD_PREPARE`, `_FOLD_GATE_FINISH`): R1
  full submissions with finishes in the last CTA of each output block (plan-owned
  arrival counters). Bitwise equal to the staged S4 path, but slower than the
  lean core alone on gfx1151 (redundant per-CTA prologue / serial tails).
- `STRIX_TRELLIS_TP_MOE_DECLARED_SWIGLU`: changes H. SwiGLU with the DeepSeek V4.1
  clamp (10) and an exp built from basic IEEE operations (`tp_moe_activation.hpp`),
  reproducible on the host.

Host references (header only, no HIP):
- `src/dot2acc_reference.hpp`: native gfx11 `v_dot2(acc)_f32_f16` arithmetic
  (model from ROCm/rocm-systems#12056 / rocjitsu `gfx11_dot2.h`): accumulator sign
  frame, opposing terms as `~m` (no increment), grid
  `max(max_lane(Ea+Eb)-24, Emax-26)`, one RNE rounding, no -0. 0 mismatches on
  7,340,032 gfx1151 cases (kernel-like, random, cancellation, subnormal, zeros).
- `src/tp_moe_s4_reference.hpp`: lane-exact CPU S4 core (prepare, H128, K2 decode,
  half FMA, dot2acc chains, 4-lane reduction, split order, finishes). Bitwise vs
  GPU for gate/up P, down P and Y; with the declared SwiGLU also H.
