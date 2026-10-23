# K2 S4 expert operator: lean core, folded R1 and host references (experimental)

`STRIX_TRELLIS_TP_MOE_PIPELINED_K2` (the pretransformed expert pipeline used by
the split creators) is OFF by default: it has not yet been compared case by
case with the non-pipelined split path. When it is ON, its defaults are the
qualified configuration: `LEAN_CORE`, `ADAPTIVE_ROWS`, `GROUPED_FINISH`,
`SPLIT_PREPARE`, `SPLIT_GATE_FINISH`, `FUSE_DOWN_PREPARE`, `PREFETCH_ROWS=4`,
`ROW_CAP=2`, `SINGLE_DEPTH=8`, `DOWN_DEPTH=6`. Each keeps its flag. None changes
the S4 identity unless stated; options that change bits or were slower in a
measured case stay OFF.

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
