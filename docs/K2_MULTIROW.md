# K2 pipeline for multiple rows

> **Type:** design note — engineering record; may describe plans, candidates or experiments that are not part of the supported API.

V11 optimizes R1 but retains the generic K2 core for R2–R8. The three API pairs
show that limitation; their timings are not attributed to the new R1 guard.
This proposal keeps the private K2 V1 decoder specialization and reuses each
weight across exactly 2, 4 or 8 rows.

Two private variants are proposed, with the same S4 and dot order:

1. Global: load four pairs of packed words early, without loading activations
   early for the four iterations. When consuming each K16, load the two
   activation pairs per row and apply the four weight pairs to all rows.
   Use two FP32 accumulators per row, as in the control.
2. LDS8: keep the qualified staging of eight A tiles per chunk, with two uniform
   barriers per chunk and participating inactive N waves. Within the chunk,
   load only four packed tiles early and consume them in ascending K order.
   Activations use the same LDS layout; their lifetime is not extended to U4.

The K2 reader, both wraps, V1 hash with sixteen earlyclobber outputs,
SAD/half2 FMA, pair order and 2/1 reductions remain unchanged. Each row retains
its accumulation chain, even when operations from independent rows interleave.
The store remains `[split,total_rows,N]`, relative to the row pointers received
by the core. U1 tails within a split and empty splits must be correct.

Initial performance domain: CB2/K2, R2/R4/R8, X aligned to four,
S4 and shapes I8192/N5120 and I1280/N32768. No changes to production, the C API,
workspace, partitions, input/output H128, prefill or output fusion.
Use the complete v11 K2 fixtures. Do not extrapolate from R1.

Independent review before implementation must cover liveness without assuming
emitted resources, A/P ownership, barriers, bounds and per-row identity. Then
run CPU tests, inspect the actual ELF for no spills, compare all partials/outputs
against v11 at the same S and independent FP64, and check offsets/padding/canaries,
all remainders/selected rows, beta0 and alpha/beta/bias. Time the core and complete
operator separately, with resident rotation >=256 MiB, eleven sweeps and two
campaigns with frozen selections. Only confirmed winners would justify proposing
guards for maintainer review and would require the full validation gate.
