# Minimal decoder replacement in WMMA

> **Type:** design note — engineering record; may describe plans, candidates or experiments that are not part of the supported API.

The blocked-K search and its prevalidated submission control gave negative
results. The next experiment keeps the qualified WMMA kernel: the same TileM
16/64, four waves, K16 per iteration, AStride, Q swizzle, two barriers,
accumulators, WMMA intrinsic and output map. Only the extraction of a lane's
eight FP16 values changes, before writing the same Q tile to LDS.

Initial domain: mul1 CB2, K5/K6, I/N multiples of 128, with dimensions bounded by
I<=8192 and N<=32768. The experiment has two variants: the project's aligned
reader with optimized mul1 pairs, and the SADDR reader with M−1 pairs already
tested in GEMV. Each pair j=0/1, 2/3, 4/5, 6/7 must produce exactly the same FP16
bits as WaveTile plus codebook_value. Write each component to its original Q
coordinate, including the XOR permutation. Do not convert any scalar half value
through an incorrectly sized bitcast. Loads remain visible to LLVM; do not hide
waits or accesses in asm.

K organization, WMMA count and A/Q reuse remain unchanged. Inactive N waves
still reach both barriers. There is no H128 fusion, activation quantization,
integer GEMM, double buffering or precision change. Private guards and submit
are prevalidated outside the timed interval. Production, API, workspace and
split selection remain unchanged.

Before the large kernel: independent review of the pair map, Q identity,
EXEC participation, alignment/provenance, arithmetic and guards. Earlier GEMV
evidence validates the helpers, not the new WMMA integration. Then check all Q
bytes, all partials against the baseline at the same S, a complete CPU reference
and canaries, WMMA impulses, M/K tails and empty splits, alpha/beta/bias, and
rejection of ineligible formats. Keep the existing tolerances.

Use at most two campaigns with frozen configurations, with GPU access through
180 s queue slots, disjoint resident weights >=256 MiB, eleven alternating
sweeps, separate complete-operator and core measurements, both geometries and
the rows actually tested. The search grid prioritizes q_b, wo_b, shared, q_a and
wo_a with M64/128/256 and their actual v9/v10 policy splits. Do not reuse the
previous experiment's universal S4 to claim an improvement in the current API.
Compile with nice/one job and audit the actually linked ELF, with no spills or
scratch. Only a confirmed gain justifies proposing a guard; maintainers would
repeat the full validation gates before enabling it.
