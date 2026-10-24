# Exact-order dense GEMV experiment

This opt-in path adds physical output tiling to a single-dispatch split-K
projection. It does not change generic GEMV, TP03, or the existing split-K API.
Build with `STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP=ON`.

Status: in use at the DeepSeek V4.1 Flash attention sites of an
inference engine, gated end to end on two ranks; operator timings are in the
[README](../README.md#dense-r1-tp-plans-decode-one-rank).

## Numerical identities

`STRIX_DENSE_R1_S4` retains four complete K16 accumulator chains, including
their original floor-divided boundaries. Each lane uses two FP32 accumulators
and FP16 dot pairs, in increasing K16 order. Its four-lane reduction adds
shuffle offsets 2, then 1. The final fold starts at FP32 +0 and adds partitions
in ascending order before output H128, svh, optional bias, alpha, and beta.
The row count, tile width, arrival order, and prefetch depth never change this
association. Every submitted row has the same arithmetic as an independent R1.

`STRIX_DENSE_R1_LEGACY` retains the existing shape-dependent order: S16 for
5120->1280/1152, S32 for 5120->512, and S4 for the other supported shapes.
These are distinct numerical identities. Switching between them is a model
identity decision; an operator tolerance check cannot authorize that switch.

Input scaling materializes FP32 before conversion to FP16, performs complete
H128 transforms, and rounds the transformed activations to FP16. A partition
may crop a transformed H128 but never transforms only its K16 intersection.
The build uses `-ffp-contract=off` and explicit dot builtins.

## Execution and ownership

Physical CTAs cover 8/16/32/64/128 outputs within an output H128 group. The
8-output wave variant decodes only the selected half of a K16xN16 trellis
tile. Several such waves can share a larger output CTA and its activation
cache, so doubling waves need not multiply H128 staging or LDS per CTA.
Thus it increases independent waves without cutting a logical accumulator.
Every CTA stages its partition's complete activations in LDS once; every
decoded weight pair is reused across its row tile. Prefetch schedules expose
one, two, or four packed readers before consumption. They keep all loads
visible to the compiler's dependency and waitcnt machinery.

The grid separates output tiles, logical partitions, and row tiles into
three dimensions. The admitted partitions have a proven constant H128 span,
so staging does not divide by a runtime span. The completing CTA prefetches
four partial planes and output metadata, then folds the planes in ascending
order with the original FP32 association.

Each CTA publishes a disjoint partial tile and releases its writes at agent
scope before a single arrival RMW. The last arriving CTA acquires all partials,
folds the declared partitions, transforms the output, and resets the counter.
There is no spinning and no second launch. The final CTA is whichever one
arrives last; its identity does not affect the reduction order.

Workspace contains `[S,R,N]` FP32 partials and one U32 counter per output H128
and row tile. It must be 256-byte aligned and used serially on the plan's
stream. Different simultaneously executing plans need disjoint workspaces.
Creation initializes counters and synchronizes; submit and graph replay do
neither. A failed/asynchronously aborted dispatch leaves workspace state
unspecified: recreate the plan before reuse. Destroy captured graphs and
complete all work before destroying the plan, weights, buffers, or stream.

## C API

Query `strix_tp_dense_r1_workspace_bytes(context, desc, rows_per_block,
identity, &bytes)`, allocate the returned workspace, then call
`strix_tp_dense_plan_create_r1(context, desc, stream, rows_per_block, identity,
&plan)`. Zero rows_per_block selects all submitted rows. Use the existing
`strix_tp_dense_plan_submit` and `strix_tp_dense_plan_destroy` entry points.
The creator validates allocation spans, aliases, image resources, and device
ownership. The plan freezes all descriptors and scalar values.

The creator selects columns per CTA, outputs per wave, and prefetch depth
from a per-shape/K/row profile. It can retain the existing single-dispatch
split-K body when that measured schedule wins and its logical split count
matches the requested identity. This never changes S4 into S16 or S32.

Supported local shapes are q_a 5120->1280, kv 5120->512, q_b 1280->16384,
o_b 4096->5120, shared gate/up 5120->1152, shared down 1152->5120, and head
5120->64640 or 129280, with K4/K5 mul1 and R1..R8. F32 row strides are in
elements, with unit channel stride. The caller owns inter-rank reductions.

## Qualification

`strix_trellis_dense_r1_probe qualify all` checks native A/P/Y bytes against
the existing scalar-decoder GPU reference, independent R1 row identities,
physical arrival permutations, graph replay, zero counters, padded views,
allocation canaries, absent bias, alpha/beta, and beta=0 over NaN old outputs.
The `tune` mode adds warm-address graph timings for every physical schedule;
`api-qualify` checks all R1..R8 with the public creator, complete workspace
partials, and two disjoint concurrent streams at R1/R6. `final` limits native
schedules to the profile and separately times the public creator's selected
body. Frozen v12 output dumps can be supplied as the fifth argument for
complete LEGACY output comparisons.

`tools/summarize_dense_r1.py` extracts tuning records and the best full-row
schedules. `tools/select_dense_r1_policy.py best-full-rows.json
src/dense_r1_policy.hpp` regenerates the profile. The existing split-K body
is retained only when it wins by more than 3% in that input profile and has
the same logical partition count. Odd rows borrow the next even-row physical
schedule and receive independent numerical qualification. The unprofiled
129280-output head uses a conservative schedule; its performance is unmeasured.
`tools/inspect_dense_r1.py` inspects linked ELF resources and publication
instructions without executing the GPU. These checks do not replace a
full-model integration gate. No model throughput claim follows from a HOT
operator measurement.
