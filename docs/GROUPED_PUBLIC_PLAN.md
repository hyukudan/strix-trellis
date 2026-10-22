# Public prepared grouped K3 plan — proposal, not implemented
> **Type:** design note — engineering record; may describe plans, candidates or experiments that are not part of the supported API.

Date: 2026-10-21. Design base: `ab062a6`, branch `design/grouped-public`.
This document is for HIGH review after the private performance gates. It does
not authorize implementation, promotion, a default change or GPU execution.
No grouped performance has been demonstrated. The qualified default remains
V14b/private9; V16/TEN (`0e60746`, integration revision `a81efca`) is pending
qualification on actual hardware. The private host-side performance harness
`7c6a07e`, included in this base, remains under HIGH review. Global `new`
interposition and weak receipt validation are known blockers under repair.
They are not considered resolved here.

## Evidence and inspection limits

The following references identify actual symbols and sources, not a
hypothetical API that is already available:

| Inspected source | Fact that constrains this proposal |
|---|---|
| [Public header](../include/strix_trellis.h), `strix_grouped_moe_desc`, handles and enums | ABI1, exact descriptor sizes and zero reserved fields; HIP-free C; opaque weights/context |
| [API](../src/api.hip), `grouped_check`, `strix_grouped_moe_workspace_bytes`, `strix_grouped_moe` | MAX workspace; complete validation before launch; sequential groups and potentially different row counts; no X sharing |
| Same source, `device_buffer`, `storage_check`, `tensor_check`, `call_check` | HIP attributes/range queries during old-style submission, arithmetic and interval alias checks; declared tensor capacity and actual extent are separate checks |
| Same source, `strix_context_destroy`, `strix_weights_prepare/destroy` | Context counts weights; prepare owns the allocation and offsets; weight destruction calls `hipFree` immediately, without a plan pin |
| [Current contract](API.md), Lifecycle, Buffer validation and Pre-routed grouped MoE | Borrowed host descriptors; caller manages GPU/stream lifetime and completion; no rollback |
| [Backend](../src/backend.hip), `linear_workspace_bytes`, `launch_linear`; [private types](../src/backend.hpp) | A aligned to 256, P after A; DeviceWeights is private, not a consumer format |
| `git show 0e60746:src/backend.hpp`, `:src/api.hip`, `:src/gemv_fused_both.hip` | TEN adds `fused_both_capable`; one additional function probe in an active context; public C ABI remains 1 |
| [Private mathematical/ownership contract](GROUPED_K3_ADAPTER.md), [R1 design](GROUPED_K3_R1.md) | Fixed old K3 implementations, NativeBatch2320, LAST validation and experiment leases; some historical default/gate statements predate this proposal |
| [Actual private ABI](../experiments/grouped_k3_math/abi.hpp), [actual owner](../experiments/grouped_k3_math/owner.hpp), `PreparedPlan`, `resources`, `launch_three`, `submit` | Batch passed by value, three functions, private owner with its own streams/allocations; that owner does not exist for borrowed public arrays |
| Native QA 618d4ae (internal record), [harness](../experiments/grouped_k3_math/native_qa.hip) | Private native PASS on gfx1151: 14 configurations, A/P/Y exactly matching original V13, FP64 references, 46 negative tests with zero nodes, two streams/graphs and owned leases; does not prove public API behavior or performance |
| Private performance harness, [heap guard](../experiments/grouped_k3_perf/heap_guard.hpp), [admission](../experiments/grouped_k3_perf/admission.py) | Private CPU infrastructure and pending gates; do not treat its mocks/receipts as hardware results or import C++ interposition into the API |

The private evidence for seven probes comprised four V13 probes plus three
grouped probes. It does not promise seven in V16: there may be five existing
plus three new probes, eight in total. The private PASS that released
Owner/Plan retained leases for its own graphs. It does not justify releasing
a public plan with borrowed arrays and externally owned HIP graphs.

## Decision: four functions and an opaque plan

The proposal provides a CPU dimension query, explicit creation that prepares
and validates, submit and destroy. A single call receiving descriptors on
every submission would have to repeat queries or accept unverified caller
certificates. A hidden cache would complicate invalidation and lifetime.
Neither option satisfies query-free submission under the requested minimal
contract. No public allocation registry, stream certificates or graph
framework is added.

Exact header addition outline; names and comments are in English.
**These are not installed declarations or implemented symbols**:

```c
typedef struct strix_grouped_k3_plan_impl *strix_grouped_k3_plan;

typedef struct strix_grouped_k3_plan_desc {
    uint32_t struct_size;
    uint32_t abi_version;
    const strix_matmul_desc *groups;
    uint64_t group_count;
    void *workspace;
    uint64_t workspace_bytes;
    uint64_t reserved[4];
} strix_grouped_k3_plan_desc;

/* CPU-only sizing; no context, handles, tensors or HIP queries. */
strix_status strix_grouped_k3_workspace_bytes(
    uint64_t input_channels, uint64_t output_channels,
    uint64_t group_count, uint64_t *out_bytes);

/* Prepare outside capture. Bind one explicit, non-NULL HIP stream. */
strix_status strix_grouped_k3_plan_create(
    strix_context context, const strix_grouped_k3_plan_desc *desc,
    const strix_stream *stream, strix_grouped_k3_plan *out_plan);

/* Enqueue only. Caller must satisfy device, stream and lifetime preconditions. */
strix_status strix_grouped_k3_plan_submit(strix_grouped_k3_plan plan);

/* Caller must finish all uses and destroy all referencing graphs first. */
strix_status strix_grouped_k3_plan_destroy(strix_grouped_k3_plan plan);
```

Proposed public layout on a host with 64-bit pointers and natural alignment 8:
struct_size@0, abi_version@4, groups@8, group_count@16, workspace@24,
workspace_bytes@32, reserved@40; sizeof=72, align=8. No packing. The handle is
pointer-sized. The new structure is a distinct type even though its layout
matches the old grouped descriptor. Do not cast between them. The future
implementation will check C11 and C++17 sizes/offsets without including HIP
in the header. It requires `struct_size == sizeof(new_type)`, abi_version=1
and all four reserved fields zero, including those in each nested
matmul/tensor/stream. No reserved bits are reused and no old descriptor bytes
change. ABI1 is retained; no arbitrary package version jump to 0.2 is required.

The symbols are additive. Future destroy protection when pins exist is an
explicit lifetime extension for the new plans. With zero plans, old calls
retain their results, validation, selection and MAX behavior.
`strix_grouped_moe` is never redirected to the prepared path, and its alias/
row acceptance is not expanded. There is no implicit public activation of
the fast path.

## Domain and numerical semantics

1..32 nonempty groups, homogeneous in I/N. All weights must be prepared by
the same context, K3 (`bits=3`), CB2/mul1 (`codebook=2`), R1. Only these
geometries are supported:

| I | N | Fixed S | A bytes | P bytes | Workspace per group |
|---:|---:|---:|---:|---:|---:|
| 5120 | 2304 | 16 | 10240 | 147456 | 157696 |
| 2304 | 5120 | 4 | 4608 | 81920 | 86528 |

X/Y are contiguous F32. Rank1 and rank2 `[1,width]` are accepted under current
stride and shape rules, with matching X/Y rank within each group. Groups do
not all need the same rank representation. F16/BF16, B=0 or B>1, mixed
geometries, other bits/CB and other dimensions return UNSUPPORTED_FORMAT,
without fallback. Each matmul must have workspace=NULL/0; only the new
descriptor supplies workspace. Alpha/beta must be finite per group. Optional
F32 bias comes from the weights.

Preserve `alpha * Linear(X) + beta * Yold`, with bias inside Linear. There is
no alpha0 shortcut. Beta +0/-0 does not read Yold; negative beta does. Preserve
input RN32/RN16 rounding, H128, fdot2, split order and epilogue as defined in
the mathematical contract. Do not reinterpret the product as exact FP64.

The future internal adapter extracts packed/suh/svh/bias pointers field by
field from its own handles. It does not expose or serialize DeviceWeights,
its nine or ten fields, capability offsets or private certificates to
consumers. TEN does not enter the kernels: only packed/sign/bias pointers
and X/A/P/Y do. The three literal baseline K3 implementations fixed in the
private contract remain the source. They are not replaced by TEN kernels,
and neither S nor mathematical instructions change.

Private NativeGroup: eight pointers, alpha/beta; 72 B align8, offsets
0/8/16/24/32/40/48/56/64/68. NativeBatch: four int@0/4/8/12,
groups@16, 2320 B align8, inactive slots and padding zeroed. Each of the three
kernels receives only Batch by value. Host/device asserts and linked ELF must
confirm size/alignment/kernarg. The budget is <=4096 B including hidden
arguments, not an architectural assumption. Block128/wave32; input/core/output
grids are respectively (10,G,1)/(36,16,G)/(5,G,1) and
(5,G,1)/(80,4,G)/(10,G,1).

## Workspace and access separation

The query checks only out, G and dimensions: null out returns INVALID_ARGUMENT;
a valid out is zeroed before validation. G0/33 or any value outside 1..32
returns INVALID_ARGUMENT; unsupported dimensions return UNSUPPORTED_FORMAT.
It does not check handles, pointers, dtype, device or resource availability.
SUCCESS means the size can be calculated, not that create will accept the
device.

`Abytes=align256(2*I)`, `Pbytes=4*N*S`, `stride=Abytes+Pbytes`,
`total=G*stride`. Both strides are already multiples of 256. The workspace
base is aligned to 256; group g uses A at `base+g*stride`, with P immediately
after A. Each group occupies the same bytes as the old K3/R1 workspace, but
the total is SUM. For G32: 5046272 and 2768896 B, calculated, not measured.
The entire supplied workspace_bytes capacity, including its tail, is checked
against its allocation and all buffers outside workspace. There is no GPU
allocation, descriptor upload or copy in submit.

Every interval is half-open `[start,start+bytes)`. Use checked integers before
forming pointers: product/sum/alignment, SIZE_MAX/UINTPTR_MAX limits,
offset<=allocation_size and bytes<=allocation_size-offset. Exact adjacency
is valid; even one overlapping element is not. A one-past pointer is not
allowed for any nonempty range. If HIP proves base+offset==end and capacity
is insufficient, return INSUFFICIENT_BUFFER. If HIP does not recognize that
pointer, return INVALID_ARGUMENT. Comparing virtual addresses alone does
not establish provenance safety.

Reads: X, packed, suh, svh and any bias from all groups. Writes: all Y, A and
P. Y remains a writer with beta0 and is also a reader if beta!=0. Read/read
sharing is allowed, including identical X and repeated weights. Every writer
must be disjoint from **all** readers and writers within and across groups.
The entire workspace allocation must also be disjoint from X, Y and every
weight allocation. The actual weight owner supplies its complete allocation,
including padding, as a protected range. Internal A/P occupy calculated
disjoint slices. No input may hide in workspace gaps/tail. The test harness
allocates external canaries; the API does not invent them.

## Create: complete validation and atomic host-plan publication

Create runs outside capture, with the context active on the thread and a
mandatory stream descriptor. It requires `hip_stream != NULL` to bind an
explicit stream. This new restriction avoids per-thread default stream
ambiguity; the old API still accepts NULL. HIP checks the current device,
hipStreamGetDevice/GetFlags and capture state. Create during capture returns
INVALID_ARGUMENT, without adding its own nodes. Implicit device switching
through hipSetDevice is not allowed.

Proposed sequence, without kernel/copy enqueues or synchronization:

1. Validate out and set it to NULL. Validate readable headers/arrays, bounds
   and context/stream. Copy descriptors into a bounded private host snapshot.
   The caller must not mutate those objects during create. Only the snapshot
   feeds later passes; publication retains no descriptor pointers.
2. Validate all G groups, including LAST, declared sizes, types, geometry,
   scalars, weight/workspace affinity, arithmetic and all aliasing.
3. Obtain actual HIP evidence for X/Y/workspace through hipPointerGetAttributes
   and hipMemGetAddressRange, once per distinct range during preparation.
   Accept only non-managed device memory on the correct ordinal. Each range
   must cover its logical bytes; workspace covers all declared bytes. Certify
   weight reads from actual allocations/offsets recorded by
   `strix_weights_prepare`. Check their extents and internal pointers without
   accepting external certificates. Deduplicating reads is a local optimization,
   not a global cache of recyclable addresses.
4. Run this context's resource preflight if it has not run yet. Build immutable
   Batch/grids, retain handles and publish only after success. Failure leaves
   out=NULL, zero launches and no persistent partial pins. Creation may allocate
   host memory and fail with OUT_OF_MEMORY; it does not allocate GPU memory.

An error discovered in LAST must leave no nodes or partial writes. A host
certificate does not prove future GPU array lifetime. These queries prove
extent/type/device only at that instant.

## Resource preflight: once per actual context

Add bounded private context state: uninitialized/ready/unavailable/failed,
with a stored status. The first valid create outside capture runs preflight,
with externally serialized creation/destruction. No preflight API is added.
Query each of the **three actual linked functions** exactly once with
hipFuncGetAttributes. Collect all three results and retain the first failure.
No submit, capture, replay, destroy or later create repeats the probes. The
five existing V16 probes are unchanged: eight successful probes are possible,
not five or seven as a universal requirement. In builds without grouped
images, create explicitly returns NOT_IMPLEMENTED, not an empty SUCCESS.

Verify gfx1151/wave32, block128 limits and maximum grids, maxThreads>=128,
LDS/local0 and image admission. Spills0/kernarg/scalarization also require
ELF/ISA inspection of the actual archive; HIP attributes do not establish
them. Device limits may be stored privately at creation or queried in this
preflight, never in submit. No hipSetAttribute.

Current context creation may accept an inactive ordinal, leaving optional
capabilities false. That does not prove kernel resources on that ordinal.
The new preflight is deferred until plan creation with the context actually
active. It can then validate grouped support even if old capabilities remain
false. It neither re-enables nor queries V16 again. A context never changes
ordinal.

Known insufficient resources/inadmissible image: cache unavailable and return
UNSUPPORTED_DEVICE from create, without sequential fallback. Unknown HIP error:
cache failed, return BACKEND (OUT_OF_MEMORY if explicit), and do not hide it
as a false capability. Old APIs remain usable under their own rules. Retrying
a failed preflight requires a new context; there are no retries on the active
context.

## Lifetime, pins and concurrency

Explicit required future internal changes: add `plan_refs` to weights and
context. Each plan increments each distinct weight once and the context
once, using checked counters without wrap. `strix_weights_destroy` must
reject with INVALID_ARGUMENT **before hipFree** if plan_refs>0, retaining
the handle and memory. `strix_context_destroy` must reject live weights or
plan_refs. Do not silently free or defer a successful destroy. With zero
pins, old code applies. Plan destruction decrements both sets and frees only
host memory, with no HIP query, synchronization or GPU free. Lifecycle is
externally serialized.

The plan does not own X/Y/workspace, stream or caller graphs. From create
until plan destruction and completion/destruction of all graphs that used
it, the caller must preserve addresses, extents, device, allocations and
stream. None may be freed, reallocated or replaced. Weights are immutable.
Descriptor bytes may be freed after create; scalars/addresses/grids are fixed.
There is no plan update; changes require another create.

Do not externally mutate X, scales, packed, bias or Y/workspace between
preparation and completion of first use, or during any later use. Between
completed uses, X and Yold contents may be updated with explicit HIP ordering
and unchanged addresses; A/P are scratch. The next execution, including
replay, again forbids concurrent mutation. This allows new tokens without
recreating a plan but does not claim automatic exclusion over borrowed memory.

Submit takes no stream: it uses the stream validated at create. The thread
must have that context's device current and keep the stream alive and usable,
as with a public HIP handle. Submit performs no hipGetDevice/stream query;
violating this precondition has no guaranteed diagnostic. Forged or dangling
C handles and invalid host storage also cannot be diagnosed safely. Fictitious
registries do not provide protection against them.

A plan does not support simultaneous host calls or executions that overlap
on its buffers. Successive uses may be enqueued in the same stream because
stream order serializes writes. Graphs using the plan are launched in that
stream and order. For two streams, use two plans, explicit streams and
disjoint writers; shared read/read access is allowed. Create checks each
plan. It cannot detect global conflicts with unrelated plans/operations;
the caller prevents them.

## Submit, errors and graphs

Submit validates a nonnull handle, reads immutable host state, launches
input/core/output in order using Batch by value and returns the first error.
It performs no host or GPU allocation, copy, synchronization,
pointer/resource/device/stream query, reconstruction or fallback. Enqueue
error checking, such as hipGetLastError in the actual adapter, is not a
metadata probe. SUCCESS confirms only accepted enqueues.

If the second/third enqueue fails, earlier ones remain queued. Stop the
sequence without rollback or freeing resources. The caller obtains
asynchronous errors and completion through HIP outside submit. Before reuse
after an error, drain or resolve the state through HIP, restore Yold if
needed and, after failed capture, invalidate/discard the graph. Failure does
not imply zero work.

Successful capture produces three mathematical nodes and two dependencies,
with no host, memcpy or alloc nodes. HIP packages Batch by value at launch;
a temporary host Batch need not survive capture. The graph retains **raw
pointers**, not public leases on their allocations. Retain plan, weights,
context, stream and buffers until all launches complete and both graph and
graphExec are destroyed. Earlier plan destruction violates the contract. It
does not return a hypothetical BUSY for a live graph: this API does not know
the edges/owners of external HIP graphs. Verifiable test: destroying weights/
context with a live plan must be rejected. Rejection of plan destruction with
a live graph/in-flight work, or proof of raw-buffer lifetime, is not promised.

Shutdown order: complete the stream and all replays, destroy graphExec/graph,
destroy plan, destroy weights, free buffers/stream, then destroy context when
no weights/plans remain. Buffers/stream may be freed after the plan and graphs;
destroy does not act as a completion mechanism.

## Proposed statuses without extending enums

Reuse exactly the ABI1 statuses, without adding BUSY or generic UNSUPPORTED:

| Condition | Status |
|---|---|
| Incorrect descriptor size/version | ABI_MISMATCH |
| Nonzero reserved fields, required NULL check failed, G out of range, nonfinite scalar, foreign owner, aliasing, alignment, invalid wrap/sum/product, incorrect stream/current device at create, active capture | INVALID_ARGUMENT |
| Unsupported dtype/rank/shape/stride/CB/bits/geometry | UNSUPPORTED_FORMAT |
| Declared capacity below logical/required workspace size, or demonstrably short HIP extent | INSUFFICIENT_BUFFER |
| Unrecognized pointer, host/managed/foreign-device memory or failed provenance/stream query | INVALID_ARGUMENT |
| Known incompatible architecture/grouped resources | UNSUPPORTED_DEVICE |
| Grouped images not compiled into that runtime | NOT_IMPLEMENTED |
| Host allocation failure or explicit HIP OOM | OUT_OF_MEMORY |
| Other backend/probe/enqueue failures | BACKEND |
| Destroy of weights/context retained by a plan | INVALID_ARGUMENT, handle intact |

Validation follows the create order described above. No priority is guaranteed
among multiple simultaneous defects. Negative tests isolate one cause at a
time. Query sets out_bytes=0 and create sets out_plan=NULL on every failure
when out is valid.

## Later test plan, without an API stub now

CPU tests with the **actual future implementation** and HIP/backend mocks:
HIP-free C11/C++17 header compilation; identical old sizes/offsets; malformed
new descriptors and the complete domain; CPU query without HIP calls; exact
SUM and overflow; host canaries; all read/write aliases and legal read/read
sharing; exact extents, one byte short, interior/one-past and invalid LAST
before any node. Do not write tests that merely mirror a stub or an invented
host certificate.

Count actual probe identities: three new probes once per context, five
preexisting V16 probes where applicable, zero probes in submit/replay/
destruction and later creates. Cover a context created inactive followed by
active preflight, cached false/failed results without fallback, active capture
and each allocation failure with pin rollback. Test pins for repeated
weights, two plans, rejected destroy and final release; the first error at
each of the three enqueues without simulating rollback. Run complete
regression tests for the old grouped MAX contract, mixed rows/empty/rejections,
and unchanged V16 selections. Instrument actual static archive code with
explicitly scoped test hooks. Do not replace global `operator new` or impose
C++ interposition on consumers/DSOs. Mocks do not test driver internals.

Future native tests: both geometries at G1/2/4/8/16/31/32 = 14 configurations.
Compare complete A/P/Y for every group and split against the actual old
baseline and CPU FP64/ordered-transform references with the already fixed
private tolerances. Preserve baseline exactness where applicable. Use 32
distinct alpha/beta pairs per geometry: negatives, +0/-0, YNaN under beta±0
and cancellation. Use different packed/scales/X in neighboring groups,
present/absent bias and subnormals/ties. Check all immutable source contents
and all canaries before/after and after LAST, not a rotating sample.

LAST negative tests: create outside capture must fail and leave plan=NULL,
with no enqueues or buffer changes. Then check an empty capture and rejection
of submit(NULL), with zero nodes. Do not attribute that rejection to LAST
validation inside capture: create during capture is rejected separately.
Enqueue two streams with two plans and disjoint writers before any wait.
Test direct execution, capture, graph/replay at least twice and LAST. Inspect
three functions/grids/two edges and complete arguments. Overwrite/destroy
temporary host Batch before instantiation. Retain the plan and public buffers
in all graphs. Test rejection of retained-owner destruction. Do not execute
UAF to imitate private dropPlan. Declare any untested asynchronous/OOM errors.

## Performance and qualification gates

First resolve the private host-side HIGH review, including new/receipts, and
run its authorized hardware gates. This only decides whether implementing
this plan is worthwhile. Neither private QA nor its three kernels establish
public performance. After implementation/review, measure the actual old C
API against the new prepared API, with preparation/uploads/allocations for
both outside timing and matching weights/S/Yold/stream/cache conditions.
The old API's validation on warm calls **is** part of its measured calls.
State its cost relative to prepared submit; do not attribute the entire
difference to GPU execution. Present the private 3G versus 3 control separately.
Comparable graphs: old 3G nodes versus prepared 3, same transforms and
epilogue, event and wall time, losses/tails, two independent campaigns and
complete commands, hardware, hashes, correctness and results. Do not invent
speedups.

Any changed archive/runtime requires new native qualification on hardware
and overall qualification. Recompile all consumers against matching TEN,
including the engine when integrated. A private V13/private9 control does
not qualify a mixed TEN archive. This document changes no default, GPU source,
CMake or public selection. Future GPU turns require separate authorization,
nice during compilation, a site-specific GPU queue script (not part of this
repository), maintenance checks and site-specific quiet hours. The limit is
<=20 minutes; an empty queue is not authorization. No executions occur in
this work.

## Engine integration: example for later downstream work

Plan5 needs grouped MoE and a connection to the engine C API. In a downstream
engine integration (not public), the caller could form homogeneous pre-routed
R1 groups, query SUM, allocate buffers, create a plan on an explicit stream
and use submit. To retain sequential/mixed-row behavior, explicitly choose
the old API and query MAX. Do not pass SUM/plan to an old call as a hidden
selector. Report unsupported preparation to the engine. Any fallback is
chosen outside capture/timing with its own workspace and lifetimes, never
inside submit.

```c
/* Proposed usage only; workspace and stream already allocated by the caller. */
uint64_t bytes = 0;
strix_grouped_k3_plan plan = NULL;
strix_status status = strix_grouped_k3_workspace_bytes(i, n, count, &bytes);
if (status != STRIX_SUCCESS) return status;
strix_grouped_k3_plan_desc d = {0};
d.struct_size = sizeof(d);
d.abi_version = STRIX_TRELLIS_ABI_VERSION;
d.groups = groups;
d.group_count = count;
d.workspace = workspace;
d.workspace_bytes = bytes;
status = strix_grouped_k3_plan_create(context, &d, &stream, &plan);
if (status != STRIX_SUCCESS) return status;
status = strix_grouped_k3_plan_submit(plan);
/* Caller completes HIP work, handles errors and destroys any graphs here. */
/* Only then: strix_grouped_k3_plan_destroy(plan). */
```

This is a documentation example, not a compilable program or engine
implementation. Changes to addresses, geometry or scalars require another
plan. Routing, scatter, top-k reduction, SiLU/SwiGLU and end-to-end MoE
performance are outside scope.

## Pending HIGH decisions, in priority order

1. **Prior evidence gate:** repair/review the private host-side harness and
   receipts and obtain hardware performance results; then authorize or reject
   implementation. Neither this document nor 618d4ae closes that gate.
2. **Accept explicit lifetime requirements:** weight/context pins are
   enforceable; raw buffers and external graphs depend on the caller. If
   reliable rejection of plan destruction with live graphs is required, this
   minimal contract does not provide it. That would require different graph
   ownership integration, outside scope.
3. **Accept a fixed explicit stream and device precondition:** this allows
   zero metadata queries in submit. Changing streams on every submission or
   diagnosing every violation would imply a different API/cost. It is not
   added silently.
4. **Review deferred preflight and cached failure:** this avoids changing the
   five V16 probes and supports contexts created inactive. The first create
   pays for three probes, and a failure requires a new context. This is not
   optional fallback.
5. **Confirm the narrow domain and SUM cost:** new shared read/read access,
   R1/CB2/K3 only. It does not replace old mixed rows or MAX. Its actual value
   for Plan5 must be measured in later engine integration.

The signatures/layout and choices above form the concrete review proposal.
They are not unresolved decisions left to an implementer without a contract.
