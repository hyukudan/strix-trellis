# Homogeneous grouped MoE experiment for K3/R1

> **Type:** design note — engineering record; may describe plans, candidates or experiments that are not part of the supported API.

Proposal for independent review before writing kernels. V12 is the qualified
production version. The existing grouped API executes groups sequentially. This
private experiment studies three launches for an entire batch of experts, without
yet changing the API, workspace queries, CMake or public policy.
It is not a persistent scheduler and does not implement routing, SiLU or full SwiGLU.

## Initial domain

- 1–32 active groups, all R1/CB2/mul1/K3, with the same I/N and S per request.
- Two geometries: I5120/N2304/S16 and I2304/N5120/S4, matching the current
  production policy for these formats. Timing trials fix G=4/8/16/32.
- Original contiguous F32 X/Y, complete F16 suh/svh, optional F32 bias,
  finite F32 alpha/beta per group. Buffers and weights are already prepared.
- Weights and inputs may share read-only ranges. No writable Y/A/P range may
  overlap reads or writes from any group. Dependencies between groups retain
  the sequential baseline and are not allowed in this experiment. No zero-alpha
  shortcuts are introduced.
- Preparation, device/stream/ownership/extent/alignment checks, allocations and
  transfers occur outside the timed interval and submit. Buffers, streams and
  weights remain alive until all uses finish. Do not mix the private eight-field
  and nine-field ABIs: rebuild controls and the harness with V12 and its archive.

## By-value arguments and memory

A private GPU descriptor has eight pointers and two floats:

```cpp
struct Group {
    const uint32_t* packed;
    const float* x;
    const uint16_t* suh;
    const uint16_t* svh;
    const float* bias;
    float* y;
    _Float16* a;
    float* partial;
    float alpha, beta;
};
struct Batch {
    int input, output, splits, count;
    Group groups[32];
};
```

Expected x64 layout: Group=72 B, Batch=2320 B, alignment eight. This calculation
alone is not sufficient: host HIP compilation static_assert checks, emitted
offsets and the actual ELF kernarg size are validation gates. The proposed limit
is 4096 B including hidden arguments, pending confirmation on this platform.
If the compiler/hardware cannot support it, the design is blocked. Do not replace
it with hipMemcpy or allocations inside submit. There is no GPU descriptor array
that needs uploading/updating on each call.

Construct and initialize the entire request, including padding/unused slots where
applicable, before timing. Each launch receives Batch by value. Captured nodes
own a copy of their launch arguments; the owner manages allocation/stream
lifetimes. A test must mutate the host copy after capture and verify that replay
retains the captured arguments. Do not retain pointers to a temporary host Batch.

Each group owns aligned, canary-protected A half[I] and P float[S,N]. A dedicated
partitioned allocation may use size_t64 offsets/alignment 256, checking sums and
products before forming pointers. Required memory is the sum of all group
allocations: do not reuse the sequential maximum. The private experiment declares
its capacities explicitly and does not change the existing public contract.
Later integration must review query and submit together. Never enqueue a batch
using the old maximum workspace.

## Three stages without internal global synchronization

1. Grouped input: block128, four H128 waves per CTA; grid.x covers
   ceil((I/128)/4), grid.y selects the group. Each wave loads its group's original
   X and suh, materializes RN32 of the product before RN16, applies the existing
   FP32 H128 and RN16 A. Waves beyond the last H do not access data. This stage
   adds no CTA barriers: each H wave is independent.
2. Grouped core: extract the project's production
   `gemv_baseline_kernel<3,true,1,false>` body from gemv_lds literally, with
   AlignedLane/V1, dot2 and its reduction order. This baseline has no LDS or CTA
   barriers. Do not replace it with gemv_lds_kernel/Chunk8 or add staging.
   Only uniform pointer/metadata selection from Batch and group=blockIdx.z
   change. Grid.x=ceil((N/16)/4), grid.y=S, grid.z=count, block128;
   four N16 tiles per CTA, with the same bounds and K partition as V12. Keep
   the original predicates and per-group P[split,N] layout. Do not introduce
   the experimental K3 decoder or change S to improve timing.
3. Grouped output: block128, four H128 waves; grid.x covers
   ceil((N/128)/4), grid.y selects the group. Sum S in ascending order from +0,
   then use the same FP32 H128, svh, bias, alpha and separate beta product/sum.
   Beta ±0 does not read Y, even if Y contains NaN. Numerical association is
   unchanged.

Enqueue the three stages on the same stream and return the first error. Submit
has no atomics, inter-CTA barriers, counters, hipDeviceSynchronize or stream sync.
Verify that Batch reads and descriptor selection are uniform within each wave.
Use readfirstlane for the required indices without treating an arbitrary
divergent address as valid. Submit has no pointer/attribute queries, allocations
or explicit copies.

## Controls and requirements before timing

Control A: G complete V12 launch_linear calls, using the actual policy and the
same descriptors/buffers/stream. Control B: G original input/core/output
sequences with identical S, to locate any stage difference. Candidate C: three
grouped stages. Keeping S and the numerical body allows byte-for-byte equality
to be required for all A/P/Y per group, alongside complete CPU references from
original X. The expected graph node count is 3G versus 3. These must be actual
kernel nodes, verified after capture.

Independent review must resolve argument size/lifetime, all three grid limits,
aliasing across all groups, canaries, A/P indexing, core barriers, each writer's
owner, rounding and enqueue errors. The CPU gate must execute the actual C++
guard and enumerate groups 1/2/4/8/16/31/32, invalid 0/33 and overflow, both
geometries, neighboring groups/slots and pointer extremes. Do not execute a HIP
program to prepare or discover GPUs outside the allocated GPU slot.

Later GPU gates: all A/P/Y, canaries and immutable inputs; distinct alpha/beta/bias
per group; distinct weights and X to detect accidental use of a neighboring
descriptor; repeated weights with shared read-only inputs; two streams with
independent requests; capture/replay and host argument copies; validation failure
in the last group before any enqueue. A descriptor error must not leave earlier
stages from other groups enqueued.

Real EAQ K3 data are available in the shared expert10 cache, w1/w2/w3, with
complete R8 CPU fixtures. Use R1 and distinct w1/w3 as groups of the same geometry.
Disjoint weight clones used for memory timing must be identified as clones,
not different trained experts. Each trial must keep >=256 MiB of disjoint packed
weights resident across requests/clones, with explicit limits of GPU1 GiB and
host2 GiB. Prepare all references and allocations before the GPU slot.

Compile with nice/one job, freeze and verify both test machines, and inspect
actual ELFs for wave32 and spills/private storage/argument loads. Resources do
not prove occupancy. Use two physical campaigns, eleven sweeps alternating A/C,
event/host timing and all outputs before/after/the final result. Freeze selection
before confirmation. Any proposal requires >=2% full-operator gain in both
campaigns. Keep regressions, tails, failures and original logs. Do not extrapolate
an isolated core result to the API or full model. Public integration is a later gate.

## Requirements clarified after independent review

The independent review identified
ambiguity in the earlier baseline. Its P1-01 applies by specifying the global
baseline body without LDS. Its LF SHA is
a5fbb60a5cde97442455e041e7f10c76397209cf2c335fc16600346c3da59503.
Chunk8 mentions in the test are counterexamples for another variant.
Keep S16/S4, total_rows=1 and literal V1 body/helpers.

The CPU test does not close the ABI/resource and ownership/capture P1 issues.
Before the large kernels, a small three-trace-kernel pilot will verify the
by-value container, HIP offsets, actual kernarg, resources, argument copies,
dependencies and two streams with live owners. The pilot does not compute GEMV
or establish resources or performance for the final mathematical functions.
Those still require their own gates.
