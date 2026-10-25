# C API v1
> **Type:** user documentation.

Source version 0.3 adds explicit fused TP dense and expert MoE plans, including
a persistent expert bank with device routing. Version 0.2 introduced
region-based preparation and prepared linear plans.
The ABI version remains 1; existing descriptor layouts and entry points are
unchanged. The TP entry points require the matching 0.3 runtime and its
experimental build option. The last archived complete GPU qualification is V16; the additions have CPU contract
coverage and a separate native probe pending execution on gfx1151.

`include/strix_trellis.h` is the public C11/C++17 interface; `src/api.hip`
implements it against `src/backend.hpp`. This is the complete linear operator, starting from original-basis activations. The CMake static target `strix_trellis::runtime` links the API and HIP backend.
The interface target `strix_trellis::api` only supplies public headers.

## Operation and orientation

Transport `rows=N` means output channels; `columns=I` means input channels.
Both are positive multiples of 128 and at most `INT_MAX`. The packed, rotated
matrix Q has mathematical shape `[I,N]`; the conventional effective weight W
has `[N,I]`. Input X and output Y are original-basis, contiguous row-major F32.

The numerical path is input F32 times stored FP16 suh, round to FP16,
normalized block H128 in F32, round to FP16, decoded-core multiplication with
F32 accumulation, normalized block H128 in F32, stored svh multiplication in
F32, and optional F32 bias. These steps define `Linear(X)`.

```
Y_new = alpha * Linear(X) + beta * Y_old
Linear_real(X) = ((X * diag(suh)) H_I Q H_N) * diag(svh) + bias
```

Bias is inside the alpha-scaled linear result. The real equation describes
orientation, not the rounding of execution. The definitive arithmetic details
and tolerances belong in `NUMERICS.md`. The backend must use the same FP16
operands for scalar and WMMA paths; their FP32 reductions may differ.

Alpha and beta must be finite; negative values and zero are supported. Neither
scalar is inferred from zero initialization. `strix_matmul_desc_init(&d)` sets
alpha=1, beta=0, descriptor headers and nested F32 tensor headers. It leaves
shapes, strides, buffers and weights unset. A caller manually initializing all
fields must set alpha=1 itself for the usual linear operator. Beta=0, including
negative zero, must not read old output; an uninitialized output buffer is valid.
Alpha=0 still follows the selected numerical path; it does not bypass validation
or promise to suppress nonfinite intermediates.

GEMV accepts rank 1 `[I] -> [N]`, or rank 2 `[B,I] -> [B,N]` for B=0..8.
GEMM accepts only rank 2, B=0..INT_MAX. Rank 1 requires shape[1]=0,
strides[0]=1, strides[1]=0. Rank 2 requires strides `[width,1]`, including
when B=0. Ranks and batch dimensions must agree between input and output.
F16/BF16 enum values are reserved; these tensor types are not supported.

B=0 is a validated empty operation: descriptors, dtype, shape, scalars, weights,
stream and workspace's declared address/alignment/alias constraints still apply.
Tensor pointers may be NULL with buffer_bytes=0. There is no device buffer
inspection, launch or synchronization, and required workspace is zero.

### GEMM core selection and environment flags

`strix_gemm` (and every GEMM-path call) uses the register-blocked WMMA core
for codebook `mul1` (2) at 4-6 bits and at least 128 rows. It gives the same
bits as the previous core: the split count, the split bounds and the ascending
K16 accumulation order per output are unchanged; only the block tiling, the
LDS layout and the block order differ. With one split it can also apply the
output transform (H128, `svh`, bias, alpha/beta) in its epilogue, with the
same arithmetic and bits as the separate transform. By default it does so
only when the call's grid has more blocks than HIP reports
multiprocessors for the device (20 on gfx1151, where HIP counts WGPs): on
one-wave grids the epilogue was slower than the separate transform.
Process-wide flags, read once:

| Variable | Effect |
|---|---|
| `STRIX_TRELLIS_WMMA_TILED=0` | use the previous WMMA core everywhere |
| `STRIX_TRELLIS_WMMA_TILED_FUSED=0` (or `off`, `false`) | separate output transform everywhere (the previous default) |
| `STRIX_TRELLIS_WMMA_TILED_FUSED=1` | apply the output transform in the core's epilogue whenever possible (one split only) |
| `STRIX_TRELLIS_WMMA_TILED_FUSED=auto` (default) | as `=1`, but only when the call's grid has at least `STRIX_TRELLIS_WMMA_TILED_FUSED_MIN_BLOCKS` blocks (default: HIP's multiprocessor count + 1, i.e. 21 on gfx1151, more than one block per WGP); same bits |
| `STRIX_TRELLIS_WMMA_RASTER=<1..64>` | block-order group size (measurement only; bits unchanged) |

## Modern EXL3 host transport

`strix_exl3_transport_desc` points to a contiguous host blob and a host-native
`strix_exl3_metadata_v1`. These structs are not wire formats and must not be
read directly out of a file. A caller's loader extracts safetensors payloads and
packs them into the blob. It must reject duplicate/ambiguous codebook markers
before producing the one enum and marker pair. This API is not a safetensors
parser and cannot recover ambiguity discarded by a loader.

Set format=`STRIX_TRANSPORT_EXL3_CANONICAL`, transport_version=1 in both
transport and metadata, metadata_bytes=`sizeof(strix_exl3_metadata_v1)`, and
metadata.struct_size to its exact size. Dimensions in both descriptors must
match. Metadata flags and every reserved field must be zero.

| Region | Payload | Exact byte count | Offset alignment |
|---|---|---|---|
| trellis | Little endian 16-bit bit containers, shape `[I/16,N/16,16*bits]` | `I*N*bits/8` | 4 |
| suh | Stored FP16 `[I]`, full signs and magnitudes | `2*I` | 2 |
| svh | Stored FP16 `[N]`, full signs and magnitudes | `2*N` | 2 |
| bias | Optional F32 `[N]` | `4*N` | 4 |

`bits_x2` accepts only the even integers 2..16 (bits=1..8). Fractional rates,
historical packed su/sv, extra scaling conventions and other versions are
explicitly unsupported. Bias is absent only when bias_offset=bias_bytes=0.
All other regions are required. Offsets are relative to the blob start, with
checked offset/length arithmetic, no overlaps, no overrun and no pointer wrap.
Gaps and unused trailing bytes are permitted; blob base alignment is unrestricted.
The implementation requires a little endian host. It does not scan payloads for
NaNs/Infs or impose additional restrictions on stored scale/bias values.

| codebook | Enum value | Required marker |
|---|---|---|
| 3INST | 0 | 0, meaning absent marker |
| MCG | 1 | `0xCBAC1FED` |
| mul1 | 2 | `0x83DCD12D` |

The API accepts one codebook enum only, never a bitmask or arbitrary multiplier.
Unknown selections and mismatched constants fail explicitly. The transport
retains its packed tile layout; preparation uploads without changing its order.
Prepared GPU layout/offsets remain private and are not an additional transport.

### Preparation from separate host regions

`strix_weights_prepare_parts` accepts a `strix_exl3_parts_desc` instead of a
contiguous blob. Set its exact `struct_size`, `abi_version=1`,
`transport_version=1`, dimensions, `bits_x2`, codebook and marker. The four
`strix_host_region` members contain independent host pointers and exact byte
counts from the table above. No metadata offsets are added to these pointers.
Use `{NULL,0}` for absent bias; all reserved fields must be zero.

Host region addresses may be unaligned. Nonempty regions must be readable,
must not overlap each other, and must not wrap the address space. Their bytes
have the same little endian representation and codebook restrictions as the
canonical blob. Validation completes before device allocation or transfer.
The runtime copies the regions directly into the same persistent, aligned HIP
allocation used by `strix_weights_prepare`, and completes the supplied stream
before returning. It does not assemble a temporary host blob or expand Q.

The loader's `transport_regions()` supplies readonly byte views for this path;
contiguous little endian payloads borrow the mmap storage. Keep all returned
views live until prepare returns. `transport_buffer()` remains useful for
consumers requiring one contiguous blob and can fill caller-owned writable
storage. `transport_blob()` retains the immutable-bytes interface.
See [LOADER.md](LOADER.md) for packing and lifetime details. These interfaces
avoid host staging copies; they do not replace persistent HIP allocation with
host-mapped GPU reads or promise a transfer speedup without measurement.

### Device span of prepared weights

`strix_weights_device_span` returns the base address and size of the single
device allocation that holds the prepared trellis, `suh`, `svh` and optional
bias, in the private prepared layout. The view is read-only and valid until
`strix_weights_destroy`. It lets a caller read the bytes ahead of use, for
example to warm caches while the GPU waits on other work; the library does not
depend on it. Both outputs are cleared on error, and a NULL handle or output
pointer returns `STRIX_ERROR_INVALID_ARGUMENT`. The macro
`STRIX_TRELLIS_HAS_WEIGHTS_SPAN` marks headers that declare it.

## Lifecycle, device and streams

All descriptors require exact struct_size, ABI version 1, and zero reserved
fields. Metadata uses transport_version instead of abi_version. Sizes and
offsets are bytes; tensor dimensions and strides are elements. Host pointers
must designate readable, sufficiently large objects, and handles must be live
handles returned by this API. Arbitrary or stale host pointers/handles are not
memory-safe inputs and cannot be diagnosed reliably by a C ABI.

1. Select a HIP device using HIP. Create a context with its explicit ordinal.
   Creation checks architecture `gfx1151` (allowing HIP feature suffixes) and
   reported wave size 32; it does not select or switch the calling thread's device.
2. Set a mandatory stream descriptor, including for the default stream:
   hip_stream=NULL means the current context device's default stream. A non-NULL
   stream is borrowed and checked with hipStreamGetDevice/hipStreamGetFlags.
3. Prepare weights. The API owns a private HIP allocation with aligned slices;
   it copies each payload and synchronizes the supplied stream before returning.
   Earlier work on that stream can therefore delay preparation. On failure after
   starting copies it also attempts to drain them before freeing the allocation.
   On successful return the caller may immediately release/reuse all transport
   descriptors and host bytes, and use the weights on another stream.
4. Allocate tensor/workspace buffers, query workspace, then submit. All device
   operations and queries require the calling thread's current device to match
   the context. Explicit streams must belong to that device.
5. Destroy any prepared plans retaining weights, after completing their work,
   then destroy the weights. Complete all work
   using a context before destroying it. Context destruction rejects remaining
   weights. Destroy never serves as the API's completion mechanism, although
   HIP allocation release may itself synchronize. NULL destroy handles are invalid.

Contexts and weights are owned by the caller as handles; stream, tensor and
workspace ownership is never transferred. Prepared weights are immutable and
belong to one context, even if another context uses the same device. Tensor
buffers, workspace, stream and weights must survive stream completion. The host
submission descriptors/group array need survive only the call. External
serialization is required for context/weights/plan lifecycle mutations; simultaneous
submissions using live immutable weights require distinct output/workspace and
normal HIP ordering. Caller synchronization is required before destruction.
Creation/preparation set output handles to NULL on failure when out is valid.

## Buffer validation and workspace

Nonempty input/output/workspace must be HIP device allocations on the context
device, recognizable by hipPointerGetAttributes and hipMemGetAddressRange.
Managed memory, host/registered-host memory and allocations that cannot be
queried this way are unsupported. Subranges of supported allocations are valid.
F32 tensors need four-byte pointer alignment; workspace needs 256-byte alignment.
Both the tensor's declared buffer_bytes and HIP allocation extent must cover
its logical accessed bytes. The entire supplied workspace_bytes must fit its
allocation. Workspace queries validate host descriptors and logical byte ranges;
they do not validate HIP allocation provenance or use workspace fields. Thus
the query needs tensor pointers and declared sizes, but not allocated workspace.

Output/input overlap, either tensor overlapping prepared weights, and workspace
overlapping either tensor or weights are rejected. Tensor checks use logical
accessed byte ranges; workspace checks use its full declared byte range. Unused
tensor capacity can share an allocation with a disjoint slice. All address and
size arithmetic is checked. No in-place operation is supported.

Query `strix_matmul_workspace_bytes` for the descriptor, regardless of whether
GEMV or GEMM will submit it; both backend modes use that upper bound. A zero
backend requirement allows NULL/zero workspace. `SIZE_MAX` from the backend
query is treated as an unsupported size, not a successful allocation request.
The backend must implement overflow-safe workspace calculations for its accepted
dimensions. Query output bytes are zero on any validation failure.

Hot submission and workspace queries allocate no memory, transfer no data and
perform no stream/device synchronization. Submission does perform HIP metadata
queries for allocation/stream validation. No runtime tensors are copied or
converted on the host. Submission status reports validation/enqueue only;
asynchronous errors and completion must be obtained through HIP by the caller.

## Prepared linear plans

For a repeated projection whose addresses, extents, shapes, scalars and stream
remain fixed, create a `strix_linear_plan` outside stream capture. Pass a fully
populated `strix_matmul_desc`, including allocated workspace, a live stream
descriptor, and `STRIX_OPERATION_GEMV` or `STRIX_OPERATION_GEMM`. Creation
performs the same descriptor, device-allocation, extent, stream and alias
validation as the ordinary entry point. It copies the descriptor and retains
the prepared weights. The selected mode preserves the existing backend policy;
this is not an autotuner and does not select the experimental K2 kernel.

```c
/* context, weights, buffers, workspace and stream_desc are already prepared. */
strix_linear_plan plan = NULL;
strix_status status = strix_linear_plan_create(
    context, &matmul_desc, &stream_desc, STRIX_OPERATION_GEMV, &plan);
if (status != STRIX_SUCCESS) { /* handle the error */ }
/* After successful creation, update input contents in HIP stream order. */
status = strix_linear_plan_submit(plan);
/* Check enqueue status, then obtain completion/errors from HIP. */
/* After the last operation has completed: */
status = strix_linear_plan_destroy(plan);
```

Submit checks the calling thread's current HIP device and enqueues the normal
linear operator. It performs no allocation, host/device transfer, synchronization,
pointer-attribute query, allocation-range query or stream query. It does not
repeat descriptor validation. Editing or releasing the original host descriptor
has no effect on the frozen copy. Tensor contents may change with correct HIP
ordering, but addresses, extents, dtype, shapes, strides, weights, alpha/beta,
workspace and stream may not. Create a new plan for those changes. Freeing and
reallocating a borrowed buffer at the same address also violates the contract.
An empty-row plan retains the existing validated no-launch behavior.

Plan handles do not own the input, output, workspace or stream. Those resources
must remain valid through plan destruction and completion of all submitted
work. `strix_weights_destroy` rejects a handle retained by any live plan;
the weights also keep their context live. Lifecycle reference counts are not
atomic: serialize context, weights and plan creation/destruction externally.
Plan destruction does not synchronize or wait for asynchronous work.

An already-created plan may be submitted during HIP stream capture. Captured
graphs and graph executables borrow the same GPU memory and may outlive the
plan. Keep weights, tensor buffers and workspace alive until the final replay
has completed and the graphs/executables have been destroyed. Destroying the
plan only removes its explicit weight retention; it does not track graph
ownership. Synchronizing the capture stream alone does not end graph lifetimes.

With `STRIX_TRELLIS_PLAN_FUSED_COMPLETION` (build option, OFF by default), a single-row GEMV plan of a
fused-input shape (for example the K6 `q_a` and K5 `o_a` projections) runs as
one launch: the last thread block of each 128-column output group folds the
split partials and applies the output transform, with the same arithmetic as
the two-launch path, so the outputs are bit-identical. Such a plan allocates a
small array of completion counters (one 128-byte line per 128 outputs). Creation zeroes
them and synchronizes the plan's stream before it returns. The counters belong
to the weights handle and are freed by `strix_weights_destroy`, so graphs
captured from the plan stay valid after the plan is destroyed, under the rules
above. Direct submissions and graph replays of one plan share its counters and
workspace and must not overlap in time; distinct plans never share counters.

The native `strix_trellis_prepared_api_probe` checks region preparation, frozen
plans, full outputs, graph replay and timing separation; build/run commands are
in [K2_LOCAL.md](K2_LOCAL.md).
Its hot-weight timings isolate host submission behavior and must not be
reported as DRAM bandwidth or cold-weight inference throughput.

## Pre-routed grouped MoE

Each host group is an ordinary matmul descriptor, rank 1 or rank 2. It must have
workspace=NULL and workspace_bytes=0. Groups may share a prepared weights handle,
but their logical input/output slices must be pairwise disjoint, including all
cross-group input/output pairs. No group's tensors may overlap another group's
weights. The enclosing descriptor supplies one workspace, and its query returns
the **maximum**, not the sum, of group requirements. Shared workspace is checked
against all groups before any launch.

All groups are validated before enqueueing the first. They are then submitted
sequentially on the supplied stream, reusing workspace in stream order. Rows<=8
prefer GEMV; larger batches prefer WMMA. GEMM's direct entry point prefers WMMA;
GEMV's direct entry point prefers the vector route. These are backend preferences,
not guarantees of instruction choice or performance.

Empty groups follow the same B=0 contract. group_count=0 allows groups=NULL and
needs zero workspace, while still validating enclosing/stream descriptors. There
is no routing, scatter, gather, router, top-k weighting or reduction. The caller
performs those operations and supplies already routed contiguous slices. If a
backend enqueue fails after earlier groups were submitted, those groups remain
enqueued; submission is not transactional.

## Errors and validation

The additive 0.3 TP interfaces are documented in [TP_RANK.md](TP_RANK.md).
They are explicitly enabled at build time and do not change the functions
above. `strix_tp_dense_plan_create[_tiled]` accepts direct F32 row views for
the K4/K5 rank geometries. `strix_tp_moe_plan_create` freezes a host top-6
window; `strix_tp_moe_plan_create_device` retains a persistent expert bank
and reads changing device routes at submission/replay. Both MoE creators
use the same two-stage submit and staged gate/up and down entry points.
The old `strix_grouped_moe` pre-routed/contiguous contract remains unchanged.

TP MoE plans own device descriptor storage. Complete their work and destroy
all graphs that capture their submissions before destroying the plan. The
caller keeps every tensor, routing buffer, workspace and borrowed stream
alive until that point. No TP submission performs a HIP allocation, transfer
or synchronization. Plan creation validates and may upload/synchronize.
In builds without `STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP`, TP creators/submits
return `STRIX_ERROR_NOT_IMPLEMENTED`; descriptor initializers remain usable.

### Common status handling

Statuses distinguish invalid arguments/reserved fields/aliasing, ABI mismatch,
unsupported format/device, insufficient storage, allocation failure and HIP or
backend failure. Unknown statuses have a stable fallback string. Implemented
entry points never return successful placeholders for unsupported work.
Memory exhaustion is distinguished when HIP explicitly reports it; other HIP
errors map to BACKEND except failed stream/allocation validation, which maps to
INVALID_ARGUMENT (or INSUFFICIENT_BUFFER for a proven short allocation).

CPU validation with a fake HIP header requires only g++ and Python. From the
repository root in Linux/WSL:

```sh
python3 - <<'PY'
from pathlib import Path
root = Path.cwd()
p = Path('/tmp/strix-api-cpu/hip')
p.mkdir(parents=True, exist_ok=True)
(p / 'hip_runtime.h').write_text(
    '#define STRIX_API_FAKE_HIP\n#include "' +
    str(root / 'tests/test_api_contract.cpp') +
    '"\n#undef STRIX_API_FAKE_HIP\n')
PY
g++ -std=c++17 -Wall -Wextra -Werror -pedantic -fsanitize=address,undefined \
  -I/tmp/strix-api-cpu -Iinclude tests/test_api_contract.cpp \
  -o /tmp/strix-api-contract
/tmp/strix-api-contract
```

These tests include the real API implementation but mock HIP and the backend.
They check metadata boundaries, independent-region uploads and cleanup, plan
retention/frozen descriptors, validation, forwarding, hot-path side effects and
grouped submission, not real GPU arithmetic or HIP behavior.
Do not compile this test against real HIP; the injected runtime header is required.
The test is registered as `api_contract` in the default Linux CMake configuration.

Compile the real object separately with HIP, then link the real backend:

```sh
nice -n 10 /opt/rocm/core-10.1/bin/hipcc --offload-arch=gfx1151 \
  -std=c++17 -Wall -Wextra -Werror -Iinclude -c src/api.hip -o api.o
```

This object compilation alone does not resolve `linear_workspace_bytes` or
`launch_linear` and does not validate kernels. No GPU execution is needed.
