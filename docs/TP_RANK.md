# Mix-B and expert TP: per-rank paths
> **Type:** user documentation.

Release 0.3 implements the requested per-rank shapes through explicit APIs.
Build with `STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP=ON` to enable them. With this
option disabled, the new functions, except initializers, return
`STRIX_ERROR_NOT_IMPLEMENTED`. Normal GEMV/GEMM dispatch and the earlier grouped
API retain their selection rules.

**Status: implementation, CPU checks and HIP compilation. The K2 expert
split plans with the lean core are qualified bitwise on the GPU and measured
as an isolated operator (see the README and
[experimental-k2-lean.md](experimental-k2-lean.md)); the other native
correctness and performance checks are pending.** The kernel, copy and memory
counts described here are code properties and contracts checked by the native
probe, not GPU results already obtained.

## 1. Dense K4/K5, R1 to R8

`strix_tp_dense_plan_create` accepts a `strix_matmul_desc` with local weights,
F32 input/output and NULL/0 workspace. Its submit runs a complete projection
in one kernel: scales, input H128, compressed decode, product, output H128,
bias and alpha/beta. `K4` and `K5` mean 4 and 5 bits per weight. They do not
specify the number of input-dimension partitions.

| Site | Local I | Local N | Relation to the full weight |
|---|---:|---:|---|
| q_b | 1,280 | 16,384 | Half of the W rows / output channels |
| o_b | 4,096 | 5,120 | Half of K / input channels |
| shared_gate, shared_up | 5,120 | 1,152 | Half of the output channels |
| shared_down | 1,152 | 5,120 | Half of the input channels |
| q_a | 5,120 | 1,280 | Replicated projection |
| kv | 5,120 | 512 | Replicated projection |

All geometries support both K4/K5 rates, the mul1 codebook and R=1..8. The
API accepts rank 1 for R1 or rank 2 for all rows. Channel stride must be 1;
row stride may exceed the local width. The pointer can refer directly to
one half of an engine activation/output. That view is not copied to another
tensor to run the projection. The validated span includes gaps between rows.
Overlap is rejected conservatively, including overlap within those gaps.

`strix_tp_dense_plan_create_tiled(..., rows_per_block, ...)` allows tests with
tiles of 1..R rows; 0 selects R. Each variant keeps one launch, with grid
`(N/128, ceil(R/tile))`, 512 threads per block and 2,560 bytes of LDS per tile
row. The all-row variant reuses weights across rows, but produces only 4
blocks for kv, 9 for shared projections and 10 for q_a. Smaller tiles allow
comparison of that reuse against greater parallelism. No variant is selected
automatically as the fastest without measurements.

The reduction preserves four logical K16 partitions. Each transforms complete
H128 blocks, then clips to its assigned K16 tiles. This is necessary for q_b:
an S4 partition has 320 channels, which is not a multiple of 128. I=1,152 also
has boundaries inside H128 blocks. Splitting the transform at those boundaries
would change the operator.

## 2. K2/K3 experts and the complete MoE block

Each expert has gate and up `[N=1152,I=5120]` and down `[N=5120,I=1152]`.
Each of its three weights can independently use K2 or K3 mul1. R1..R8 and
exactly six distinct experts per token are supported.

The local operation is:

```text
hidden[r,s] = SiLU(gate_e(X[r])) * up_e(X[r])
Y[r] = alpha * sum_s(score[r,s] * down_e(hidden[r,s])) + beta * Y[r]
```

Each projection includes its two H128 transforms, scales and bias. SiLU uses
a stable sigmoid evaluation. The sum visits slots 0..5 in order, starting at
+0, without atomics or a reduction order that depends on other blocks. An
exactly zero score skips its down projection; neither its hidden values nor
its down weights are read. Scores are not renormalized and must be finite
and nonnegative.

The complete submit uses **two kernels**:

1. Gate/up grouped by expert, reusing weights across tokens in the window
   that select that expert. Both projections and SiLU are fused. Inputs are
   read directly from X using its stride.
2. Down fused with routing weights and the top-6 reduction. Each block
   produces a final output. There is no later scatter or global per-expert
   partial output.

The only global intermediate tensor is native F32 hidden `[R,6,1152]`,
consumed directly by down. It requires `R*6*1152*4` bytes: 27,648 at R1 and
221,184 at R8, with a 256-byte-aligned base. It is not an input/output adapter
buffer. Submit allocates no other tensors.

### Persistent bank and GPU routing

`strix_tp_moe_plan_create_device` is the entry point for routes that change
during inference. It receives a `strix_tp_moe_device_desc` with a host bank
of 6..256 experts and two contiguous device buffers `[R,6]`: U32 IDs and F32
scores. The ID is the bank index; `experts[i].expert_id` must be i. The plan
uploads the bank once and retains its weights.

For each gate/up, candidate blocks correspond to the R*6 slots. The first
occurrence of each expert in that window groups its tokens. Blocks for
repeated occurrences exit without repeating the projection. The union of up
to 48 experts is discovered inside the kernel. There is no compaction phase,
host expert loop or routing copy in submit. Down reads device IDs/scores
directly.

The engine may write new routes, scores and X to its buffers before the next
submit, including when reusing a graph. It must order these writes in the
same stream or through events. The data must remain stable until the kernels
that read them complete. R, the bank, pointers and alpha/beta are fixed.
Changing them requires another plan.

Spans, types, owners, alignment and aliasing are checked when the plan is
created. Routing values are not downloaded to the host. The kernel detects
IDs outside the bank, duplicate IDs within a token and nonfinite/negative
scores. In these cases, the affected token's entire output is NaN and no
access goes outside the bank. Valid tokens are still computed. The caller
must treat a NaN output as a data failure, never as an inference result.

### Host union and fixed routes

`strix_tp_moe_plan_create` accepts a `strix_tp_moe_desc` with the exact union
of 6..48 experts and host routes/scores. IDs may be sparse and the expert
array may have any order. Extra experts, missing IDs, duplicates and invalid
scores are rejected before the table is allocated/uploaded. The plan copies
the table and synchronizes its upload during creation. The host arrays may
then be freed. Fixed routes are useful for reproducing a window and comparing
it with the device version. Recreate the plan if those routes change; use
the persistent bank for variable windows.

Both plan types offer `submit`, `gate_up` and `down`. Stage entry points allow
hidden values to be checked and measured and errors to be isolated before
testing the complete block. Each enqueues one kernel. Down requires valid
hidden data ordered in the stream; it does not check when they were produced.

### Resource lifetime and TP across machines

Plan creation must occur outside capture. The plan retains each weight
handle once, even if the bank shares it. Destroying those weights returns an
error while the plan exists. Submit does not allocate, copy, synchronize or
query pointer/stream attributes. It does check the current device and
propagates enqueue errors.

The device bank/table belongs to the MoE plan, and captured graphs retain
its address. **Complete all submissions and destroy all their graphs before
destroying the MoE plan.** X/Y/hidden/routes/stream belong to the caller and
must remain alive throughout that use. Do not free and reallocate them at
the same address. Serialize creation/destruction externally.

The o_b, shared_down and expert_down outputs are local contributions. The
engine must sum them across ranks. Bias for these projections must reside on
only one rank; a global beta residual must not be added twice either. Output
partitions are concatenated. q_a and kv are replicas and are not summed.
This library does not implement transport between the test machines,
attention, the KV cache, the router that selects experts or the complete
generation loop.

## 3. Per-rank weight preparation

```python
from strix_trellis import load_exl3, rank_partition

full = load_exl3("model.safetensors", "layers.0.attn.wo_b")
local = rank_partition(full, "o_b", rank=1, bias_rank=0)
regions, metadata = local.transport_regions()
# Use these regions with strix_weights_prepare_parts.
```

Supported names are `q_b`, `o_b`, `shared_gate`, `shared_up`, `shared_down`,
`q_a`, `kv`, `expert_gate`, `expert_up` and `expert_down`. The helper validates
the full shapes and the rates above and supports two ranks (0/1).
`slice_inputs(begin,end,include_bias=...)` also exposes general H128 slicing
and requires an explicit choice of bias handling.

Partitions are views of compressed arrays, without expanding Q. An output
partition usually needs trellis compaction when exported because of its tile
stride. This happens once during weight preparation. A contiguous K partition
preserves borrowed storage. The zero adapter copy guarantee concerns runtime
activation submission, not the removal of all initial weight transfers.

## 4. Download, build and measure on the test machines

Use a scratch directory. Prepare matching paths, sources, binaries and
fixtures on both test machines: the runner chooses where to execute. Use a
separate checkout to preserve the qualified V16 archive. In the commands
below, replace `<scratch dir>` with that directory.

```sh
git clone --branch master https://github.com/hyukudan/strix-trellis.git \
  "<scratch dir>/qualification-v03"
cd "<scratch dir>/qualification-v03"
```

### Build and CPU references

```sh
set -euo pipefail
mkdir -p build-qualification-v03/.tmp build-qualification-v03/.cache/comgr
export TMPDIR="$PWD/build-qualification-v03/.tmp"
export XDG_CACHE_HOME="$PWD/build-qualification-v03/.cache"
export AMD_COMGR_CACHE_DIR="$XDG_CACHE_HOME/comgr"

python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ./python
bash tools/ci_cpu.sh build-cpu-v03

nice -n 10 cmake -S . -B build-qualification-v03 \
  -DCMAKE_BUILD_TYPE=Release -DSTRIX_TRELLIS_ENABLE_HIP=ON \
  -DSTRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm/core-10.1 \
  -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1 \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151
nice -n 10 cmake --build build-qualification-v03 --parallel 1 \
  --target strix_trellis_rank_fused_probe

nice -n 10 python3 tools/make_rank_fixtures.py \
  --out "$PWD/build-qualification-v03/rank-fixtures"
```

The generator creates 14 dense projections, six expert bases and three window
oracles (shared, overlapping and disjoint unions), with seeds and hashes. It
decodes in blocks, does not store expanded Q and does not download shards.
The reference uses explicit F16/F32 stages, an FP64 core and numerical checks.
It does not emulate or replace a GPU kernel run.

### Three turns, three logs

```sh
set -euo pipefail
run_dir="$PWD/build-qualification-v03/results/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$run_dir"
git rev-parse HEAD > "$run_dir/commit.txt"
git status --porcelain > "$run_dir/worktree.txt"
/opt/rocm/core-10.1/bin/hipcc --version > "$run_dir/compiler.txt" 2>&1
sha256sum build-qualification-v03/strix_trellis_rank_fused_probe \
  > "$run_dir/binaries.sha256"
sha256sum build-qualification-v03/rank-fixtures/*.strank \
  > "$run_dir/fixtures.sha256"
cp build-qualification-v03/rank-fixtures/manifest.json "$run_dir/manifest.json"

for suite in dense moe moe-device; do
  bash tools/run_gpu_turn.sh --seconds 1180 \
    --maintenance-check "$PWD/tools/check_maintenance.sh" \
    --queue-arg "strix-v03-$suite" -- \
    "$PWD/build-qualification-v03/strix_trellis_rank_fused_probe" \
    "$suite" "$PWD/build-qualification-v03/rank-fixtures" 4 \
    2>&1 | tee "$run_dir/$suite.log"
done
```

`tools/run_gpu_turn.sh` wraps a site-specific GPU queue script (not part of
this repository). The wrapper enforces site-specific quiet hours,
maintenance and the turn limit. The loop advances only when the previous
turn finishes successfully. It retains the log on error or timeout and
confirms shutdown before retrying. Do not run the binary outside the runner
or invoke `--inside` directly. Do not change binaries/fixtures while queued.

### Interpretation

The dense campaign checks every K4/K5 shape and R1..R8, padded views, row
tiles, alpha/beta, weight lifetime and graphs. The fixed MoE campaign covers
unions of 6..48 experts, K2/K3 mixtures, hidden values, isolated down, the
complete block, zero scores and graphs. The device routing campaign changes
routes and scores without recreating the plan/graph and checks invalid
values and recovery.

The probe records `TP_PREPARE` separately from `TP_SAMPLE` and `TP_RESULT`.
The eleven timing sweeps use fixed warm addresses and separate host
submission time, GPU events and completion. They do not measure DRAM
streaming, full model loading, router time or tokens/s. The dense control
uses the earlier plan on contiguous tensors, excluding any cost of packing
engine views. Compare correctness first, then complete operator times for
the same shape/R/rate. Do not mix stage measurements with complete block
measurements.

A dense capture must contain one kernel node; the complete MoE capture must
contain two. Copy/allocation nodes are rejected within these captures. Keep
the final PASS markers, all checks and samples, together with machine,
power, clocks, temperature and hashes. Repeat in another independent turn
and inspect ISA/LDS/registers/spills before promoting a variant. Historical
README figures do not qualify these new kernels.
