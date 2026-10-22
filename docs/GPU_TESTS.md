# HIP correctness harness
> **Type:** user documentation.

`tests/gpu_runtime.hip` consumes self-contained external fixtures, prepares
weights through the real C API, and checks the complete operator from
original-basis F32 input. It does not substitute a different backend.
CMake links `strix_trellis_gpu_runtime` with the real library.
The harness checks correctness and does not measure performance.

## Reproducible CPU generation

From the checkout root, with Python and NumPy available:

```powershell
python tools/make_gpu_fixtures.py --out "<fixture output dir>" --real-directory "$env:STRIX_TRELLIS_FIXTURES"
```

Set `STRIX_TRELLIS_FIXTURES` to a local fixture directory.
This also works on Linux with Linux directory paths.
It does not download data. EAQ loading uses `tests/fixtures/eaq_v1.json`:
it checks size, SHA-256 and the mul1 marker before generation.
Missing or corrupt data is an error; `--skip-real` explicitly excludes it.
Binary fixtures and their manifest remain outside Git.
`manifest.json` records NumPy, seed, dimensions, sizes and SHA-256.
Generation uses PCG64 with base seed 20261020 and recorded individual seeds.

There are 48 synthetic cases: integer bits 1..8, codebooks 3inst/mcg/mul1,
and both rectangular orientations I=128,N=256 and I=256,N=128.
Trellis containers are valid uint16 values transported as little-endian bits.
Scales are F16 with signs, zeros, negative zero and magnitudes from 0.0625 to 2.
Odd bit widths include F32 bias; even widths test absent bias.
Each case has 32 rows; prefixes allow different sizes to be tested without
duplicating weights. The first row has asymmetric impulses at lane/block
boundaries; the others are random.
The real case is wkv EAQ K6 mul1, I=5120,N=512, without bias, with 16 rows.

The reference is `matmul_fp32_stages`, the same independent implementation
used by `matmul_fp32`, without BLAS or GPU code.
It preserves RN16 before and after H128, scalar CPU accumulation in increasing
I order, output H128 in F32, svh and bias.
The harness applies alpha/beta in F32 to that reference, with separate
multiplication and addition. Logical Q and the core checkpoint are stored explicitly.

## STGPU001 v1 binary format

All integers and payloads are little endian. C/C++ structs are not serialized.
The reader limits files to 256 MiB and validates versions, duplicates, lengths,
dimensions, marker, required sections and absence of trailing bytes.
The manifest allows SHA-256 integrity checks outside the executable;
the C++ reader does not implement a checksum.
Use generated and verified fixtures.

Header: 8-byte ASCII magic `STGPU001`, uint32 version=1, uint32 section count.
Each consecutive section contains an 8-byte ASCII tag padded with NUL,
a uint64 length, and exactly that many payload bytes, without padding.

| Tag | Content |
|---|---|
| meta | 8 uint32: I,N,bits,codebook enum,marker,rows,bias present,seed |
| trellis | uint16 [I/16,N/16,16*bits] |
| suh, svh | F16 [I] and [N] |
| bias | F32 [N], or zero length |
| x | Original F32 [rows,I] |
| old | F32 [rows,N], previous output for nonzero beta |
| scaled | F16 [rows,I], RN16(X*suh) |
| inh | F32 [rows,I], H128 of scaled |
| a | F16 [rows,I], core input |
| q | F16 [I,N], Q decoded in logical order |
| core | F32 [rows,N], CPU product of a and q |
| ref | F32 [rows,N], Linear(X), including bias |

## Executable coverage

```sh
gpu_runtime [--route all|gemv|gemm|grouped|stages] fixture.stgpu [...]
```

The default is `all`. Any API, HIP, format, stage or tolerance failure
produces a nonzero exit status. It prints the actual device, architecture,
wave size, case, route, rows, rank, alpha/beta, maxabs, maxref, nrmse and
worst index. `--route stages` checks stages; its PASS does not establish
validation of the complete matmul.

- GEMV: rows 0..8, rank 1 for one row and rank 2 for all sizes.
- GEMM: rank 2, rows 0..8,16,17,32,64,65,128,129,256 as fixture capacity allows.
  In addition to the requested 1/2/4/8/16, the EAQ case includes the available
  small prefixes.
- Alpha/beta: (1,0), (-0.75,0.25), (0,-0.5) and (1,-0).
  Zero beta is tested with newly allocated uninitialized memory and,
  separately, memory filled with NaN.
- Groups: disjoint contiguous slices with 1/2/4/8/16 rows and an empty group.
  A second prepared expert retains trellis/suh/bias and sets svh to zero;
  its reference is pure bias. This detects an ignored handle for another expert.
  Each group is compared with its individual matmul through the same route
  and must have identical bytes, in addition to its individual comparison
  against the Python reference.
- Q: an isolated probe uses production `WaveTile` and `codebook_value`
  to decode every weight and compare it bit for bit with logical Python Q.
  Rectangular dimensions and asymmetric impulses detect permutations;
  the complete comparison also verifies the actual core kernel's indexing.
- Input: a checkpoint probe using the production H128 helper requires identical
  bytes for scaled FP16, H128 FP32 and FP16 core input.
  `input_transform` is also tested directly.
  Signed zeros and RN16 errors are not hidden.
- Output: `output_transform` receives the known Python core and is compared
  with ref. This separates H128/svh/bias errors from product errors.
  The backend's internal core has no public extraction point:
  it is validated through isolated Q and the complete operator,
  without inventing a debugging API or another matmul kernel.

The context and allocations belong to device 0.
An explicit nonblocking stream is used for uploads, launches and downloads,
and synchronized before reading results or destroying weights.
Prepare receives native metadata and a blob with aligned offsets,
F16 for suh/svh and F32 for bias.
Temporary host storage is released when prepare returns.
Each route queries and allocates its workspace (alignment 256);
groups share one allocation and their individual descriptors have NULL workspace.
Zero rows use NULL buffers and must require zero workspace.
The test checks ordering and results on the stream; it does not measure
whether the host blocks during submission.
Exhaustive negative tests remain in the existing CPU mock.

## Numerical criterion

Both conditions are required for complete and isolated output:

```text
maxabs = max(abs(actual-reference)) <= 5e-4 * max(1,max(abs(reference)))
nrmse = sqrt(sum((actual-reference)^2) / max(sum(reference^2),1e-30)) <= 3e-5
```

Any nonfinite value is rejected. Bitwise identity of matmul is not required:
a GPU tree/WMMA reduction can differ from scalar CPU summation even when
FP16 products are exactly representable in FP32.
These thresholds are proposed acceptance criteria for these fixtures,
pending GPU measurement; they are not a universal bound
or an approved logit tolerance.
`NUMERICS.md` explains gamma bounds and their dependence on cancellation
and scales. The dual condition limits maximum error and total error energy.
Input stages and Q do require isolated exactness, to avoid attributing
decoding/rounding errors to core reduction.
Groups on the same route must match their individual execution bit for bit.

## Building and running

Compile the object without GPU execution in a dedicated scratch directory:

```sh
cd "<scratch dir>"
TMPDIR=$PWD/tmp XDG_CACHE_HOME=$PWD/cache AMD_COMGR_CACHE_DIR=$PWD/cache \
  nice -n 10 /opt/rocm/core-10.1/bin/hipcc --offload-arch=gfx1151 \
  -std=c++17 -Wall -Wextra -Werror -ffp-contract=off -Iinclude \
  -c tests/gpu_runtime.hip -o gpu_runtime.o
```

Link this object with the real C API/backend using hipcc,
retaining `-ffp-contract=off` for stage probes. Do not link mocks.
After creating the executable and copying fixtures to the scratch directory,
the exact payload command for the shared GPU turn is:

```sh
./gpu_runtime --route all fixtures/*.stgpu > gpu-runtime.log 2>&1
```

At the time of this record, the object and 49 fixtures had been staged
in a test scratch directory, with fixtures under `fixtures/`.
The queue wrapper's interface was read without executing it.
After checking site-specific quiet hours and maintenance,
the outer invocation from that directory uses the repository wrapper:

```sh
bash tools/run_gpu_turn.sh --seconds 1180 \
  --maintenance-check "$PWD/tools/check_maintenance.sh" \
  --queue-arg trellis-gpu-tests -- \
  timeout --signal=TERM --kill-after=5s 1180s \
  ./gpu_runtime --route all fixtures/*.stgpu > gpu-runtime.log 2>&1
```

`tools/run_gpu_turn.sh` wraps a site-specific GPU queue script
(not part of this repository). Use it for all GPU execution,
with a maximum of 20 minutes, outside site-specific quiet hours
and subject to maintenance.
The wrapper was not invoked for this delivery.
To isolate a failure, select `--route stages`, `gemv`, `gemm` or `grouped`
and a single file. Record the outer turn command, commit/backend, compiler,
printed hardware and complete log before claiming a GPU result.

## Integrated runtime qualification

`runtime-v4` checks 56 fixtures: the original 49 and seven real dense groups
up to M256. It includes the backend's GEMV/LDS and WMMA M64 selection.
Six rejection cases with real HIP check host memory, managed memory,
a subrange exceeding its allocation, and short/misaligned/overlapping workspace;
they verify that output remains intact.
For the subrange, INVALID_ARGUMENT or INSUFFICIENT_BUFFER is accepted because
aliasing with adjacent allocations can take precedence.
The same valid explicit stream runs the positive cases.
No foreign device or invalid stream was fabricated that could cause the driver to fail.

The seven larger fixtures are generated by `tools/make_dense_gpu_fixtures.py`
from previously shared ranges. `--maxrows 256` produces 591,593,728 bytes,
with a maximum of 214,764,288 per file.
External manifests retain SHA-256.
The full-layer benchmark is separate: `bench/api_linear_bench.hip`,
CMake option `STRIX_TRELLIS_BUILD_BENCHMARKS=ON`;
see [Measured performance](../README.md#measured-performance) for its method.

## Later v7 baseline

Runtime v7 repeats 80 fixtures (the earlier 56 +24 for subnormals/range),
5,076 comparisons and 400 exact stages.
The full API extends the sweep to M32/128/129: 67 cases and 6,401 comparisons.
Graphs: 16 cases, 10,272 comparisons.
Sources, binaries and metrics were pinned in internal qualification records
(not published).
These results do not carry over to a new policy/kernel.
