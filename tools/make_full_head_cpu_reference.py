"""Stream a complete real-head CPU oracle without materializing the full Q."""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from strix_trellis.linear import input_stages_fp32, core_dot_fp32, output_stages_fp32
from audit_ours_packs import independent_lut, decode_block, check_tile, low_priority
from make_ours_pack_fixtures import peak_working_set


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(actual, expected):
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Invalid full-output comparison")
    error = float(np.max(np.abs(a-b)))
    nrms = float(np.sqrt(np.sum((a-b)**2) / max(float(np.sum(b*b)), 1e-30)))
    if error > 5e-4 * max(1., float(np.max(np.abs(b)))) or nrms > 3e-5:
        raise ValueError(f"Full-output numerical gate failed: {error}, {nrms}")
    per_row=[dict(maxabs=float(np.max(np.abs(x-y))),
                    nrms=float(np.sqrt(np.sum((x-y)**2)/max(float(np.sum(y*y)), 1e-30))))
                         for x, y in zip(a, b)]
    for row, ref in zip(per_row, b):
        if row['maxabs'] > 5e-4*max(1.,float(np.max(np.abs(ref)))) or row['nrms'] > 3e-5:
            raise ValueError('Per-row numerical gate failed')
    return dict(values=int(a.size), maxabs=error, nrms=nrms, per_row=per_row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--cuda-reference", type=Path)
    parser.add_argument("--amd-reference", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    low_priority()
    args.out.mkdir(exist_ok=False)
    started = time.perf_counter()
    manifest_path = args.head / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if any(manifest[k] != v for k, v in dict(input=5120, output=129280,
            rows=8, codebook="mul1", actual_model_values=True).items()) or manifest['bits'] not in (4,5,6):
        raise ValueError("Expected a frozen complete real K4/K5/K6 head")
    initial = {}
    for name, entry in manifest["files"].items():
        path = args.head / name
        initial[name] = (path.stat().st_size, path.stat().st_mtime_ns)
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError("Head source integrity failure: " + name)
    i, n, rows, bits = 5120, 129280, 8, int(manifest['bits'])
    packed_bytes = i*n*bits//8
    t = np.memmap(args.head/"head.blob", mode="r", dtype="<u2", shape=(i//16, n//16, 16*bits))
    u = np.memmap(args.head/"head.blob", mode="r", dtype="<f2", offset=packed_bytes, shape=(i,))
    v = np.memmap(args.head/"head.blob", mode="r", dtype="<f2", offset=packed_bytes+2*i, shape=(n,))
    x = np.fromfile(args.head/"input.f32", dtype="<f4").reshape(rows, i)
    stages = input_stages_fp32(x, u)
    a = stages["core_input_fp16"]
    cuda_a_equal = None if args.cuda_reference is None else a.tobytes()==(args.cuda_reference/"r8-a.f16").read_bytes()
    if cuda_a_equal is False:
        raise ValueError("CPU A differs from all original-input CUDA A bytes")
    table = independent_lut("mul1")
    ordered = np.empty((rows, n), dtype="<f4")
    high_precision = np.empty_like(ordered)
    a64 = a.astype(np.float64)
    checks, blocks, weights, max_bound_ratio = 0, 0, 0, 0.
    for begin in range(0, n, 512):
        end = min(begin+512, n)
        packed = t[:, begin//16:end//16]
        q = decode_block(packed, bits, table)
        for ti, tj in {(0, 0), (i//32, packed.shape[1]//2), (i//16-1, packed.shape[1]-1)}:
            check_tile(packed[ti, tj], bits, table, "mul1")
            checks += 1
        core = core_dot_fp32(a, q)
        q64 = q.astype(np.float64)
        exact = a64 @ q64
        mass = np.abs(a64) @ np.abs(q64)
        gamma = (2*i*2**-24)/(1-2*i*2**-24)
        err = np.abs(core.astype(np.float64)-exact)
        if not np.isfinite(exact).all() or np.any(err > gamma*mass+1e-30):
            raise ValueError("Ordered FP32 core exceeds full FP64 error bound")
        max_bound_ratio = max(max_bound_ratio, float(np.max(err/np.maximum(mass, 1e-30))))
        ordered[:, begin:end] = output_stages_fp32(core, v[begin:end])["output_fp32"]
        high_precision[:, begin:end] = output_stages_fp32(exact.astype(np.float32), v[begin:end])["output_fp32"]
        blocks += 1
        weights += q.size
        peak_working_set()
        if blocks % 16 == 0 or end == n:
            print("CPU_PROGRESS", end, n, "elapsed_s", time.perf_counter()-started, flush=True)
    sampled = np.fromfile(args.head/"expected.f32", dtype="<f4").reshape(3, rows, 128)
    for index, begin in enumerate(manifest["sampled_H128_blocks"]):
        if ordered[:, begin:begin+128].tobytes() != sampled[index].tobytes():
            raise ValueError("Complete CPU output differs from prior sampled ordered oracle")
    metrics = dict(ordered_vs_high_precision=compare(ordered, high_precision))
    if args.cuda_reference:
        cuda = np.fromfile(args.cuda_reference/"r8-y.f32", dtype="<f4").reshape(rows, n)
        metrics.update(cuda_vs_ordered=compare(cuda, ordered), cuda_vs_high_precision=compare(cuda, high_precision))
    if args.amd_reference:
        amd = np.fromfile(args.amd_reference/"r8-y.f32", dtype="<f4").reshape(rows, n)
        metrics['amd_v12_vs_ordered']=compare(amd, ordered)
    for name, view in [("r8-y-ordered.f32", ordered), ("r8-y-fp64-core.f32", high_precision),
                       ("r8-a.f16", a), ("r8-inh.f32", stages["input_hadamard_fp32"])]:
        view.tofile(args.out/name)
    for name, identity in initial.items():
        path = args.head/name
        if identity != (path.stat().st_size, path.stat().st_mtime_ns):
            raise ValueError("Source changed during CPU generation")
    result = dict(passed=True, device="CPU only", gpu_executed=False, numpy=np.__version__,
        input=i, output=n, rows=rows, bits=bits, codebook="mul1", decoded_weights=weights,
        output_values=ordered.size, blocks=blocks, independent_full_tiles_checked=checks,
        all_codebook_states_checked=65536, A_all_bytes_equal_CUDA=cuda_a_equal,
        source_weights=manifest.get('source_weights'),
        sampled_prior_ordered_outputs_all_bytes_equal=True, max_core_error_over_abs_products=max_bound_ratio,
        elapsed_cpu_seconds=time.perf_counter()-started, host_peak_working_set_bytes=peak_working_set(),
        source_manifest_sha256=sha(manifest_path), source_files=manifest["files"],
        generator_sha256=sha(Path(__file__)), decoder_sha256=sha(ROOT/"tools/audit_ours_packs.py"),
        reference_sha256=sha(ROOT/"python/strix_trellis/linear.py"), metrics=metrics,
        reference_contract="Original F32 X, RN32 scale then RN16, staged F32 H128/RN16 A; entire ordered increasing-K separate F32 multiply/add; staged F32 output H/svh. Additional whole-core FP64 dot reference independently rounds to F32 before the same output stage.",
        files={p.name:dict(bytes=p.stat().st_size, sha256=sha(p)) for p in args.out.iterdir() if p.is_file()})
    (args.out/"manifest.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8", newline="\n")
    print("PASS_complete_CPU_head", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
