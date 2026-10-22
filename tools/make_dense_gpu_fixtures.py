#!/usr/bin/env python3
"""Generate external real dense STGPU001 checkpoints, with bounded CPU checks."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from strix_trellis.linear import matmul_fp32_stages, input_stages_fp32, output_stages_fp32
from strix_trellis.codec import decode_tile
from fetch_dense_fixtures import PREFIXES

SEED = 20261020
FILE_LIMIT = 256 * 1024**2
TOTAL_LIMIT = 1024**3


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def load_group(metadata, directory, prefix):
    tensors, provenance = {}, {}
    for suffix, dtype in (("trellis", "<i2"), ("suh", "<f2"), ("svh", "<f2"), ("mul1", "<i4")):
        name = prefix + "." + suffix
        info = metadata["tensors"][name]
        path = directory / info["file"]
        size = math.prod(info["shape"]) * np.dtype(dtype).itemsize
        if info["dtype"] != {"<i2": "I16", "<f2": "F16", "<i4": "I32"}[dtype]:
            raise ValueError(f"Wrong dtype: {name}")
        if size != info["bytes"] or path.stat().st_size != size or digest(path) != info["sha256"]:
            raise ValueError(f"Source integrity failure: {name}")
        tensors[suffix] = np.memmap(path, mode="r", dtype=dtype, shape=tuple(info["shape"]))
        provenance[name] = info
    t, u, v = (tensors[key] for key in ("trellis", "suh", "svh"))
    if int(tensors["mul1"]) & 0xffffffff != 0x83DCD12D:
        raise ValueError("Unknown codebook marker")
    if t.ndim != 3 or u.ndim != 1 or v.ndim != 1 or t.shape[:2] != (u.size // 16, v.size // 16):
        raise ValueError("Invalid tensor geometry")
    if u.size % 128 or v.size % 128 or t.shape[2] not in range(16, 129, 16):
        raise ValueError("Unsupported geometry/rate")
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError("Nonfinite source scales")
    return t, u, v, provenance


def check_stages(x, t, u, v, stages):
    """Check full transform stages and independent scalar FP64 sampled dots.

    fsum has a different implementation/reduction from the ordered FP32 oracle.
    Scalar circular bit extraction and struct half rounding also check sampled Q
    without using the codec lookup table or unpack_states.
    """
    for name, value in input_stages_fp32(x, u).items():
        if value.tobytes() != stages[name].tobytes():
            raise ValueError(f"Input checkpoint mismatch: {name}")
    for name, value in output_stages_fp32(stages["core_fp32"], v).items():
        if value.tobytes() != stages[name].tobytes():
            raise ValueError(f"Output checkpoint mismatch: {name}")
    q = stages["rotated_fp16"]
    bits = t.shape[-1] // 16
    for i in sorted({0, 1, 15, 16, 127, len(u) - 1}):
        for j in sorted({0, 1, 15, 16, 127, len(v) - 1}):
            words = t[i // 16, j // 16]
            ring = 0
            for k in range(0, len(words), 2):
                ring = (ring << 32) | (int(words[k]) & 65535) | ((int(words[k+1]) & 65535) << 16)
            r, c = i % 16, j % 16
            pos = ((c & 7) << 5) | ((r & 6) << 2) | (r & 1) | ((r & 8) >> 2) | ((c & 8) >> 1)
            size = 256 * bits
            state = (((ring << size) | ring) >> (size - (pos + 1) * bits)) & 65535
            mixed = (state * 0x83DCD12D) & 0xffffffff
            numerator = (1024 + sum(mixed.to_bytes(4, "little"))) * 1774 - 1329 * 2048
            expected = struct.pack("<e", numerator / 2**18)
            if q[i, j].tobytes() != expected:
                raise ValueError(f"Independent scalar Q mismatch at {i},{j}")
    a = stages["core_input_fp16"]
    columns = sorted({0, 1, 15, 16, 31, 127, len(v)//2, len(v)-1})
    errors, ratios = [], []
    gamma = (2 * len(u) * 2**-24) / (1 - 2 * len(u) * 2**-24)
    for row in range(len(x)):
        for col in columns:
            products = [float(av) * float(qv) for av, qv in zip(a[row], q[:, col])]
            exact = math.fsum(products)
            mass = math.fsum(map(abs, products))
            error = abs(float(stages["core_fp32"][row, col]) - exact)
            if error > gamma * mass + 1e-30:
                raise ValueError("Core FP64 bound exceeded")
            # Independent ordered scalar sum must reproduce the stored core bits.
            acc = np.float32(0)
            for product in products:
                acc = np.float32(acc + np.float32(product))
            if acc.tobytes() != stages["core_fp32"][row, col].tobytes():
                raise ValueError("Scalar ordered FP32 mismatch")
            errors.append(error)
            ratios.append(error / mass if mass else 0)
    return dict(full_input_and_output_stage_bytes_equal=True,
                independent_q_scalar_samples=36, fp64_core_samples=len(errors),
                sampled_columns=columns, max_core_fp64_abs_error=max(errors),
                max_core_error_over_abs_products=max(ratios), gamma_2I=gamma,
                ordered_scalar_core_bytes_equal=True)


def write_stream(path, sections):
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite fixture: {path}")
    size = 16 + sum(16 + value.nbytes for value in sections.values())
    if size > FILE_LIMIT:
        raise ValueError(f"File exceeds 256 MiB: {size}")
    temporary = path.with_suffix(".partial")
    with temporary.open("xb") as stream:
        stream.write(b"STGPU001" + struct.pack("<II", 1, len(sections)))
        for tag, value in sections.items():
            stream.write(tag.encode().ljust(8, b"\0") + struct.pack("<Q", value.nbytes))
            # tofile streams contiguous arrays without a bytes/bytearray copy.
            np.ascontiguousarray(value).tofile(stream)
    if temporary.stat().st_size != size:
        raise ValueError("Wrong serialized size")
    # Read every serialized section back and compare exact bytes in bounded chunks.
    with temporary.open("rb") as stream:
        if stream.read(16) != b"STGPU001" + struct.pack("<II", 1, len(sections)):
            raise ValueError("Invalid file header")
        for tag, value in sections.items():
            if stream.read(16) != tag.encode().ljust(8, b"\0") + struct.pack("<Q", value.nbytes):
                raise ValueError("Invalid serialized section")
            view = memoryview(np.ascontiguousarray(value)).cast("B")
            for offset in range(0, len(view), 1024**2):
                if stream.read(min(1024**2, len(view)-offset)) != view[offset:offset+1024**2]:
                    raise ValueError("Serialized checkpoint differs")
        if stream.read(1):
            raise ValueError("Trailing bytes")
    temporary.replace(path)
    return size, digest(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("external-data/gpu-dense-tests"))
    parser.add_argument("--real-directory", type=Path, default=Path("external-data/fixtures"))
    parser.add_argument("--metadata", type=Path, default=ROOT / "tests/fixtures/eaq_dense.json")
    parser.add_argument("--maxrows", type=int, choices=(1, 2, 4, 8, 16, 64, 129, 256), default=16)
    parser.add_argument("--prefix", action="append", help="Exact manifest group; repeat for multiple groups")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.metadata.read_text())
    started, records = time.perf_counter(), []
    selected = args.prefix if args.prefix else PREFIXES
    if len(selected) != len(set(selected)):
        raise ValueError("Duplicate requested tensor group")
    for index, prefix in enumerate(selected):
        if not prefix or "/" in prefix or "\\" in prefix or ".." in prefix:
            raise ValueError("Invalid group name")
        t, u, v, provenance = load_group(metadata, args.real_directory, prefix)
        rows, seed = args.maxrows, SEED + 20000 + index
        rng = np.random.Generator(np.random.PCG64(seed))
        x = rng.normal(0, 0.25, (rows, len(u))).astype("<f4")
        x[0] = 0
        x[0, [0, 1, 31, 32, 63, 64, 127, len(u)-1]] = [1, -.3, .7, -1, .2, .9, -.4, .6]
        old = rng.normal(0, .3, (rows, len(v))).astype("<f4")
        stages = matmul_fp32_stages(x, t, u, v, t.shape[-1]//16, "mul1")
        checks = check_stages(x, t, u, v, stages)
        sections = dict(meta=np.frombuffer(struct.pack("<8I", len(u), len(v), t.shape[-1]//16,
                           2, 0x83DCD12D, rows, 0, seed), dtype="u1"),
                        trellis=t, suh=u, svh=v, bias=np.empty(0, dtype="<f4"), x=x, old=old)
        for tag, name in (("scaled", "scaled_fp16"), ("inh", "input_hadamard_fp32"),
                          ("a", "core_input_fp16"), ("q", "rotated_fp16"),
                          ("core", "core_fp32"), ("ref", "output_fp32")):
            sections[tag] = stages[name]
        predicted = 16 + sum(16 + value.nbytes for value in sections.values())
        if sum(r["bytes"] for r in records) + predicted > TOTAL_LIMIT:
            raise ValueError("Aggregate exceeds 1 GiB")
        path = args.out / (prefix + ".stgpu")
        size, sha = write_stream(path, sections)
        records.append(dict(file=path.name, bytes=size, sha256=sha, prefix=prefix,
                            input=len(u), output=len(v), bits=t.shape[-1]//16, codebook="mul1",
                            marker="0x83dcd12d", rows=rows, seed=seed, bias=False,
                            cpu_checks=checks, source_tensors=provenance))
        print(f"{prefix}: {size} bytes; CPU checks passed", flush=True)
        del sections, stages
    manifest = dict(format="STGPU001", version=1, model=metadata["model"], revision=metadata["revision"],
                    source_url=metadata["url"], source_manifest_sha256=digest(args.metadata),
                    generator_sha256=digest(Path(__file__)), numpy=np.__version__,
                    reference="strix_trellis.linear.matmul_fp32_stages", fixtures=records,
                    total_bytes=sum(r["bytes"] for r in records), max_file_bytes=max(r["bytes"] for r in records),
                    max_input=max(r["input"] for r in records), max_output=max(r["output"] for r in records),
                    reference_rows=[r for r in (1, 2, 4, 8, 16, 64, 65, 128, 129, 256) if r <= args.maxrows],
                    gpu_validation="Not executed", elapsed_cpu_seconds=time.perf_counter()-started)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("total_bytes", "max_file_bytes", "elapsed_cpu_seconds")}))


if __name__ == "__main__":
    main()
