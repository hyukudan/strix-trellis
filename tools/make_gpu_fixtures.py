#!/usr/bin/env python3
"""Generate bounded external GPU fixtures using our independent staged CPU oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from strix_trellis.fixture import load_fixture
from strix_trellis.linear import matmul_fp32_stages

MAGIC = b"STGPU001"
MARKERS = (0, 0xCBAC1FED, 0x83DCD12D)
SEED = 20261020


def write_fixture(path, trellis, suh, svh, bias, codebook, seed, rows, *, x=None):
    bits = trellis.shape[2] // 16
    inputs, outputs = len(suh), len(svh)
    rng = np.random.Generator(np.random.PCG64(seed))
    if x is None:
        x = rng.normal(0, 0.25, (rows, inputs)).astype("<f4")
        # Distinct channel and row probes make transpose mistakes visible.
        x[0] = 0
        x[0, [0, 1, 31, 32, 63, 64, 127, inputs - 1]] = [1, -0.3, 0.7, -1, 0.2, 0.9, -0.4, 0.6]
    else:
        x = np.asarray(x, dtype="<f4")
        if x.shape != (rows, inputs) or not np.isfinite(x).all():
            raise ValueError("Custom fixture input must be finite F32 [rows, inputs]")
    old = rng.normal(0, 0.3, (rows, outputs)).astype("<f4")
    stages = matmul_fp32_stages(x, trellis, suh, svh, bits, codebook, bias)
    cb = ("3inst", "mcg", "mul1").index(codebook)
    sections = {"meta": struct.pack("<8I", inputs, outputs, bits, cb,
                                    MARKERS[cb], rows, int(bias is not None), seed),
                "trellis": np.asarray(trellis, dtype="<u2").tobytes(),
                "suh": np.asarray(suh, dtype="<f2").tobytes(),
                "svh": np.asarray(svh, dtype="<f2").tobytes(),
                "bias": b"" if bias is None else np.asarray(bias, dtype="<f4").tobytes(),
                "x": x.tobytes(), "old": old.tobytes()}
    for tag, name, dtype in [("scaled", "scaled_fp16", "<f2"),
                             ("inh", "input_hadamard_fp32", "<f4"),
                             ("a", "core_input_fp16", "<f2"),
                             ("q", "rotated_fp16", "<f2"),
                             ("core", "core_fp32", "<f4"),
                             ("ref", "output_fp32", "<f4")]:
        sections[tag] = np.asarray(stages[name], dtype=dtype).tobytes()
    data = bytearray(MAGIC + struct.pack("<II", 1, len(sections)))
    for tag, payload in sections.items():
        data += tag.encode("ascii").ljust(8, b"\0") + struct.pack("<Q", len(payload)) + payload
    path.write_bytes(data)
    return dict(file=path.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                input=inputs, output=outputs, bits=bits, codebook=codebook,
                rows=rows, seed=seed, bias=bias is not None)


def scales(rng, size):
    values = rng.choice(np.array([-2, -0.5, -0.125, 0, 0.0625, 0.75, 1.5]), size).astype("<f2")
    values[::17] = 0
    values[1] = np.float16(-0.0)
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("external-data/gpu-tests"))
    parser.add_argument("--real-directory", type=Path, default=Path("external-data/fixtures"))
    parser.add_argument("--skip-real", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = []
    for cb, name in enumerate(("3inst", "mcg", "mul1")):
        for bits in range(1, 9):
            for inputs, outputs in ((128, 256), (256, 128)):
                seed = SEED + cb * 1000 + bits * 10 + int(inputs == 256)
                rng = np.random.Generator(np.random.PCG64(seed))
                trellis = rng.integers(0, 65536, (inputs // 16, outputs // 16, 16 * bits), dtype=np.uint16)
                suh, svh = scales(rng, inputs), scales(rng, outputs)
                bias = rng.normal(0, 0.1, outputs).astype("<f4") if bits % 2 else None
                path = args.out / f"synthetic-{name}-k{bits}-i{inputs}-n{outputs}.stgpu"
                records.append(write_fixture(path, trellis, suh, svh, bias, name, seed, 32))
    if not args.skip_real:
        trellis, suh, svh, rate = load_fixture(ROOT / "tests/fixtures/eaq_v1.json", args.real_directory)
        if rate != 6 or trellis.dtype.kind != "i" or trellis.dtype.itemsize != 2:
            raise ValueError("Expected the shared I16 EAQ wkv K6 fixture")
        records.append(write_fixture(args.out / "eaq-wkv-k6-mul1.stgpu", trellis, suh, svh,
                                     None, "mul1", SEED + 9999, 16))
    total = sum(record["bytes"] for record in records)
    if total >= 50_000_000:
        raise ValueError(f"Fixture budget exceeded: {total}")
    manifest = dict(format="STGPU001", version=1, reference="strix_trellis.linear.matmul_fp32_stages",
                    numpy=np.__version__, seed=SEED, total_bytes=total, fixtures=records)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(records)} fixtures, {total} bytes, {time.perf_counter()-started:.3f} seconds")


if __name__ == "__main__":
    main()
