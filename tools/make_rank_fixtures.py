#!/usr/bin/env python3
"""Bounded synthetic mix-B/expert-TP fixtures and independent CPU MoE oracles.

No model shards, GPU initialization or downloads are required.  Packed matrices
are decoded in 128-column blocks.  The compact STRANK01 container intentionally
does not store expanded weights.  All references use the documented input
rounding stages, a binary64 matrix product, and the FP32 output stages; they do
not emulate a GPU reduction tree.
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from strix_trellis.codec import decode_states, decode_tile
from strix_trellis.linear import input_stages_fp32, output_stages_fp32

MAGIC = b"STRANK01"
SEED = 20261021
DENSE_SHAPES = (("q_b", 1280, 16384), ("o_b", 4096, 5120),
                ("shared_gate", 5120, 1152), ("shared_up", 5120, 1152),
                ("shared_down", 1152, 5120), ("q_a", 5120, 1280),
                ("kv", 5120, 512))
EXPERT_SHAPES = (("gate", 5120, 1152), ("up", 5120, 1152),
                 ("down", 1152, 5120))


def decode_columns(trellis, bits):
    """Vectorized ring extraction checked against the separate integer codec."""
    t = np.asarray(trellis, dtype="<u2")
    words = np.ascontiguousarray(t).view("<u4")
    width = 256 * bits
    result = np.empty((t.shape[0], 16, t.shape[1], 16), dtype="<f2")
    for row in range(16):
        for col in range(16):
            position = ((col & 7) << 5) | ((row & 6) << 2)
            position |= (row & 1) | ((row & 8) >> 2) | ((col & 8) >> 1)
            start = (bits * (position + 1) - 16) % width
            index, shift = divmod(start, 32)
            pair = ((words[..., index].astype(np.uint64) << 32)
                    | words[..., (index + 1) % (width // 32)])
            state = ((pair >> (48 - shift)) & 65535).astype(np.uint16)
            result[:, row, :, col] = decode_states(state)
    return result.reshape(t.shape[0] * 16, t.shape[1] * 16)


def write_sections(path, sections):
    """Write without concatenating or duplicating the full compressed payload."""
    with path.open("wb") as handle:
        handle.write(MAGIC + struct.pack("<II", 1, len(sections)))
        for name, values in sections.items():
            if not name.isascii() or not 1 <= len(name) <= 8:
                raise ValueError("Section names must be 1..8 ASCII characters")
            payload = values if isinstance(values, bytes) else memoryview(np.ascontiguousarray(values)).cast("B")
            handle.write(name.encode().ljust(8, b"\0") + struct.pack("<Q", len(payload)))
            handle.write(payload)
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": digest}


def input_values(width, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    x = rng.normal(0, 0.025, (8, width)).astype("<f4")
    x[0, :8] = [0., -0., 2**-24, -2**-24, 2**-14, -2**-14, .125, -.0625]
    return x


def gamma(operations):
    value = operations * 2**-24
    return value / (1 - value)


def reference(trellis, bits, suh, svh, bias, x):
    """FP64 core oracle with staged input/output and an explicit roundoff bound."""
    inputs, outputs = len(suh), len(svh)
    activation = input_stages_fp32(x, suh)["core_input_fp16"].astype(np.float64)
    core = np.empty((len(x), outputs), dtype="<f4")
    bounds = np.empty((len(x), outputs), dtype="<f8")
    for col in range(0, outputs, 128):
        q = decode_columns(trellis[:, col // 16:col // 16 + 8], bits).astype(np.float64)
        exact = activation @ q
        absolute = np.abs(activation) @ np.abs(q)
        core[:, col:col + 128] = exact
        # Two valid FP32 dot orders, followed by H128 and output scaling.  The
        # extra underflow allowance is tiny for these bounded finite fixtures.
        core_error = 2 * gamma(2 * inputs) * absolute + 1e-12
        block = (core_error.sum(axis=1) + 2 * gamma(9) * np.abs(exact).sum(axis=1)) / np.sqrt(128)
        bounds[:, col:col + 128] = block[:, None] * np.abs(svh[col:col + 128].astype(np.float64))
    stages = output_stages_fp32(core, svh, bias)
    bounds += 4 * 2**-24 * (np.abs(stages["output_fp32"].astype(np.float64)) + np.abs(bias)[None, :]) + 1e-7
    return stages["output_fp32"], stages["scaled_output_fp32"], bounds


def make_fixture(path, inputs, outputs, bits, seed, *, x=None):
    rng = np.random.Generator(np.random.PCG64(seed))
    trellis = rng.integers(0, 65536, (inputs // 16, outputs // 16, bits * 16), dtype=np.uint16)
    choices = np.array([-.5, -.25, -.125, -0., 0., .0625, .125, .25], dtype="<f2")
    suh = rng.choice(choices, inputs)
    svh = rng.choice(choices, outputs)
    bias = rng.normal(0, .01, outputs).astype("<f4")
    if x is None:
        x = input_values(inputs, seed + 7000)
    old = rng.normal(0, .03, (8, outputs)).astype("<f4")
    np.testing.assert_array_equal(decode_columns(trellis[:2, :2], bits),
                                  decode_tile(trellis[:2, :2], bits).transpose(0, 2, 1, 3).reshape(32, 32))
    ref, base, bound = reference(trellis, bits, suh, svh, bias, x)
    arrays = dict(trellis=trellis, suh=suh, svh=svh, bias=bias, x=x, old=old,
                  ref=ref, base=base, bound=bound)
    sections = {"meta": struct.pack("<8I", inputs, outputs, bits, 2, 0x83DCD12D, 8, 1, seed), **arrays}
    record = write_sections(path, sections)
    record.update(input=inputs, output=outputs, bits=bits, rows=8, seed=seed)
    return record, arrays


def expert_id(index):
    return 17 + 3 * index


def expert_bits(index, projection):
    if projection == "gate":
        return 2 + (index % 2)
    if projection == "up":
        return 3 if index % 3 == 0 else 2
    return 2 if index % 3 == 1 else 3


def expert_bias(base, index, projection):
    factor = {"gate": 1., "up": -.5, "down": .25}[projection]
    return (base.astype(np.float32) + np.float32((expert_id(index) + 1) * factor / 512)).astype("<f4")


def route_case(name):
    if name == "shared":
        index = np.array([[(row + slot) % 6 for slot in range(6)] for row in range(8)], dtype=np.uint32)
    elif name == "overlap":
        index = np.array([[3 * row + slot for slot in range(6)] for row in range(8)], dtype=np.uint32)
    elif name == "disjoint":
        index = np.arange(48, dtype=np.uint32).reshape(8, 6)
    else:
        raise ValueError("Unknown route case")
    scores = np.array([[1 + (row % 3) + slot for slot in range(6)] for row in range(8)], dtype="<f4")
    scores[1::3, 5] = 0
    scores /= scores.sum(axis=1, keepdims=True)
    return index, scores


def make_moe_oracle(path, name, fixtures):
    indices, scores = route_case(name)
    hidden = np.empty((8, 6, 1152), dtype="<f4")
    individual = np.empty((8, 6, 5120), dtype="<f4")
    # Cache only the two 11.25 MiB half down matrices, independently decoded.
    down = {bits: np.concatenate([decode_columns(fixtures[("down", bits)]["trellis"][:, c:c + 8], bits)
                                 for c in range(0, 5120 // 16, 8)], axis=1)
            for bits in (2, 3)}
    for index in sorted(int(i) for i in np.unique(indices)):
        positions = np.argwhere(indices == index)
        rows = positions[:, 0]
        projected = []
        for projection in ("gate", "up"):
            f = fixtures[(projection, expert_bits(index, projection))]
            projected.append((f["base"][rows] + expert_bias(f["bias"], index, projection)[None, :]).astype(np.float32))
        gate, up = projected
        # Separate FP32 operations; HIP expf is allowed a measured tolerance.
        exponential = np.exp(-np.abs(gate)).astype(np.float32)
        sigmoid = np.where(gate < 0, exponential / (np.float32(1) + exponential),
                           np.float32(1) / (np.float32(1) + exponential)).astype(np.float32)
        values = ((gate * sigmoid).astype(np.float32) * up).astype("<f4")
        bits = expert_bits(index, "down")
        f = fixtures[("down", bits)]
        activation = input_stages_fp32(values, f["suh"])["core_input_fp16"].astype(np.float64)
        core = (activation @ down[bits].astype(np.float64)).astype(np.float32)
        result = output_stages_fp32(core, f["svh"], expert_bias(f["bias"], index, "down"))["output_fp32"]
        for offset, (row, slot) in enumerate(positions):
            hidden[row, slot] = values[offset]
            individual[row, slot] = result[offset]
    result = np.zeros((8, 5120), dtype="<f4")
    for slot in range(6):
        # A zero score contributes +0 and skips the projection in the kernel.
        contribution = individual[:, slot] * scores[:, slot, None]
        contribution[scores[:, slot] == 0] = 0
        result += contribution
    rng = np.random.Generator(np.random.PCG64(SEED + 88000))
    old = rng.normal(0, .03, result.shape).astype("<f4")
    sections = {"meta": struct.pack("<8I", 5120, 5120, 0, 2, 0x83DCD12D, 8, 0, SEED),
                "routes": (17 + 3 * indices).astype("<u4"), "scores": scores,
                "hidden": hidden, "each": individual, "ref": result, "old": old}
    record = write_sections(path, sections)
    record.update(kind="moe", case=name, union=int(np.unique(indices).size), top_k=6,
                  unique_experts_per_prefix=[int(np.unique(indices[:rows]).size) for rows in range(1, 9)])
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--suite", choices=("all", "dense", "expert"), default="all")
    args = parser.parse_args()
    if hasattr(os, "nice"):
        os.nice(10)
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    if args.suite in ("all", "dense"):
        for site, inputs, outputs in DENSE_SHAPES:
            for bits in (4, 5):
                seed = SEED + 1000 * bits + next(i for i, s in enumerate(DENSE_SHAPES) if s[0] == site)
                record, _ = make_fixture(args.out / f"dense-{site}-k{bits}.strank", inputs, outputs, bits, seed)
                record.update(kind="dense", site=site)
                records.append(record)
                print(json.dumps(record), flush=True)
    if args.suite in ("all", "expert"):
        fixtures = {}
        common_x = input_values(5120, SEED + 9000)
        for projection, inputs, outputs in EXPERT_SHAPES:
            for bits in (2, 3):
                seed = SEED + 1000 * bits + (0, 1, 2)[("gate", "up", "down").index(projection)] + 100
                record, arrays = make_fixture(args.out / f"expert-{projection}-k{bits}.strank", inputs, outputs,
                                              bits, seed, x=common_x if projection != "down" else None)
                fixtures[(projection, bits)] = arrays
                record.update(kind="expert", site=projection)
                records.append(record)
                print(json.dumps(record), flush=True)
        for name in ("shared", "overlap", "disjoint"):
            record = make_moe_oracle(args.out / f"moe-{name}.strank", name, fixtures)
            records.append(record)
            print(json.dumps(record), flush=True)
    manifest = dict(format="STRANK01", version=1, synthetic=True, numpy=np.__version__,
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    reference="RN32 -> RN16 -> H32 -> RN16; FP64 core -> RN32 -> H32 -> FP32 epilogue",
                    limitation="CPU oracle is not a GPU reduction emulator; no HIP or speed evidence",
                    expert_id="17 + 3 * index", bias_offsets="(id+1)/512 times gate=1,up=-.5,down=.25",
                    files=records, total_bytes=sum(r["bytes"] for r in records))
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
