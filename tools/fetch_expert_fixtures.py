"""Fetch one real K3 expert with pinned headers and bounded HTTP ranges."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from fetch_fixture import MODEL, REVISION, SHARD, RangeReader
from fetch_dense_fixtures import HEADER_SHA


def fetch(header_path, directory):
    raw = header_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HEADER_SHA:
        raise ValueError("Pinned shard-one header mismatch")
    header = json.loads(raw)
    selected = {}
    for projection, dims in (("w1", (5120, 2304)), ("w2", (2304, 5120)),
                             ("w3", (5120, 2304))):
        prefix = "layers.0.ffn.experts.10." + projection
        i, n = dims
        expected = {"trellis": ("I16", [i//16, n//16, 48]),
                    "suh": ("F16", [i]), "svh": ("F16", [n]),
                    "mul1": ("I32", [])}
        for suffix, (dtype, shape) in expected.items():
            name = prefix + "." + suffix
            entry = header[name]
            begin, end = entry["data_offsets"]
            size = math.prod(shape) * (4 if dtype == "I32" else 2)
            if entry["dtype"] != dtype or entry["shape"] != shape or end-begin != size:
                raise ValueError("Unexpected expert tensor contract: " + name)
            selected[name] = entry
    budget = 15_000_000
    if sum(v["data_offsets"][1]-v["data_offsets"][0] for v in selected.values()) > budget:
        raise ValueError("Expert payload exceeds budget")
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{SHARD}"
    reader = RangeReader(url, budget=budget)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "manifest.json"
    manifest = json.loads(output.read_text()) if output.exists() else dict(
        model=MODEL, revision=REVISION, shard=SHARD, header_sha256=HEADER_SHA,
        fixture_directory=str(directory), url=url, tensors={},
        evidence="Actual expert payload; no model-quality claim")
    if any(manifest[k] != v for k, v in dict(model=MODEL, revision=REVISION,
            shard=SHARD, header_sha256=HEADER_SHA).items()):
        raise ValueError("Existing manifest has different provenance")
    for name, entry in selected.items():
        begin, end = entry["data_offsets"]
        absolute = [8+len(raw)+begin, 8+len(raw)+end]
        path = directory / (name + ".bin")
        known = manifest["tensors"].get(name)
        if path.exists():
            data = path.read_bytes()
            if not known or known["absolute_range"] != absolute or any(
                    known[k] != entry[k] for k in ("dtype", "shape", "data_offsets")):
                raise ValueError("Existing data lacks matching provenance: " + name)
            if len(data) != end-begin or hashlib.sha256(data).hexdigest() != known["sha256"]:
                raise ValueError("Existing tensor failed integrity: " + name)
        else:
            data = reader.read(*absolute)
            if known and hashlib.sha256(data).hexdigest() != known["sha256"]:
                raise ValueError("Resumed download differs from recorded tensor: " + name)
            if name.endswith(".mul1") and data != struct.pack("<I", 0x83DCD12D):
                raise ValueError("Unknown codebook marker")
            with path.open("xb") as handle:
                handle.write(data)
        manifest["tensors"][name] = dict(entry, file=path.name, bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(), absolute_range=absolute)
        manifest["downloaded_bytes_total"] = sum(v["bytes"] for v in manifest["tensors"].values())
        # Save after each tensor, allowing verified resumes without duplicate downloads.
        pending = directory / "manifest.pending.json"
        pending.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8", newline="\n")
        pending.replace(output)
        print("VERIFIED", name, len(data), flush=True)
    print("PASS real K3 expert; downloaded bytes this invocation:", reader.used, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    fetch(args.header, args.directory)
