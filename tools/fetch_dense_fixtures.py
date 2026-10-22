"""Fetch pinned dense EAQ groups with bounded, exact HTTP range requests."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
from fetch_fixture import MODEL, REVISION, SHARD, RangeReader

PREFIXES = ["layers.0.attn.wq_a", "layers.0.attn.wq_b", "layers.0.attn.wo_b",
            "layers.0.attn.wo_a.slice.0", "layers.0.ffn.shared_experts.w1",
            "layers.0.ffn.shared_experts.w2", "layers.0.ffn.shared_experts.w3"]
HEADER_SHA = "f942eb8ca490f99c673b222a0df95398995da6c2a9a82cc52e670ed44c0aaa61"


def fetch(directory, output):
    directory.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{SHARD}"
    reader = RangeReader(url, budget=90_000_000)
    cache = directory / "eaq-v1-shard1-header.json"
    if cache.exists():
        raw = cache.read_bytes()
    else:
        n = struct.unpack("<Q", reader.read(0, 8))[0]
        if n > 8_000_000:
            raise ValueError("Header exceeds bound")
        raw = reader.read(8, 8+n)
    if hashlib.sha256(raw).hexdigest() != HEADER_SHA:
        raise ValueError("Pinned header hash mismatch")
    if not cache.exists():
        cache.write_bytes(raw)
    header = json.loads(raw)
    manifest = json.loads(output.read_text()) if output.exists() else dict(
        model=MODEL, revision=REVISION, shard=SHARD, url=url,
        header_sha256=HEADER_SHA, fixture_directory=str(directory), tensors={})
    for prefix in PREFIXES:
        for suffix in (".trellis", ".suh", ".svh", ".mul1", ".bias"):
            name = prefix+suffix
            if suffix == ".bias" and name not in header:
                continue
            info = header[name]
            begin, end = info["data_offsets"]
            path = directory / (name+".bin")
            known = manifest["tensors"].get(name)
            if path.exists():
                if not known:
                    raise FileExistsError(f"Existing fixture has no manifest: {path}")
                data = path.read_bytes()
                if len(data) != end-begin or hashlib.sha256(data).hexdigest() != known["sha256"]:
                    raise ValueError(f"Existing fixture failed integrity: {path}")
            else:
                data = reader.read(8+len(raw)+begin, 8+len(raw)+end)
                if suffix == ".mul1" and data != struct.pack("<I", 0x83DCD12D):
                    raise ValueError("Unknown mul1 marker")
                path.write_bytes(data)
            manifest["tensors"][name] = dict(info, file=path.name, bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                absolute_range=[8+len(raw)+begin, 8+len(raw)+end])
            manifest.update(downloaded_bytes_this_run=reader.used, ranges_this_run=reader.records)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        print(prefix, "verified", flush=True)
    print("Downloaded bytes:", reader.used, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("external-data/fixtures"))
    parser.add_argument("--metadata", type=Path, default=Path("tests/fixtures/eaq_dense.json"))
    args = parser.parse_args()
    fetch(args.directory, args.metadata)
