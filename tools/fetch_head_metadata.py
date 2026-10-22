#!/usr/bin/env python3
"""Discover actual pinned head metadata; never download a shard body."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import urllib.request

from fetch_fixture import MODEL, REVISION, RangeReader
from fetch_dense_fixtures import HEADER_SHA

BUDGET = 20_000_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("external-data/fixtures"))
    parser.add_argument("--metadata", type=Path, default=Path("tests/fixtures/eaq_head_metadata.json"))
    parser.add_argument("--budget", type=int, default=BUDGET)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/"
    budget = min(args.budget, BUDGET)
    used, evidence = 0, []

    def small(name, filename):
        nonlocal used
        path = args.directory / filename
        cached = path.exists()
        if cached:
            raw = path.read_bytes()
        else:
            with urllib.request.urlopen(base+name, timeout=60) as response:
                size = int(response.headers.get("Content-Length", budget+1))
                if size > budget-used:
                    raise ValueError("Metadata Content-Length exceeds remaining budget")
                raw = response.read(size)
            used += len(raw)
            if used > budget or len(raw) != size:
                raise ValueError("Metadata download budget exceeded")
            path.write_bytes(raw)
        evidence.append(dict(url=base+name, file=path.name, bytes=len(raw),
                             sha256=hashlib.sha256(raw).hexdigest(), cached=cached,
                             request="small metadata GET; bounded read"))
        return raw

    # The complete index is 15.5 MB: exact prefix/suffix ranges are sufficient
    # to find head entries while preserving the total payload budget.
    index_url = base+"model.safetensors.index.json"
    with urllib.request.urlopen(urllib.request.Request(index_url, method="HEAD"), timeout=60) as response:
        index_size = int(response.headers["Content-Length"])
    reader = RangeReader(index_url, budget=budget-used)
    fragments = []
    for start, stop in ((0, min(65536, index_size)), (max(65536, index_size-65536), index_size)):
        if start >= stop:
            continue
        path = args.directory / f"eaq-head-index-range-{start}-{stop}.json.part"
        cached = path.exists()
        raw = path.read_bytes() if cached else reader.read(start, stop)
        if len(raw) != stop-start:
            raise ValueError("Invalid cached index range")
        if not cached:
            path.write_bytes(raw)
        fragments.append(raw.decode("utf-8"))
        evidence.append(dict(url=index_url, file=path.name, bytes=len(raw), cached=cached,
                             absolute_range=[start, stop], sha256=hashlib.sha256(raw).hexdigest()))
    used += reader.used
    weight_map = {}
    for fragment in fragments:
        weight_map.update(re.findall(r'"((?:head|lm_head|output)\.[^"\n]+)"\s*:\s*"([^"\n]+\.safetensors)"', fragment))
    config = json.loads(small("config.json", "eaq-head-pinned-config.json"))
    card = small("README.md", "eaq-head-pinned-model-card.md").decode("utf-8")
    names = [name for name in weight_map if name.startswith(("head.", "lm_head.", "output."))]
    if not names:
        # A bounded metadata index is still tiny compared with a model shard.
        # Some checkpoints place the head between decoder and MTP entries rather
        # than in the index prefix/suffix. Never infer its shape from config.
        full = json.loads(small("model.safetensors.index.json", "eaq-head-pinned-full-index.json"))
        weight_map = full["weight_map"]
        names = [name for name in weight_map if name.startswith(("head.", "lm_head.", "output."))]
    if not names:
        raise ValueError("No actual head found in checkpoint weight_map")
    heads, headers = {}, {}
    for shard in sorted({weight_map[name] for name in names}):
        url = base + shard
        reader = RangeReader(url, budget=budget-used)
        path = args.directory / ("eaq-head-" + shard + "-header.json")
        shared = args.directory / "eaq-v1-shard1-header.json"
        cached = True
        if shard == "model-00001-of-00039.safetensors" and shared.exists():
            raw = shared.read_bytes()
            if hashlib.sha256(raw).hexdigest() != HEADER_SHA:
                raise ValueError("Shared header hash mismatch")
            path = shared
        elif path.exists():
            raw = path.read_bytes()
        else:
            cached = False
            size = struct.unpack("<Q", reader.read(0, 8))[0]
            if size > budget-used-8:
                raise ValueError("Head header exceeds remaining budget")
            raw = reader.read(8, 8+size)
            path.write_bytes(raw)
        header = json.loads(raw)
        used += reader.used
        evidence.append(dict(url=url, file=path.name, bytes=len(raw), cached=cached,
                             sha256=hashlib.sha256(raw).hexdigest(),
                             absolute_range=[8, 8+len(raw)], ranges_this_run=reader.records.copy()))
        headers[shard] = dict(header_sha256=hashlib.sha256(raw).hexdigest(), header_bytes=len(raw),
                             safetensors_metadata=header.get("__metadata__"))
        for name in names:
            if weight_map[name] != shard:
                continue
            info = header[name]
            begin, end = info["data_offsets"]
            heads[name] = dict(info, shard=shard, absolute_range=[8+len(raw)+begin, 8+len(raw)+end])
            if name.endswith((".mul1", ".mcg")):
                if end-begin != 4:
                    raise ValueError("Unexpected marker size")
                marker_path = args.directory / ("eaq-head-" + name + ".bin")
                marker_cached = marker_path.exists()
                if marker_cached:
                    marker = marker_path.read_bytes()
                else:
                    reader.budget = reader.used + budget-used
                    marker = reader.read(8+len(raw)+begin, 8+len(raw)+end)
                    used += len(marker)
                    marker_path.write_bytes(marker)
                if len(marker) != 4:
                    raise ValueError("Invalid marker cache")
                heads[name].update(marker_hex=f"0x{struct.unpack('<I', marker)[0]:08x}",
                                   marker_sha256=hashlib.sha256(marker).hexdigest())
                evidence.append(dict(url=url, file=marker_path.name, bytes=4, cached=marker_cached,
                                     absolute_range=heads[name]["absolute_range"],
                                     sha256=hashlib.sha256(marker).hexdigest()))
    geometry = {}
    prefixes = sorted({name.rsplit(".", 1)[0] for name in names if name.endswith(".trellis")})
    for prefix in prefixes:
        t, u, v = (heads[prefix+suffix] for suffix in (".trellis", ".suh", ".svh"))
        inputs, outputs = u["shape"][0], v["shape"][0]
        if t["shape"][:2] != [inputs//16, outputs//16]:
            raise ValueError("Head geometry mismatch")
        marker = heads.get(prefix+".mul1", {}).get("marker_hex")
        geometry[prefix] = dict(input=inputs, output=outputs, bits=t["shape"][2]/16,
                                codebook="mul1" if marker == "0x83dcd12d" else "unknown",
                                marker_hex=marker, scales_dtype=[u["dtype"], v["dtype"]],
                                scale_values="unknown: no scale payload downloaded",
                                head_weight_values="not downloaded")
    license_lines = [line for line in card.splitlines() if line.lower().startswith("license:")]
    manifest = dict(model=MODEL, revision=REVISION, download_budget_bytes=budget,
                    downloaded_bytes_this_run=used, evidence=evidence, head_weight_map={n: weight_map[n] for n in names},
                    headers=headers, tensors=heads, geometry=geometry,
                    config=config, model_card_license_metadata=license_lines,
                    gpu_validation="Not executed; metadata only")
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(downloaded_bytes=used, head_names=names, geometry=geometry,
                         license=license_lines), indent=2), flush=True)


if __name__ == "__main__":
    main()
