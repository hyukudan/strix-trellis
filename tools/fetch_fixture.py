"""Fetch one pinned EXL3 tensor group; never accept a full shard response."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import urllib.request

MODEL = "Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw"
REVISION = "45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b"
SHARD = "model-00001-of-00039.safetensors"
BUDGET = 90_000_000


class RangeReader:
    def __init__(self, url, budget=BUDGET):
        self.url, self.budget, self.used, self.records = url, budget, 0, []

    def read(self, start, stop):
        size = stop - start
        if start < 0 or size <= 0 or self.used + size > self.budget:
            raise ValueError("Invalid range or fixture budget exceeded")
        # Separate cache keys prevent a cached first range replacing later ranges.
        request = urllib.request.Request(
            self.url + f"?range_start={start}&range_stop={stop}",
            headers={"Range": f"bytes={start}-{stop - 1}", "Accept-Encoding": "identity"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            cr = response.headers.get("Content-Range", "")
            if response.status != 206 or not re.fullmatch(
                rf"bytes {start}-{stop - 1}/[0-9]+", cr
            ):
                raise RuntimeError(f"Server did not honor exact range: {response.status} {cr}")
            data = response.read(size + 1)
        self.used += len(data)
        if len(data) != size:
            raise RuntimeError("Unexpected range length")
        self.records.append({"start": start, "stop_exclusive": stop,
                             "bytes": size, "sha256": hashlib.sha256(data).hexdigest()})
        return data


def fetch(destination, metadata):
    url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{SHARD}"
    reader = RangeReader(url)
    header_size = struct.unpack("<Q", reader.read(0, 8))[0]
    if header_size > 8_000_000:
        raise ValueError("Unexpectedly large safetensors header")
    header_data = reader.read(8, 8 + header_size)
    header = json.loads(header_data)
    candidates = []
    for name, info in header.items():
        if name.endswith(".trellis"):
            prefix = name[:-8]
            keys = [prefix + suffix for suffix in (".trellis", ".suh", ".svh", ".mul1")]
            if all(key in header for key in keys):
                total = sum(header[key]["data_offsets"][1] - header[key]["data_offsets"][0]
                            for key in keys)
                candidates.append((total, prefix, keys))
    total, prefix, keys = min(candidates)
    if reader.used + total >= BUDGET:
        raise ValueError("No small tensor group fits the budget")
    destination.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for key in keys:
        info = header[key]
        begin, end = info["data_offsets"]
        data = reader.read(8 + header_size + begin, 8 + header_size + end)
        filename = key + ".bin"
        path = destination / filename
        if path.exists() and path.read_bytes() != data:
            raise FileExistsError(f"Refusing to replace different fixture: {path}")
        path.write_bytes(data)
        tensors[key] = dict(info, file=filename, bytes=len(data),
                            sha256=hashlib.sha256(data).hexdigest(),
                            absolute_range=[8 + header_size + begin, 8 + header_size + end])
    manifest = dict(model=MODEL, revision=REVISION, shard=SHARD, url=url,
                    prefix=prefix, fixture_directory=str(destination), tensors=tensors,
                    header_bytes=header_size, header_sha256=hashlib.sha256(header_data).hexdigest(),
                    downloaded_bytes=reader.used, ranges=reader.records,
                    evidence="Actual tensor bytes; no upstream numerical parity established")
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"prefix": prefix, "downloaded_bytes": reader.used,
                      "metadata": str(metadata)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("external-data/fixtures"))
    parser.add_argument("--metadata", type=Path, default=Path("tests/fixtures/eaq_v1.json"))
    args = parser.parse_args()
    fetch(args.destination, args.metadata)
