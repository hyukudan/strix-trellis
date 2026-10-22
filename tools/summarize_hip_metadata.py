#!/usr/bin/env python3
"""Summarize llvm-readelf AMDGPU notes without loading or executing a GPU binary."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def parse_notes(path):
    kernels = []
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "  - .args:":
            if current is not None:
                kernels.append(current)
            current = {}
        elif current is not None:
            match = re.fullmatch(r"    \.([a-z_]+):\s*(.*)", line)
            if match and match[2]:
                key, value = match.groups()
                if re.fullmatch(r"[0-9]+", value):
                    value = int(value)
                elif value in ("true", "false"):
                    value = value == "true"
                current[key] = value
    if current is not None:
        kernels.append(current)
    if not kernels or any("name" not in kernel for kernel in kernels):
        raise ValueError(f"Missing kernel records or names: {path}")
    for kernel in kernels:
        for key in ("vgpr_count", "sgpr_count", "vgpr_spill_count",
                    "sgpr_spill_count", "private_segment_fixed_size",
                    "group_segment_fixed_size", "wavefront_size"):
            if not isinstance(kernel.get(key), int):
                raise ValueError(f"Missing integer {key}: {path}: {kernel['name']}")
    return kernels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notes", nargs="+", type=Path)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.archive_sha256):
        parser.error("Expected a lowercase SHA-256 digest")
    modules = []
    for path in sorted(args.notes):
        kernels = parse_notes(path)
        modules.append({
            "notes_file": path.name,
            "notes_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "kernel_count": len(kernels),
            "kernels": kernels,
        })
    all_kernels = [kernel for module in modules for kernel in module["kernels"]]
    result = {
        "archive_sha256": args.archive_sha256,
        "method": "CPU-only parsing of llvm-readelf --notes; no occupancy measurement",
        "kernel_count": len(all_kernels),
        "nonzero_vgpr_spill_count": sum(k["vgpr_spill_count"] != 0 for k in all_kernels),
        "nonzero_sgpr_spill_count": sum(k["sgpr_spill_count"] != 0 for k in all_kernels),
        "nonzero_private_segment_count": sum(k["private_segment_fixed_size"] != 0 for k in all_kernels),
        "wavefront_sizes": sorted({k["wavefront_size"] for k in all_kernels}),
        "modules": modules,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: value for key, value in result.items() if key != "modules"}))


if __name__ == "__main__":
    main()
