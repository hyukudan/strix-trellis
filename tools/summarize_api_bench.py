#!/usr/bin/env python3
"""Export complete-operator RESULT records to a reproducible CSV (CPU only)."""
import argparse
import csv
import math
from pathlib import Path
import sys


def records(path):
    fixture = None
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[-1] != "PASS complete_operator":
        raise ValueError("Benchmark log is incomplete or failed")
    for number, line in enumerate(lines, 1):
        if not line.startswith(("FIXTURE ", "RESULT ")):
            continue
        fields = dict(part.split("=", 1) for part in line.split()[1:])
        if line.startswith("FIXTURE "):
            fixture = fields
            continue
        if fixture is None or fields.get("mode") != "cold":
            raise ValueError(f"Missing fixture or unexpected benchmark mode at line {number}")
        row = {"fixture": Path(fixture["path"]).name, "input_size": fixture["I"],
               "output_size": fixture["N"], "bits": fixture["bits"],
               "codebook": fixture["codebook"]}
        for key in ("api", "rows", "clones", "cloned_bytes", "workspace_bytes",
                    "event_min_us", "event_median_us", "event_p90_us",
                    "wall_min_us", "wall_median_us", "wall_p90_us",
                    "compressed_GBs", "useful_TOPS"):
            row[key] = fields[key]
        for key in ("event_median_us", "wall_median_us", "compressed_GBs", "useful_TOPS"):
            if not math.isfinite(float(row[key])) or float(row[key]) <= 0:
                raise ValueError(f"Invalid measurement at line {number}: {key}")
        yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        rows = list(records(args.log))
    except (OSError, ValueError, KeyError) as e:
        parser.error(str(e))
    if not rows:
        parser.error("No complete-operator results found")
    output = args.out.open("w", encoding="utf-8", newline="") if args.out else sys.stdout
    try:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.out:
            output.close()


if __name__ == "__main__":
    main()
