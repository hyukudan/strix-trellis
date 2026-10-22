#!/usr/bin/env python3
"""Generate tiny, bounded fixtures that exercise FP16 subnormals and range."""
import argparse
import json
from pathlib import Path
import numpy as np
from make_gpu_fixtures import MARKERS, SEED, write_fixture


def probes():
    x = np.zeros((8, 128), dtype="<f4")
    x[0] = np.float32(4e-6)  # H128 produces a nonzero FP16 subnormal at DC.
    x[1] = np.where(np.arange(128) & 1, -4e-6, 4e-6)
    x[2] = np.linspace(-3e-5, 3e-5, 128, dtype=np.float32)
    x[3, ::2] = np.float32(2**-24)  # Smallest positive FP16 input.
    x[4] = 4000  # Finite RN16 after H128, close to the high FP16 range.
    x[5] = np.where(np.arange(128) & 1, -4000, 4000)
    x[6] = np.linspace(-6000, 6000, 128, dtype=np.float32)
    x[7, ::2] = np.float32(-0.0)
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    x = probes()
    for cb, codebook in enumerate(("3inst", "mcg", "mul1")):
        for bits in range(1, 9):
            seed = SEED + 20000 + cb*100 + bits
            rng = np.random.Generator(np.random.PCG64(seed))
            trellis = rng.integers(0, 65536, (8, 16, 16*bits), dtype=np.uint16)
            suh = np.ones(128, dtype="<f2")
            svh = np.where(np.arange(256)&1, -0.5, 0.5).astype("<f2")
            path = args.out / f"numerical-{codebook}-k{bits}.stgpu"
            records.append(write_fixture(path, trellis, suh, svh, None,
                                         codebook, seed, len(x), x=x))
    manifest = dict(format="STGPU001", version=1, seed=SEED,
                    reference="strix_trellis.linear.matmul_fp32_stages",
                    probes=["subnormal_dc", "subnormal_ac", "subnormal_ramp",
                            "minimum_half", "large_dc", "large_ac", "large_ramp", "signed_zero"],
                    numpy=np.__version__, total_bytes=sum(v["bytes"] for v in records),
                    fixtures=records)
    if manifest["total_bytes"] > 5000000:
        raise ValueError("Numerical fixture budget exceeded")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(f"Generated {len(records)} numerical fixtures, {manifest['total_bytes']} bytes")


if __name__ == "__main__":
    main()
