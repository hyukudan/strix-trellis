"""Load and verify the external, ranged EAQ fixture described by a manifest."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .codec import MUL1_MULTIPLIER


def load_fixture(manifest, directory=None):
    """Return (trellis, suh, svh, k); reject corrupt bytes or unknown mul1 markers."""
    metadata = json.loads(Path(manifest).read_text(encoding="utf-8"))
    directory = Path(directory or metadata["fixture_directory"])
    tensors = {}
    dtypes = {"I16": "<i2", "I32": "<i4", "F16": "<f2"}
    for name, info in metadata["tensors"].items():
        path = directory / info["file"]
        raw = path.read_bytes()
        if len(raw) != info["bytes"] or hashlib.sha256(raw).hexdigest() != info["sha256"]:
            raise ValueError(f"Fixture integrity check failed: {name}")
        tensors[name.rsplit(".", 1)[1]] = np.frombuffer(raw, dtype=dtypes[info["dtype"]]).reshape(info["shape"])
    if (int(tensors["mul1"].item()) & 0xFFFFFFFF) != MUL1_MULTIPLIER:
        raise ValueError("Unsupported mul1 multiplier")
    trellis = tensors["trellis"]
    return trellis, tensors["suh"], tensors["svh"], trellis.shape[-1] / 16
