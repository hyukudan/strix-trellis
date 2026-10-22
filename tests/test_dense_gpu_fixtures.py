"""CPU integrity gates for dense fixtures; no GPU execution."""
import hashlib
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_dense_gpu_fixtures as dense


class DenseFixtureTests(unittest.TestCase):
    def test_source_corruption_marker_and_nonfinite_scales(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            metadata = {"tensors": {}}
            values = dict(trellis=np.zeros((8, 8, 80), dtype="<i2"),
                          suh=np.ones(128, dtype="<f2"), svh=np.ones(128, dtype="<f2"),
                          mul1=np.array(-2082672339, dtype="<i4"))

            def save():
                for key, value in values.items():
                    raw = value.tobytes()
                    filename = key + ".bin"
                    (directory / filename).write_bytes(raw)
                    metadata["tensors"]["test."+key] = dict(file=filename, bytes=len(raw),
                        shape=list(value.shape), dtype={"trellis": "I16", "suh": "F16", "svh": "F16", "mul1": "I32"}[key],
                        sha256=hashlib.sha256(raw).hexdigest())
            # Use the marker's unsigned representation without NumPy overflow casts.
            values["mul1"] = np.frombuffer(struct.pack("<I", 0x83DCD12D), dtype="<i4").reshape(())
            save()
            dense.load_group(metadata, directory, "test")
            with (directory / "trellis.bin").open("r+b") as stream:
                stream.write(b"\x01")
            with self.assertRaisesRegex(ValueError, "integrity"):
                dense.load_group(metadata, directory, "test")
            values["mul1"] = np.array(0, dtype="<i4")
            save()
            with self.assertRaisesRegex(ValueError, "marker"):
                dense.load_group(metadata, directory, "test")
            values["mul1"] = np.frombuffer(struct.pack("<I", 0x83DCD12D), dtype="<i4").reshape(())
            values["suh"][7] = np.inf
            save()
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                dense.load_group(metadata, directory, "test")

    def test_independent_core_gate_detects_changed_checkpoint(self):
        rng = np.random.default_rng(428)
        t = rng.integers(-32768, 32768, (8, 8, 80), dtype=np.int16)
        u = rng.uniform(-1, 1, 128).astype(np.float16)
        v = rng.uniform(-1, 1, 128).astype(np.float16)
        x = rng.normal(0, .25, (2, 128)).astype(np.float32)
        stages = dense.matmul_fp32_stages(x, t, u, v, 5, "mul1")
        dense.check_stages(x, t, u, v, stages)
        stages["core_fp32"][0, 0] += np.float32(1)
        stages.update(dense.output_stages_fp32(stages["core_fp32"], v))
        with self.assertRaisesRegex(ValueError, "FP64 bound"):
            dense.check_stages(x, t, u, v, stages)


if __name__ == "__main__":
    unittest.main()
