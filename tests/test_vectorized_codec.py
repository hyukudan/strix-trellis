"""Vectorized trellis decode and BF16 mapping against the scalar reference."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from strix_trellis import (SafetensorsFile, dequantize_rotated, dequantize_rotated_vectorized,
                           unpack_states, unpack_states_vectorized)

RATES = (1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8)


class VectorizedCodecTest(unittest.TestCase):
    def test_states_match_scalar_for_every_rate(self):
        rng = np.random.default_rng(7)
        for k in RATES:
            words = rng.integers(-32768, 32768, size=(5, 3, int(16 * k)), dtype=np.int16)
            for chunk in (1, 4, 8192):
                np.testing.assert_array_equal(unpack_states_vectorized(words, k, chunk),
                                              unpack_states(words, k), err_msg=f"k={k}")

    def test_dequantize_bitwise_and_unsigned_words(self):
        rng = np.random.default_rng(11)
        for k in RATES:
            words = rng.integers(0, 65536, size=(2, 4, int(16 * k)), dtype=np.uint16)
            for codebook in ("mul1",) if k % 1 else ("mul1", "mcg", "3inst"):
                expected = dequantize_rotated(words, k, codebook)
                actual = dequantize_rotated_vectorized(words, k, codebook, chunk_tiles=3)
                self.assertEqual(actual.shape, expected.shape)
                self.assertEqual(actual.view(np.uint16).tobytes(), expected.view(np.uint16).tobytes())

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            unpack_states_vectorized(np.zeros((1, 47), np.int16), 3)
        with self.assertRaises(ValueError):
            unpack_states_vectorized(np.zeros((1, 48), np.int32), 3)
        with self.assertRaises(ValueError):
            dequantize_rotated_vectorized(np.zeros((1, 1, 40), np.int16), 2.5, "mcg")


class Bf16MappingTest(unittest.TestCase):
    def test_bf16_requires_opt_in_and_widens_exactly(self):
        values = np.array([1.0, -2.5, 3.140625, 0.0], dtype=np.float32)
        bits = (values.view(np.uint32) >> 16).astype("<u2")
        header = {"x": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]}}
        raw = json.dumps(header).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.safetensors"
            path.write_bytes(struct.pack("<Q", len(raw)) + raw + bits.tobytes())
            file = SafetensorsFile(path)
            with self.assertRaises(ValueError):
                file.tensor("x")
            np.testing.assert_array_equal(file.tensor("x", raw_bf16=True), bits)
            np.testing.assert_array_equal(file.tensor_float32("x"), values)
            del file


if __name__ == "__main__":
    unittest.main()
