import os
"""Independent byte/coordinate oracles and algebraic CPU reference tests."""
import json
from pathlib import Path
import unittest

import numpy as np

from strix_trellis import (decode_states, decode_tile, dequantize_rotated, hadamard,
                           load_fixture, matmul, reconstruct_weight, step_widths, unpack_states)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests/fixtures/eaq_v1.json"


def bit_oracle(tile, k):
    """Read physical bytes bit by bit, then advance a 16-bit shift register."""
    raw = np.asarray(tile, dtype="<u2").tobytes()
    bits = []
    for offset in range(0, len(raw), 4):
        for byte in raw[offset:offset + 4][::-1]:
            bits.extend(int(digit) for digit in f"{byte:08b}")
    state = int("".join(str(bit) for bit in bits[-16:]), 2)
    states = []
    cursor = 0
    for i in range(256):
        count = int(k) + (i % 2 if k % 1 else 0)
        for bit in bits[cursor:cursor + count]:
            state = ((state << 1) + bit) % 65536
        states.append(state)
        cursor += count
    return np.array(states, dtype=np.uint16)


def explicit_hadamard(n):
    # H[r,c] follows the parity of the binary dot product, no butterfly.
    return np.array([[(-1) ** ((r & c).bit_count()) for c in range(n)]
                     for r in range(n)], dtype=np.float64) / np.sqrt(n)


class CodecTests(unittest.TestCase):
    def test_all_rates_against_byte_shift_register(self):
        rng = np.random.default_rng(9823)
        for k in [*range(1, 9), 1.5, 2.5, 3.5]:
            with self.subTest(k=k):
                words = rng.integers(0, 65536, (3, int(16 * k)), dtype=np.uint16)
                actual = unpack_states(words, k)
                for i in range(3):
                    np.testing.assert_array_equal(actual[i], bit_oracle(words[i], k))
                np.testing.assert_array_equal(actual, unpack_states(words.view(np.int16), k))
                np.testing.assert_array_equal(actual, unpack_states(words.astype(">u2"), k))

    def test_integer_state_dtypes(self):
        expected = decode_states(np.array([0, 1, 65535], dtype=np.int64))
        for dtype in (np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64):
            values = np.array([0, 1, 65535], dtype=np.int64).astype(dtype)
            np.testing.assert_array_equal(decode_states(values), expected)

    def test_physical_word_order_and_ring_wrap(self):
        words = np.zeros(16, dtype=np.uint16)
        words[1] = 0x8000  # MSB of physical uint32, ring bit zero
        states = unpack_states(words, 1)
        np.testing.assert_array_equal(states[:16], 1 << np.arange(16))
        self.assertEqual(int(states[16]), 0)
        words[:] = 0
        words[-2] = 1  # last ring bit becomes bit one of first window
        self.assertEqual(int(unpack_states(words, 1)[0]), 2)

    def test_fragment_coordinate_mapping(self):
        # Exercise every logical coordinate with nonuniform stream values.
        words = np.random.default_rng(12).integers(0, 65536, 128, dtype=np.uint16)
        values = decode_states(bit_oracle(words, 8))
        tile = decode_tile(words, 8)
        seen = set()
        for lane in range(32):
            for fragment in range(8):
                row = 2 * (lane % 4) + fragment % 2 + 8 * ((fragment // 2) % 2)
                col = lane // 4 + 8 * (fragment // 4)
                self.assertEqual(tile[row, col], values[8 * lane + fragment])
                seen.add((row, col))
        self.assertEqual(len(seen), 256)

    def test_codebook_bit_vectors(self):
        states = np.array([0, 1, 2, 255, 256, 32767, 32768, 65535])
        expected = {
            "mul1": [0xc2e8, 0x3921, 0xb72d, 0x3ab3, 0xb3db, 0x3fca, 0xbaf2, 0xb936],
            "mcg": [0x3f60, 0x304e, 0xba13, 0xbd26, 0xbaf0, 0x4172, 0xb915, 0x3acd],
            "3inst": [0x3a25, 0xbb5b, 0x3b74, 0x3cd1, 0x3c3f, 0xba96, 0xb7f8, 0xb110],
        }
        for name, bits in expected.items():
            with self.subTest(codebook=name):
                result = decode_states(states, name)
                np.testing.assert_array_equal(result.view(np.uint16), bits)
                np.testing.assert_array_equal(result, decode_states(states + 65536, name))
                np.testing.assert_array_equal(decode_states(np.array([-1]), name), result[-1:])

    def test_mul1_single_rounding_exhaustive(self):
        # Independent vector oracle using exact integer numerator of the FMA.
        # 0x1eee = 1774/2**18; 0xc931 = -1329/128.
        states = np.arange(65536, dtype=np.uint64)
        products = (states * 2212286765) % (1 << 32)
        byte_sum = np.zeros(65536, dtype=np.int64)
        for shift in (0, 8, 16, 24):
            byte_sum += ((products >> shift) % 256).astype(np.int64)
        numerator = (1024 + byte_sum) * 1774 - 1329 * 2048
        expected = (numerator.astype(np.float64) / 262144).astype(np.float16)
        np.testing.assert_array_equal(decode_states(states).view(np.uint16), expected.view(np.uint16))

    def test_reject_unsupported_formats(self):
        for rate in [0, 9, 4.5, 1.25, float("nan"), float("inf"), True, "3"]:
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                step_widths(rate)
        for data in [np.zeros(47, dtype=np.int16), np.zeros(48, dtype=np.int32)]:
            with self.assertRaises(ValueError):
                unpack_states(data, 3)
        with self.assertRaises(ValueError):
            decode_tile(np.zeros(24, dtype=np.int16), 1.5, "mcg")
        with self.assertRaises(ValueError):
            decode_states([1], "unknown")
        with self.assertRaises(ValueError):
            decode_states([1.0])


class TransformTests(unittest.TestCase):
    def test_hadamard_matches_parity_matrix_and_is_involution(self):
        rng = np.random.default_rng(45)
        for n in (1, 2, 8, 128):
            x = rng.normal(size=(3, 2 * n))
            expected = x.reshape(3, 2, n) @ explicit_hadamard(n)
            np.testing.assert_allclose(hadamard(x, n), expected.reshape(x.shape), atol=2e-14)
            np.testing.assert_allclose(hadamard(hadamard(x, n), n), x, atol=2e-14)
        x = rng.normal(size=(256, 3))
        np.testing.assert_allclose(hadamard(x, axis=0), hadamard(x.T).T, atol=1e-14)

    def test_signed_scales_bias_and_orientation(self):
        rng = np.random.default_rng(41)
        packed = rng.integers(-32768, 32767, (8, 16, 32), dtype=np.int16)
        left = rng.normal(size=128)
        right = rng.normal(size=256)
        x = rng.normal(size=(2, 3, 128))
        bias = rng.normal(size=256)
        rotated = dequantize_rotated(packed).astype(np.float64)
        hleft = explicit_hadamard(128)
        hright = np.kron(np.eye(2), hleft)
        dense = left[:, None] * (hleft @ rotated @ hright) * right
        np.testing.assert_allclose(reconstruct_weight(packed, left, right), dense, atol=2e-14)
        np.testing.assert_allclose(matmul(x, packed, left, right, bias=bias), x @ dense + bias,
                                   rtol=1e-12, atol=3e-13)
        np.testing.assert_allclose(matmul(x[0, 0], packed, left, right), x[0, 0] @ dense,
                                   rtol=1e-12, atol=3e-13)

    def test_dimension_errors(self):
        for size in (0, 3, True):
            with self.assertRaises(ValueError):
                hadamard(np.ones(128), size)
        with self.assertRaises(ValueError):
            hadamard(np.ones(129))
        packed = np.zeros((8, 8, 16), dtype=np.int16)
        with self.assertRaises(ValueError):
            matmul(np.ones(127), packed, np.ones(128), np.ones(128))
        with self.assertRaises(ValueError):
            matmul(np.ones(128), packed, np.ones(128), np.ones(127))


class RealFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        metadata = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if not Path(os.environ.get("STRIX_TRELLIS_FIXTURES") or metadata["fixture_directory"]).exists():
            raise unittest.SkipTest("Fetch the external fixture using tools/fetch_fixture.py")
        cls.packed, cls.left, cls.right, cls.k = load_fixture(MANIFEST, os.environ.get("STRIX_TRELLIS_FIXTURES"))

    def test_actual_shape_multiplier_and_ranged_budget(self):
        self.assertEqual(self.packed.shape, (320, 32, 96))
        self.assertEqual(self.k, 6)
        meta = json.loads(MANIFEST.read_text())
        self.assertLess(meta["downloaded_bytes"], 100_000_000)
        self.assertEqual(sum(r["bytes"] for r in meta["ranges"]), meta["downloaded_bytes"])

    def test_actual_tiles_against_byte_oracle(self):
        for row, col in [(0, 0), (0, 31), (319, 0), (319, 31), (167, 13)]:
            tile = self.packed[row, col]
            np.testing.assert_array_equal(unpack_states(tile, self.k), bit_oracle(tile, self.k))

    def test_full_actual_tensor_matmul(self):
        x = np.random.default_rng(1017).normal(size=(2, 5120))
        dense = reconstruct_weight(self.packed, self.left, self.right)
        self.assertEqual(dense.shape, (5120, 512))
        self.assertTrue(np.isfinite(dense).all())
        output = matmul(x, self.packed, self.left, self.right)
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_allclose(output, x @ dense, rtol=1e-11, atol=2e-11)


if __name__ == "__main__":
    unittest.main()
