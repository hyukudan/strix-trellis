import os
"""Independent numerical stage oracles; no upstream code execution."""
import hashlib
import json
from pathlib import Path
import struct
import unittest

import numpy as np

from strix_trellis import (core_dot_fp32, decode_legacy_signs, decode_states,
                           dequantize_rotated, hadamard_fp32, input_stages_fp32,
                           matmul, matmul_fp32, matmul_fp32_stages, output_stages_fp32)
from strix_trellis import load_fixture


def parity_matrix(n):
    return np.array([[1 if (i & j).bit_count() % 2 == 0 else -1
                      for j in range(n)] for i in range(n)], dtype=np.float64)


def mathematical_codebook_bytes(name):
    """Dyadic integer mathematics + stdlib binary16 rounding, independent of NumPy LUT."""
    def half_units(bits):
        # Exact signed value in units of 2**-24, including subnormals.
        exponent, fraction = (bits >> 10) & 31, bits & 1023
        magnitude = fraction if exponent == 0 else (1024 + fraction) << (exponent - 1)
        return -magnitude if bits & 32768 else magnitude

    result = bytearray()
    for state in range(65536):
        if name == "mul1":
            p = (state * 2212286765) & 0xffffffff
            total = sum((p >> shift) & 255 for shift in (0, 8, 16, 24))
            numerator = (1024 + total) * 1774 - 1329 * 2048
            value = numerator / 262144
        else:
            multiplier, offset = (89226354, 64248484) if name == "3inst" else (3417055213, 0)
            p = ((state * multiplier + offset) & 0xffffffff & 0x8fff8fff) ^ 0x3b603b60
            value = (half_units(p & 65535) + half_units(p >> 16)) / 16777216
        result.extend(struct.pack("<e", value))
    return bytes(result)


class NumericalTests(unittest.TestCase):
    def test_prescale_double_rounding_regression(self):
        x = np.zeros(128, dtype=np.float32)
        scales = np.ones(128, dtype=np.float16)
        x[0] = np.float32(0.2001299113035202)
        scales[0] = np.float16(0.02642822265625)
        stage = input_stages_fp32(x, scales)["scaled_fp16"]
        self.assertEqual(int(stage.view(np.uint16)[0]), 0x1D6A)
        direct = np.float16(float(x[0]) * float(scales[0]))
        self.assertEqual(int(direct.view(np.uint16)), 0x1D6B)

    def test_real_fixture_production_stages(self):
        manifest = Path(__file__).parent / "fixtures/eaq_v1.json"
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if not Path(os.environ.get("STRIX_TRELLIS_FIXTURES") or metadata["fixture_directory"]).exists():
            self.skipTest("External fixture unavailable; no network access in tests")
        packed, left, right, k = load_fixture(manifest, os.environ.get("STRIX_TRELLIS_FIXTURES"))
        x = np.random.default_rng(1017).normal(size=(2, 5120)).astype(np.float32)
        stages = matmul_fp32_stages(x, packed, left, right, k)
        self.assertEqual(stages["output_fp32"].shape, (2, 512))
        for value in stages.values():
            self.assertTrue(np.isfinite(value).all())
        a = stages["core_input_fp16"].astype(np.float64)
        q = stages["rotated_fp16"].astype(np.float64)
        gamma = (2 * 5120 * 2**-24) / (1 - 2 * 5120 * 2**-24)
        self.assertTrue(np.all(np.abs(stages["core_fp32"] - a @ q) <= gamma * (np.abs(a) @ np.abs(q))))
        np.testing.assert_array_equal(stages["scaled_fp16"], (x * left.astype(np.float32)).astype(np.float16))
        np.testing.assert_array_equal(stages["output_fp32"], stages["output_hadamard_fp32"] * right.astype(np.float32))

    def test_all_codebook_hashes_from_independent_mathematics(self):
        expected = {
            "3inst": "ad88de2dcff5bb05fff2faec37cceceb43aa1c2e8203f9d81a6fbd6dd3393eea",
            "mcg": "95383563929baa9d1ae5de0be22490284382673620bdd85929feea7bdbef04a9",
            "mul1": "bc48d02cb1c14939dc47b90f870dd689d63e3b1ab69a157e65538db68677a6c8",
        }
        for name, digest in expected.items():
            with self.subTest(codebook=name):
                independent = mathematical_codebook_bytes(name)
                self.assertEqual(len(independent), 131072)
                self.assertEqual(hashlib.sha256(independent).hexdigest(), digest)
                actual = decode_states(np.arange(65536), name).astype("<f2").tobytes()
                self.assertEqual(actual, independent)
                self.assertEqual(hashlib.sha256(actual).hexdigest(), digest)

    def test_all_hadamard_impulses_and_block_isolation(self):
        x = np.eye(128, dtype=np.float32)
        factor = np.float32(1 / np.sqrt(128))
        np.testing.assert_array_equal(hadamard_fp32(x), parity_matrix(128).astype(np.float32) * factor)
        blocks = np.concatenate([x, np.zeros_like(x)], axis=1)
        np.testing.assert_array_equal(hadamard_fp32(blocks)[:, 128:], 0)
        np.testing.assert_array_equal(hadamard_fp32(blocks)[:, :128], hadamard_fp32(x))
        np.testing.assert_array_equal(hadamard_fp32(blocks.T, axis=0), hadamard_fp32(blocks).T)

    def test_explicit_half_boundaries_ties_subnormals_and_signed_scales(self):
        x = np.zeros(128, dtype=np.float32)
        x[:6] = [1 + 2**-11, 1 + 3 * 2**-11, 2**-24, 2**-25, 1.0003, -3.25]
        scales = np.ones(128, dtype=np.float16)
        scales[4:7] = [-1.5, 0, 2]
        stages = input_stages_fp32(x, scales)
        np.testing.assert_array_equal(stages["scaled_fp32"], x * scales.astype(np.float32))
        self.assertEqual(stages["scaled_fp16"][0], 1)
        self.assertEqual(stages["scaled_fp16"][1], 1 + 2**-9)
        self.assertEqual(stages["scaled_fp16"][2], 2**-24)
        self.assertEqual(stages["scaled_fp16"][3], 0)
        self.assertEqual(stages["scaled_fp16"][5], 0)
        expected = stages["scaled_fp16"].astype(np.float64) @ parity_matrix(128) / np.sqrt(128)
        np.testing.assert_allclose(stages["input_hadamard_fp32"], expected, rtol=2e-6, atol=2e-7)
        np.testing.assert_array_equal(stages["core_input_fp16"], stages["input_hadamard_fp32"].astype(np.float16))
        self.assertTrue(np.any(stages["core_input_fp16"].astype(np.float32) != stages["input_hadamard_fp32"]))

    def test_core_fixed_order_and_fp32_accumulation(self):
        a = np.array([[2048, 1, 2048], [1, -2, 0.5]], dtype=np.float16)
        q = np.array([[2048, 2], [0.125, -3], [-2048, 4]], dtype=np.float16)
        expected = np.zeros((2, 2), dtype=np.float32)
        for b in range(2):
            for o in range(2):
                acc = np.float32(0)
                for i in range(3):
                    acc = np.float32(acc + np.float32(np.float32(a[b, i]) * np.float32(q[i, o])))
                expected[b, o] = acc
        np.testing.assert_array_equal(core_dot_fp32(a, q), expected)
        self.assertEqual(expected[0, 0], 0)  # 0.125 is lost in the FP32 partial sum.

    def test_rectangular_full_layer_both_orientations_and_bias(self):
        rng = np.random.default_rng(220)
        for inputs, outputs in ((128, 256), (256, 128)):
            with self.subTest(shape=(inputs, outputs)):
                packed = rng.integers(0, 65536, (inputs // 16, outputs // 16, 32), dtype=np.uint16)
                x = rng.normal(size=(2, 1, inputs)).astype(np.float32)
                left = rng.uniform(-2, 2, inputs).astype(np.float16)
                right = rng.uniform(-3, 3, outputs).astype(np.float16)
                left[::7], right[::5] = 0, 0
                bias = rng.normal(size=outputs).astype(np.float32)
                stages = matmul_fp32_stages(x, packed, left, right, bias=bias)
                np.testing.assert_array_equal(stages["rotated_fp16"], dequantize_rotated(packed))
                a, q = stages["core_input_fp16"], stages["rotated_fp16"]
                exact_core = a.astype(np.float64) @ q.astype(np.float64)
                bound = (2 * inputs * 2**-24 / (1 - 2 * inputs * 2**-24)) * (np.abs(a.astype(np.float64)) @ np.abs(q.astype(np.float64)))
                self.assertTrue(np.all(np.abs(stages["core_fp32"] - exact_core) <= bound))
                h = parity_matrix(128) / np.sqrt(128)
                expected_h = (stages["core_fp32"].astype(np.float64).reshape(-1, 128) @ h).reshape(2, 1, outputs)
                np.testing.assert_allclose(stages["output_hadamard_fp32"], expected_h, rtol=2e-5, atol=1e-5)
                np.testing.assert_array_equal(stages["scaled_output_fp32"], stages["output_hadamard_fp32"] * right.astype(np.float32))
                np.testing.assert_array_equal(stages["output_fp32"], stages["scaled_output_fp32"] + bias)
                np.testing.assert_array_equal(stages["output_fp32"][..., ::5], np.broadcast_to(bias[::5], (2, 1, len(bias[::5]))))
                np.testing.assert_array_equal(matmul_fp32(x, packed, left, right, bias=bias), stages["output_fp32"])
                np.testing.assert_array_equal(matmul_fp32(x[0, 0], packed, left, right, bias=bias), stages["output_fp32"][0, 0])
                for name, value in stages.items():
                    self.assertEqual(value.dtype, np.float16 if name.endswith("fp16") else np.float32)
                self.assertEqual(matmul(x, packed, left, right).dtype, np.float64)
                self.assertEqual(stages["output_fp32"].dtype, np.float32)

    def test_rounding_contract_differs_from_mathematical_oracle(self):
        packed = np.zeros((8, 8, 16), dtype=np.uint16)
        x = np.zeros(128, dtype=np.float32)
        x[0] = 1 + 2**-11
        actual = matmul_fp32(x, packed, np.ones(128), np.ones(128))
        mathematical = matmul(x, packed, np.ones(128), np.ones(128))
        self.assertGreater(np.max(np.abs(actual - mathematical)), 1e-4)

    def test_legacy_sign_bit_order_exhaustive_and_signed_storage(self):
        for bit in range(16):
            words = np.array([1 << bit, 0], dtype=np.uint16)
            expected = np.ones(32, dtype=np.float16)
            expected[bit] = -1
            np.testing.assert_array_equal(decode_legacy_signs(words), expected)
            np.testing.assert_array_equal(decode_legacy_signs(words.view(np.int16)), expected)
        words = np.arange(65536, dtype=np.uint16)
        signs = decode_legacy_signs(words).reshape(-1, 16)
        repacked = np.sum((signs < 0).astype(np.uint32) * (1 << np.arange(16)), axis=1)
        np.testing.assert_array_equal(repacked, words)
        np.testing.assert_array_equal(decode_legacy_signs(np.array([0x8001, 0x0002], dtype=">u2")),
                                      [-1] + [1] * 14 + [-1, 1, -1] + [1] * 14)

    def test_reject_nonfinite_conversion_overflow_and_bad_shapes(self):
        for bad in (np.nan, np.inf, -np.inf, 1e100, 1j):
            with self.subTest(bad=bad):
                x = np.ones(128, dtype=np.complex128 if bad == 1j else np.float64)
                x[0] = bad
                for call in (lambda: input_stages_fp32(x, np.ones(128)),
                             lambda: input_stages_fp32(np.ones(128), x),
                             lambda: output_stages_fp32(x, np.ones(128)),
                             lambda: output_stages_fp32(np.ones(128), x),
                             lambda: output_stages_fp32(np.ones(128), np.ones(128), x),
                             lambda: core_dot_fp32(x, np.ones((128, 2))),
                             lambda: core_dot_fp32(np.ones(128), x.reshape(128, 1))):
                    with self.assertRaises(ValueError):
                        call()
        for call in (lambda: input_stages_fp32(np.full(128, 65504), np.full(128, 2)),
                     lambda: hadamard_fp32(np.full(128, np.finfo(np.float32).max)),
                     lambda: input_stages_fp32(np.ones(127), np.ones(127)),
                     lambda: output_stages_fp32(np.ones(128), np.ones(127)),
                     lambda: output_stages_fp32(np.ones(128), np.ones(128), np.ones(1)),
                     lambda: core_dot_fp32(np.ones(2), np.ones((3, 4))),
                     lambda: decode_legacy_signs(np.ones(2, dtype=np.int32)),
                     lambda: decode_legacy_signs(np.ones((1, 2), dtype=np.int16))):
            with self.assertRaises(ValueError):
                call()


if __name__ == "__main__":
    unittest.main()
