from dataclasses import replace
import unittest

import numpy as np

from strix_trellis import Exl3Tensor, dequantize_rotated, rank_partition


def small_group(bits=4, bias=True):
    rng = np.random.default_rng(20261021 + bits)
    return Exl3Tensor(
        rng.integers(-32768, 32768, (16, 16, bits * 16), dtype=np.int16),
        rng.uniform(-0.5, 0.5, 256).astype(np.float16),
        rng.uniform(-0.5, 0.5, 256).astype(np.float16), bits, "mul1",
        np.linspace(-0.25, 0.25, 256, dtype=np.float32) if bias else None,
        "synthetic", "small")


def independent_h128():
    # Matrix definition from parity, independent of butterfly implementation.
    return np.array([[(-1.0 if (i & j).bit_count() & 1 else 1.0)
                      for j in range(128)] for i in range(128)]) / np.sqrt(128.0)


def lightweight_group(inputs, outputs, bits, bias=True):
    # Only input-tile indices are stored; the packed logical shape can describe
    # q_b without allocating a full model tensor or decoding any large weight.
    packed = np.broadcast_to(np.arange(inputs // 16, dtype=np.int16)[:, None, None],
                             (inputs // 16, outputs // 16, bits * 16))
    return Exl3Tensor(packed,
        np.linspace(-0.5, 0.5, inputs, dtype=np.float16),
        np.linspace(0.125, -0.125, outputs, dtype=np.float16), bits, "mul1",
        np.linspace(-0.25, 0.25, outputs, dtype=np.float32) if bias else None,
        "synthetic", "site")


class TpPartitionTests(unittest.TestCase):
    def test_small_input_partitions_recombine_with_bias_exactly_once(self):
        h = np.kron(np.eye(2), independent_h128())
        x = np.random.default_rng(19).normal(0, 0.25, (3, 256))
        for bits in (2, 3, 4, 5):
            group = small_group(bits)
            q = dequantize_rotated(group.trellis, bits, "mul1").astype(np.float64)
            expected = ((x * group.suh) @ h @ q @ h) * group.svh + group.bias
            for bias_rank in (0, 1):
                parts = [group.slice_inputs(128 * rank, 128 * (rank + 1),
                                            include_bias=rank == bias_rank)
                         for rank in (0, 1)]
                output = sum(parts[rank].matmul(x[:, rank * 128:(rank + 1) * 128])
                             for rank in (0, 1))
                with self.subTest(bits=bits, bias_rank=bias_rank):
                    np.testing.assert_allclose(output, expected, rtol=1e-12, atol=1e-12)
                    self.assertEqual(sum(part.bias is not None for part in parts), 1)
            duplicated = (group.slice_inputs(0, 128, include_bias=True).matmul(x[:, :128]) +
                          group.slice_inputs(128, 256, include_bias=True).matmul(x[:, 128:]))
            np.testing.assert_allclose(duplicated - expected,
                                       np.broadcast_to(group.bias, expected.shape),
                                       rtol=1e-12, atol=1e-12)

    def test_small_output_partitions_concatenate_in_original_basis(self):
        group = small_group(5)
        x = np.random.default_rng(31).normal(0, 0.25, (2, 256))
        h = np.kron(np.eye(2), independent_h128())
        q = dequantize_rotated(group.trellis, 5, "mul1").astype(np.float64)
        expected = ((x * group.suh) @ h @ q @ h) * group.svh + group.bias
        halves = [group.slice_outputs(rank * 128, (rank + 1) * 128).matmul(x)
                  for rank in (0, 1)]
        np.testing.assert_allclose(np.concatenate(halves, axis=-1), expected,
                                   rtol=1e-12, atol=1e-12)

    def test_input_slices_borrow_contiguous_payloads_and_share_output_scales(self):
        group = small_group()
        part = group.slice_inputs(128, 256, include_bias=True)
        self.assertTrue(np.shares_memory(group.trellis, part.trellis))
        self.assertTrue(np.shares_memory(group.suh, part.suh))
        self.assertIs(part.svh, group.svh)
        self.assertIs(part.bias, group.bias)
        self.assertTrue(part.trellis.flags.c_contiguous)
        np.testing.assert_array_equal(part.trellis, group.trellis[8:])
        np.testing.assert_array_equal(part.suh, group.suh[128:])
        regions, metadata = part.transport_regions()
        self.assertEqual((metadata["columns"], metadata["rows"]), (128, 256))
        for name, region in regions.items():
            self.assertTrue(region.readonly)
            self.assertTrue(np.shares_memory(np.frombuffer(region, dtype=np.uint8),
                                            getattr(group, name)))
        excluded = group.slice_inputs(0, 128, include_bias=False)
        self.assertIsNone(excluded.bias)
        regions, metadata = excluded.transport_regions()
        self.assertIsNone(regions["bias"])
        self.assertEqual(metadata["bias_bytes"], 0)
        absent = small_group(bias=False).slice_inputs(0, 128, include_bias=True)
        self.assertIsNone(absent.bias)

    def test_output_slices_repack_only_when_exported(self):
        group = small_group()
        part = group.slice_outputs(128, 256)
        self.assertTrue(np.shares_memory(group.trellis, part.trellis))
        self.assertFalse(part.trellis.flags.c_contiguous)
        regions, metadata = part.transport_regions()
        self.assertFalse(np.shares_memory(np.frombuffer(regions["trellis"], dtype=np.uint8),
                                         group.trellis))
        expected = group.trellis[:, 8:, :].astype("<i2").tobytes(order="C")
        self.assertEqual(regions["trellis"].tobytes(), expected)
        self.assertEqual(metadata["trellis_bytes"], len(expected))
        for name in ("suh", "svh", "bias"):
            self.assertTrue(np.shares_memory(np.frombuffer(regions[name], dtype=np.uint8),
                                            getattr(group, name)))

    def test_input_slice_requires_explicit_boolean_and_h128_bounds(self):
        group = small_group()
        with self.assertRaises(TypeError):
            group.slice_inputs(0, 128)
        for begin, end in ((-128, 128), (0, 0), (128, 128), (128, 384),
                           (1, 128), (0, 129), (True, 128), (0, 128.0),
                           (np.int64(0), 128)):
            with self.subTest(begin=begin, end=end), self.assertRaises(ValueError):
                group.slice_inputs(begin, end, include_bias=False)
        for include in (None, 0, 1, "yes", np.bool_(True)):
            with self.subTest(include=include), self.assertRaises(ValueError):
                group.slice_inputs(0, 128, include_bias=include)

    def test_all_exact_site_shapes_ranks_and_bias_owners_without_large_weights(self):
        cases = (
            ("q_b", 1280, 32768, 1280, 16384, "output", (4, 5)),
            ("o_b", 8192, 5120, 4096, 5120, "input", (4, 5)),
            ("shared_gate", 5120, 2304, 5120, 1152, "output", (4, 5)),
            ("shared_up", 5120, 2304, 5120, 1152, "output", (4, 5)),
            ("shared_down", 2304, 5120, 1152, 5120, "input", (4, 5)),
            ("q_a", 5120, 1280, 5120, 1280, "replicate", (4, 5)),
            ("kv", 5120, 512, 5120, 512, "replicate", (4, 5)),
            ("expert_gate", 5120, 2304, 5120, 1152, "output", (2, 3)),
            ("expert_up", 5120, 2304, 5120, 1152, "output", (2, 3)),
            ("expert_down", 2304, 5120, 1152, 5120, "input", (2, 3)),
        )
        for site, inputs, outputs, local_i, local_o, axis, rates in cases:
            for bits in rates:
                group = lightweight_group(inputs, outputs, bits)
                self.assertLess(group.trellis.base.nbytes, 4096)
                for bias_rank in (0, 1):
                    for rank in (0, 1):
                        with self.subTest(site=site, bits=bits, rank=rank, bias_rank=bias_rank):
                            part = rank_partition(group, site, rank, bias_rank=bias_rank)
                            self.assertEqual((part.input_size, part.output_size), (local_i, local_o))
                            self.assertEqual((part.bits, part.codebook), (bits, "mul1"))
                            self.assertTrue(np.shares_memory(part.trellis, group.trellis))
                            if axis == "replicate":
                                self.assertIs(part, group)
                            elif axis == "output":
                                np.testing.assert_array_equal(part.svh,
                                    group.svh[rank * local_o:(rank + 1) * local_o])
                                np.testing.assert_array_equal(part.bias,
                                    group.bias[rank * local_o:(rank + 1) * local_o])
                                self.assertIs(part.suh, group.suh)
                            else:
                                np.testing.assert_array_equal(part.suh,
                                    group.suh[rank * local_i:(rank + 1) * local_i])
                                np.testing.assert_array_equal(part.trellis[:, 0, 0],
                                    group.trellis[rank * (local_i // 16):(rank + 1) * (local_i // 16), 0, 0])
                                self.assertIs(part.svh, group.svh)
                                self.assertIs(part.bias, group.bias if rank == bias_rank else None)
                            if axis != "replicate":
                                with self.assertRaises(ValueError):
                                    rank_partition(part, site, rank)

    def test_partition_validation_rejects_wrong_site_rank_format_and_shapes(self):
        group = lightweight_group(1280, 32768, 4)
        for site in (None, "", "qb", "Q_B", "expert_down", 1, []):
            with self.subTest(site=site), self.assertRaises(ValueError):
                rank_partition(group, site, 0)
        for rank in (-1, 2, True, 0.0, "0", np.int64(0), None):
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                rank_partition(group, "q_b", rank)
            with self.subTest(bias_rank=rank), self.assertRaises(ValueError):
                rank_partition(group, "q_b", 0, bias_rank=rank)
        with self.assertRaises(ValueError):
            rank_partition(None, "q_b", 0)
        for bits in (True, 2, 3, 4.5, 6, float("nan"), "4"):
            with self.subTest(bits=bits), self.assertRaises(ValueError):
                rank_partition(replace(group, bits=bits), "q_b", 0)
        for codebook in ("mcg", "3inst", None):
            with self.subTest(codebook=codebook), self.assertRaises(ValueError):
                rank_partition(replace(group, codebook=codebook), "q_b", 0)
        malformed = (
            replace(group, trellis=np.zeros((80, 2048), dtype=np.int16)),
            replace(group, trellis=np.broadcast_to(np.array(0, dtype=np.int16), (80, 2048, 80))),
            replace(group, trellis=np.broadcast_to(np.array(0, dtype=np.uint16), (80, 2048, 64))),
            replace(group, suh=group.suh[:-1]),
            replace(group, suh=group.suh.astype(np.float32)),
            replace(group, svh=np.full(32768, np.nan, dtype=np.float16)),
            replace(group, bias=group.bias[:-1]),
            replace(group, bias=group.bias.astype(np.float16)),
            replace(group, bias=np.full(32768, np.inf, dtype=np.float32)),
        )
        for invalid in malformed:
            with self.assertRaises(ValueError):
                rank_partition(invalid, "q_b", 0)
        experts = lightweight_group(5120, 2304, 4)
        with self.assertRaises(ValueError):
            rank_partition(experts, "expert_gate", 0)

    def test_partition_without_bias_preserves_absence_on_every_rank(self):
        group = lightweight_group(2304, 5120, 2, bias=False)
        for rank in (0, 1):
            for bias_rank in (0, 1):
                part = rank_partition(group, "expert_down", rank, bias_rank=bias_rank)
                self.assertIsNone(part.bias)


if __name__ == "__main__":
    unittest.main()
