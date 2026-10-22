from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import struct
import tempfile
import tracemalloc
import unittest
import weakref

import numpy as np

from strix_trellis.container import Exl3Tensor, SafetensorsFile, load_exl3
from strix_trellis.linear import matmul_fp32
from strix_trellis.fixture import load_fixture


def tensors(bits=3, marker="mul1", empty_marker=False):
    rng = np.random.default_rng(73)
    result = {
        "projection.trellis": rng.integers(-32768, 32768, (8, 16, int(16*bits)), dtype=np.int16),
        "projection.suh": np.linspace(-1, 1, 128, dtype=np.float16),
        "projection.svh": np.linspace(0.25, -0.5, 256, dtype=np.float16),
        "projection.bias": np.arange(256, dtype=np.float16) / np.float16(256),
    }
    if marker:
        word = {"mul1": 0x83DCD12D, "mcg": 0xCBAC1FED}[marker]
        result["projection."+marker] = (np.empty(0, dtype=np.float16) if empty_marker
                                          else np.array(word, dtype=np.uint32).view(np.int32))
    return result


def write_file(path, values, mutate=None):
    dtypes = {"int16": "I16", "int32": "I32", "float16": "F16", "float32": "F32"}
    data, header = bytearray(), {}
    for name, array in values.items():
        payload = array.tobytes()
        header[name] = dict(dtype=dtypes[array.dtype.name], shape=list(array.shape),
                            data_offsets=[len(data), len(data)+len(payload)])
        data.extend(payload)
    if mutate:
        mutate(header)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw))+raw+data)


def independent_transport(group):
    """Legacy wire construction, independent of the optimized layout/copy path."""
    result, fields = bytearray(), {}
    for name, dtype in (("trellis", "<i2"), ("suh", "<f2"), ("svh", "<f2"), ("bias", "<f4")):
        values = getattr(group, name)
        if values is None:
            fields[name+"_offset"] = fields[name+"_bytes"] = 0
            continue
        while len(result) % 4:
            result.append(0)
        raw = np.asarray(values).astype(dtype).tobytes(order="C")
        fields[name+"_offset"], fields[name+"_bytes"] = len(result), len(raw)
        result.extend(raw)
    fields.update(rows=group.output_size, columns=group.input_size, bits_x2=int(group.bits)*2,
                  codebook={"3inst": 0, "mcg": 1, "mul1": 2}[group.codebook],
                  marker={"3inst": 0, "mcg": 0xCBAC1FED, "mul1": 0x83DCD12D}[group.codebook],
                  transport_version=1)
    return bytes(result), fields


class ContainerTests(unittest.TestCase):
    def test_regions_borrow_readonly_mapped_payloads_and_keep_them_alive(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"regions.safetensors"
            source = tensors()
            source["projection.bias"] = source["projection.bias"].astype(np.float32)
            write_file(path, source)
            group = load_exl3(path, "projection")
            expected, fields = independent_transport(group)
            regions, metadata = group.transport_regions()
            self.assertEqual(metadata, fields)
            self.assertIsInstance(group.bias, np.memmap)
            self.assertFalse(group.bias.flags.writeable)
            owner = weakref.ref(group.trellis)
            for name, region in regions.items():
                self.assertTrue(region.readonly)
                self.assertTrue(region.c_contiguous)
                self.assertEqual((region.format, region.itemsize), ("B", 1))
                self.assertTrue(np.shares_memory(np.frombuffer(region, dtype=np.uint8), getattr(group, name)))
                begin, count = fields[name+"_offset"], fields[name+"_bytes"]
                self.assertEqual(region.tobytes(), expected[begin:begin+count])
                with self.assertRaises(TypeError):
                    region[0] = 0
            del group
            gc.collect()
            self.assertIsNotNone(owner())
            self.assertEqual(regions["trellis"].tobytes(), expected[:fields["trellis_bytes"]])
            del region, regions  # Release maps before Windows deletes the file.

    def test_regions_repack_only_strided_trellis_partition(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"sliced.safetensors"
            write_file(path, tensors())
            group = load_exl3(path, "projection")
            self.assertFalse(group.bias.flags.writeable)
            part = group.slice_outputs(128, 256)
            expected, fields = independent_transport(part)
            regions, metadata = part.transport_regions()
            self.assertEqual(metadata, fields)
            self.assertFalse(np.shares_memory(np.frombuffer(regions["trellis"], dtype=np.uint8), part.trellis))
            for name in ("suh", "svh", "bias"):
                self.assertTrue(np.shares_memory(np.frombuffer(regions[name], dtype=np.uint8), getattr(part, name)))
            for name, region in regions.items():
                offset, count = fields[name+"_offset"], fields[name+"_bytes"]
                self.assertEqual(region.tobytes(), expected[offset:offset+count])
            buffer, meta = part.transport_buffer()
            self.assertIsInstance(buffer, bytearray)
            self.assertEqual((bytes(buffer), meta), (expected, fields))
            self.assertEqual(part.transport_blob(), (expected, fields))
            del group, part, region, regions

    def test_transport_buffers_convert_endianness_and_support_absent_bias(self):
        values = tensors()
        for bias in (None, values["projection.bias"].astype(">f4")):
            group = Exl3Tensor(values["projection.trellis"].astype(">i2"),
                               values["projection.suh"].astype(">f2"),
                               values["projection.svh"].astype(">f2"),
                               3, "mul1", bias, "synthetic", "projection")
            expected, fields = independent_transport(group)
            regions, meta = group.transport_regions()
            self.assertEqual(meta, fields)
            if bias is None:
                self.assertIsNone(regions["bias"])
                self.assertEqual((meta["bias_offset"], meta["bias_bytes"]), (0, 0))
            for name, region in regions.items():
                if region is not None:
                    offset, count = fields[name+"_offset"], fields[name+"_bytes"]
                    self.assertEqual(region.tobytes(), expected[offset:offset+count])
                    self.assertTrue(region.readonly)
            destination = np.full((len(expected)//4 + 8, 1), 0xA5A5A5A5, dtype=np.uint32)
            view, metadata = group.transport_buffer(destination)
            self.assertIsInstance(view, memoryview)
            self.assertFalse(view.readonly)
            self.assertEqual((view.nbytes, view.tobytes(), metadata), (len(expected), expected, fields))
            self.assertEqual(memoryview(destination).cast("B")[len(expected):].tobytes(), b"\xa5"*32)
            view[0] ^= 1
            self.assertEqual(memoryview(destination).cast("B")[0], expected[0] ^ 1)

    def test_destination_validation_rejects_aliases_before_any_write(self):
        values = tensors()
        group = Exl3Tensor(values["projection.trellis"], values["projection.suh"],
                           values["projection.svh"], 3, "mul1",
                           values["projection.bias"].astype(np.float32), "synthetic", "projection")
        expected, _ = independent_transport(group)
        short = bytearray(b"\xa5"*(len(expected)-1))
        with self.assertRaisesRegex(ValueError, "too small"):
            group.transport_buffer(short)
        self.assertEqual(short, b"\xa5"*len(short))
        with self.assertRaisesRegex(ValueError, "writable"):
            group.transport_buffer(bytes(len(expected)))
        noncontiguous = bytearray(b"\xa5"*(2*len(expected)))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            group.transport_buffer(memoryview(noncontiguous)[::2])
        self.assertEqual(noncontiguous, b"\xa5"*len(noncontiguous))
        for name, dtype in (("trellis", "<i2"), ("bias", "<f4")):
            storage = bytearray(b"\xa5"*(len(expected)+32))
            original = getattr(group, name)
            alias = np.ndarray(original.shape, dtype=dtype, buffer=storage)
            alias[...] = original
            aliased = replace(group, **{name: alias})
            before = bytes(storage)
            with self.subTest(source=name), self.assertRaisesRegex(ValueError, "overlap"):
                aliased.transport_buffer(storage)
            self.assertEqual(bytes(storage), before)

    def test_transport_padding_is_zero_and_destination_tail_is_preserved(self):
        # The serializer historically accepts caller-constructed array sizes;
        # native prepare independently validates their shape contract.
        group = Exl3Tensor(np.arange(16, dtype=np.int16).reshape(1, 1, 16),
                           np.array([1.25], dtype=np.float16),
                           np.array([-0.5], dtype=np.float16), 1, "3inst",
                           np.array([2.0], dtype=np.float32), "synthetic", "padding")
        expected, fields = independent_transport(group)
        destination = bytearray(b"\xa5"*(len(expected)+13))
        view, metadata = group.transport_buffer(destination)
        self.assertEqual((view.tobytes(), metadata), (expected, fields))
        self.assertEqual(destination[-13:], b"\xa5"*13)
        self.assertEqual(destination[34:36], b"\0\0")
        self.assertEqual(destination[38:40], b"\0\0")

    def test_contiguous_regions_do_not_allocate_a_weight_sized_copy(self):
        group = Exl3Tensor(np.zeros((128, 256, 80), dtype="<i2"),
                           np.ones(2048, dtype="<f2"), np.ones(4096, dtype="<f2"),
                           5, "mul1", None, "synthetic", "large")
        tracing = tracemalloc.is_tracing()
        if not tracing:
            tracemalloc.start()
        baseline, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        try:
            regions, metadata = group.transport_regions()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            if not tracing:
                tracemalloc.stop()
        self.assertEqual(regions["trellis"].nbytes, 5_242_880)
        self.assertEqual(metadata["trellis_bytes"], 5_242_880)
        self.assertTrue(np.shares_memory(np.frombuffer(regions["trellis"], dtype=np.uint8), group.trellis))
        self.assertLess(peak-baseline, 256*1024)

    def test_real_eaq_group_in_a_local_safetensors_container(self):
        manifest = Path(__file__).parent / "fixtures/eaq_v1.json"
        metadata = json.loads(manifest.read_text())
        directory = os.environ.get("STRIX_TRELLIS_FIXTURES") or metadata["fixture_directory"]
        if not Path(directory).exists():
            self.skipTest("External fixture unavailable; no network in tests")
        trellis, suh, svh, _ = load_fixture(manifest, directory)
        prefix = metadata["prefix"]
        values = {prefix+".trellis": trellis, prefix+".suh": suh, prefix+".svh": svh,
                  prefix+".mul1": np.array(0x83DCD12D, dtype=np.uint32).view(np.int32)}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"selected-real-group.safetensors"
            write_file(path, values)
            group = load_exl3(path, prefix)
            self.assertEqual((group.input_size, group.output_size, group.bits), (5120, 512, 6))
            np.testing.assert_array_equal(group.trellis, trellis)
            np.testing.assert_array_equal(group.suh, suh)
            np.testing.assert_array_equal(group.svh, svh)
            del group

    def test_modern_groups_map_and_export_without_materializing_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"weights.safetensors"
            for marker in (None, "mcg", "mul1"):
                source = tensors(marker=marker)
                write_file(path, source)
                group = SafetensorsFile(path).exl3("projection")
                self.assertIsInstance(group.trellis, np.memmap)
                self.assertFalse(group.trellis.flags.writeable)
                self.assertEqual((group.input_size, group.output_size), (128, 256))
                self.assertEqual(group.codebook, marker or "3inst")
                np.testing.assert_array_equal(group.trellis, source["projection.trellis"])
                x = np.arange(128, dtype=np.float32)/128
                expected = matmul_fp32(x, group.trellis, group.suh, group.svh, 3,
                                      group.codebook, bias=group.bias)
                np.testing.assert_array_equal(group.matmul(x, production=True), expected)
                for begin in (0, 128):
                    part = group.slice_outputs(begin, begin+128)
                    np.testing.assert_array_equal(part.matmul(x, production=True), expected[begin:begin+128])
                    payload, fields = part.transport_blob()
                    self.assertEqual(fields["rows"], 128)
                    self.assertEqual(fields["trellis_bytes"], part.trellis.size*2)
                    del part
                with self.assertRaises(ValueError):
                    group.slice_outputs(1, 129)
                blob, meta = group.transport_blob()
                self.assertEqual((meta["rows"], meta["columns"], meta["bits_x2"]), (256, 128, 6))
                for name, dtype in (("trellis", "<i2"), ("suh", "<f2"), ("svh", "<f2"), ("bias", "<f4")):
                    begin, size = meta[name+"_offset"], meta[name+"_bytes"]
                    self.assertEqual(begin % 4, 0)
                    np.testing.assert_array_equal(np.frombuffer(blob[begin:begin+size], dtype=dtype),
                                                  np.asarray(getattr(group, name), dtype=dtype).ravel())
                del group

    def test_half_rate_cpu_and_historical_empty_marker_are_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"weights.safetensors"
            write_file(path, tensors(2.5, empty_marker=True))
            group = load_exl3(path, "projection")
            self.assertEqual(group.bits, 2.5)
            for export in (group.transport_blob, group.transport_buffer, group.transport_regions):
                with self.assertRaisesRegex(ValueError, "integer bitrates"):
                    export()
            del group  # Release read-only maps before Windows removes the file.

    def test_reject_corrupt_container_and_ambiguous_semantics(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"weights.safetensors"
            invalid = [
                lambda h: h["projection.trellis"].update(data_offsets=[0, 99999999]),
                lambda h: h["projection.trellis"].update(shape=[True, 16, 48]),
                lambda h: h["projection.trellis"].update(shape=[7, 16, 48]),
                lambda h: h["projection.trellis"].update(dtype="F16"),
                lambda h: h["projection.svh"].update(data_offsets=h["projection.suh"]["data_offsets"]),
            ]
            for mutate in invalid:
                write_file(path, tensors(), mutate)
                with self.assertRaises(ValueError):
                    load_exl3(path, "projection")
            source = tensors()
            source["projection.mcg"] = np.array(0, dtype=np.int32)
            write_file(path, source)
            with self.assertRaisesRegex(ValueError, "Simultaneous"):
                load_exl3(path, "projection")
            source = tensors()
            source["projection.mul1"] = np.array(1, dtype=np.int32)
            write_file(path, source)
            with self.assertRaisesRegex(ValueError, "marker"):
                load_exl3(path, "projection")
            source = tensors()
            source["projection.out_scales"] = np.ones(256, dtype=np.float16)
            write_file(path, source)
            with self.assertRaisesRegex(ValueError, "scale contract"):
                load_exl3(path, "projection")
            source = tensors()
            source["projection.suh"][4] = np.nan
            write_file(path, source)
            with self.assertRaisesRegex(ValueError, "scale"):
                load_exl3(path, "projection")
            path.write_bytes(b"bad")
            with self.assertRaises(ValueError):
                SafetensorsFile(path)
            duplicate = b'{"x":{},"x":{}}'
            path.write_bytes(struct.pack("<Q", len(duplicate))+duplicate)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                SafetensorsFile(path)


if __name__ == "__main__":
    unittest.main()
