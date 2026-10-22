"""Independent decode/oracle checks and hostile inputs for the native parser."""
import importlib.util
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rank_fixture_generator", ROOT / "tools/make_rank_fixtures.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


class RankFixtureTests(unittest.TestCase):
    def test_vector_decode_matches_integer_ring(self):
        rng = np.random.default_rng(17)
        for bits in (2, 3, 4, 5):
            packed = rng.integers(0, 65536, (4, 3, bits * 16), dtype=np.uint16)
            expected = generator.decode_tile(packed, bits).transpose(0, 2, 1, 3).reshape(64, 48)
            np.testing.assert_array_equal(generator.decode_columns(packed, bits), expected)

    def test_oracle_encloses_separate_fp32_core(self):
        from strix_trellis.linear import matmul_fp32
        with tempfile.TemporaryDirectory() as temporary:
            for bits in (2, 3, 4, 5):
                _, data = generator.make_fixture(Path(temporary) / "case.strank", 128, 128, bits, 77 + bits)
                other = matmul_fp32(data["x"], data["trellis"], data["suh"], data["svh"], bits, bias=data["bias"])
                error = np.abs(other.astype(np.float64) - data["ref"].astype(np.float64))
                self.assertTrue(np.all(error <= data["bound"]))
                zero = data["svh"] == 0
                np.testing.assert_array_equal(data["ref"][:, zero], np.broadcast_to(data["bias"][zero], (8, int(zero.sum()))))

    def test_routes_cover_union_extremes(self):
        sizes = {}
        for name in ("shared", "overlap", "disjoint"):
            indices, scores = generator.route_case(name)
            self.assertEqual(indices.shape, (8, 6))
            self.assertTrue(all(len(set(row)) == 6 for row in indices))
            self.assertTrue(np.all(scores >= 0))
            np.testing.assert_allclose(scores.sum(axis=1), 1, atol=1e-7)
            sizes[name] = np.unique(indices).size
        self.assertEqual(sizes, {"shared": 6, "overlap": 27, "disjoint": 48})
        self.assertEqual({generator.expert_bits(i, p) for i in range(48) for p in ("gate", "up", "down")}, {2, 3})

    def test_native_parser_rejects_malformed_sections(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "parser.cpp"
            source.write_text('#include "rank_fixture.hpp"\nint main(int argc,char**argv){try{if(argc!=2)return 2;rank_qa::Fixture f(argv[1]);return 0;}catch(...){return 1;}}\n')
            binary = directory / "parser"
            subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(ROOT / "tests"), str(source), "-o", str(binary)], check=True, capture_output=True)
            good = directory / "good.strank"
            generator.make_fixture(good, 128, 128, 2, 88)
            self.assertEqual(subprocess.run([str(binary), str(good)]).returncode, 0)
            raw = good.read_bytes()
            malformed = [b"", raw[:15], raw[:-1], raw + b"x", b"BADMAGIC" + raw[8:]]
            item = bytearray(raw); struct.pack_into("<I", item, 12, 33); malformed.append(item)
            item = bytearray(raw); struct.pack_into("<Q", item, 24, 2**64 - 1); malformed.append(item)
            item = bytearray(raw); struct.pack_into("<I", item, 32 + 20, 9); malformed.append(item)
            item = bytearray(raw); struct.pack_into("<I", item, 32 + 8, 6); malformed.append(item)
            item = bytearray(raw); item[16:24] = b"meta\0x\0\0"; malformed.append(item)
            # The second section starts after the 32-byte metadata payload.
            item = bytearray(raw); item[64:72] = b"meta\0\0\0\0"; malformed.append(item)
            for index, data in enumerate(malformed):
                path = directory / f"bad-{index}.strank"; path.write_bytes(data)
                self.assertNotEqual(subprocess.run([str(binary), str(path)]).returncode, 0, index)
            indices, scores = generator.route_case("disjoint")
            sections = {"meta": struct.pack("<8I", 5120, 5120, 0, 2, 0x83DCD12D, 8, 0, 1),
                        "routes": (17 + 3 * indices).astype("<u4"), "scores": scores,
                        "hidden": np.zeros((8, 6, 1152), dtype="<f4"),
                        "each": np.zeros((8, 6, 5120), dtype="<f4"),
                        "ref": np.zeros((8, 5120), dtype="<f4"), "old": np.zeros((8, 5120), dtype="<f4")}
            moe = directory / "moe.strank"; generator.write_sections(moe, sections)
            self.assertEqual(subprocess.run([str(binary), str(moe)]).returncode, 0)
            for field, value, index in (("routes", 16, (0, 0)), ("routes", 20, (0, 0)),
                                        ("scores", -1, (0, 0)), ("scores", np.nan, (0, 0))):
                saved = sections[field][index]; sections[field][index] = value
                generator.write_sections(moe, sections)
                self.assertNotEqual(subprocess.run([str(binary), str(moe)]).returncode, 0, (field, value))
                sections[field][index] = saved


if __name__ == "__main__":
    unittest.main()
