"""Run the real K2 probe's fixture parser on CPU, without HIP dependencies.

Run from the repository root with:
    python3 -m unittest discover -s tests -p test_k2_fixture_parser.py -v

The nine test methods exercise 28 files. g++ is a required CPU CI dependency;
an unavailable compiler is an error, never a skipped parser regression test.
"""

from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest


def encode(sections):
    """Construct STGPU framing; the C++ parser supplies all validation."""
    return b"STGPU001" + struct.pack("<II", 1, len(sections)) + b"".join(
        tag.encode("ascii").ljust(8, b"\0") + struct.pack("<Q", len(data)) + data
        for tag, data in sections
    )


class K2FixtureParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("g++")
        if compiler is None:
            raise RuntimeError("g++ is required for the CPU K2 fixture parser tests")
        cls.temporary = tempfile.TemporaryDirectory(prefix="strix-k2-parser-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)

        source = (Path(__file__).parent / "k2_core_probe.hip").read_text()
        begin = "struct Fixture {"
        end = "\nstatic void large("
        if source.count(begin) != 1 or source.count(end) != 1:
            raise RuntimeError("K2 Fixture source boundaries changed; update extraction")
        start = source.index(begin)
        parser = source[start:source.index(end, start)]
        # Extract the production assertion helper as well as the complete
        # parser. Only standard headers and the process driver are test code.
        require = [line for line in source.splitlines()
                   if line.startswith("static void require(")]
        if len(require) != 1:
            raise RuntimeError("K2 require source boundary changed; update extraction")
        wrapper = """#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
""" + require[0] + "\n" + parser + """
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    try {
        Fixture fixture(argv[1]);
        std::cout << fixture.I << ' ' << fixture.N << '\\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\\n';
        return 1;
    }
}
"""
        translation_unit = cls.root / "fixture_parser.cpp"
        translation_unit.write_text(wrapper)
        cls.executable = cls.root / "fixture_parser"
        result = subprocess.run(
            [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic",
             str(translation_unit), "-o", str(cls.executable)],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode:
            raise RuntimeError("Real K2 parser compilation failed:\n" + result.stdout + result.stderr)

        # Tiny, valid R8/K2 arrays with the same layouts as make_k2_fixtures.py.
        cls.sections = {
            "meta": struct.pack("<8I", 128, 128, 2, 2, 0x83DCD12D, 8, 0, 1),
            "trellis": bytes(128 * 128 // 4),
            "a": bytes(8 * 128 * 2), "x": bytes(8 * 128 * 4),
            "suh": bytes(128 * 2), "svh": bytes(128 * 2),
            "old": bytes(8 * 128 * 4), "ref": bytes(8 * 128 * 4),
            "p64": bytes(4 * 128 * 8), "pabs": bytes(4 * 128 * 8),
            "bias": b"",
        }
        cls.valid = encode(list(cls.sections.items()))

    def check_file(self, name, payload, *, accepted=False, error=None):
        path = self.root / (name + ".stgpu")
        path.write_bytes(payload)
        result = subprocess.run(
            [str(self.executable), str(path)],
            capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(result.returncode, 0 if accepted else 1,
                         msg=result.stdout + result.stderr)
        if accepted:
            self.assertEqual(result.stdout, "128 128\n")
        else:
            self.assertIn(error, result.stderr)

    def test_accept_valid_with_and_without_bias(self):
        for bias in (0, 1):
            with self.subTest(bias=bias):
                sections = dict(self.sections)
                sections["meta"] = struct.pack("<8I", 128, 128, 2, 2, 0x83DCD12D, 8, bias, 1)
                sections["bias"] = bytes(128 * 4) if bias else b""
                self.check_file("valid_bias_" + str(bias), encode(list(sections.items())), accepted=True)

    def test_reject_short_metadata(self):
        self.check_file("short_meta", encode([("meta", bytes(23))]), error="fixture metadata size")

    def test_reject_duplicate_section(self):
        self.check_file("duplicate", encode(list(self.sections.items()) + [("meta", self.sections["meta"])]),
                        error="duplicate fixture section")

    def test_reject_truncated_file(self):
        self.check_file("truncated", self.valid[:-1], error="section header")

    def test_reject_trailing_bytes(self):
        self.check_file("trailing", self.valid + b"!", error="trailing fixture bytes")

    def test_reject_oversized_section(self):
        payload = b"STGPU001" + struct.pack("<II", 1, 1) + b"meta\0\0\0\0" + struct.pack("<Q", 1 << 63)
        self.check_file("huge_section", payload, error="section length")

    def test_reject_wrong_logical_sizes(self):
        for tag, data in self.sections.items():
            if tag == "meta":
                continue
            with self.subTest(section=tag):
                sections = dict(self.sections)
                sections[tag] = data[:-1] if data else b"x"
                self.check_file("bad_size_" + tag, encode(list(sections.items())),
                                error="fixture logical section size")

    def test_reject_missing_sections(self):
        for tag in self.sections:
            if tag == "meta":
                continue
            with self.subTest(section=tag):
                sections = dict(self.sections)
                del sections[tag]
                self.check_file("missing_" + tag, encode(list(sections.items())),
                                error="fixture logical section size")

    def test_reject_invalid_bias_flag(self):
        sections = dict(self.sections)
        sections["meta"] = struct.pack("<8I", 128, 128, 2, 2, 0x83DCD12D, 8, 2, 1)
        self.check_file("bad_bias_flag", encode(list(sections.items())), error="fixture contract")


if __name__ == "__main__":
    unittest.main()
