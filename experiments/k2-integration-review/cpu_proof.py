"""Independent, bounded CPU model; no HIP, models, network, or file writes."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import struct


def lower_priority():
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        assert kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000)
    else:
        os.nice(10)


def half(value):
    return struct.unpack("<e", struct.pack("<H", value))[0]


def fields(line):
    return dict(re.findall(r"(\w+)=(\S+)", line))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    lower_priority()
    counts = dict(reader=0, hash=0, half=0, split=0, guard=0, bounds=0)
    # Basis completeness proves a linear bit-selection map for every ring.
    for bit in range(512):
        words = [0] * 16
        words[bit // 32] = 1 << (31 - bit % 32)
        for lane in range(32):
            start = 498 + 16 * lane
            index, shift = start // 32 - 16, start % 32
            ia, ib = index % 16, (index + 1) % 16
            assert -1 <= index <= 15 and shift in (2, 18)
            a, b = words[ia], words[ib]
            high = ((a << shift) | (b >> (32 - shift))) & 0xffffffff
            window = a if shift == 2 else ((a << 16) | (b >> 16)) & 0xffffffff
            for j in range(8):
                expected = 0
                for t in range(16):
                    pos = (2 * (8 * lane + j + 1) - 16 + t) % 512
                    expected = expected * 2 + ((words[pos // 32] >> (31 - pos % 32)) & 1)
                assert (window >> (14 - 2 * j)) & 65535 == expected
                assert (high >> (16 - 2 * j)) & 65535 == expected
                counts["reader"] += 2
    digest = hashlib.sha256()
    for state in range(65536):
        product = state * 0x83dcd12d & 0xffffffff
        byte_sum = sum((product >> t) & 255 for t in (0, 8, 16, 24))
        exact = ((1024 + byte_sum) * 1774 - 1329 * 2048) / 2**18
        bits = struct.pack("<e", exact)
        digest.update(bits)
        for j in range(8):
            container = state | ((0xa55a + j) << 16)
            # Model the two u16 sources and the full-width MAD addend.
            s = container & 65535
            packed_high = ((s * 0x83dc) & 65535) << 16
            staged = (s * 0xd12d + packed_high) & 0xffffffff
            assert staged == product
            sad = sum((staged >> t) & 255 for t in (0, 8, 16, 24))
            assert struct.pack("<e", half(0x6400 + sad) * half(0x1eee) + half(0xc931)) == bits
            counts["hash"] += 1
            counts["half"] += 1
    assert digest.hexdigest() == "bc48d02cb1c14939dc47b90f870dd689d63e3b1ab69a157e65538db68677a6c8"
    for tiles, splits in itertools.product(list(range(36)) + [80, 320, 512], range(1, 34)):
        visited = []
        for split in range(splits):
            begin, end = tiles * split // splits, tiles * (split + 1) // splits
            k, sequence = begin, []
            while k + 4 <= end:
                sequence.extend(range(k, k + 4))
                k += 4
            sequence.extend(range(k, end))
            assert sequence == list(range(begin, end))
            visited.extend(sequence)
            counts["split"] += 1
        assert visited == list(range(tiles))
    measured = set()
    audit = []
    for turn in ("search", "confirmation"):
        lines = (args.experiment / "reports/gemv-pipeline-raw" / (turn + ".log")).read_text().splitlines()
        assert "PASS gemv_pipeline" in lines and not any(x.startswith("FAIL") for x in lines)
        samples = {}
        results = [fields(x) for x in lines if x.startswith("RESULT ")]
        for line in lines:
            if line.startswith("SAMPLE "):
                r = fields(line)
                samples.setdefault((r["shape"], r["protocol"], r["mode"]), []).append(float(r["us"]))
        for r in results:
            measured.add(tuple(int(r[k]) for k in ("I", "N", "K", "R", "splits")))
            s = sorted(samples[r["shape"], r["protocol"], r["mode"]])
            assert len(s) == 11
            assert [s[0], s[5], s[9]] == [float(r[k]) for k in ("min_us", "median_us", "p90_us")]
        assert len(results) == 60 and sum(map(len, samples.values())) == 660
        audit.append(dict(turn=turn, results=len(results), samples=660))
    geometries = {(8192, 5120), (1280, 32768), (5120, 2304), (5120, 129280)}
    assert measured == {(i, n, 2, 1, 4) for i, n in geometries}
    dimensions = geometries | {(i + d, n) for i, n in geometries for d in (-16, 16)}
    dimensions |= {(i, n + d) for i, n in geometries for d in (-16, 16)}
    for (i, n), bits, cb, rows, total, splits, offset in itertools.product(
            sorted(dimensions), range(1, 9), range(3), range(10), (0, 1, 3, 9), (1, 3, 4, 11, 16, 32), (0, 2)):
        chosen = cb == 2 and bits == 2 and rows == 1 and total >= rows and offset == 0 and splits == 4 and (i, n) in geometries
        expected = (i, n, bits, rows, splits) in measured and cb == 2 and total >= rows and offset == 0
        assert chosen == expected
        counts["guard"] += 1
    for i, n in geometries:
        for k, tn, lane in itertools.product((0, i // 16 - 1), (0, n // 16 - 1), range(32)):
            index = (498 + 16 * lane) // 32 - 16
            for word in (index % 16, (index + 1) % 16):
                address = (k * (n // 16) + tn) * 16 + word
                assert 0 <= address < i * n // 16
            assert k * 16 + 2 * (lane & 3) + 8 + 1 < i
            counts["bounds"] += 1
        for total, row, split, col in itertools.product((1, 3, 9), (0,), range(4), (0, n - 1)):
            assert 0 <= (split * total + row) * n + col < 4 * total * n
    fixtures = [fields(x) for x in (Path(__file__).resolve().parents[2] / "reports/runtime-v9-data/correctness.log").read_text().splitlines() if x.startswith("fixture=")]
    assert len(fixtures) == 82
    hits = [r for r in fixtures if (int(r["I"]), int(r["N"])) in geometries and r["bits"] == r["codebook"] == "2"]
    assert not hits
    print(json.dumps(dict(status="PASS", scope="CPU mathematical model and recorded text only", counts=counts,
                         mul1_sha256=digest.hexdigest(), raw=audit, measured=sorted(measured),
                         v9_fixtures=len(fixtures), v9_guard_hits=len(hits)), indent=2))


if __name__ == "__main__":
    main()
