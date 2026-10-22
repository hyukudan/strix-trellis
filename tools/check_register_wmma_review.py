#!/usr/bin/env python3
"""Independent bounded CPU review of register-WMMA contracts; no HIP imports."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import struct


def coordinates(state):
    group, part = divmod(state, 8)
    return (2 * (group % 4) + part % 2 + 8 * ((part // 2) % 2),
            group // 4 + 8 * (part // 4))


def check_mapping():
    inverse = {coordinates(s): s for s in range(256)}
    assert len(inverse) == 256
    summaries = []
    for bits in (5, 6):
        owners, shifts, zero_high = [], set(), 0
        word_indices = []
        for lane in range(32):
            base = (lane & 7) * bits - 1
            loaded = [(base + j) % (8 * bits) for j in range(bits + 1)]
            assert -1 <= base and base + bits < 8 * bits
            assert all(base + j >= 0 for j in range(1, bits + 1))
            word_indices.extend(loaded)
            for pair in range(4):
                state = (32 * (lane & 7) + 2 * (lane >> 4)
                         + 4 * ((lane >> 3) & 1) + 8 * pair)
                for half in (0, 1):
                    assert inverse[8 * (lane // 16) + 2 * pair + half,
                                   lane % 16] == state + half
                    owners.append(state + half)
                # This selector has no input data dependency. Trace each output
                # bit to a physical source bit, without evaluating packed data.
                constant = 16 + (1 + 8 * pair) * bits
                start = constant + (2 * (lane >> 4) + 4 * ((lane >> 3) & 1)) * bits
                first = constant // 32
                select = int(start >= 32 * (first + 1))
                assert start // 32 == first + select
                index, shift = first + select, start % 32
                assert 0 <= index <= bits and 0 < shift < 32
                shifts.add(shift)
                if index + 1 > bits:
                    zero_high += 1
                    assert shift + 16 + bits <= 32
                for half in (0, 1):
                    for output_bit in range(16):
                        window_bit = shift + half * bits + output_bit
                        word = index + window_bit // 32
                        assert 0 <= word <= bits
                        physical = 32 * loaded[word] + window_bit % 32
                        expected = ((inverse[8 * (lane // 16) + 2 * pair + half,
                                             lane % 16] + 1) * bits - 16 + output_bit) % (256 * bits)
                        assert physical == expected
        assert Counter(owners) == Counter(range(256))
        # Model an XOR16 shuffle of each source's own P, followed by ordering.
        owned = [[inverse[8 * (lane // 16) + k, lane % 16] for k in range(8)]
                 for lane in range(32)]
        fragments = []
        for lane in range(32):
            peer = owned[lane ^ 16]
            fragment = owned[lane] + peer if lane < 16 else peer + owned[lane]
            assert fragment == [inverse[k, lane % 16] for k in range(16)]
            fragments.append(fragment)
        assert fragments[:16] == fragments[16:]
        summaries.append(dict(bits=bits, owned_states=len(owners),
                              source_bit_checks=32 * 4 * 2 * 16,
                              shifts=sorted(shifts), zero_high_pairs=zero_high,
                              loaded_word_range=[min(word_indices), max(word_indices)],
                              requested_words_per_wave=32 * (bits + 1)))
    return summaries


def check_a_stores_splits():
    a_reads = stores = 0
    for rows in range(1, 17):
        cells = [(2 * j + lane // 16, lane % 16)
                 for lane in range(32) for j in range(8)
                 if 2 * j + lane // 16 < rows]
        assert Counter(cells) == Counter((r, c) for r in range(rows) for c in range(16))
        stores += len(cells)
        for inputs in (128, 256, 5120):
            for tile in range(inputs // 16):
                for lane in range(32):
                    row = lane % 16
                    if row < rows:
                        offset = (row * inputs + tile * 16) * 2
                        assert offset % 32 == 0
                        assert 0 <= offset and offset + 32 <= rows * inputs * 2
                        a_reads += 1
                    # An inactive row must take the zero branch before loading.
    partitions = empty = 0
    for tiles in range(1, 65):
        for splits in range(1, 97):
            spans = [(tiles * s // splits, tiles * (s + 1) // splits)
                     for s in range(splits)]
            assert spans[0][0] == 0 and spans[-1][1] == tiles
            assert all(spans[s][1] == spans[s + 1][0] for s in range(splits - 1))
            assert [k for begin, end in spans for k in range(begin, end)] == list(range(tiles))
            empty += sum(begin == end for begin, end in spans)
            partitions += 1
    max_dimension = ((1 << 31) - 1) // 128 * 128
    # Boundary arithmetic for the proposed R<=16 domain, not allocation tests.
    assert (max_dimension // 16) * ((1 << 31) - 1) < (1 << 63)
    assert max_dimension * max_dimension * 6 // 8 < (1 << 64)
    assert 16 * max_dimension * 4 * 32 < (1 << 64)
    oversized_workspace = 16 * max_dimension * 4 * ((1 << 31) - 1)
    assert oversized_workspace >= (1 << 64)
    return dict(rows=list(range(1, 17)), a_vector_reads=a_reads,
                unique_predicated_stores=stores, split_partitions=partitions,
                empty_spans=empty, arithmetic_only_max_dimension=max_dimension,
                rejected_unbounded_split_workspace_bytes=oversized_workspace)


def check_mul1_endian():
    to_half = lambda x: struct.pack('<e', x)
    from_half = lambda b: struct.unpack('<e', struct.pack('<H', b))[0]
    scale, bias = from_half(0x1EEE), from_half(0xC931)
    assert scale == 1774 / 262144 and bias == -1329 / 128
    raw = bytearray()
    byte_sums = set()
    for state in range(65536):
        product = state * 0x83DCD12D & 0xFFFFFFFF
        total = sum(product.to_bytes(4, 'little'))
        byte_sums.add(total)
        operand = from_half(0x6400 + total)
        assert operand == 1024 + total
        exact = ((1024 + total) * 1774 - 1329 * 2048) / 262144
        assert operand * scale + bias == exact
        # The exact dyadic numerator fits FP32: the old FP32 FMA introduces no
        # rounding before RN16 for mul1. This does not execute device hfma2.
        assert struct.unpack('<f', struct.pack('<f', exact))[0] == exact
        value = to_half(exact)
        raw.extend(value)
        # Exhaust every state in each position of a deliberately unequal pair.
        other = to_half(-exact if exact else 1.0)
        for low, high in ((value, other), (other, value)):
            word = int.from_bytes(low + high, 'little')
            assert word & 65535 == int.from_bytes(low, 'little')
            assert word >> 16 == int.from_bytes(high, 'little')
            assert struct.pack('<I', word) == low + high
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == 'bc48d02cb1c14939dc47b90f870dd689d63e3b1ab69a157e65538db68677a6c8'
    # Logical words serialize low16 first, while bit extraction starts at bit31.
    word = 0x0123ABCD
    assert struct.pack('<I', word) == struct.pack('<HH', 0xABCD, 0x0123)
    return dict(states=65536, half_pair_checks=131072, sha256=digest,
                observed_byte_sum_range=[min(byte_sums), max(byte_sums)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    inputs = ['AGENTS.md', 'docs/DESIGN.md', 'docs/FORMAT.md', 'docs/NUMERICS.md',
              'docs/REGISTER_WMMA.md', 'tools/check_register_wmma_map.py',
              'reports/register-wmma-map.json', 'src/gemm.hip',
              'src/trellis_optimized.hpp', 'src/gemv_lds.hip',
              'src/trellis_device.hpp', 'src/backend.hip', 'src/launch_policy.hpp',
              'src/wmma_probe.hip', 'reports/wmma-debug.md']
    result = dict(status='passed_cpu_only', gpu_executed=False,
                  python=platform.python_version(), platform=platform.platform(),
                  mapping=check_mapping(), a_stores_splits=check_a_stores_splits(),
                  mul1_endian=check_mul1_endian(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  input_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                                for p in inputs})
    output = json.dumps(result, indent=2) + '\n'
    if args.json:
        args.json.write_text(output, encoding='utf-8', newline='\n')
    print(output, end='')


if __name__ == '__main__':
    main()
