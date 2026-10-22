"""Small CPU proof for the proposed TP partition contract; no decoder/GPU model."""

import struct
from collections import Counter


def intervals(tiles, splits):
    return [(tiles * s // splits, tiles * (s + 1) // splits)
            for s in range(splits)]


def cropped_h_tiles(begin, end):
    # Transform complete H128 blocks, but consume only the logical K16 interval.
    return [k for h in range(begin // 8, (end + 7) // 8)
            for k in range(8 * h, 8 * h + 8) if begin <= k < end]


def prove_partition(tiles, splits):
    actual = intervals(tiles, splits)
    # Independent ownership formula: the first split whose right edge exceeds k.
    owners = [(splits * (k + 1) - 1) // tiles for k in range(tiles)]
    all_consumed = []
    transformed = Counter()
    for s, (begin, end) in enumerate(actual):
        expected = [k for k, owner in enumerate(owners) if owner == s]
        consumed = cropped_h_tiles(begin, end)
        assert consumed == expected, (tiles, splits, s, consumed, expected)
        assert consumed == list(range(begin, end))
        assert 0 <= begin <= end <= tiles
        for h in range(begin // 8, (end + 7) // 8):
            assert 0 <= h < tiles // 8
            transformed[h] += 1
        all_consumed.extend(consumed)
    assert all_consumed == list(range(tiles))
    assert Counter(all_consumed) == Counter(range(tiles))
    return sum(count - 1 for count in transformed.values())


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def fold(values):
    result = f32(0.0)
    for value in values:
        result = f32(result + value)
    return result


def main():
    # Forty q_b split configurations, including non-H128-aligned boundaries.
    duplicates = [prove_partition(80, splits) for splits in range(1, 41)]
    assert any(duplicates)
    qb = intervals(80, 4)
    assert qb == [(0, 20), (20, 40), (40, 60), (60, 80)]
    old_h = [(8 * (10 * s // 4), 8 * (10 * (s + 1) // 4))
             for s in range(4)]
    assert old_h == [(0, 16), (16, 40), (40, 56), (56, 80)]
    assert qb != old_h
    assert prove_partition(80, 4) == 2  # H2 and H7 are transformed twice.

    boundary_terms = [0.0] * 80
    boundary_terms[15:19] = [float(2**25), 1.0, -float(2**25), 1.0]
    canonical_boundary = fold([fold(boundary_terms[b:e]) for b, e in qb])
    old_boundary = fold([fold(boundary_terms[b:e]) for b, e in old_h])
    assert (canonical_boundary, old_boundary) == (1.0, 0.0)

    # Main TP shapes, alternative N half, and next integrator row classes.
    shapes = [(1280, 16384, 4), (4096, 5120, 4), (8192, 2560, 4),
              (5120, 64640, 4), (4096, 1024, 16),
              (5120, 2304, 16), (2304, 5120, 4)]
    for width, output, splits in shapes:
        prove_partition(width // 16, splits)
        assert output % 128 == 0
        assert sum(end - begin for begin, end in intervals(width // 16, splits)) == width // 16

    # Logical P slots must survive physical G2/G4 scheduling at every row count.
    # This proves ownership/order only, not native fdot2 arithmetic or row bits.
    for rows in range(1, 9):
        for groups in (2, 4):
            slots = {(s, r): None for s in range(4) for r in range(rows)}
            for phase in range(4 // groups):
                for group in range(groups):
                    s = group + phase * groups
                    for r in range(rows):
                        assert slots[s, r] is None
                        slots[s, r] = (s, r)
            for r in range(rows):
                assert [slots[s, r] for s in range(4)] == [(s, r) for s in range(4)]

    # Same number of partials is insufficient: changing grouping changes FP32.
    partial = [float(2**25), 1.0, -float(2**25), 1.0]
    ascending = fold(partial)
    adjacent_groups = fold([fold(partial[:2]), fold(partial[2:])])
    interleaved_groups = fold([fold(partial[::2]), fold(partial[1::2])])
    assert (ascending, adjacent_groups, interleaved_groups) == (1.0, 0.0, 2.0)

    # Producer order cannot be changed merely because every tile is visited once.
    tiles = [0.0] * 80
    tiles[16:20] = partial
    canonical = fold([fold(tiles[begin:end]) for begin, end in qb])
    reversed_inner = fold([fold(list(reversed(tiles[begin:end]))) for begin, end in qb])
    assert canonical != reversed_inner
    print('PASS: 40 q_b split configurations; exact coverage and H128 crop/order')
    print('PASS: 7 canonical shapes; G2/G4 logical P ownership at R1..R8')
    print('PASS: FP32 counterexamples: ascending=1, adjacent=0, interleaved=2')
    print('PASS: q_b boundary counterexample: canonical=1, old H partition=0')
    print('LIMIT: no native fdot2, packed decode, GPU row-invariance or performance proof')


if __name__ == '__main__':
    main()
