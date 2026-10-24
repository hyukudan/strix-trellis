#!/usr/bin/env python3
"""Generate the dense GEMV profile from summarized project-owned HOT timings."""
import argparse
import json
from pathlib import Path

SHAPES = {
    'q_a': (5120, 1280), 'kv': (5120, 512), 'q_b': (1280, 16384),
    'o_b': (4096, 5120), 'shared_gate': (5120, 1152),
    'shared_down': (1152, 5120), 'head': (5120, 64640),
}


def generate(points):
    groups = {}
    for point in points:
        group = groups.setdefault((point['site'], point['splits']), {})
        key = point['bits'], point['rows']
        if key in group:
            raise ValueError(f'duplicate point: {key}')
        group[key] = point
    lines = [
        '#pragma once', 'namespace strix_trellis {',
        'struct DenseR1Schedule { int columns, prefetch, outputs_per_wave; bool use_splitk04; };',
        '// Empirical HOT schedule; numerical identity is fixed separately by splits.',
        '// Odd row counts borrow the next even-row schedule; correctness is qualified independently.',
        'inline DenseR1Schedule dense_r1_schedule(int input,int output,int bits,int rows,int splits) {',
        '    if((bits!=4 && bits!=5) || rows<1 || rows>8) return {64,2,16,false};',
    ]
    for (site, splits), cells in sorted(groups.items()):
        input_size, output_size = SHAPES[site]
        required = {(bits, rows) for bits in (4, 5) for rows in (1, 2, 4, 6, 8)}
        if set(cells) != required:
            raise ValueError(f'incomplete profile for {site}, S{splits}')
        legacy = 32 if site == 'kv' else 16 if site in ('q_a', 'shared_gate') else 4
        lines += [f'    if(input=={input_size} && output=={output_size} && splits=={splits}) {{',
                  '        static constexpr DenseR1Schedule table[2][8] = {']
        for bits in (4, 5):
            row = []
            for rows in (1, 2, 4, 4, 6, 6, 8, 8):
                point = cells[bits, rows]
                retain = (splits == legacy and 'splitk04_us' in point and
                          point['splitk04_us'] < 0.97 * point['median_us'])
                physical = ','.join(str(point[key]) for key in
                                    ('columns', 'prefetch', 'outputs_per_wave'))
                row.append('{' + physical + ',' + str(retain).lower() + '}')
            lines.append('            {' + ','.join(row) + '},')
        lines += ['        };', '        return table[bits-4][rows-1];', '    }']
    lines += ['    return {output<=512?32:64,2,16,false};', '}',
              '} // namespace strix_trellis', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('points', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(generate(json.loads(args.points.read_text())))
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
