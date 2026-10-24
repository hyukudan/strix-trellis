#!/usr/bin/env python3
"""Summarize project-owned dense GEMV HOT operator logs."""
import argparse
import csv
import json
from pathlib import Path


def parse(paths):
    records = []
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.startswith('DENSE_RESULT '):
                continue
            fields = dict(part.split('=', 1) for part in line.split()[1:] if '=' in part)
            for key in ('bits', 'rows', 'tile', 'splits', 'columns', 'prefetch', 'outputs_per_wave', 'samples'):
                if key in fields:
                    fields[key] = int(fields[key])
            for key in ('median_us', 'pmax_us'):
                fields[key] = float(fields[key])
            fields['log'] = str(path)
            records.append(fields)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('logs', type=Path, nargs='+')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    records = parse(args.logs)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'records.json').write_text(json.dumps(records, indent=2) + '\n')
    groups = {}
    baselines = {}
    api_values = {}
    for r in records:
        if r['backend'] == 'dense_r1_api':
            api_values.setdefault((r['site'],r['bits'],r['rows'],r['splits']),[]).append(r['median_us'])
        if r['backend'] != 'dense_r1':
            baselines.setdefault((r['site'], r['bits'], r['rows'], r['backend']), []).append(r['median_us'])
        else:
            groups.setdefault((r['site'], r['bits'], r['rows'], r['splits']), []).append(r)
    points = []
    for key, variants in sorted(groups.items()):
        full = [r for r in variants if r['tile'] == r['rows']]
        best = min(full, key=lambda r: r['median_us'])
        any_tile = min(variants, key=lambda r: r['median_us'])
        point = {k: best[k] for k in ('site', 'bits', 'rows', 'splits', 'tile', 'columns', 'prefetch', 'outputs_per_wave', 'median_us')}
        point.update(any_tile_us=any_tile['median_us'], any_tile=any_tile['tile'])
        values=sorted(api_values.get(key,[]))
        if values:
            point['dense_r1_api_us']=values[len(values)//2]
        for backend in ('v12', 'v16_api', 'splitk04', 'tp03'):
            values = baselines.get((*key[:3], backend), [])
            if values:
                values.sort()
                point[backend + '_us'] = values[len(values)//2]
        points.append(point)
    (args.out / 'best-full-rows.json').write_text(json.dumps(points, indent=2) + '\n')
    keys = ['site', 'bits', 'rows', 'splits', 'tile', 'columns', 'prefetch', 'outputs_per_wave', 'median_us',
            'any_tile_us', 'any_tile', 'v12_us', 'v16_api_us', 'splitk04_us', 'tp03_us', 'dense_r1_api_us']
    with (args.out / 'points.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(points)
    print(json.dumps({'results': len(records), 'groups': len(points), 'out': str(args.out)}, indent=2))


if __name__ == '__main__':
    main()
