"""Preserve every paired result, raw sample and loss from both fixed turns."""
import json
import math
from pathlib import Path
import sys

source = Path(sys.argv[1])
target = Path(sys.argv[2])
target.mkdir(parents=True, exist_ok=True)
records, diagnostics, gates, hardware = [], [], [], []
completion = {}
for turn in ('first', 'confirm'):
    lines = (source / f'{turn}.log').read_text().splitlines()
    complete = any(x.startswith('PASS mode=--search') for x in lines)
    completion[turn] = dict(payload_complete=complete,
                            outer_status=(source/f'{turn}.outer').read_text().strip())
    if not complete:
        # Only completed pairs are emitted after full post-checks. A truncated
        # first turn is not a completed campaign and is never silently promoted.
        assert turn == 'first'
        assert 'exit=124' in (source/f'{turn}.outer').read_text()
        assert any(x.startswith('PASS mode=--validate') for x in lines)
        gates.append(dict(turn=turn, raw='INCOMPLETE outer timeout; completed pairs only'))
    assert not any(x.startswith('FAIL') for x in lines), turn
    for line in lines:
        if line.startswith('HARDWARE '):
            hardware.append(dict(turn=turn, raw=line))
        if line.startswith(('GATE ', 'VALIDATE_DONE ', 'PASS ', 'SKIP')):
            gates.append(dict(turn=turn, raw=line))
        if not line.startswith(('RESULT ', 'DIAGNOSTIC ')):
            continue
        row = dict(part.split('=', 1) for part in line.split()[1:])
        row['turn'] = turn
        row['shape'] = ('q_b' if 'wq_b.' in row['path'] else
                        'wo_b' if 'wo_b.' in row['path'] else 'shared')
        for key in ('rows', 'splits', 'gpu_bound', 'cloned_bytes', 'workspace_bytes', 'core_partial_bytes'):
            if key in row:
                row[key] = int(row[key])
        for key in ('min_us', 'median_us', 'p90_us'):
            row[key] = float(row[key])
        row['samples'] = [float(x) for x in row['samples'].split(',') if x]
        assert len(row['samples']) == 11
        ordered = sorted(row['samples'])
        for key, index in (('min_us', 0), ('median_us', 5), ('p90_us', 9)):
            assert math.isclose(row[key], ordered[index], rel_tol=1e-8)
        assert row['correctness'] == row['canaries'] == 'PASS'
        (records if line.startswith('RESULT ') else diagnostics).append(row)

paired = []
for row in records:
    if row['candidate'].startswith('original-'):
        continue
    matches = [r for r in records if r['turn'] == row['turn'] and r['path'] == row['path']
               and r['rows'] == row['rows'] and r['splits'] == row['splits']
               and r['mode'] == row['mode'] and r['pair'] == row['pair']
               and r['candidate'].startswith('original-')]
    assert len(matches) == 1
    baseline = matches[0]
    paired.append(dict(turn=row['turn'], shape=row['shape'], rows=row['rows'],
                       splits=row['splits'], mode=row['mode'], candidate=row['candidate'],
                       ratio=row['median_us']/baseline['median_us'],
                       baseline=baseline, measured=row))

confirmed = []
for row in paired:
    if row['turn'] != 'first':
        continue
    match = [r for r in paired if r['turn'] == 'confirm' and all(r[k] == row[k]
             for k in ('shape', 'rows', 'splits', 'mode', 'candidate'))]
    assert len(match) <= 1
    if match and row['ratio'] <= .98 and match[0]['ratio'] <= .98:
        confirmed.append(dict(shape=row['shape'], rows=row['rows'], splits=row['splits'],
                              mode=row['mode'], candidate=row['candidate'],
                              first_ratio=row['ratio'], confirm_ratio=match[0]['ratio']))

result = dict(records=records, diagnostics=diagnostics, paired=paired,
              confirmed=confirmed, gates=gates, hardware=hardware, completion=completion)
if all(r['payload_complete'] for r in completion.values()):
    assert len(records) == 432 and len(paired) == 216 and len(diagnostics) == 72
try:
    import psutil
    result['analysis_process_memory_bytes'] = psutil.Process().memory_info()._asdict()
except ImportError:
    pass
(target/'results.json').write_text(json.dumps(result, indent=2)+'\n')
table = ['| Forma | M | Variante | Core T1/T2 | Operador T1/T2 |',
         '|---|---:|---|---:|---:|']
for shape in ('q_b', 'wo_b', 'shared'):
    for rows in (64, 128, 256):
        for decoder in ('slow', 'fast'):
            for bk in (16, 32, 64):
                candidate = f'M64-BK{bk}-{decoder}'
                cells = []
                for mode in ('core', 'device_operator'):
                    values = []
                    for turn in ('first', 'confirm'):
                        match = [r for r in paired if r['shape'] == shape and r['rows'] == rows
                                 and r['candidate'] == candidate and r['mode'] == mode and r['turn'] == turn]
                        assert len(match) <= 1
                        values.append(f"{match[0]['ratio']:.4f}" if match else 'omitido')
                    cells.append(' / '.join(values))
                table.append(f'| {shape} | {rows} | BK{bk} {decoder} | {cells[0]} | {cells[1]} |')
(target/'ratios.md').write_text('\n'.join(table)+'\n')
table = ['| Forma | M | Core T1/T2 | Operador T1/T2 |',
         '|---|---:|---:|---:|']
for shape in ('q_b', 'wo_b', 'shared'):
    for rows in (64, 128, 256):
        cells = []
        for mode in ('core', 'device_operator'):
            values = []
            for turn in ('first', 'confirm'):
                subset = [r for r in diagnostics if r['shape'] == shape and r['rows'] == rows
                          and r['mode'] == mode and r['turn'] == turn]
                checked = next(r for r in subset if r['launcher'] == 'checked')
                lean = next(r for r in subset if r['launcher'] == 'prevalidated')
                values.append(f"{checked['median_us']/lean['median_us']:.4f}")
            cells.append(' / '.join(values))
        table.append(f'| {shape} | {rows} | {cells[0]} | {cells[1]} |')
(target/'launcher-ratios.md').write_text('\n'.join(table)+'\n')
print(json.dumps(dict(records=len(records), pairs=len(paired), diagnostics=len(diagnostics),
                      confirmed=confirmed, omissions=[g for g in gates if g['raw'].startswith('SKIP')]), indent=2))
