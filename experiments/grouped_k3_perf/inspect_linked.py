"""Inspect all images from the linked executable's fatbin. CPU only."""
from pathlib import Path
import hashlib
import json
import re
import struct
import sys

NAMES = ('grouped_input_transform', 'grouped_baseline_core', 'grouped_output_transform')
if '--extract' in sys.argv:
    blob = Path('build/linked.fatbin').read_bytes()
    marker = b'__CLANG_OFFLOAD_BUNDLE__'
    pos = 0
    images = []
    while True:
        p = blob.find(marker, pos)
        if p < 0:
            break
        q = p + len(marker)
        count, = struct.unpack_from('<Q', blob, q)
        q += 8
        end = q
        for _ in range(count):
            offset, size, length = struct.unpack_from('<QQQ', blob, q)
            q += 24
            target = blob[q:q+length].decode()
            q += length
            end = max(end, p+offset+size)
            if 'gfx1151' in target:
                image = blob[p+offset:p+offset+size]
                assert image[:4] == b'\x7fELF'
                path = Path(f'build/linked-{len(images):02d}.elf')
                path.write_bytes(image)
                images.append(dict(path=str(path), target=target, bytes=len(image),
                    sha256=hashlib.sha256(image).hexdigest()))
        pos = end
    assert images
    Path('build/images.json').write_text(json.dumps(images, indent=2)+'\n')
    print(json.dumps(images, indent=2))
    sys.exit(0)

images = json.loads(Path('build/images.json').read_text())
resources = []
counts = []
for image in images:
    path = Path(image['path'])
    notes = path.with_suffix('.notes').read_text()
    isa = path.with_suffix('.isa').read_text()
    count = 0
    for record in notes.split('  - .args:')[1:]:
        def field(key):
            match = re.search(r'\.'+re.escape(key)+r':\s+(\S+)', record)
            assert match, key
            return match[1]
        name = field('name')
        count += 1
        selected = [n for n in NAMES if n in name]
        if not selected:
            continue
        assert len(selected) == 1
        keys = ('kernarg_segment_size','kernarg_segment_align','group_segment_fixed_size',
                'private_segment_fixed_size','wavefront_size','max_flat_workgroup_size',
                'sgpr_count','vgpr_count','sgpr_spill_count','vgpr_spill_count')
        r = {k:int(field(k)) for k in keys}
        r.update(name=name, stage=selected[0], image=image)
        argtext = re.split(r'\n    \.', record, maxsplit=1)[0]
        args = []
        for a in re.split(r'\n\s*- \.', '\n'+argtext):
            values = dict(re.findall(r'\.?([A-Za-z_]+):\s*([^\n]+)',a))
            if 'offset' in values and 'size' in values:
                args.append({k:v.strip() for k,v in values.items()})
        r['args'] = args
        r['hidden_reserved_alignment_bytes'] = r['kernarg_segment_size'] - 2320
        # Preserve each function's actual instructions for source/ISA review.
        begin = isa.index('<'+name+'>:')
        tail = isa[begin:]
        next_function = re.search(r'\n[0-9a-f]+ <', tail[1:])
        function = tail[:next_function.start()+1] if next_function else tail
        Path('build/'+selected[0]+'.isa').write_text(function)
        r['forbidden_isa'] = re.findall(r'\bscratch_\w+|\bs_barrier\b|\bds_(?:load|store)\w+', function)
        resources.append(r)
    counts.append(dict(image=image['path'], total_kernels=count))

violations = []
if sorted(r['stage'] for r in resources) != sorted(NAMES):
    violations.append('missing/duplicate mathematical functions')
for r in resources:
    args = r['args']
    if not args or int(args[0]['offset']) != 0 or int(args[0]['size']) != 2320 or args[0]['value_kind'] != 'by_value':
        violations.append(r['stage']+': Batch ABI')
    if any(not a['value_kind'].startswith('hidden_') for a in args[1:]):
        violations.append(r['stage']+': extra explicit argument')
    if r['wavefront_size'] != 32 or r['max_flat_workgroup_size'] != 128:
        violations.append(r['stage']+': wave/block')
    if r['kernarg_segment_size'] > 4096 or r['kernarg_segment_align'] != 8:
        violations.append(r['stage']+': proposed kernarg budget/alignment')
    if any(r[k] for k in ('group_segment_fixed_size','private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count')) or r['forbidden_isa']:
        violations.append(r['stage']+': STOP LDS/private/scratch/spills/barrier')
    if any(int(a['offset'])+int(a['size']) > r['kernarg_segment_size'] for a in args):
        violations.append(r['stage']+': argument extent')
print(json.dumps(dict(resources=resources, image_counts=counts, violations=violations,
    linked_binary_sha256=hashlib.sha256(Path('bin/perf_native').read_bytes()).hexdigest(),
    proposed_budget=4096, budget_is_hardware_limit=False,
    scalar_selection_review='OPEN: inspect emitted per-function ISA',
    native_admission='OPEN'), indent=2), flush=True)
if violations:
    raise SystemExit(2)
