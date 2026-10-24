#!/usr/bin/env python3
"""Inspect linked dense GEMV images without opening a GPU device."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('binary', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--llvm', type=Path, required=True)
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    binary = out / args.binary.name
    shutil.copy2(args.binary, binary)
    result = subprocess.check_output([str(args.llvm / 'llvm-objdump'), '--offloading', str(binary)], cwd=out, text=True)
    (out / 'offloading.txt').write_text(result)
    records = []
    counts = {}
    for image in sorted(out.glob('*hip-amdgcn*')):
        if not image.is_file() or image.open('rb').read(4) != b'\x7fELF':
            continue
        metadata = subprocess.check_output([str(args.llvm / 'llvm-readelf'), '-n', str(image)], text=True)
        if 'dense_r1' not in metadata:
            continue
        (out / (image.name + '.metadata')).write_text(metadata)
        isa = subprocess.check_output([str(args.llvm / 'llvm-objdump'), '-d', str(image)], text=True)
        (out / (image.name + '.isa')).write_text(isa)
        for op in ('s_waitcnt_vscnt', 'buffer_gl0_inv', 'buffer_gl1_inv', 'global_atomic_add_u32', 'v_(?:dual_)?dot2(?:acc)?_f32_f16'):
            counts[op] = counts.get(op, 0) + len(re.findall(r'\b' + op + r'\b', isa))
        for block in metadata.split('  - .args:')[1:]:
            name = re.search(r'\.name:\s+(\S+)', block)
            if not name or 'dense_r1' not in name[1]:
                continue
            record = {'name': name[1], 'image': image.name}
            for field in ('vgpr_count', 'sgpr_count', 'group_segment_fixed_size', 'private_segment_fixed_size',
                          'vgpr_spill_count', 'sgpr_spill_count', 'wavefront_size', 'max_flat_workgroup_size'):
                match = re.search(r'\.' + field + r':\s+(\d+)', block)
                if match:
                    record[field] = int(match[1])
            shape = re.search(r'projectionILi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)ELi(\d+)', name[1])
            if shape:
                record.update(zip(('input', 'splits', 'bits', 'rows', 'columns', 'prefetch'), map(int, shape.groups())))
                half = re.search(r'ELi\d+ELb([01])E', name[1])
                record['outputs_per_wave'] = 8 if half and half[1] == '1' else 16
            records.append(record)
    need = ('private_segment_fixed_size', 'vgpr_spill_count', 'sgpr_spill_count')
    spills = [r for r in records if any(r.get(f, -1) != 0 for f in need)]
    result = {'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(), 'kernels': len(records),
              'max_vgpr': max((r['vgpr_count'] for r in records), default=0),
              'max_lds': max((r['group_segment_fixed_size'] for r in records), default=0),
              'spills': spills, 'instructions': counts, 'resources': records}
    (out / 'resources.json').write_text(json.dumps(result, indent=2) + '\n')
    assert records and all(r['wavefront_size'] == 32 for r in records)
    assert all(r['max_flat_workgroup_size'] == r['columns'] * (4 if r['outputs_per_wave']==8 else 2) for r in records)
    assert not spills, spills
    assert all(counts.values()), counts
    print(json.dumps({k: v for k, v in result.items() if k != 'resources'}, indent=2))
    print('PASS dense_r1_linked_images')


if __name__ == '__main__':
    main()
