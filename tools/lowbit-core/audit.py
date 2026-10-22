"""Read-only mechanical gates on the linked candidate ELF disassembly/metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import re

p=argparse.ArgumentParser();p.add_argument('isa',type=Path);p.add_argument('notes',type=Path)
p.add_argument('source',type=Path);args=p.parse_args()
isa=args.isa.read_text();notes=args.notes.read_text();source=args.source.read_text()
staged=source[source.index('__device__ __forceinline__ void staged'):source.index('template<int K,bool Coop> __device__ __forceinline__ void consume')]
assert staged.count('"=&v"')==16
assert staged.count('v_pk_mul_lo_u16')==8 and staged.count('v_mad_u32_u16')==8
assert not re.search(r'load|store|wait|exec|memory',staged.split('asm(')[1].split(');')[0],re.I)
assert '__syncthreads' not in source and '__shared__' not in source
assert not re.search(r'\b(?:scratch_|s_barrier|ds_write|ds_read)',isa)
resources=[]
for record in notes.split('  - .args:')[1:]:
    name=re.search(r'\.name:\s+(\S+)',record).group(1)
    if '6kernelI' not in name:continue
    values={key:int(re.search(r'\.'+key+r':\s+(\d+)',record).group(1)) for key in
            ('group_segment_fixed_size','private_segment_fixed_size','sgpr_count','vgpr_count',
             'sgpr_spill_count','vgpr_spill_count','wavefront_size','max_flat_workgroup_size')}
    assert values['wavefront_size']==32 and values['max_flat_workgroup_size']==128
    for key in ('group_segment_fixed_size','private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count'):
        assert values[key]==0
    resources.append(dict(symbol=name,**values))
assert len(resources)==5
print(json.dumps(dict(passed=True,resources=resources,sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest()
      for f in (args.isa,args.notes,args.source)},
      scope='Mechanical gates only; control flow, hoisting and sender provenance also inspected manually.'),indent=2))
