"""Inspect the ELF extracted from the ACTUALLY LINKED executable. CPU only."""
from pathlib import Path
import hashlib
import json
import re
import struct
import sys

if '--extract' in sys.argv:
    blob=Path('build/linked.fatbin').read_bytes()
    marker=b'__CLANG_OFFLOAD_BUNDLE__'
    p=blob.index(marker);q=p+len(marker)
    count,=struct.unpack_from('<Q',blob,q);q+=8
    images=[]
    for _ in range(count):
        offset,size,length=struct.unpack_from('<QQQ',blob,q);q+=24
        target=blob[q:q+length].decode();q+=length
        if 'gfx1151' in target:
            image=blob[p+offset:p+offset+size]
            assert image[:4]==b'\x7fELF', 'not a linked ELF'
            images.append((target,image))
    assert len(images)==1, [(t,len(b)) for t,b in images]
    Path('build/linked.elf').write_bytes(images[0][1])
    print('EXACT_LINKED_ELF',images[0][0],hashlib.sha256(images[0][1]).hexdigest())
    sys.exit(0)

notes=Path('build/linked-notes.txt').read_text()
isa=Path('build/linked-isa.txt').read_text()
resources=[]
for record in notes.split('  - .args:')[1:]:
    def field(key):
        match=re.search(r'\.'+re.escape(key)+r':\s+(\S+)',record)
        assert match, key
        return match[1]
    name=field('name')
    assert name in ('trace_inputs','trace_core','trace_outputs'), name
    keys=('kernarg_segment_size','kernarg_segment_align','group_segment_fixed_size',
          'private_segment_fixed_size','wavefront_size','max_flat_workgroup_size',
          'sgpr_count','vgpr_count','sgpr_spill_count','vgpr_spill_count')
    r={k:int(field(k)) for k in keys};r['name']=name
    args=[]
    # Preserve every explicit/hidden argument entry, including offsets and sizes.
    argtext=re.split(r'\n    \.',record,maxsplit=1)[0]
    for a in re.split(r'\n\s*- \.', '\n'+argtext):
        values=dict(re.findall(r'\.?([A-Za-z_]+):\s*([^\n]+)',a))
        if 'offset' in values and 'size' in values:
            args.append({k:v.strip() for k,v in values.items()})
    r['args']=args
    r['explicit_argument_bytes']=2328
    r['extra_trace_argument_bytes']=8
    r['hidden_and_alignment_bytes']=r['kernarg_segment_size']-2328
    resources.append(r)
assert len(resources)==3, resources
print(json.dumps({'resources':resources,'proposed_budget_bytes':4096,
    'budget_is_architectural_fact':False,'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (Path('bin/abi_pilot'),Path('build/linked.elf'))}},indent=2),flush=True)
for r in resources:
    assert r['wavefront_size']==32 and r['max_flat_workgroup_size']>=128,r
    assert r['group_segment_fixed_size']==0,r
    assert r['private_segment_fixed_size']==r['sgpr_spill_count']==r['vgpr_spill_count']==0, ('STOP private/spills',r)
    assert r['kernarg_segment_size']<=4096,r
    args=r['args'];assert len(args)>=2, r
    assert int(args[0]['offset'])==0 and int(args[0]['size'])==2320,r
    assert args[0]['value_kind']=='by_value',r
    assert int(args[1]['offset'])==2320 and int(args[1]['size'])==8,r
    assert args[1]['value_kind']=='global_buffer',r
    for a in args:
        assert int(a['offset'])+int(a['size'])<=r['kernarg_segment_size'],r
assert not re.search(r'\bscratch_|\bs_barrier\b|\bds_(?:load|store)',isa),'STOP scratch/LDS/barrier ISA'
print('PASS linked trace kernels only; future Batch-only math ELFs remain OPEN')
