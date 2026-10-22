"""Audit the actual linked device images; no GPU or original object access."""
import json
from pathlib import Path
import re

isa=Path('build/linked-0.isa').read_text()
notes=Path('build/linked-0.notes').read_text()
metadata={}
for block in notes.split('  - .args:')[1:]:
    name=re.search(r'\.name:\s+(\S+)',block).group(1)
    fields={key:int(re.search(r'\.'+key+r':\s+(\d+)',block).group(1)) for key in
            ('group_segment_fixed_size','private_segment_fixed_size','sgpr_count',
             'vgpr_count','sgpr_spill_count','vgpr_spill_count','wavefront_size')}
    assert fields['wavefront_size']==32
    assert not any(fields[k] for k in ('private_segment_fixed_size','sgpr_spill_count','vgpr_spill_count'))
    metadata[name]=fields
cores=[]
for name,body in re.findall(r'<([^>]+)>:\n(.*?)(?=\n[0-9a-f]+ <|\Z)',isa,re.S):
    if 'decoder_core' not in name:
        continue
    bits,mode,m,x,p=map(int,re.search(r'ILi(\d)ELi(\d)ELi(\d+)ELb(\d)ELi(\d)',name).groups())
    fma=body.count('v_pk_fma_f16')
    wmma=[line.strip().split('//')[0] for line in body.splitlines() if 'v_wmma_f32' in line]
    stores=[line.strip().split('//')[0] for line in body.splitlines() if 'ds_store_b16' in line]
    loads=[line.strip().split('//')[0] for line in body.splitlines() if 'global_load_b32' in line]
    assert body.count('s_barrier')==2 and len(wmma)==m//16
    assert len(stores)==9 # One A instruction in its loop, eight separate Q stores.
    assert metadata[name]['group_segment_fixed_size']==2*(m*(16+p)+1024)
    assert not re.search(r'(scratch_|private_)',body)
    if mode:
        assert fma==4 and len(loads)==3
        assert sum('ds_store_b16_d16_hi' in line for line in stores)==4
        assert 's_waitcnt vmcnt(0)' in body and 's_waitcnt lgkmcnt(0)' in body
        if mode==2:
            assert 'v_readfirstlane_b32' in body
            assert all(re.search(r'global_load_b32 v\d+, v\d+, s\[',line) for line in loads)
    # Each instruction accumulates into its own unchanged register vector.
    for line in wmma:
        operands=re.findall(r'v\[\d+:\d+\]',line)
        assert len(operands)==4 and operands[0]==operands[3]
    cores.append(dict(symbol=name,bits=bits,mode=mode,tile_m=m,xor=bool(x),pad_a=p,
                      resources=metadata[name],half2_fma=fma,CTA_barriers=2,
                      wmma_in_accumulator_order=wmma,separate_half_stores=stores,
                      visible_packed_loads=loads))
assert len(cores)==18
Path('build/isa-audit.json').write_text(json.dumps(dict(status='PASS',kernels=cores),indent=2)+'\n')
print('PASS actual linked ELF: 18 cores, wave32, no private/spills, four packed FMA, eight Q half stores, two barriers')
