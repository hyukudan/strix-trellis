"""Inspect the candidate code object extracted from the exact linked executable."""
from pathlib import Path
import hashlib
import json
import re
import sys

build=Path(sys.argv[1])
notes=(build/'linked-notes.txt').read_text()
isa=(build/'linked-isa.txt').read_text()
parts=re.split(r'\n[0-9a-f]+ <([^>]+)>:\n',isa)
bodies=dict(zip(parts[1::2],parts[2::2]))
rows=[]
for metadata in notes.split('    .group_segment_fixed_size:')[1:]:
    name=re.search(r'\.name:\s+(\S+)',metadata).group(1)
    bits,mode=map(int,re.search(r'bothILi([56])ELNS0_4ModeE([012])',name).groups())
    value=lambda k:int(re.search(r'\.'+k+r':\s+(\d+)',metadata).group(1))
    static=int(metadata.split()[0]);body=bodies[name]
    assert value('max_flat_workgroup_size')==1024
    assert value('wavefront_size')==32
    assert value('private_segment_fixed_size')==0
    assert value('sgpr_spill_count')==value('vgpr_spill_count')==0
    assert static==(0 if mode==0 else 2048)
    assert body.count('s_barrier')==2
    assert len(re.findall(r'\b(?:v_|v_dual_)dot2acc_f32_f16',body))==4
    assert 'ds_load_2addr_b32' in body and 'ds_load_u16' not in body
    assert len(re.findall(r'global_load_b32[^\n]*s\[',body))==3
    assert body.count('v_pk_mul_lo_u16')==8
    assert 'v_mad_u32_u16' in body and 'v_pk_fma_f16' in body
    assert 'v_fma_mixlo_f16' not in body and 'v_fma_mixhi_f16' not in body
    assert 'v_fma_mix_f32' in body and 'v_cvt_f16_f32' in body
    assert '3DB504F3' in body and 's_waitcnt' in body
    assert not re.search(r'\bscratch_',body)
    if mode:
        assert re.search(r'add_f32[^\n]*, 0, v',body)
        assert 's_cmp_eq_f32' in body and 's_cmp_neq_f32' in body
    # Every backedge in these kernels is a scalar loop. Neither variable loop
    # encloses a barrier; branch target addresses come from this linked ELF.
    instructions=[]
    for line in body.splitlines():
        match=re.search(r'// ([0-9A-F]+):',line)
        if match:instructions.append((int(match.group(1),16),line))
    backedges=[]
    for pc,line in instructions:
        target=re.search(r'\bs_cbranch_\w+[^<]* ([0-9a-f]+) <',line)
        if target and int(target.group(1),16)<pc:
            dest=int(target.group(1),16)
            assert not any('s_barrier' in text for addr,text in instructions if dest<=addr<=pc)
            backedges.append([hex(pc),hex(dest)])
    rows.append(dict(bits=bits,mode=mode,static_bytes=static,dynamic_max=16384,
        total_max=static+16384,threads=1024,wave=32,private=0,spills=0,
        vgpr=value('vgpr_count'),sgpr=value('sgpr_count'),barriers=2,
        dot2_per_iteration=4,backedges=backedges,
        body_sha256=hashlib.sha256(body.encode()).hexdigest()))
assert len(rows)==6
print(json.dumps(dict(images=rows,linked_notes_sha256=hashlib.sha256(notes.encode()).hexdigest(),
    linked_isa_sha256=hashlib.sha256(isa.encode()).hexdigest(),
    limitation='Emission checks and source/CFG review; no occupancy inference or physical qualification'),indent=2))
