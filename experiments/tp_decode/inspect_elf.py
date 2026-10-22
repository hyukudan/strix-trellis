"""Whole linked GPU ELF identity, emitted inventory/resources and raw ISA, CPU only."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import struct
import subprocess
import sys
build=Path(sys.argv[1]);llvm='/opt/rocm/core-10.1/lib/llvm/bin'
def sha(b):return hashlib.sha256(b).hexdigest()
def extract(blob,prefix):
    marker=b'__CLANG_OFFLOAD_BUNDLE__';pos=0;images=[]
    while True:
        p=blob.find(marker,pos)
        if p<0:break
        q=p+len(marker);count,=struct.unpack_from('<Q',blob,q);q+=8;end=q
        assert 0<count<100
        for _ in range(count):
            offset,size,length=struct.unpack_from('<QQQ',blob,q);q+=24
            target=blob[q:q+length].decode();q+=length;end=max(end,p+offset+size)
            if 'gfx1151' in target:
                image=blob[p+offset:p+offset+size];assert image[:4]==b'\x7fELF'
                path=build/f'{prefix}-{len(images):02d}.elf';path.write_bytes(image)
                images.append(dict(path=path.name,bytes=len(image),sha256=sha(image),target=target))
        pos=end
    return images
linked=extract((build/'linked.fatbin').read_bytes(),'linked')
old=extract(Path('lib/libstrix_trellis_v16.a').read_bytes(),'old')
assert len(old)==11,('actual V16 old image inventory',len(old))
assert not (Counter(x['sha256'] for x in old)-Counter(x['sha256'] for x in linked)), 'old whole ELF mutation'
oldhash={x['sha256'] for x in old};new=[x for x in linked if x['sha256'] not in oldhash]
resources=[]
for image in linked:
    path=build/image['path']
    notes=subprocess.check_output([llvm+'/llvm-readelf','--notes',path]).decode()
    isa=subprocess.check_output([llvm+'/llvm-objdump','-d','--mcpu=gfx1151',path]).decode()
    path.with_suffix('.notes').write_text(notes);path.with_suffix('.isa').write_text(isa)
    if image['sha256'] in oldhash:continue
    for record in notes.split('  - .args:')[1:]:
        def field(key):
            match=re.search(r'\.'+re.escape(key)+r':\s+(\S+)',record);assert match,key;return match[1]
        name=field('name');demangled=subprocess.check_output([llvm+'/llvm-cxxfilt',name]).decode().strip()
        if 'tp_decode::' not in demangled:raise ValueError('Unexpected new function '+demangled)
        r={k:int(field(k)) for k in ('group_segment_fixed_size','private_segment_fixed_size','wavefront_size','max_flat_workgroup_size','sgpr_count','vgpr_count','sgpr_spill_count','vgpr_spill_count')}
        r.update(name=name,demangled=demangled,image=image['path'])
        match=re.search(r'tp_core<(\d+), \(tp_decode::Variant\)(\d+)>',demangled)
        if match:r.update(rows=int(match[1]),variant={2:'A',3:'B',4:'C'}[int(match[2])],diagnostic=False)
        elif 'tp_both<' in demangled:
            match=re.search(r'tp_both<(\d+), (true|false)>',demangled);assert match
            r.update(rows=int(match[1]),variant='D',diagnostic=match[2]=='true')
        else:r.update(rows=0,variant='input' if 'tp_input' in demangled else 'output',diagnostic=False)
        begin=isa.index('<'+name+'>:');tail=isa[begin:];next_fn=re.search(r'\n[0-9a-f]+ <',tail[1:]);fn=tail[:next_fn.start()+1] if next_fn else tail
        filename=f"{r['variant']}-R{r['rows']}"+('-diagnostic' if r['diagnostic'] else '')+'.isa'
        (build/filename).write_text(fn)
        r['isa_sha256']=sha(fn.encode());r['scratch_instructions']=len(re.findall(r'\bscratch_\w+',fn))
        r['vmem_loads']=len(re.findall(r'\b(?:global|flat|buffer)_load\w*',fn))
        r['waits']=len(re.findall(r'\bs_wait\w*',fn));r['fdot2']=len(re.findall(r'\bv_dot2\w*',fn))
        r['retirement_reasons']=[]
        if r['wavefront_size']!=32:r['retirement_reasons'].append('wave32')
        expected_threads=(1024 if r['rows']==1 else 512) if r['variant']=='D' else 128
        if r['max_flat_workgroup_size']!=expected_threads:r['retirement_reasons'].append('maxThreads')
        if r['private_segment_fixed_size'] or r['sgpr_spill_count'] or r['vgpr_spill_count'] or r['scratch_instructions']:r['retirement_reasons'].append('private/spills')
        limit=64 if r['variant'] in ('A','B') and r['rows']==6 else 96 if r['variant'] in ('C','D') else 256
        if r['vgpr_count']>limit:r['retirement_reasons'].append(f'VGPR>{limit}')
        r['resource_status']='RETIRED' if r['retirement_reasons'] else 'ELF_PASS_actual_attributes_OPEN'
        if r['variant']=='B':
            # Static instruction ordering only. This does NOT measure hidden cycles.
            start=fn.index('s_waitcnt vmcnt(3)')
            stop=fn.index('s_waitcnt vmcnt(0)',start)
            before=fn[:start];body=fn[start:stop]
            loads=[line for line in before.splitlines() if re.search(r'global_load_b32 .*s\[',line)]
            assert len(loads)>=6 and 'v_pk_mul_lo_u16' in body and 'dot2' in body,'premature NEXT wait before CURRENT hash/dot'
            r['latency_hiding_proof']='PASS_STATIC_ORDER; measured latency hidden remains OPEN'
            r['prefetch_evidence']=dict(last_six_raw_loads=loads[-6:],current_wait='vmcnt(3)',
                current_hash_before_next_full_wait=True,current_dot_before_next_full_wait=True,
                current_dot_count_before_next_full_wait=body.count('dot2'),
                next_full_wait=fn[stop:].splitlines()[0],excerpt_sha256=sha(body.encode()))
        resources.append(r)
assert len(resources)==30,(len(resources),resources)
assert Counter(r['variant'] for r in resources)==Counter(A=8,B=8,C=8,D=4,input=1,output=1)
report=dict(old_whole_ELFs=old,linked_whole_ELFs=linked,new_whole_ELFs=new,old11_byte_identity='PASS',
            principal_bodies=26,diagnostic_bodies=2,transform_bodies=2,actual_new_functions=len(resources),resources=resources,
            source_is_not_latency_proof=True,actual_attributes='OPEN',native='OPEN')
(build/'resources.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='resources'},indent=2))
print('RETIRED',[(r['variant'],r['rows'],r['diagnostic'],r['retirement_reasons']) for r in resources if r['retirement_reasons']])
