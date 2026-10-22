from pathlib import Path
import re,hashlib,json,struct
root=Path('.')
def body(text,signature):
    p=text.index('{',text.index(signature));depth=1;q=p+1
    while depth:
        depth+=(text[q]=='{')-(text[q]=='}');q+=1
    return text[p+1:q-1]
def clean(s):return re.sub(r'\s+','',re.sub(r'//[^\n]*','',s))
old=Path('src/gemv_k2.hip').read_text();new=Path('src/gemv_k2_rows.hip').read_text()
checks={}
for signature in ('uint32_t word_load','Pair sad_pair','void staged'):
    assert clean(body(old,signature))==clean(body(new,signature)),signature
    checks[signature]='identical own V1 body'
assert clean(body(old,'void states'))==clean(body(new,'void states'))
assert body(new,'void staged').count('"=&v"')==16
consume=body(new,'void consume')
assert consume.index('staged(s,h)')<consume.index('Pair activations')
assert 'RawPacked raw[4]' in new and 'k+4<=end && k+4<=chunk_end' in new
assert new.count('__syncthreads()')==2 and 'int64_t(input/16)' in new
# Verify exact code object inclusion in the linked executable, not a parallel .s.
fat=Path('build/exe.fatbin').read_bytes()
def bundles(blob):
    marker=b'__CLANG_OFFLOAD_BUNDLE__';out=[];pos=0
    while True:
        p=blob.find(marker,pos)
        if p<0:break
        q=p+len(marker);n,=struct.unpack_from('<Q',blob,q);q+=8
        records=[]
        for _ in range(n):
            offset,size,namelen=struct.unpack_from('<QQQ',blob,q);q+=24
            target=blob[q:q+namelen].decode();q+=namelen
            records.append((offset,size,target))
        for offset,size,target in records:
            out.append((target,blob[p+offset:p+offset+size]))
        pos=p+max([q-p]+[off+size for off,size,_ in records])
    return out
images=bundles(fat);assert images
for name in ('rows','control'):
    expected=Path('build/'+name+'.elf').read_bytes()
    assert any(b==expected for _,b in images),('linked exact ELF',name)
checks['linked_exact_candidate_and_control_ELFs']=True
isa=Path('build/rows-isa.txt').read_text();notes=Path('build/rows-notes.txt').read_text()
resources=[]
for record in notes.split('  - .args:')[1:]:
    def field(name):
        match=re.search(r'\.'+re.escape(name)+r':\s+(\S+)',record);assert match,name;return match[1]
    name=field('name')
    if 'kernel' not in name:continue
    r={k:int(field(k)) for k in ('vgpr_count','sgpr_count','group_segment_fixed_size','private_segment_fixed_size','wavefront_size','max_flat_workgroup_size','sgpr_spill_count','vgpr_spill_count')}
    r['name']=name
    assert r['wavefront_size']==32 and r['max_flat_workgroup_size']==128
    assert r['private_segment_fixed_size']==r['sgpr_spill_count']==r['vgpr_spill_count']==0,r
    rows=int(re.search(r'kernelILi([248])',name)[1]);lds='Lb1' in name
    assert r['group_segment_fixed_size']==(rows*8*8*4 if lds else 0),r
    resources.append(r)
assert len(resources)==6,resources
assert not re.search(r'\bscratch_|\b(s|v)_.*spill',isa)
assert 'v_pk_fma_f16' in isa and 'v_dot2acc_f32_f16' in isa and 'v_pk_mul_lo_u16' in isa and 'v_mad_u32_u16' in isa
# Actual instruction ordering and V1 register dataflow in all five bodies per kernel.
def instructions(text):
    return [line.split('//')[0].strip() for line in text.splitlines() if line.startswith('\t') and not line.strip().startswith('.')]
def dots(lines):
    return sum(line.count('dot2acc_f32_f16') for line in lines)
order=[]
for resource in resources:
    name=resource['name'];R=int(re.search(r'kernelILi([248])',name)[1]);LDS='Lb1' in name
    fragment=isa.split('<'+name+'>:',1)[1].split('\n\n',1)[0]
    ops=instructions(fragment)
    starts=[j for j,line in enumerate(ops) if 'v_pk_mul_lo_u16' in line and (j==0 or 'v_pk_mul_lo_u16' not in ops[j-1])]
    assert len(starts)==5,(name,len(starts))
    for j in starts:
        block=ops[j:j+16];assert all('v_pk_mul_lo_u16' in z for z in block[:8]) and all('v_mad_u32_u16' in z for z in block[8:])
        temps=[];inputs=[];outputs=[]
        for n in range(8):
            pk=re.search(r'v_pk_mul_lo_u16 (v[0-9]+), (v[0-9]+), s[0-9]+ op_sel_hi:\[0,1\]',block[n]);assert pk,block[n]
            mad=re.search(r'v_mad_u32_u16 (v[0-9]+), (v[0-9]+)\.l, s[0-9]+, (v[0-9]+)',block[n+8]);assert mad,block[n+8]
            assert pk[1]==mad[3] and pk[2]==mad[2]
            temps.append(pk[1]);inputs.append(pk[2]);outputs.append(mad[1])
        assert len(set(temps+outputs))==16 and not set(temps+outputs)&set(inputs)
    prefix=ops[:starts[0]]
    assert sum('global_load_b32' in line for line in prefix)==8+(1 if LDS else 0),(name,'B4 prefetch')
    assert not any('ds_load' in line for line in prefix)
    for j,k in zip(starts[:3],starts[1:4]):
        current=ops[j+16:k]
        if LDS:assert sum('ds_load_2addr_b32' in line for line in current)==R
        else:assert sum('global_load_b32' in line for line in current)==2*R
        assert dots(current)==4*R,(name,'current K dots',dots(current))
    assert sum('s_barrier' in line for line in ops)==(2 if LDS else 0)
    assert 's_waitcnt' in fragment
    order.append(dict(name=name,U4_B_words=8,hash_blocks=5,current_K_A_only=True,hash_before_A=True,barriers=2 if LDS else 0))
checks['actual_instruction_dataflow_and_order']=order
control_notes=Path('build/control-notes.txt').read_text();control_isa=Path('build/control-isa.txt').read_text()
selected=[]
for R in (2,4,8):
    name=f'_ZN13strix_trellis20gemv_baseline_kernelILi2ELb1ELi{R}ELb0EEEvPKjPKDF16_Pfiiiiii'
    assert name in control_notes and '<'+name+'>:' in control_isa
    selected.append(name)
Path('build/control-selected-isa.txt').write_text('\n\n'.join(control_isa.split('<'+name+'>:',1)[1].split('\n\n',1)[0] for name in selected))
checks['immutable_V11_generic_baseline_symbols']=selected

checks['resource_gate']='PASS all six no spills/private'
checks['resources']=resources
checks['sha256']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ('src/gemv_k2_rows.hip','src/gemv_k2_rows.hpp','build/rows.elf','build/control.elf','bin/k2_rows_probe','reference/libstrix_trellis_v11.a')}
print(json.dumps(checks,indent=2))
