"""Inspect actual archives and linked ELF bytes without executing device code."""
from pathlib import Path
import hashlib
import importlib.util
import json
import struct
import subprocess

root=Path.home()/'gpu-site/scratch-trellis'
own=root/'runtime-v16';qa=root/'runtime-v16-qa';engine=root/'engine-v16'
assert Path.cwd()==own and not (own/'frozen.sha256').exists()
out=own/'metadata';out.mkdir(exist_ok=True)
llvm=Path('/opt/rocm/core-10.1/lib/llvm/bin')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(args):return subprocess.check_output(list(map(str,args)))
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def functions(p):
    raw=p.read_bytes();assert raw[:6]==b'\x7fELF\x02\x01'
    h=struct.unpack_from('<16sHHIQQQIHHHHHH',raw)
    ss=[struct.unpack_from('<IIQQQQIIQQ',raw,h[6]+i*h[11]) for i in range(h[12])];result={}
    for s in ss:
        if s[1]!=2:continue
        st=ss[s[6]];names=raw[st[4]:st[4]+st[5]]
        for pos in range(s[4],s[4]+s[5],s[9]):
            name,info,_,index,value,size=struct.unpack_from('<IBBHQQ',raw,pos)
            if info&15!=2 or not size or not 0<index<len(ss):continue
            label=names[name:names.index(b'\0',name)].decode();sec=ss[index];a=sec[4]+value-sec[3]
            assert sec[4]<=a and a+size<=sec[4]+sec[5];result[label]=raw[a:a+size]
    return result

def extract(archive,module,prefix):
    p=out/(prefix+'-'+module+'.o');p.write_bytes(run(['ar','p',archive,module+'.hip.o']))
    (out/(prefix+'-'+module+'.offloading')).write_bytes(run([llvm/'llvm-objdump','--offloading',p]))
    images=list(out.glob(p.name+'.*.hip-amdgcn-amd-amdhsa--gfx1151'));assert len(images)==1
    return images[0]

spec=importlib.util.spec_from_file_location('notes',own/'tools/summarize_hip_metadata.py')
notes=importlib.util.module_from_spec(spec);spec.loader.exec_module(notes)
def metadata(image,tag):
    p=out/(tag+'.notes');p.write_bytes(run([llvm/'llvm-readelf','--notes',image]))
    return sorted(notes.parse_notes(p),key=lambda r:r['name'])

archive=own/'build-hip/libstrix_trellis.a'
old=root/'runtime-v14b/build-hip/libstrix_trellis.a'
v15=root/'runtime-v15/build-hip/libstrix_trellis.a'
assert sha(old)=='14ed324103eef6b957651a61cc8f0435251022eb1fc1767f2a9afe60ae8d3c0c'
assert sha(v15)=='ae21eaf19762b544396f2c40b7d68aa83bcb3f7fdfcdd6398f0e5c6bf44a5ade'
modules=['backend','gemm','gemv','gemv_lds','gemv_k2','gemv_fused','gemv_output_direct','gemv_grouped_direct','gemv_k2_multirow','wmma_packed','gemv_fused_both']
production={};records=[];kernels=[]
for name in modules:
    image=extract(archive,name,'v16');rows=metadata(image,'v16-'+name);kernels+=rows
    for r in rows:assert r['wavefront_size']==32 and r['private_segment_fixed_size']==r['vgpr_spill_count']==r['sgpr_spill_count']==0,r
    prior=extract(v15 if name=='gemv_fused_both' else old,name,'prior')
    before=metadata(prior,'prior-'+name)
    assert functions(image)==functions(prior) and rows==before,name
    # Production images must retain full byte identity, including CUID sections.
    assert image.read_bytes()==prior.read_bytes(),('Whole production ELF differs',name)
    production[name]=image
    records.append(dict(module=name,sha256=sha(image),reference_archive_sha256=sha(v15 if name=='gemv_fused_both' else old),whole_ELF_identical=True,kernels=len(rows),metadata=rows))
    if name=='gemv_fused_both':
        fs={n:b for n,b in functions(image).items() if 'bothILi' in n};assert len(fs)==1
        symbol,body=next(iter(fs.items()));assert 'bothILi5E' in symbol and 'ModeE1' in symbol
        assert len(body)==3660 and hashlib.sha256(body).hexdigest()=='86d22fc496c5df7d6a07f82f6a0bc7d3e5f2897fb3aa807558d3e5ae0f5a0fc5'
        direct=dict(symbol=symbol,FUNC_bytes=len(body),FUNC_sha256=hashlib.sha256(body).hexdigest(),whole_ELF_sha256=sha(image),metadata=rows[0],actual_runtime_max_dynamic='OPEN_UNTIL_ROOT_LEASE')
assert len(kernels)==433 and len(production)==11

def linked_images(binary):
    assert any(binary.is_relative_to(p) for p in (own,qa,engine))
    (out/(binary.name+'.offloading')).write_bytes(run([llvm/'llvm-objdump','--offloading',binary]))
    return sorted(binary.parent.glob(binary.name+'.*.hip-amdgcn-amd-amdhsa--gfx1151'))

bins=[own/'build-hip'/n for n in ('strix_trellis_gpu_runtime','strix_trellis_api_linear_bench','strix_trellis_graph_linear_bench','strix_trellis_head_linear_bench','head_crosscheck')]
bins += [qa/'v14/runtime_v16_v14b_qa',qa/'direct/build/runtime_v16_complete',engine/'build/engine/strix-trellis-dense-example']
linked=[]
required={sha(p) for name,p in production.items() if name!='gemv'}
for binary in bins:
    images=linked_images(binary);seen={sha(p) for p in images}
    assert required<=seen,('Missing matching production images',binary,required-seen)
    rows=[];aux=[]
    for i,image in enumerate(images):
        m=metadata(image,binary.name+'-'+str(i))
        for r in m:
            assert r['wavefront_size']==32 and r['vgpr_spill_count']==r['sgpr_spill_count']==0,r
            if r['private_segment_fixed_size']:
                assert binary.name=='runtime_v16_v14b_qa' and r['name']=='_Z12native_probePfiii' and r['private_segment_fixed_size']==288,r
                baseline=root/'runtime-v14b-qa/complete-v1/runtime_v14_qa'
                prior=next(p for p in baseline.parent.glob(baseline.name+'.*.hip-amdgcn-amd-amdhsa--gfx1151') if functions(p).keys()==functions(image).keys())
                assert functions(prior)==functions(image) and metadata(prior,'historical-native-probe')==m

        if sha(image) not in required:
            aux.append(dict(sha256=sha(image),functions={n:hashlib.sha256(b).hexdigest() for n,b in functions(image).items()},metadata=m))
        rows+=m
    literal=[]
    if binary.name=='runtime_v16_complete':
        prior_binary=root/'runtime-v15-qa/complete-v2/build-complete/runtime_v15_complete'
        # Use already-frozen extracted auxiliary images, never extract beside old binaries.
        old_images=list(prior_binary.parent.glob(prior_binary.name+'.*.hip-amdgcn-amd-amdhsa--gfx1151'))
        for image in images:
            fs=functions(image)
            if not any('qa_literal' in n for n in fs):continue
            prior=next(p for p in old_images if functions(p).keys()==fs.keys())
            assert fs==functions(prior)
            assert metadata(image,'literal-new')==metadata(prior,'literal-old')
            literal.append(dict(new_ELF_sha256=sha(image),old_ELF_sha256=sha(prior),whole_ELF_identical=sha(image)==sha(prior),all_FUNC_and_metadata_identical=True))
        assert len(literal)==1
    linked.append(dict(binary=str(binary),sha256=sha(binary),image_count=len(images),kernel_count=len(rows),matching_required_production_images=len(required),auxiliary=aux,literal=literal))

controls=[]
for name in ('v14b_control','v14b_q1_control'):
    binary=qa/'direct/build'/name;images=linked_images(binary);seen={sha(p) for p in images}
    expected={r['sha256'] for r in records if r['module'] not in ('gemv','gemv_fused_both')}
    assert expected<=seen
    assert sha(production['gemv_fused_both']) not in seen
    controls.append(dict(binary=str(binary),sha256=sha(binary),private_fields=9,C_ABI_only=True,separate_process=True,image_count=len(images)))
save(out/'audit.json',dict(state='CPU_ELF_AUDIT_PASS_NO_HIP_EXECUTION',archive_sha256=sha(archive),old_control_sha256=sha(old),private_fields=10,C_ABI=1,kernel_count=len(kernels),device_ELF_count=len(production),old_TEN_whole_ELFs_identical=True,new_Direct_whole_ELF_identical_V15=True,all_wave32_zero_private_spills=True,new_function=direct,production=records,linked=linked,controls=controls))
print(json.dumps(dict(state='CPU_ELF_AUDIT_PASS_NO_HIP_EXECUTION',archive_sha256=sha(archive),kernels=len(kernels),images=len(production),Direct=direct,linked_binaries=len(linked)),indent=2))
