"""Read-only CPU audit of R1/R2 frozen evidence. Never executes HIP or queues work."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
if os.name=='nt':
    import ctypes
    ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
    ctypes.windll.kernel32.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
    assert ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
from pathlib import Path
import argparse
import hashlib
import json
import re
import struct
import sys
import tarfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from audit_grouped_k3_math import sections,images,metadata,manifest
from audit_grouped_k3_math_review import archive_images,verify_words


def sha(raw):return hashlib.sha256(raw).hexdigest()


def members(path):
    with tarfile.open(path) as t:
        return {m.name:t.extractfile(m).read() for m in t.getmembers() if m.isfile()}


def emit(label,value):print(label,json.dumps(value,sort_keys=True),flush=True)


def function_bytes(elf):
    off,=struct.unpack_from('<Q',elf,40)
    size,count,_=struct.unpack_from('<HHH',elf,58)
    rows=[struct.unpack_from('<IIQQQQIIQQ',elf,off+j*size) for j in range(count)]
    result={}
    for r in rows:
        if r[1]!=2:continue
        names=rows[r[6]];strings=elf[names[4]:names[4]+names[5]]
        for pos in range(r[4],r[4]+r[5],r[9]):
            name,info,_,index,address,length=struct.unpack_from('<IBBHQQ',elf,pos)
            if info&15!=2 or not index:continue
            s=rows[index];raw=elf[s[4]+address-s[3]:s[4]+address-s[3]+length]
            key=strings[name:].split(b'\0',1)[0].decode()
            result[key]=dict(bytes=length,sha256=sha(raw))
    return result


def historical(old):
    pins=manifest((old/'source.sha256').read_text())
    source=members(old/'source.tar');assert set(source)==set(pins)
    for name,raw in source.items():assert sha(raw)==pins[name],name
    exceptions=[]
    for file in ('resource-bounds.json','cpu-prepare-frozen.log','pilot-audit-cpu.log','lexical-cpu.log'):
        name='reports/grouped-k3-math-data/'+file
        frozen=source[name];checkout=(ROOT/name).read_bytes()
        if frozen!=checkout:
            assert frozen.replace(b'\r\n',b'\n')==checkout
            exceptions.append(dict(file=name,frozen_sha256=sha(frozen),checkout_sha256=sha(checkout),only_CRLF_to_LF=True))
    # No newline exceptions for code, helpers, layout, precision or private views.
    unchanged=[name for name in source if name.startswith(('src/','include/'))]
    unchanged+=['experiments/grouped_k3_math/'+name for name in
                ('kernels.hip','abi.hpp','private_views.hpp','extract.py','first_error.hpp')]
    for name in unchanged:assert (ROOT/name).read_bytes()==source[name],name
    emit('HISTORICAL',dict(source_tar_sha256=sha((old/'source.tar').read_bytes()),
         strict_unchanged_code_files=len(unchanged),auxiliary_report_exceptions=exceptions))


def cpu(data,cpu_path,old):
    m=json.loads((data/'cpu-manifest.json').read_text())
    assert (cpu_path/'manifest.json').read_bytes()==(data/'cpu-manifest.json').read_bytes()
    assert m['full_original_q_values']==35389440 and m['full_derived_q_values']==11796480
    assert m['peak_host_bytes']<2*1024**3 and m['threads']==1
    proof_raw=(cpu_path/'unique-scalar-proof.json').read_bytes()
    assert sha(proof_raw)==m['unique_scalar_proof_sha256']
    assert proof_raw==(data/'unique-scalar-proof.json').read_bytes()
    proof=json.loads(proof_raw);assert len(proof['cases'])==64
    oldcases={c['file']:c for c in json.loads((old/'cpu-manifest.json').read_text())['cases']}
    pairs={5120:set(),2304:set()};changed={'a':0,'p64':0,'y':0,'y64':0};rotations=[]
    for c in m['cases']:
        raw=(cpu_path/c['file']).read_bytes();assert sha(raw)==c['sha256'] and len(raw)==c['bytes']
        assert raw[:8]==b'GK3CPU01'
        i,n,s,g,bits,count=struct.unpack_from('<6I',raw,8)
        assert (i,n,s,g)==(c['input'],c['output'],c['splits'],c['group']) and bits==3 and count==12
        at=32;arrays={}
        for _ in range(count):
            tag=raw[at:at+8].rstrip(b'\0').decode();size,=struct.unpack_from('<Q',raw,at+8);at+=16
            b=raw[at:at+size];at+=size;arrays[tag]=b
            assert len(b)==c['tensors'][tag]['bytes'] and sha(b)==c['tensors'][tag]['sha256']
        assert at==len(raw)
        pair=arrays['scalars'];alpha,beta=struct.unpack('<2f',pair)
        assert np.isfinite([alpha,beta]).all() and pair.hex()==proof['pair_bytes'][g]
        assert pair not in pairs[i];pairs[i].add(pair)
        old_y=np.frombuffer(arrays['old'],dtype='<f4')
        assert bool(np.isnan(old_y).all()) if beta==0 else bool(np.isfinite(old_y).all())
        # Reconstruct the ordered epilogue directly from full independent P64.
        # This never uses a grouped/native candidate as its own expected value.
        total=np.zeros(n,np.float32)
        for split in np.frombuffer(arrays['p64'],dtype='<f8').reshape(s,n):total+=split.astype(np.float32)
        transformed=total.copy()
        for step in (1,2,4,8,16,32,64):
            v=transformed.reshape(-1,2,step);left,right=v[:,0].copy(),v[:,1].copy()
            v[:,0]=left+right;v[:,1]=left-right
        transformed*=np.uint32(0x3db504f3).view(np.float32)
        base=transformed*np.frombuffer(arrays['svh'],dtype='<f2').astype(np.float32)
        if arrays['bias']:base+=np.frombuffer(arrays['bias'],dtype='<f4')
        expected=base*np.float32(alpha)
        if beta!=0:expected+=np.float32(beta)*old_y
        assert expected.tobytes()==arrays['y'],(i,g,'complete ordered Y')
        aa,bb=struct.unpack('<2f',bytes.fromhex(proof['pair_bytes'][(g+4)%32]))
        mutation=base*np.float32(aa)
        if bb!=0:mutation+=np.float32(bb)*old_y
        error=mutation.astype(np.float64)-expected.astype(np.float64)
        maximum=float(np.max(np.abs(error)));maxref=float(np.max(np.abs(expected)))
        nrms=float(np.sqrt((error@error)/max(float(expected.astype(np.float64)@expected.astype(np.float64)),1e-30)))
        assert mutation.tobytes()!=expected.tobytes()
        assert maximum>5e-4*max(1.,maxref) or nrms>3e-5
        rotations.append(dict(input=i,group=g,maxabs=maximum,nrms=nrms,rejected=True))
        for name in changed:changed[name]+=c['tensors'][name]['sha256']!=oldcases[c['file']]['tensors'][name]['sha256']
        for name in ('packed','x','suh','svh','bias','old','a','p64','sumabs'):
            assert c['tensors'][name]==oldcases[c['file']]['tensors'][name],(i,g,name)
    assert all(len(p)==32 for p in pairs.values()) and changed==dict(a=0,p64=0,y=62,y64=62)
    emit('CPU',dict(cases=64,original_q=35389440,rotated_q=11796480,unique_pairs_each=32,
        all_tensors_verified=True,changed_tensors_vs_original=changed,scalar_rotation_rejections=rotations,
        generation_peak_host_bytes=m['peak_host_bytes'],original_payload_before_after_hashes=m['payload_pins']))


def archives(data,old):
    source=members(data/'source.tar');pins=manifest((data/'source.sha256').read_text())
    assert set(source)==set(pins)
    for name,raw in source.items():assert sha(raw)==pins[name]==sha((ROOT/name).read_bytes()),name
    results={}
    for node in ('node1','node2'):
        a=members(data/(node+'-evidence.tar.gz'));baseline=members(old/(node+'-evidence.tar.gz'))
        frozen=manifest(a['frozen.sha256'].decode());checked=0
        assert a['frozen.sha256']==(data/(node+'-frozen.sha256')).read_bytes()
        for name,raw in a.items():
            if name in frozen:assert sha(raw)==frozen[name],(node,name);checked+=1
        for name,raw in source.items():assert a[name]==raw,(node,name)
        assert a['source.tar']==(data/'source.tar').read_bytes()
        library=a['lib/libstrix_trellis.a']
        assert sha(library)=='0b1453a3c9c25e3f7e4f12251d310a3a9f2c3b53c76d78c314e9624ba763e3b3'
        assert library==baseline['lib/libstrix_trellis.a']
        fat=sections(a['bin/math_native_qa'])['.hip_fatbin'];assert fat==a['build/linked.fatbin']
        emitted=images(fat);assert len(emitted)==8
        counts=[len(metadata(e)) for e in emitted];assert counts==[3,2,72,252,1,32,2,2]
        old_images={sha(e) for e in archive_images(library)};wordcount=0
        for j,e in enumerate(emitted):
            assert e==a[f'build/linked-{j:02d}.elf']
            wordcount+=verify_words(e,a[f'build/linked-{j:02d}.isa'].decode())
            if j:assert e==baseline[f'build/linked-{j:02d}.elf'] and sha(e) in old_images
        math=emitted[0];prior=baseline['build/linked-00.elf'];oldfunc=function_bytes(prior);newfunc=function_bytes(math)
        assert len(newfunc)==5 and newfunc==oldfunc
        mathfunc={k:v for k,v in newfunc.items() if k.startswith('_ZN13strix_trellis')}
        assert len(mathfunc)==3
        assert set(newfunc)-set(mathfunc)=={'__cxa_pure_virtual','__cxa_deleted_virtual'}
        assert sections(math)['.text']==sections(prior)['.text'] and metadata(math)==metadata(prior)
        changed_sections=[k for k,v in sections(math).items() if v!=sections(prior).get(k)]
        assert set(changed_sections)=={'.dynsym','.gnu.hash','.hash','.dynstr','.strtab'}
        oldstrings=set(sections(prior)['.strtab'].split(b'\0'));newstrings=set(sections(math)['.strtab'].split(b'\0'))
        removed=oldstrings-newstrings;added=newstrings-oldstrings
        assert len(removed)==len(added)==1
        assert all(v.startswith(b'__hip_cuid_') for v in removed|added)
        resources=json.loads(a['build/resources.json']);assert not resources['violations']
        for k in metadata(math):
            assert k['.args']==[{'.offset':0,'.size':2320,'.value_kind':'by_value'}]
            for key,value in dict(kernarg_segment_size=2320,kernarg_segment_align=8,wavefront_size=32,max_flat_workgroup_size=128,
                                  group_segment_fixed_size=0,private_segment_fixed_size=0,sgpr_spill_count=0,vgpr_spill_count=0).items():
                assert k['.'+key]==value
        assert json.loads(a['build/scalar-selection.json'])['mutation_tests_rejected']==6
        link=json.loads(a['build/attribute-link.json']);assert link['status']=='PASS_BUILD_ONLY'
        assert link['binary_sha256']==sha(a['bin/math_native_qa']) and len(link['actual_linked_V13_relocations'])==4
        assert b'PASS same forwarding hook:' in a['build/attribute-cpu.log']
        assert not re.search(b'amdhip|hsa-runtime',a['build/attribute-cpu-dependencies.txt'])
        result=dict(scope='~/gpu-site/scratch-trellis/grouped-k3-math-r1',
            frozen_manifest_sha256=sha(a['frozen.sha256']),binary_sha256=sha(a['bin/math_native_qa']),
            library_sha256=sha(library),math_ELF_sha256=sha(math),prior_math_ELF_sha256=sha(prior),
            math_whole_image_identical=math==prior,math_function_hashes=mathfunc,compiler_support_function_hashes={k:v for k,v in newfunc.items() if k not in mathfunc},
            math_metadata_sha256=sha(json.dumps(metadata(math),sort_keys=True).encode()),
            math_metadata_identical=True,changed_math_sections=changed_sections,
            prior_CUID=next(iter(removed)).decode(),new_CUID=next(iter(added)).decode(),
            whole_image_differing_bytes=sum(x!=y for x,y in zip(math,prior))+abs(len(math)-len(prior)),
            seven_control_ELFs_byte_identical=True,kernel_counts=counts,verified_ISA_instructions=wordcount,
            linkmap_sha256=sha(a['build/link.map']),evidence_archive_sha256=sha((data/(node+'-evidence.tar.gz')).read_bytes()),
            archived_frozen_files_verified=checked,actual_attribute_observations='OPEN_NATIVE',
            canonical_ROOT_payload_command='cd ~/gpu-site/scratch-trellis/grouped-k3-math-r1 && '
                'ROOT_GROUPED_K3_MATH_R1_LABEL=<fresh_ROOT-grouped-k3-math-r1-label> '
                'ROOT_GROUPED_K3_MATH_R1_ADMISSION_SHA256='+sha(a['frozen.sha256'])+
                ' bash experiments/grouped_k3_math/native_payload.sh --inside-root-lease')
        results[node]=result;emit('BUILD_'+node,result)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cpu-data',type=Path,required=True);p.add_argument('--write-handoff',action='store_true');args=p.parse_args()
    data=ROOT/'reports/grouped-k3-math-r1-data';old=ROOT/'reports/grouped-k3-math-data'
    historical(old);cpu(data,args.cpu_data,old);nodes=archives(data,old)
    if args.write_handoff:
        result=dict(schema=1,state='PREPARED_NOT_GPU_PASS',root_review_required=True,
            review='HIGH62ca8a4',worktree=str(ROOT),branch='performance/grouped-k3-math-r1',
            R1='CLOSED_CPU_BUILD_ONLY_NATIVE_OBSERVATION_OPEN',R2='CLOSED_CPU_REGENERATED_64_CASES',
            actual_contexts_from_owner_source=1,contexts_per_geometry=False,queries_required_per_actual_admitted_context=7,
            observation_scope='link boundary of user/static-library calls, not all internal HIP driver queries',
            gpu_executed=False,HIP_executable_executed=False,GPU_requests=0,
            baseline='V13 private9/56B/C1 publicABI1',nodes=nodes,
            cpu_manifest_sha256=sha((data/'cpu-manifest.json').read_bytes()),
            scalar_proof_sha256=sha((data/'unique-scalar-proof.json').read_bytes()),
            resource_bounds=json.loads((data/'resource-bounds.json').read_text()),
            native_gates=dict(P1_01='OPEN',P1_02='OPEN',P1_03='OPEN',actual_attributes='OPEN',
                numeric_A_P_Y_FP64='OPEN',ownership_capture_LAST='OPEN',actual_peak_RSS_and_ledger='OPEN',
                HIP_async_errors='OPEN_NOT_INJECTED',actual_allocation_recycling='NOT_CLAIMED'),
            native_configs=14,groups=[1,2,4,8,16,31,32],inner_timeout_seconds=145,physical_lease_limit_seconds=180,
            ROOT_must_obtain_lease_and_check_local_time_and_maintenance_before_and_after_wait=True,
            performance='OPEN_TWO_CAMPAIGNS_NOT_AUTHORIZED',weight_data='QA-derived variants, not newly trained experts')
        (data/'handoff.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print('PASS CPU/build R1/R2 repairs; PREPARED_NOT_GPU_PASS; all native gates OPEN')


if __name__=='__main__':main()
