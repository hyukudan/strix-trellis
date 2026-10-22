"""Freeze this private CPU build and references; never runs the HIP executable."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys
root=Path.home()/'gpu-site/scratch-trellis/tp-decode-v1/host-r2'
assert Path.cwd()==root and os.getpriority(os.PRIO_PROCESS,0)>=10 and not (root/'frozen.sha256').exists()
attempt=sys.argv[1];build=root/'build'/attempt
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,value):p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
compiled=read(build/'build.json');resource=read(build/'resources.json');evidence=read(root/'evidence/cpu-validation.json')
assert compiled['cpu_tests']=='PASS' and compiled['executed_hip'] is False
assert sha(root/'lib/libstrix_trellis_v16.a')==compiled['archive_sha256']
assert sha(build/'tp_native')==compiled['binary_sha256']
assert sha(Path('/opt/rocm/core-10.1/lib/libamdhip64.so'))==compiled['HIP_runtime_library_sha256']
assert read(build/'link-audit.json')['passed']
assert 'PASS private CPU' in (build/'sanitized-cpu.log').read_text()
assert resource['old11_byte_identity']=='PASS' and resource['actual_new_functions']==30
retired=[r for r in resource['resources'] if r['retirement_reasons']]
assert not retired,'STOP retire resource variant before any physical campaign'
assert all(r['latency_hiding_proof'].startswith('PASS_STATIC_ORDER') for r in resource['resources'] if r['variant']=='B')
assert evidence['passed'] and evidence['full_worker_validation']['passed'] and len(evidence['full_worker_validation']['fixtures'])==14
fixtures=read(root/'fixtures/manifest.json');assert fixtures['passed'] and len(fixtures['fixtures'])==12
for item in fixtures['fixtures']:assert sha(root/'fixtures'/item['file'])==item['sha256']
expected=read(root/'evidence/source-files.json')
for relative,digest in expected.items():assert sha(root/relative)==digest,relative
(root/'bin').mkdir(exist_ok=False);shutil.copyfile(build/'tp_native',root/'bin/tp_native');(root/'bin/tp_native').chmod(0o555)
(root/'runs').mkdir(exist_ok=False)
admission=dict(state='CPU_PREPARED_NATIVE_OPEN',base_commit='205ad212c0c311be72ccad88ae5dd6106cf3690e',
    candidate_binary_sha256=sha(root/'bin/tp_native'),control_archive_sha256=compiled['archive_sha256'],
    HIP_runtime_library_sha256=compiled['HIP_runtime_library_sha256'],
    linked_whole_ELFs=resource['linked_whole_ELFs'],old11_identity='PASS',principal_bodies=26,diagnostic_bodies=2,transform_bodies=2,
    full_fixture_validation='PASS',cpu_references='PASS',fixture_manifest_sha256=sha(root/'fixtures/manifest.json'),
    cpu_validation_sha256=sha(root/'evidence/cpu-validation.json'),source_fingerprints_sha256=sha(root/'evidence/source-files.json'),
    retired=retired,isa_prefetch_gate='PASS_STATIC_ORDER',actual_function_attributes='OPEN',
    native_A_P_Y_byte_identity='OPEN',ownership_graph_capture='OPEN',actual_HIP_extent_query='not implemented; own hipMalloc extents tracked by private factory',
    primary_performance=['L0.q_b.rank0','L0.q_b.rank1','L0.o_b.rank0','L0.o_b.rank1'],
    holdouts='L21/L39 same matrix; head and N-half remain second phase',
    campaign1=dict(state='OPEN',actual_outer_lease=None,receipt=None),campaign2=dict(state='OPEN',actual_outer_lease=None,receipt=None),
    root_entrypoint='bash experiments/tp_decode/root_payload.sh --inside-root-lease ROOT_RECEIPT.json',inner_timeout_seconds=850,outer_timeout_seconds=900,
    memory=dict(explicit_GPU_hard_cap_bytes=1<<30,host_RSS_gate_bytes=2<<30,resident_packed_bytes=275251200,
                expected_GPU_peak_upper_bound_bytes=450<<20,expected_host_peak_upper_bound_bytes=300<<20,
                estimates_are_unmeasured=True,actual_GPU_RSS='OPEN'),benchmark_gains='UNMEASURED')
save(root/'admission.json',admission)
paths=[]
for prefix in ('src','include','experiments/tp_decode','evidence','fixtures','bin','lib'):
    paths.extend(p for p in (root/prefix).rglob('*') if p.is_file())
paths.extend(p for p in build.rglob('*') if p.is_file());paths.append(root/'admission.json')
with (root/'frozen.sha256').open('x') as f:
    for p in sorted(set(paths)):f.write(sha(p)+'  '+p.relative_to(root).as_posix()+'\n')
handoff=dict(admission=admission,frozen_sha256=sha(root/'frozen.sha256'),frozen_files=len(set(paths)),
             scratch=str(root),build_attempt=attempt,source_files=expected,build=compiled,resources_file='build/'+attempt+'/resources.json')
save(root/'handoff.json',handoff)
print('PASS frozen CPU-only private candidate; native/campaign1/campaign2 OPEN')
