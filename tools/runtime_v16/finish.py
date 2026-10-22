"""Freeze only NEW V16 trees and emit a CPU-only ROOT handoff."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import tarfile

root=Path.home()/'gpu-site/scratch-trellis';own=root/'runtime-v16';qa=root/'runtime-v16-qa';engine=root/'engine-v16'
assert Path.cwd()==own and os.getpriority(os.PRIO_PROCESS,0)==10
assert not (own/'frozen.sha256').exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def check(command,cwd=own):subprocess.run(command,cwd=cwd,check=True,stdout=subprocess.DEVNULL)
pin=json.loads((own/'source-pin.json').read_text())
assert len(pin['commit'])==40
for relative,digest in pin['LF_files'].items():
    assert hashlib.sha256((own/relative).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest,relative
for directory in ('src','include'):
    for p in (own/directory).glob('*'):
        if p.is_file():assert p.read_bytes()==(qa/directory/p.name).read_bytes()
for name in ('runtime_v15_complete.hip','runtime_v15_case.hpp','q1_complete.hpp','negative.hpp','v13_q1_control.hip','control.hip'):
    assert (own/'tools/runtime_v16/native'/name).read_bytes()==(qa/'direct'/name).read_bytes()
assert (qa/'build/qa_private_layout.hpp').read_bytes()==(qa/'direct/build/qa_private_layout.hpp').read_bytes()
assert 'fused_both_capable' in (qa/'build/qa_private_layout.hpp').read_text()
for name in ('native-root.sh','root-payload.sh'):
    check(['bash','-n','tools/runtime_v16/'+name])
check(['bash','-n','tools/run_runtime_v16.sh'])
audit=json.loads((own/'metadata/audit.json').read_text())
assert audit['kernel_count']==433 and audit['device_ELF_count']==11
assert audit['old_TEN_whole_ELFs_identical'] and audit['new_Direct_whole_ELF_identical_V15']
assert '100% tests passed, 0 tests failed out of 9' in (own/'build/ctest.log').read_text()
python=(own/'build/python.log').read_text();assert 'Ran 33 tests' in python and '\nOK\n' in python and 'skipped' not in python
assert 'mixed_g1_flag_cases=8' in (own/'build/high.log').read_text()
assert json.loads((own/'build/shell-proof/proof.json').read_text())['engine_cases_reached']==2
references=json.loads((own/'build/reference-validation.json').read_text());assert len(references['records'])==3

# Freeze all named historical manifests and all inputs consumed by native/global.
# Verification runs in the original cwd, read-only; no old extraction or build.
preserved=[];bound=set()
for scope in ('runtime-v13','runtime-v14','runtime-v14b','runtime-v15','runtime-v14b-qa','engine-v14b','runtime-v15-qa'):
    directory=root/scope;manifest=directory/'frozen.sha256'
    check(['sha256sum','-c','frozen.sha256'],directory)
    preserved.append(dict(scope=scope,manifest_sha256=sha(manifest),verified=True));bound.add(manifest)
    for line in manifest.read_text().splitlines():
        digest,name=line.split('  ',1);p=Path(name);p=p if p.is_absolute() else directory/p
        assert p.is_relative_to(root) or p.is_relative_to('/opt/rocm'),p
        bound.add(p)
for name in ('frozen.sha256','frozen-complete.sha256','frozen-root.sha256'):
    p=root/'runtime-v15-qa/complete-v2'/name;check(['sha256sum','-c',name],p.parent);bound.add(p)
for record in json.loads((qa/'build/reference-inputs.json').read_text()):
    p=Path(record['path']);assert sha(p)==record['sha256'];bound.add(p)
for record in references['records']:bound.add(Path(record['source']))
for path in ('runtime-v14b/build-hip/libstrix_trellis.a','runtime-v15/build-hip/libstrix_trellis.a','runtime-v14b/build-hip/strix_trellis_api_linear_bench','runtime-v14b/build-hip/strix_trellis_head_linear_bench','runtime-v14b/build-hip/head_crosscheck','input.f32','expected-plain.f32','expected-bias.f32'):
    bound.add(root/path)
preserved_lines=''.join(sha(p)+'  '+str(p)+'\n' for p in sorted(bound))
(qa/'preserved-inputs.sha256').write_text(preserved_lines,encoding='utf-8')
(own/'preserved-inputs.sha256').write_text(preserved_lines,encoding='utf-8')
save(own/'metadata/preserved.json',preserved)
save(own/'metadata/resource-bounds.json',dict(native_aggregate_seconds=600,host_limit_bytes=2<<30,explicit_GPU_limit_bytes=900<<20,V14b_weight_reserve_bytes=128<<20,actual_runtime_counter_and_RSS='OPEN_UNTIL_ROOT_LEASE',driver_graph_overhead_not_in_explicit_counter=True,CPU_reference_peak_bytes=references['host_peak_bytes'],private_fields=10,private_size=56,old_cap_offset=48,new_cap_offset=49,expected_creation_queries=5,after_context_query_delta=0,Q1_nodes=[3,4,4,5],WMMA_qualified_guards=13,WMMA_fallbacks=3,Direct_whole_workspace_unused=True))
# Keep the exact final compiled byte spelling, including historical CRLF headers.
with tarfile.open(own/'source-compiled-final.tar','w') as archive:
    for directory in ('src','include'):
        archive.add(own/directory,arcname=directory)

def freeze(directory):
    files=[]
    for p in directory.rglob('*'):
        relative=p.relative_to(directory)
        if not p.is_file() or any(x in relative.parts for x in ('.tmp','.cache','__pycache__','runs','delivery')):continue
        if relative.as_posix() in ('frozen.sha256','handoff.json','build/finish.log'):continue
        files.append(p)
    manifest=directory/'frozen.sha256'
    manifest.write_text(''.join(sha(p)+'  '+p.relative_to(directory).as_posix()+'\n' for p in sorted(files)),encoding='utf-8')
    check(['sha256sum','-c','frozen.sha256'],directory)
    return dict(path=str(manifest),sha256=sha(manifest),files=len(files))

freezes=[freeze(p) for p in (qa,engine,own)]
node=os.uname().nodename
handoff=dict(state='PREPARED_CPU_ONLY_NOT_GPU_QUALIFIED',node=node,base_commit='2a3ad4e',source_commit=pin['commit'],qualified_default='V14b',default_changed=False,private_branch='integration/runtime-v16',no_GPU_execution_discovery_or_request=True,no_push_merge_or_agents=True,library=dict(path=str(own/'build-hip/libstrix_trellis.a'),sha256=audit['archive_sha256'],private_fields=10,C_ABI=1,kernels=433,images=11),Direct=audit['new_function'],consumers=audit['linked'],separate_private9_controls=audit['controls'],freezes=freezes,source_archives={p.name:sha(p) for p in (own/'source-final.tar',own/'source-compiled-final.tar')},native_command=f'cd "{qa}" && ROOT_GPU_LEASE=1 bash native-root.sh "$PWD/root-v16-native-001"',global_command=f'cd "{own}" && ROOT_GPU_LEASE=1 timeout --signal=TERM --kill-after=3s 550s bash tools/run_runtime_v16.sh --payload root-v16-001',combined_ROOT_payload=f'cd "{own}" && ROOT_GPU_LEASE=1 bash tools/runtime_v16/root-payload.sh review001',controller_seconds=3600,payload_max_seconds=1180,native_max_seconds=600,expected_queries=5,after_context_query_delta=0,CPU_CTest=9,CPU_Python=33,open_gates=['actual five unique HIP resources and maxDynamic','native 52 suites/156 negative captures plus full Direct/Q1','native measured RSS/explicit allocation counters','global raw EXIT0 fixtures/API/graphs/head/CPU/CUDA/engine','three actual V14b/V16 API process pairs and R1 performance'],native_GPU_results='NOT_RUN',performance_V16='NOT_MEASURED')
save(own/'handoff.json',handoff)
delivery=own/'delivery'
for source in (own/'handoff.json',own/'source-pin.json',own/'source-byte-alignment.json',own/'metadata/audit.json',own/'metadata/preserved.json',own/'metadata/resource-bounds.json',own/'build/reference-validation.json'):
    shutil.copyfile(source,delivery/source.name)
for scope in (own,qa,engine):
    target=delivery/scope.name;target.mkdir(exist_ok=True)
    for p in scope.rglob('*.log'):
        if any(x in p.relative_to(scope).parts for x in ('.tmp','.cache','delivery')) or p==own/'build/finish.log':continue
        dest=target/p.relative_to(scope);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    shutil.copyfile(scope/'frozen.sha256',delivery/(scope.name+'-frozen.sha256'))
print(json.dumps(handoff,indent=2))
