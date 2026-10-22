"""CPU-only audit of the frozen private performance harness; never loads HIP."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,tarfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from audit_grouped_k3_math import sections,images,metadata,manifest
from audit_grouped_k3_math_review import archive_images,verify_words
def sha(raw):return hashlib.sha256(raw).hexdigest()
def members(path):
    with tarfile.open(path) as tar:return {m.name:tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
def load(name):
    sys.path.insert(0,str(ROOT/'experiments/grouped_k3_perf'))
    spec=importlib.util.spec_from_file_location('perf_'+name,ROOT/'experiments/grouped_k3_perf'/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cpu-data',type=Path,required=True);p.add_argument('--write-handoff',action='store_true');args=p.parse_args()
    data=ROOT/'reports/grouped-k3-perf-r1-data';handoff=json.loads((ROOT/'reports/grouped-k3-math-r1-data/handoff.json').read_text());reference=handoff['nodes']['node1']
    sources=members(data/'source.tar');pins=manifest((data/'source.sha256').read_text());assert sources.keys()==pins.keys()
    for name,raw in sources.items():assert sha(raw)==pins[name]==sha((ROOT/name).read_bytes()),name
    identity=load('identity');nodes={}
    for node in ('node1','node2'):
        a=members(data/(node+'-evidence.tar.gz'));frozen=manifest(a['frozen.sha256'].decode());checked=0
        assert a['frozen.sha256']==(data/(node+'-frozen.sha256')).read_bytes()
        assert a['source.tar']==(data/'source.tar').read_bytes() and a['source.sha256']==(data/'source.sha256').read_bytes()
        for name,raw in a.items():
            if name in frozen:assert sha(raw)==frozen[name],(node,name);checked+=1
        for name,raw in sources.items():assert sha(raw)==frozen[name],(node,name)
        lib=a['lib/libstrix_trellis.a'];assert sha(lib)==reference['library_sha256']
        binary=a['bin/perf_native'];fat=sections(binary)['.hip_fatbin'];assert fat==a['build/linked.fatbin']
        linked=images(fat);assert len(linked)==8
        archive_set={sha(elf) for elf in archive_images(lib)};words=0
        counts=[]
        for j,elf in enumerate(linked):
            assert elf==a[f'build/linked-{j:02d}.elf'];counts.append(len(metadata(elf)))
            words+=verify_words(elf,a[f'build/linked-{j:02d}.isa'].decode())
            if j:assert sha(elf) in archive_set
        assert counts==[3,2,72,252,1,32,2,2]
        funcs={k:v for k,v in identity.function_bytes(linked[0]).items() if k.startswith('_ZN13strix_trellis')}
        assert funcs==reference['math_function_hashes']
        md=metadata(linked[0]);assert sha(json.dumps(md,sort_keys=True).encode())==reference['math_metadata_sha256']
        for k in md:
            assert k['.args']==[{'.offset':0,'.size':2320,'.value_kind':'by_value'}]
            for key,value in dict(kernarg_segment_size=2320,kernarg_segment_align=8,wavefront_size=32,max_flat_workgroup_size=128,group_segment_fixed_size=0,private_segment_fixed_size=0,sgpr_spill_count=0,vgpr_spill_count=0).items():assert k['.'+key]==value
        proof=json.loads(a['build/identity.json']);assert proof['binary_sha256']==sha(binary) and proof['reference_binary']==reference['binary_sha256']
        assert all(r['whole_image_identical'] for r in proof['images'][1:])
        assert not json.loads(a['build/resources.json'])['violations']
        assert json.loads(a['build/attribute-link.json'])['status']=='PASS_BUILD_ONLY'
        assert json.loads(a['build/baseline-binding.json'])['calls']
        for exe in ('perf_cpu_test','attribute_cpu_test','high_guard','submit_hooks_test'):
            assert b'PASS' in a[f'build/{exe}.log']
            assert b'libamdhip' not in a[f'build/{exe}.dependencies.txt'] and b'hsa-runtime' not in a[f'build/{exe}.dependencies.txt']
        for filename in ('heap-link.json','provenance-test.log','raw-fixture-test.log','identity.json','resources.json','attribute-link.json','baseline-binding.json','toolchain.txt','toolchain.sha256','scalar-selection.json','cpu-audit.log','perf_cpu_test.log','attribute_cpu_test.log','high_guard.log','submit_hooks_test.log'):
            (data/(node+'-'+filename)).write_bytes(a['build/'+filename])
        nodes[node]=dict(scope='~/gpu-site/scratch-trellis/grouped-k3-perf-r1/runs/host-r2',frozen_manifest_sha256=sha(a['frozen.sha256']),
            source_tar_sha256=sha(a['source.tar']),source_manifest_sha256=sha(a['source.sha256']),binary_sha256=sha(binary),library_sha256=sha(lib),
            math_ELF_sha256=sha(linked[0]),math_text_sha256=sha(sections(linked[0])['.text']),math_metadata_sha256=reference['math_metadata_sha256'],math_functions=funcs,
            whole_math_image_identical_to_R1=proof['images'][0]['whole_image_identical'],seven_control_images_identical_to_R1_and_actual_V13=True,
            archived_frozen_files_verified=checked,source_files_verified=len(sources),ISA_words_verified=words,
            evidence_archive_sha256=sha((data/(node+'-evidence.tar.gz')).read_bytes()),
            toolchain_manifest_sha256=sha(a['build/toolchain.sha256']),
            canonical_ROOT_payload_entrypoint='cd ~/gpu-site/scratch-trellis/grouped-k3-perf-r1/runs/host-r2 && ROOT_GROUPED_K3_PERF_LABEL=<fresh_ROOT-grouped-k3-perf-r1-label> bash experiments/grouped_k3_perf/native_payload.sh --inside-root-lease')
    assert (args.cpu_data/'manifest.json').read_bytes()==(data/'cpu-manifest.json').read_bytes()
    load('verify_data').verify(args.cpu_data)
    cpu=json.loads((data/'cpu-manifest.json').read_text())
    for c in cpu['cases']:assert frozen['data/cpu/'+c['file']]==c['sha256']
    result=dict(schema=1,state='CPU_ONLY_NATIVE_QA_PASS_REQUIRED',base_commit='7c6a07e6ab687867c300631d93be443e8a39c4bf',branch='performance/grouped-k3-perf-r1',
        benchmark_executed=False,HIP_executable_executed=False,GPU_requests=0,performance_execution_authorized=False,
        ROOT_actual_first_math_QA='ACTUAL_R1_QA_PINNED_NOT_NEW_POOL_QA',ROOT_host_pool_timing_review='REQUIRED',QA_binary_required=reference['binary_sha256'],
        baseline='actual V13 archive 0b1453a3 private9/56B/C1 publicABI1; prepared individual launch_linear SUM workspace',
        current_ROOT_default='MAIN V16 qualified 8eb47b09 / TEN / C1; report 21a1ea7 README 573567f; private performance control stays V13 old0b / NINE / C1',
        selection=dict(input_output_splits=[[5120,2304,16],[2304,5120,4]],G=[4,8,16,32],cold_clones=[16,8,4,2],weights_per_geometry=64,
                       packed_bytes_each=4423680,resident_packed_bytes=283115520,alpha=1,beta=0,bias=False,primary='prepared DIRECT old3G vs new3',secondary='GRAPH old3G chain vs new3',sweeps=11),
        weights='Distinct disjoint GPU copies of real expert10 originals (w1/w3 alternating forward, repeated w2 inverse), not 64 distinct trained experts; no L2 physical counter claim',
        CPU_manifest_sha256=sha((data/'cpu-manifest.json').read_bytes()),bounds=load('resource_bounds').bounds(cpu),nodes=nodes,
        counters='One real C API context per geometry; 4 old+3 math once per actual context, actual count from R1 fixed trace; zero late deltas; not global driver queries',
        future_limits=dict(physical_campaigns=2,physical_seconds_each=300,inner_seconds=270,queue_CPU_wait_seconds_allowed=3600,first_failure='STOP; no identical retry'),
        proposal_gate='Two distinct ROOT physical full PASS receipts; all fixed primary direct cells event AND wall median gain >=2% in BOTH; graph losses kept, negative graph => direct-only; no posthoc selection',
        native_gates='OPEN: actual attributes, numerics, graph inspection, ownership LAST, pointer validation, peak RSS/ledger, timing fit',
        claim_scope='No routing, SwiGLU, end-to-end MoE, tokens/s, quality, model or public API/default claim')
    if args.write_handoff:(data/'handoffCPU_ONLY_NEW_POOL_NATIVE_GATES_OPEN.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2,ensure_ascii=False));print('PASS CPU/build/freeze only; CPU_ONLY_NATIVE_QA_PASS_REQUIRED; performance BLOCKED')
if __name__=='__main__':main()
