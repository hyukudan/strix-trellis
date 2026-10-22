"""Serial nice10 CPU/build preparation. Never starts a HIP executable."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

root = Path.home()/'gpu-site/scratch-trellis'
own = root/'runtime-v16'
qa = root/'runtime-v16-qa'
engine = root/'engine-v16'
assert Path.cwd() == own and os.getpriority(os.PRIO_PROCESS, 0) == 10
assert not (own/'frozen.sha256').exists()
phase = sys.argv[1]
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MAX_JOBS='1', PYTHONDONTWRITEBYTECODE='1')
hipcc = '/opt/rocm/core-10.1/bin/hipcc'
library = own/'build-hip/libstrix_trellis.a'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(p, value):
    p.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')

def run(command, log, cwd=own, env=None):
    assert not log.exists(), log
    with log.open('wb') as out:
        out.write(('COMMAND '+json.dumps(list(map(str, command)))+'\n').encode())
        out.flush()
        subprocess.run(list(map(str, command)), cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)

def caches(directory):
    for key, rel in dict(TMPDIR='.tmp', XDG_CACHE_HOME='.cache', HIP_CACHE_DIR='.cache/hip', AMD_COMGR_CACHE_DIR='.cache/comgr').items():
        p=directory/rel;p.mkdir(parents=True, exist_ok=True);os.environ[key]=str(p)

caches(own)
if phase in ('build','build-tail','build-complete'):
    build=own/'build'
    if phase == 'build':
        run(['cmake', '-S', '.', '-B', 'build-hip', '-DCMAKE_BUILD_TYPE=Release', '-DSTRIX_TRELLIS_ENABLE_HIP=ON', '-DSTRIX_TRELLIS_BUILD_BENCHMARKS=ON', '-DSTRIX_TRELLIS_BUILD_TESTING=ON', '-DCMAKE_HIP_COMPILER=/opt/rocm/core-10.1/lib/llvm/bin/clang++', '-DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm/core-10.1', '-DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1', '-DCMAKE_HIP_ARCHITECTURES=gfx1151'], build/'configure.log')
        run(['cmake', '--build', 'build-hip', '--parallel', '1'], build/'compile.log')
        run(['ctest', '--test-dir', 'build-hip', '--output-on-failure'], build/'ctest.log')
        env=dict(os.environ, STRIX_TRELLIS_FIXTURES=str(own/'cpu-fixtures'), PYTHONPATH=str(own/'python'))
        run(['python3','-B','-m','unittest','discover','-s','tests','-p','test_*.py','-v'], build/'python.log', env=env)
        run(['python3','-B','tools/runtime_v16/high_cpu.py','--host','--counterexamples'],build/'high.log')
    if phase != 'build-complete':
        run(['python3','-B','tools/runtime_v16/test_shell_cpu.py',str(build/'shell-proof')],build/'shell-final.log')
    s=(own/'bench/head_linear_bench.hip').read_text();(build/'head_crosscheck_helpers.hpp').write_text(s[:s.index('int main(')],encoding='utf-8')
    run([hipcc,'-O3','-ffp-contract=off','-std=c++17','--offload-arch=gfx1151','-Iinclude','-Ibuild','-Ibench','bench/head_crosscheck.hip','-x','none',library,'-o','build-hip/head_crosscheck'],build/'head-final-build.log')
    (build/'private-abi.hip').write_text('#include "backend.hpp"\nstatic_assert(sizeof(strix_trellis::DeviceWeights)==56);\nstatic_assert(offsetof(strix_trellis::DeviceWeights,codebook)==44);\nstatic_assert(offsetof(strix_trellis::DeviceWeights,fused_output_capable)==48);\nstatic_assert(offsetof(strix_trellis::DeviceWeights,fused_both_capable)==49);\n',encoding='utf-8')
    run([hipcc,'-std=c++17','--offload-arch=gfx1151','-Isrc','-c','build/private-abi.hip','-o','build/private-abi.o'],build/'private-abi.log')
elif phase == 'qa':
    assert not qa.exists()
    qa.mkdir();(qa/'build').mkdir();(qa/'v14').mkdir();direct=qa/'direct';direct.mkdir();(direct/'build').mkdir()
    caches(qa)
    for name in ('src','include','tests','tools','python','experiments'):
        shutil.copytree(own/name,qa/name)
    shutil.copytree(root/'runtime-v14b-qa/reference',qa/'reference')
    for name in ('control_gemm.hip','control_gemv_lds.hip','v14_helpers.hpp','v14_native_probe.hpp','reference-inputs.json','control-provenance.json'):
        shutil.copyfile(root/'runtime-v14b-qa/build'/name,qa/'build'/name)
    api=(own/'src/api.hip').read_text();layout=api[api.index('struct strix_context_impl {'):api.index('namespace {')]
    (qa/'build/qa_private_layout.hpp').write_text(layout,encoding='utf-8')
    # All in-process baseline factories compile against TEN matching fields.
    s=(own/'tests/runtime_v14b_complete_qa.hip').read_text().replace('queried[4]','queried[5]').replace('i<4','i<5').replace('attributes==4','attributes==5').replace('attributes=4','attributes=5')
    s='#include <sys/resource.h>\n'+s
    terminal='no_queries();r.close();std::cout<<'
    measure='rusage usage{};require(getrusage(RUSAGE_SELF,&usage)==0 && uint64_t(usage.ru_maxrss)*1024<(2ull<<30),"Host bound");std::cout<<"HOST_PEAK="<<uint64_t(usage.ru_maxrss)*1024<<" GPU_EXPLICIT_PEAK_WITH_WEIGHT_RESERVE="<<qa_peak+(128ull<<20)<<\'\\n\';'
    assert terminal in s
    s=s.replace(terminal,measure+terminal)
    (qa/'v14/runtime_v16_v14b_qa.hip').write_text(s,encoding='utf-8')
    shutil.copyfile(own/'tests/runtime_v14_extent.hpp',qa/'v14/runtime_v14_extent.hpp')
    # Add actual bounded-allocation and host RSS counters to the existing harness.
    helper=(qa/'build/v14_helpers.hpp').read_text()
    budget='static size_t qa_live=0,qa_peak=0;\nstatic void qa_reserve(size_t n){if(n+qa_live+(128ull<<20)>=(900ull<<20))throw std::runtime_error("Explicit GPU bound");qa_live+=n;qa_peak=std::max(qa_peak,qa_live);}\n'
    helper=helper.replace('struct Device {',budget+'struct Device {\n size_t bytes=0;')
    helper=helper.replace('explicit Device(size_t n) { if (n) hip_check(hipMalloc(&p, n)); }','explicit Device(size_t n):bytes(n){qa_reserve(n);try{if(n)hip_check(hipMalloc(&p,n));}catch(...){qa_live-=n;throw;}}')
    helper=helper.replace('~Device() { if (p) (void)hipFree(p); }','~Device(){if(p)(void)hipFree(p);qa_live-=bytes;}')
    (qa/'build/v14_helpers.hpp').write_text(helper,encoding='utf-8')
    run([hipcc,'-O3','-ffp-contract=off','-std=c++17','--offload-arch=gfx1151','-Iinclude','-Isrc','-Ibuild','v14/runtime_v16_v14b_qa.hip','build/control_gemm.hip','build/control_gemv_lds.hip','-x','none',library,'-Wl,--wrap=hipFuncGetAttributes','-Wl,-Map,v14/link.map','-o','v14/runtime_v16_v14b_qa'],qa/'v14/build.log',qa)
    previous=root/'runtime-v15-qa/complete-v2'
    # Canonical complete-v2 source, including negatives and actual G1 Q1 refs.
    for name in ('runtime_v15_complete.hip','runtime_v15_case.hpp','q1_complete.hpp','negative.hpp','v13_q1_control.hip'):
        shutil.copyfile(own/'tools/runtime_v16/native'/name,direct/name)
    (direct/'src').symlink_to(qa/'src',target_is_directory=True)
    (direct/'include').symlink_to(qa/'include',target_is_directory=True)
    for name in ('v15_helpers.hpp','literal_producer.hip'):
        shutil.copyfile(previous/'build'/name,direct/'build'/name)
    (direct/'build/qa_private_layout.hpp').write_text(layout,encoding='utf-8')
    # Frozen compact inputs preserve actual original first-eight F32 rows.
    shutil.copytree(previous/'inputs',direct/'inputs')
    for name in ('inputs.json','linked.json'):
        shutil.copyfile(previous/'build'/name,direct/'build'/('inherited-'+name))
    inputs=json.loads((direct/'build/inherited-inputs.json').read_text())
    for name,digest in inputs['inputs'].items():assert sha(direct/'inputs'/name)==digest
    run([hipcc,'-O3','-ffp-contract=off','-std=c++17','--offload-arch=gfx1151','-Iinclude','-Ibuild','runtime_v15_complete.hip','build/literal_producer.hip','-x','none','-Wl,--whole-archive',library,'-Wl,--no-whole-archive','-Wl,--wrap=hipFuncGetAttributes','-Wl,-Map,build/qa.map','-o','build/runtime_v16_complete'],direct/'build/compile.log',direct)
    # Old V14b archive only in separate opaque-C-ABI control processes.
    shutil.copyfile(own/'tools/runtime_v16/native/control.hip',direct/'control.hip')
    for source,binary in [('control.hip','v14b_control'),('v13_q1_control.hip','v14b_q1_control')]:
        run([hipcc,'-O3','-ffp-contract=off','-std=c++17','--offload-arch=gfx1151','-Iinclude','-Ibuild',source,'-x','none','-Wl,--whole-archive',root/'runtime-v14b/build-hip/libstrix_trellis.a','-Wl,--no-whole-archive','-o','build/'+binary],direct/'build'/(binary+'.log'),direct)
    shutil.copyfile(own/'tools/runtime_v16/native-root.sh',qa/'native-root.sh')
    run(['bash','-n','native-root.sh'],qa/'native-shell.log',qa)
elif phase == 'qa-refresh':
    direct=qa/'direct';caches(qa)
    run(['python3','-B','tools/runtime_v16/verify_inputs.py'],own/'build/reference-validation.log')
    for name in ('runtime_v15_complete.hip','runtime_v15_case.hpp','q1_complete.hpp','negative.hpp','v13_q1_control.hip','control.hip'):
        shutil.copyfile(own/'tools/runtime_v16/native'/name,direct/name)
    run([hipcc,'-O3','-ffp-contract=off','-std=c++17','--offload-arch=gfx1151','-Iinclude','-Ibuild','runtime_v15_complete.hip','build/literal_producer.hip','-x','none','-Wl,--whole-archive',library,'-Wl,--no-whole-archive','-Wl,--wrap=hipFuncGetAttributes','-Wl,-Map,build/qa-final.map','-o','build/runtime_v16_complete'],direct/'build/compile-final.log',direct)
    for source,binary in [('control.hip','v14b_control'),('v13_q1_control.hip','v14b_q1_control')]:
        run([hipcc,'-O3','-ffp-contract=off','-std=c++17','--offload-arch=gfx1151','-Iinclude','-Ibuild',source,'-x','none','-Wl,--whole-archive',root/'runtime-v14b/build-hip/libstrix_trellis.a','-Wl,--no-whole-archive','-o','build/'+binary],direct/'build'/(binary+'-final.log'),direct)
    shutil.copyfile(own/'tools/runtime_v16/native-root.sh',qa/'native-root.sh')
    run(['bash','-n','native-root.sh'],qa/'native-shell-final.log',qa)
elif phase == 'engine':
    assert not engine.exists();engine.mkdir();caches(engine)
    shutil.copyfile(root/'engine-v14b/CMakeLists.txt',engine/'CMakeLists.txt')
    shutil.copytree(root/'engine-v14b/engine',engine/'engine')
    run(['cmake','-S','.','-B','build','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_PREFIX_PATH=/opt/rocm/core-10.1',f'-DSTRIX_TRELLIS_INCLUDE_DIR={own}/include',f'-DSTRIX_TRELLIS_LIBRARY={library}'],engine/'configure.log',engine)
    run(['cmake','--build','build','--target','strix-trellis-dense-example','--parallel','1'],engine/'compile.log',engine)
elif phase in ('audit','audit-final','audit-v3','audit-v4','finish'):
    run(['python3','-B','tools/runtime_v16/'+('audit' if phase.startswith('audit') else phase)+'.py'],own/'build'/(phase+'.log'))
else:
    raise ValueError(phase)
print('PREPARED_CPU_ONLY_PHASE='+phase)
