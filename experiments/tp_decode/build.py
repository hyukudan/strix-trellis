"""Serial nice10 CPU-only build/ELF inspection. Never executes a HIP-linked file."""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys
import time

root=Path.home()/'gpu-site/scratch-trellis/tp-decode-v1/host-r2'
assert Path.cwd()==root and os.getpriority(os.PRIO_PROCESS,0)>=10
assert not (root/'frozen.sha256').exists()
attempt=sys.argv[1];assert re.fullmatch('[a-z0-9-]+',attempt)
build=root/'build'/attempt;build.mkdir(parents=True,exist_ok=False)
for key,relative in dict(TMPDIR='.tmp',XDG_CACHE_HOME='.cache',HIP_CACHE_DIR='.cache/hip',AMD_COMGR_CACHE_DIR='.cache/comgr').items():
    path=root/relative;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
os.environ.update(MAX_JOBS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
src='experiments/tp_decode';hipcc='/opt/rocm/core-10.1/bin/hipcc';llvm='/opt/rocm/core-10.1/lib/llvm/bin'
def run(args,log):
    with (build/log).open('xb') as f:
        f.write(('COMMAND '+json.dumps(list(map(str,args)))+'\n').encode());f.flush()
        subprocess.run(list(map(str,args)),stdout=f,stderr=subprocess.STDOUT,check=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
run(['hostname'],'host.txt');run([hipcc,'--version'],'compiler.txt')
run(['bash','-n',src+'/root_payload.sh'],'payload-syntax.log')
lib=root/'lib/libstrix_trellis_v16.a'
assert sha(lib)=='8eb47b095b6ffdc3444e61da5d52dc2cf922d87da8aaec3ef4b4db8c7fed1db8'
run(['gcc','-std=c11','-Wall','-Wextra','-Werror',src+'/layout_test.c','-o',build/'layout_cpu'],'layout-build.log')
run(['g++','-std=c++17','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-Wno-misleading-indentation',src+'/cpu_test.cpp','-o',build/'contract_cpu'],'cpu-build.log')
heap=['malloc','calloc','realloc','free','_Znwm','_Znam','_ZdlPv','_ZdaPv','_ZdlPvm','_ZdaPvm']
run(['g++','-std=c++17','-O2','-fPIC','-shared',src+'/heap_dso.cpp','-o',build/'libheap_dso.so'],'heap-dso-build.log')
run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-Wno-misleading-indentation',src+'/hooks_cpu.cpp',
     *['-Wl,--wrap='+n for n in heap],'-L'+str(build),'-lheap_dso','-Wl,-rpath,'+str(build),'-o',build/'hooks_cpu'],'hooks-build.log')
for name in ('layout_cpu','contract_cpu','hooks_cpu'):
    run(['ldd',build/name],name+'-dependencies.log')
    deps=(build/(name+'-dependencies.log')).read_text();assert not re.search('amdhip|hsa-runtime',deps,re.I)
    run([build/name],name+'.log')
run(['g++','-std=c++17','-O1','-g','-ffp-contract=off','-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer',
     '-Wall','-Wextra','-Werror','-Wno-misleading-indentation',src+'/cpu_test.cpp','-o',build/'contract_sanitized_cpu'],'sanitized-build.log')
run(['ldd',build/'contract_sanitized_cpu'],'sanitized-dependencies.log')
assert not re.search('amdhip|hsa-runtime',(build/'sanitized-dependencies.log').read_text(),re.I)
run([build/'contract_sanitized_cpu'],'sanitized-cpu.log')
flags=['-std=c++17','-O2','-ffp-contract=off','--offload-arch=gfx1151','-Iinclude','-Isrc','-Wall','-Wextra','-Werror','-Wno-misleading-indentation']
run([hipcc,*flags,'-M',src+'/native.hip',src+'/kernels.hip',src+'/hooks.hip'],'dependencies.d')
wraps=['hipMalloc','hipFree','hipMemcpy','hipMemcpyAsync','hipMemset','hipMemsetAsync','hipStreamSynchronize','hipDeviceSynchronize',
       'hipEventSynchronize','hipFuncGetAttributes','hipGetDeviceProperties','hipGetDevice','hipPointerGetAttributes','hipMemGetAddressRange',
       'hipEventRecord','hipEventElapsedTime','malloc','calloc','realloc','free','_Znwm','_Znam','_ZdlPv','_ZdaPv','_ZdlPvm','_ZdaPvm']
started=time.monotonic()
run(['/usr/bin/time','-v',hipcc,*flags,src+'/native.hip',src+'/kernels.hip',src+'/hooks.hip','-x','none',
     '-Wl,--whole-archive',lib,'-Wl,--no-whole-archive',*['-Wl,--wrap='+n for n in wraps],
     '-Wl,-Map='+str(build/'link.map')+',--cref,--emit-relocs','-o',build/'tp_native'],'hip-build.log')
run([llvm+'/llvm-objcopy','--dump-section','.hip_fatbin='+str(build/'linked.fatbin'),build/'tp_native'],'extract.log')
run(['python3','-B',src+'/inspect_elf.py',build],'inspect.log')
run([llvm+'/llvm-objdump','-dr','--demangle',build/'tp_native'],'host-disassembly.txt')
run([llvm+'/llvm-readelf','--symbols','--relocations','--dynamic',build/'tp_native'],'host-symbols.txt')
run([llvm+'/llvm-objdump','-r',lib],'archive-relocations.txt')
manifest=dict(cpu_only=True,executed_hip=False,build_seconds=time.monotonic()-started,
              archive_sha256=sha(lib),binary_sha256=sha(build/'tp_native'),wraps=wraps,
              compiler_sha256=sha(Path(llvm+'/clang++')),cpu_tests='PASS',native_admission='OPEN')
manifest['HIP_runtime_library_sha256']=sha(Path('/opt/rocm/core-10.1/lib/libamdhip64.so'))
(build/'build.json').write_text(json.dumps(manifest,indent=2)+'\n')
run(['python3','-B',src+'/audit_link.py',build],'link-audit.log')
print(json.dumps(manifest,indent=2))
