"""Verify the finite V15 addition and preserved qualified V14b source."""
from pathlib import Path
import hashlib
import json
import subprocess

repo=Path(__file__).resolve().parents[2]
def raw(path):return (repo/path).read_bytes().replace(b'\r\n',b'\n')
def git(rev,path):return subprocess.check_output(['git','show',rev+':'+path],cwd=repo).replace(b'\r\n',b'\n')
def sha(raw):return hashlib.sha256(raw).hexdigest()
base='2a3ad4e';v15='f451d1f'
additions={'src/api.hip','src/backend.hip','src/backend.hpp','src/fused_both_policy.hpp','src/gemv_fused_both.hpp','src/gemv_fused_both.hip'}
for name in additions:assert raw(name)==git(v15,name),name
unchanged=[]
for directory in ('src','include'):
    for p in sorted((repo/directory).glob('*')):
        if not p.is_file():continue
        relative=p.relative_to(repo).as_posix()
        if relative not in additions:
            assert raw(relative)==git(base,relative),relative
            unchanged.append(dict(path=relative,LF_sha256=sha(raw(relative))))
backend=raw('src/backend.hip').decode()
stripped=backend.replace('#include "gemv_fused_both.hpp"\n','').replace('#include "fused_both_policy.hpp"\n','')
a=stripped.index('    FusedBothLayout direct{};');b=stripped.index('    auto* transformed',a)
assert stripped[:a]+stripped[b:]==git(base,'src/backend.hip').decode()
cmake=raw('CMakeLists.txt').decode().replace(' src/gemv_fused_both.hip','')
assert cmake==git(base,'CMakeLists.txt').decode()
prototype=git('b79f058','experiments/fused_both/kernel.hip').decode()
body=prototype[prototype.index('using Pair'):prototype.index('\nstatic size_t dynamic_bytes')]
assert body in raw('src/gemv_fused_both.hip').decode()
assert raw('tools/runtime_v16/high_checks.cpp')==git('5012c48','tools/runtime_v15_review_checks.cpp')
runner=raw('tools/run_runtime_v16.sh').decode()
for token in ('order=(v16 v14b)','order=(v14b v16)','$root/runtime-v14b/build-hip/','"$TRELLIS_SOURCE/frozen.sha256"','(cd "$root/engine-v16" && sha256sum -c --quiet frozen.sha256'):
    assert token in runner,token
assert 'runtime-v13/build-hip/' not in runner
api=raw('src/api.hip').decode();launch=backend[backend.index('hipError_t launch_linear'):]
assert launch.index('bytes==SIZE_MAX')<launch.index('fused_both_layout')<launch.index('input_transform<<<')
assert api.count('query_fused_both_resources(')==1
for forbidden in ('hipFuncSetAttribute','hipSetDevice'):
    assert forbidden not in api+raw('src/gemv_fused_both.hip').decode()
result=dict(state='CPU_FINITE_SOURCE_PROVENANCE_PASS',base=base,V15_addition=v15,production_additions=sorted(additions),preserved=unchanged,prototype_body_sha256=sha(body.encode()),WMMA13_K2_R8_and_public_storage_unchanged=True,CMake_preserves_both_V14b_objects=True,source_math_rewrite=False)
print(json.dumps(result,indent=2))
