"""Local CPU transport/controller. Never runs the HIP executable outside run.sh.

Invoke from the repository root. Build attempts and raw diagnostics are retained.
"""
from pathlib import Path
import hashlib
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'reports/grouped-k3-abi-data'
REMOTE='gpu-site/scratch-trellis/grouped-k3-abi'
ENV='export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" HIP_CACHE_DIR="$PWD/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr" MAX_JOBS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; '
def run(args,log):
    print('COMMAND',repr(args),flush=True)
    with (DATA/log).open('wb') as out:
        r=subprocess.run(args,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
    print('EXIT',r.returncode,'LOG',log,flush=True)
    if r.returncode: raise SystemExit(r.returncode)

DATA.mkdir(exist_ok=True)
action=sys.argv[1]
if action in ('build','rebuild'):
    sources=list((ROOT/'experiments/grouped-k3-abi').glob('*'))
    sources += [ROOT/p for p in ('reports/grouped_k3_private_guard.hpp','reports/grouped_k3_guard_test.cpp',
        'reports/grouped_k3_cpu_proof.py','reports/grouped-k3-review.md','reports/grouped-k3-cpu-proof.txt',
        'docs/GROUPED_K3_R1.md','src/gemv_policy.hpp','AGENTS.md')]
    with tarfile.open(DATA/'source.tar','w') as tar:
        for p in sources:
            if p.is_file():tar.add(p,arcname=p.relative_to(ROOT).as_posix())
    (DATA/'source.sha256').write_text('\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(ROOT).as_posix() for p in sources if p.is_file())+'\n')
    for node in ('node1','node2'):
        if action=='build':
            run(['ssh','-o','BatchMode=yes',node,
                f'test ! -e ~/{REMOTE} && mkdir -p ~/{REMOTE}/{{.tmp,.cache/hip,.cache/comgr,build,bin,runs,attempts}}'],f'{node}-bootstrap.log')
        run(['scp','-q',str(DATA/'source.tar'),f'{node}:{REMOTE}/source.tar'],f'{node}-upload.log')
        run(['ssh','-o','BatchMode=yes',node,f'cd ~/{REMOTE} && '+ENV+'nice -n 10 tar -xf source.tar'],f'{node}-extract.log')
    previous=list(DATA.glob('build-*.log'))
    run(['ssh','-o','BatchMode=yes','node1',f'cd ~/{REMOTE} && '+ENV+
        'nice -n 10 bash experiments/grouped-k3-abi/build.sh'], f'build-{len(previous)+1:02d}.log')
elif action=='collect':
    for node in ('node1','node2'):
        (DATA/node).mkdir(exist_ok=True)
        run(['scp','-qr',f'{node}:{REMOTE}/build',f'{node}:{REMOTE}/runs',f'{node}:{REMOTE}/attempts',str(DATA/node)],f'{node}-collect.log')
elif action=='queue':
    run(['ssh','-o','BatchMode=yes','node1',f'cd ~/{REMOTE} && '+ENV+
        'nice -n 10 bash experiments/grouped-k3-abi/run.sh'],'controller.log')
elif action=='freeze':
    run(['ssh','-o','BatchMode=yes','node1',f'cd ~/{REMOTE} && '+ENV+
        'nice -n 10 bash experiments/grouped-k3-abi/freeze.sh'],'mirror-freeze.log')
else:
    raise SystemExit('actions: build, collect, queue')
