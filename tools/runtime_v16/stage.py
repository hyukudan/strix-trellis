"""Stage tracked source bytes and run CPU/build phases in new V16 scopes only."""
from pathlib import Path
import hashlib
import io
import json
import subprocess
import sys
import tarfile

repo = Path(__file__).resolve().parents[2]
node, phase = sys.argv[1:]
assert node in ('node1', 'node2')
assert phase in ('stage', 'build', 'build-tail', 'build-complete', 'qa', 'qa-refresh', 'engine', 'audit', 'audit-final', 'audit-v3', 'audit-v4', 'finish', 'collect')
data = repo / 'reports/runtime-v16-data'
data.mkdir(exist_ok=True)
remote = 'gpu-site/scratch-trellis/runtime-v16'
log = data / (node + '-' + phase + '.log')
assert not log.exists(), 'Keep every prior attempt log'

def call(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with log.open('ab') as output:
        output.write(('COMMAND ' + json.dumps(command) + '\n').encode('utf-8'))
        output.write(result.stdout)
    print(result.stdout.decode('utf-8', errors='replace'), flush=True)
    result.check_returncode()

if phase == 'stage':
    paths = subprocess.check_output(['git', 'ls-files'], cwd=repo).decode().splitlines()
    paths = [p for p in paths if p == 'CMakeLists.txt' or p.startswith(
        ('src/', 'include/', 'tests/', 'bench/', 'tools/', 'python/', 'docs/',
         'experiments/fused_both/', 'reports/runtime-v14b-data/'))]
    paths += [str(p.relative_to(repo)).replace('\\', '/') for p in (repo/'tools/runtime_v16').glob('*') if p.is_file()]
    paths += ['tools/run_runtime_v16.sh', 'src/fused_both_policy.hpp', 'src/gemv_fused_both.hpp', 'src/gemv_fused_both.hip']
    archive = data / 'source.tar'
    records = {}
    with tarfile.open(archive, 'w') as tar:
        for relative in sorted(set(paths)):
            p = repo / relative
            raw = p.read_bytes().replace(b'\r\n', b'\n')
            records[relative] = hashlib.sha256(raw).hexdigest()
            info = tarfile.TarInfo(relative)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
    (data/'local-source.json').write_text(json.dumps(records, indent=2)+'\n', encoding='utf-8')
    call(['ssh', '-o', 'BatchMode=yes', node, f'test ! -e ~/{remote} && mkdir ~/{remote}'])
    call(['scp', str(archive), f'{node}:{remote}/source.tar'])
    call(['ssh', '-o', 'BatchMode=yes', node, f'cd ~/{remote} && tar -xf source.tar && nice -n 10 python3 -B tools/runtime_v16/prepare.py'])
elif phase == 'collect':
    target = data / node
    target.mkdir(exist_ok=True)
    call(['scp', '-r', f'{node}:{remote}/delivery/.', str(target)])
else:
    command = f'cd ~/{remote} && export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" HIP_CACHE_DIR="$PWD/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 && nice -n 10 python3 -B tools/runtime_v16/remote.py {phase}'
    call(['ssh', '-o', 'BatchMode=yes', node, command])
