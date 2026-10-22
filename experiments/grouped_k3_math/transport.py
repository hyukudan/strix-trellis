"""CPU-only transport. No queue, GPU entry point, or HIP execution action."""
from pathlib import Path
import hashlib
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'reports/grouped-k3-math-r1-data'
REMOTE = 'gpu-site/scratch-trellis/grouped-k3-math-r1'
DATA.mkdir(exist_ok=True)


def run(args, log):
    print('COMMAND', repr(args), flush=True)
    r = subprocess.run(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (DATA / log).write_bytes(r.stdout.replace(b'\r\n',b'\n'))
    print('EXIT', r.returncode, 'LOG', log, flush=True)
    return r.returncode


def source_bundle():
    sources = sorted(p for folder in ('src', 'include', 'experiments/grouped_k3_math')
                     for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    sources += [ROOT / p for p in ('AGENTS.md', 'tools/verify_grouped_k3_adapter.py',
        'reports/grouped-k3-adapter-pins.json', 'reports/grouped_k3_private_guard.hpp',
        'reports/grouped_k3_guard_test.cpp', 'reports/grouped_k3_cpu_proof.py',
        'reports/real-k3-expert-data/fixture-manifest.json',
        'reports/real-k3-expert-data/source-manifest.json',
        'docs/GROUPED_K3_ADAPTER.md', 'docs/GROUPED_K3_R1.md', 'docs/API.md', 'docs/NUMERICS.md',
        'reports/grouped-k3-followup.md', 'reports/grouped-k3-math-review.md',
        'tools/audit_grouped_k3_math_r1.py', 'tools/audit_grouped_k3_math.py',
        'tools/audit_grouped_k3_math_review.py', 'tools/audit_grouped_k3_pilot.py')]
    sources += [p for name in ('cpu-manifest.json','resource-bounds.json','cpu-prepare.log',
        'unique-scalar-proof.json','lexical-cpu.log') if (p:=DATA/name).is_file()]
    with tarfile.open(DATA / 'source.tar', 'w') as tar:
        for p in sources:
            tar.add(p, arcname=p.relative_to(ROOT).as_posix())
    (DATA / 'source.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest() +
        '  ' + p.relative_to(ROOT).as_posix() + '\n' for p in sources), newline='\n')


action = sys.argv[1]
if action == 'bootstrap':
    source_bundle()
    for node in ('node1', 'node2'):
        cmd = (f'test ! -e ~/{REMOTE} && mkdir -p ~/{REMOTE}/{{.tmp,.cache/hip,.cache/comgr,build,bin,lib,data,runs,attempts}} && '
               f'cp ~/gpu-site/scratch-trellis/runtime-v13/build-hip/libstrix_trellis.a ~/{REMOTE}/lib/')
        assert run(['ssh', '-o', 'BatchMode=yes', node, cmd], node + '-bootstrap.log') == 0
        assert run(['scp', '-q', str(DATA / 'source.tar'), str(DATA / 'source.sha256'), node + ':' + REMOTE + '/'], node + '-upload.log') == 0
        assert run(['ssh', '-o', 'BatchMode=yes', node, f'cd ~/{REMOTE} && nice -n 10 tar -xf source.tar && sha256sum -c source.sha256'], node + '-extract.log') == 0
elif action == 'build':
    # Both nodes are built sequentially, with one compiler process/job each.
    for node in ('node1', 'node2'):
        code = run(['ssh', '-o', 'BatchMode=yes', node,
                    f'cd ~/{REMOTE} && nice -n 10 bash experiments/grouped_k3_math/build.sh'], node + '-build.log')
        print('BUILD_RESULT', node, code, flush=True)
        assert code==0
elif action == 'refresh':
    source_bundle()
    for node in ('node1', 'node2'):
        attempt = sys.argv[2]
        assert attempt.isdecimal()
        assert run(['ssh', '-o', 'BatchMode=yes', node, f'cd ~/{REMOTE} && test ! -e frozen.sha256 && test ! -e attempts/{attempt} && mkdir attempts/{attempt} && cp -a build source.tar source.sha256 attempts/{attempt}/'], node + '-retain-'+attempt+'.log') == 0
        assert run(['scp', '-q', str(DATA / 'source.tar'), str(DATA / 'source.sha256'), node + ':' + REMOTE + '/'], node + '-refresh.log') == 0
        assert run(['ssh', '-o', 'BatchMode=yes', node, f'cd ~/{REMOTE} && nice -n 10 tar -xf source.tar && sha256sum -c source.sha256'], node + '-refresh-extract.log') == 0
        assert run(['ssh','-o','BatchMode=yes',node,f'cd ~/{REMOTE} && if test -e experiments/grouped_k3_math/inspect.py; then test ! -e attempts/legacy-inspect.py && mv experiments/grouped_k3_math/inspect.py attempts/legacy-inspect.py; fi'],node+'-legacy-name.log')==0
elif action == 'collect':
    for node in ('node1', 'node2'):
        target = DATA / node
        target.mkdir(exist_ok=True)
        assert run(['scp', '-qr', node + ':' + REMOTE + '/build', str(target)], node + '-collect.log') == 0
elif action == 'data':
    cpu=Path(sys.argv[2]).resolve()
    with tarfile.open(DATA/'cpu-data.tar','w') as tar:
        for p in sorted(cpu.glob('*')):tar.add(p,arcname='data/cpu/'+p.name)
        payloads=Path('external-data/eaq-k3-expert10-ref')
        for p in sorted(payloads.glob('*.bin')):tar.add(p,arcname='data/payloads/'+p.name)
    for node in ('node1','node2'):
        assert run(['scp','-q',str(DATA/'cpu-data.tar'),node+':'+REMOTE+'/'],node+'-data-upload.log')==0
        cmd=(f'cd ~/{REMOTE} && test ! -e frozen.sha256 && test ! -e data/cpu && '
             'nice -n 10 tar -xf cpu-data.tar && mkdir data/originals && '
             'cp ~/gpu-site/scratch-trellis/real-k3-v12/data/layers.0.ffn.experts.10.w{1,2,3}.stgpu data/originals/ && '
             'nice -n 10 python3 -I -S -B experiments/grouped_k3_math/verify_data.py')
        assert run(['ssh','-o','BatchMode=yes',node,cmd],node+'-data-verify.log')==0
elif action == 'evidence':
    for node in ('node1','node2'):
        cmd=(f'cd ~/{REMOTE} && nice -n 10 tar -czf evidence.tar.gz '
             'frozen.sha256 source.tar source.sha256 AGENTS.md experiments tools reports docs src include '
             'bin lib build data/cpu/manifest.json data/cpu/unique-scalar-proof.json && '
             'sha256sum evidence.tar.gz && nice -n 10 sha256sum -c --quiet frozen.sha256')
        assert run(['ssh','-o','BatchMode=yes',node,cmd],node+'-evidence.log')==0
        assert run(['scp','-q',node+':'+REMOTE+'/evidence.tar.gz',str(DATA/(node+'-evidence.tar.gz'))],node+'-evidence-collect.log')==0
elif action == 'freeze':
    for node in ('node1','node2'):
        assert run(['ssh','-o','BatchMode=yes',node,f'cd ~/{REMOTE} && nice -n 10 bash experiments/grouped_k3_math/freeze.sh'],node+'-freeze.log')==0
        assert run(['scp','-q',node+':'+REMOTE+'/frozen.sha256',str(DATA/(node+'-frozen.sha256'))],node+'-freeze-collect.log')==0
else:
    raise SystemExit('CPU-only actions: bootstrap, build, collect')
