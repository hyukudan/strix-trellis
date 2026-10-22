"""Private SSH driver. CPU operations only unless the explicit runner queues a lease."""
import argparse
from pathlib import Path
import subprocess
import base64
import io
import tarfile
import shlex

ROOT = Path(__file__).resolve().parents[1]
PREFIX = '''set -euo pipefail
own="$HOME/gpu-site/scratch-trellis/fused-both-confirm"
export TMPDIR="$own/.tmp" TMP="$own/.tmp" TEMP="$own/.tmp"
export XDG_CACHE_HOME="$own/.cache" HIP_CACHE_DIR="$own/.cache/hip"
export AMD_COMGR_CACHE_DIR="$own/.cache/comgr" MAX_JOBS=1
mkdir -p "$TMPDIR" "$HIP_CACHE_DIR" "$AMD_COMGR_CACHE_DIR"
cd "$own"
'''

def remote(node, script):
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', node, 'nice -n 10 bash -s'],
                          input=(PREFIX + script).encode('utf-8'), check=True)

def command(script):
    return 'nice -n 10 bash -c '+shlex.quote(PREFIX+script)

def mirror(source, target, artifacts_only=False):
    # Direct cluster-to-cluster stream; the workstation is not a data relay.
    archive='tar -cf - src experiments '+('' if artifacts_only else 'fixtures ')+'bin build references build-confirm.log prepare-confirm.log'
    remote(source,archive+' | ssh -o BatchMode=yes '+target+' '+shlex.quote(command('tar -xf -'))+'\n')

def fetch(node):
    script="for dir in build runs controller; do if [[ -d $dir ]];then find \"$dir\" -type f \\( -name '*.json' -o -name '*.jsonl' -o -name '*.txt' -o -name '*.sha256' -o -name '*.log' -o -name '*.stderr' -o -name '*.started' -o -name '*.pid' -o -name '*.cpp' \\) -print0;fi;done | tar --null -T - -czf -"
    data=subprocess.check_output(['ssh','-o','BatchMode=yes',node,command(script)])
    destination=ROOT/'reports/fused-both-confirm-data'/node
    destination.mkdir(parents=True,exist_ok=True)
    tarfile.open(fileobj=io.BytesIO(data),mode='r:gz').extractall(destination,filter='data')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('node', choices=['node1', 'node2'])
    p.add_argument('script', type=Path, nargs='?')
    p.add_argument('--stage', action='store_true')
    p.add_argument('--mirror-from', choices=['node1','node2'])
    p.add_argument('--artifacts-only',action='store_true')
    p.add_argument('--fetch', action='store_true')
    args = p.parse_args()
    if args.fetch:
        fetch(args.node)
    elif args.mirror_from:
        mirror(args.mirror_from,args.node,args.artifacts_only)
    elif args.stage:
        data=io.BytesIO()
        with tarfile.open(fileobj=data,mode='w:gz') as archive:
            files=list((ROOT/'src').glob('*'))+list((ROOT/'experiments/fused_both').glob('*'))+list((ROOT/'experiments/fused_both_confirm').glob('*'))
            files += [ROOT/'reports/fused-both-review-proof.py', ROOT/'reports/resources-v11.json']
            for file in files:
                if file.is_file():
                    content=file.read_bytes().replace(b'\r\n',b'\n')
                    info=tarfile.TarInfo(file.relative_to(ROOT).as_posix()); info.size=len(content)
                    archive.addfile(info,io.BytesIO(content))
            for name in ('r8-y-fp64-core.f32','manifest.json'):
                archive.add(Path('external-data/work/head-full-cpu-v13')/name,arcname='references/'+name)
        encoded=base64.b64encode(data.getvalue()).decode()
        remote(args.node,"python3 - <<'PY'\nimport base64,io,tarfile\ntarfile.open(fileobj=io.BytesIO(base64.b64decode('"+encoded+"')),mode='r:gz').extractall('.',filter='data')\nPY\n")
    else:
        # Run the mirrored file with closed stdin. Scheduler-internal SSH calls
        # must never consume the remaining controller commands from bash -s.
        relative=args.script.resolve().relative_to(ROOT).as_posix()
        remote(args.node, 'bash '+shlex.quote(relative)+' </dev/null\n')
