"""CPU/build transport only. No GPU discovery, queue, request or execution action."""
from pathlib import Path
import argparse,hashlib,json,subprocess,tarfile
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'reports/grouped-k3-perf-r1-data'
REMOTE='gpu-site/scratch-trellis/grouped-k3-perf-r1/runs/host-r2'
def run(args,log):
    r=subprocess.run(args,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if (DATA/log).exists():
        n=1
        while (DATA/(log+f'.previous-{n}')).exists():n+=1
        (DATA/log).rename(DATA/(log+f'.previous-{n}'))
    (DATA/log).write_bytes(r.stdout.replace(b'\r\n',b'\n'))
    print(json.dumps(dict(command=args,exit=r.returncode,log=log)),flush=True)
    if r.returncode:raise RuntimeError(log)
def bundle():
    files=[]
    for directory in ('src','include','experiments/grouped_k3_math','experiments/grouped_k3_perf'):
        files.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for name in ('AGENTS.md','tools/verify_grouped_k3_adapter.py','reports/grouped_k3_private_guard.hpp',
                 'reports/grouped_k3_guard_test.cpp','reports/grouped-k3-adapter-pins.json',
                 'reports/grouped-k3-math-r1-data/handoff.json'):
        files.append(ROOT/name)
    with tarfile.open(DATA/'source.tar','w') as tar:
        for p in sorted(files):tar.add(p,arcname=p.relative_to(ROOT).as_posix())
    (DATA/'source.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(ROOT).as_posix()+'\n' for p in sorted(files)),encoding='utf-8',newline='\n')
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('bootstrap','refresh','build','data','freeze','collect'))
    parser.add_argument('--cpu-data',type=Path)
    parser.add_argument('--attempt',type=int)
    args=parser.parse_args();DATA.mkdir(exist_ok=True)
    if args.action in ('bootstrap','refresh'):bundle()
    if args.action=='data':
        assert args.cpu_data and (args.cpu_data/'manifest.json').is_file()
        with tarfile.open(DATA/'cpu-data.tar','w') as tar:
            for p in sorted(args.cpu_data.glob('*')):tar.add(p,arcname='data/cpu/'+p.name)
    for node in ('node1','node2'):
        prefix=f'cd ~/{REMOTE} && '
        if args.action=='bootstrap':
            cmd=f'test ! -e ~/{REMOTE} && mkdir -p ~/{REMOTE}/{{.tmp,.cache/hip,.cache/comgr,build,bin,lib,data,runs,attempts}} && cp ~/gpu-site/scratch-trellis/grouped-k3-math-r1/lib/libstrix_trellis.a ~/{REMOTE}/lib/ && nice -n 10 cp -a ~/gpu-site/scratch-trellis/grouped-k3-perf/data/cpu ~/{REMOTE}/data/'
            run(['ssh','-o','BatchMode=yes',node,cmd],node+'-bootstrap.log')
        if args.action=='refresh':
            assert args.attempt is not None and args.attempt>0
            run(['ssh','-o','BatchMode=yes',node,prefix+f'test ! -e frozen.sha256 && test ! -e attempts/{args.attempt} && mkdir attempts/{args.attempt} && cp -a source.tar source.sha256 attempts/{args.attempt}/ && mv build bin attempts/{args.attempt}/ && mkdir build bin'],node+f'-retain-{args.attempt}.log')
        if args.action in ('bootstrap','refresh'):
            run(['scp','-rq',str(DATA/'evidence/qa'),str(DATA/'evidence/review'),node+':'+REMOTE+'/data/'],node+'-qa-review-upload.log')
            run(['scp','-q',str(DATA/'source.tar'),str(DATA/'source.sha256'),node+':'+REMOTE+'/'],node+'-upload.log')
            run(['ssh','-o','BatchMode=yes',node,prefix+'test ! -e frozen.sha256 && nice -n 10 tar -xf source.tar && sha256sum -c source.sha256'],node+'-extract.log')
        elif args.action=='build':
            run(['ssh','-o','BatchMode=yes',node,prefix+'nice -n 10 bash experiments/grouped_k3_perf/build.sh'],node+'-build.log')
        elif args.action=='data':
            run(['scp','-q',str(DATA/'cpu-data.tar'),node+':'+REMOTE+'/'],node+'-data-upload.log')
            run(['ssh','-o','BatchMode=yes',node,prefix+'test ! -e frozen.sha256 && test ! -e data/cpu && nice -n 10 tar -xf cpu-data.tar && nice -n 10 python3 -I -S -B experiments/grouped_k3_perf/verify_data.py'],node+'-data.log')
        elif args.action=='freeze':
            run(['ssh','-o','BatchMode=yes',node,prefix+'nice -n 10 bash experiments/grouped_k3_perf/freeze.sh'],node+'-freeze.log')
            run(['scp','-q',node+':'+REMOTE+'/frozen.sha256',str(DATA/(node+'-frozen.sha256'))],node+'-freeze-collect.log')
        elif args.action=='collect':
            cmd=prefix+'nice -n 10 tar -czf runs/evidence.tar.gz source.tar source.sha256 frozen.sha256 bin lib build experiments/grouped_k3_perf data/cpu/manifest.json && nice -n 10 sha256sum -c --quiet frozen.sha256 && sha256sum runs/evidence.tar.gz'
            run(['ssh','-o','BatchMode=yes',node,cmd],node+'-collect.log')
            run(['scp','-q',node+':'+REMOTE+'/runs/evidence.tar.gz',str(DATA/(node+'-evidence.tar.gz'))],node+'-evidence.log')
if __name__=='__main__':main()
