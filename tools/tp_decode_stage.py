"""Stage only this experiment and read-only production dependencies into a NEW scope."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import subprocess
import tarfile

repo=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('node',choices=('node1','node2'));parser.add_argument('phase',choices=('stage','refresh','build','collect','fixtures'));parser.add_argument('--attempt',default='initial')
args=parser.parse_args();remote='gpu-site/scratch-trellis/tp-decode-v1/host-r2'
data=repo/'reports/tp-decode-private-data';data.mkdir(exist_ok=True)
external=Path('external-data/work/tp-decode-v1-build');external.mkdir(exist_ok=True)
log=data/(args.node+'-'+args.phase+'-'+args.attempt+'.log');assert not log.exists()
def call(cmd):
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    with log.open('ab') as f:f.write(('COMMAND '+json.dumps(cmd)+'\n').encode());f.write(p.stdout)
    print(p.stdout.decode('utf-8',errors='replace'),flush=True);p.check_returncode()
if args.phase in ('stage','refresh'):
    paths=[p for prefix in ('src','include','experiments/tp_decode') for p in (repo/prefix).rglob('*') if p.is_file()]
    archive=external/(args.node+'-'+args.attempt+'.tar');records={}
    with tarfile.open(archive,'w') as tar:
        for p in paths:
            relative=p.relative_to(repo).as_posix();raw=p.read_bytes().replace(b'\r\n',b'\n');records[relative]=hashlib.sha256(raw).hexdigest()
            info=tarfile.TarInfo(relative);info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    (data/(args.node+'-sources-'+args.attempt+'.json')).write_text(json.dumps(records,indent=2)+'\n')
    if args.phase=='stage':
        call(['ssh','-o','BatchMode=yes',args.node,'test ! -e ~/'+remote+' && mkdir ~/'+remote])
    call(['scp',str(archive),args.node+':'+remote+'/source-'+args.attempt+'.tar'])
    command='cd ~/'+remote+' && test ! -e frozen.sha256 && tar -xf source-'+args.attempt+'.tar'
    # Remove only our renamed, previously staged files inside this NEW scope.
    command+=" && python3 -I -S -B -c 'from pathlib import Path; root=Path.cwd(); assert root==Path.home()/\"gpu-site/scratch-trellis/tp-decode-v1/host-r2\"; [p.unlink() for p in (root/\"experiments/tp_decode/inspect.py\",root/\"experiments/tp_decode/statistics.py\") if p.is_file()]'"
    if args.phase=='stage':command+=' && mkdir lib && cp ~/gpu-site/scratch-trellis/build-hip/libstrix_trellis_fused_v16.a lib/libstrix_trellis_v16.a'
    call(['ssh','-o','BatchMode=yes',args.node,command])
elif args.phase=='build':
    call(['ssh','-o','BatchMode=yes',args.node,'cd ~/'+remote+' && nice -n 10 python3 -B experiments/tp_decode/build.py '+args.attempt])
elif args.phase=='collect':
    target=data/args.node/args.attempt;target.mkdir(parents=True,exist_ok=False)
    # Small artifacts only; binaries, fatbins, ISA and large maps remain outside Git.
    raw=external/args.node/args.attempt;raw.mkdir(parents=True,exist_ok=False)
    call(['scp','-r',args.node+':'+remote+'/build/'+args.attempt+'/.',str(raw)])
    for p in raw.iterdir():
        if p.suffix in ('.json','.log','.txt') and p.stat().st_size<256000:(target/p.name).write_bytes(p.read_bytes())
elif args.phase=='fixtures':
    manifest=Path('external-data/work/tp-decode-v1-cpu/refs/manifest.json')
    m=json.loads(manifest.read_text(encoding='utf-8'));assert m['passed'] and len(m['fixtures'])==12
    call(['ssh','-o','BatchMode=yes',args.node,'cd ~/'+remote+' && test ! -e fixtures && mkdir fixtures'])
    for item in m['fixtures']:
        p=manifest.parent/item['file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
        call(['scp',str(p),args.node+':'+remote+'/fixtures/'+p.name])
    call(['scp',str(manifest),args.node+':'+remote+'/fixtures/manifest.json'])
    checks=''.join(x['sha256']+'  '+x['file']+'\n' for x in m['fixtures'])
    sums=external/'fixtures.sha256';sums.write_text(checks,encoding='ascii')
    call(['scp',str(sums),args.node+':'+remote+'/fixtures/fixtures.sha256'])
    call(['ssh','-o','BatchMode=yes',args.node,'cd ~/'+remote+'/fixtures && sha256sum -c fixtures.sha256 && chmod a-w *.tp manifest.json fixtures.sha256'])
