"""Freeze all actual inputs and compiled artifacts before the first GPU request."""
from pathlib import Path
import hashlib
import json
import tarfile

root=Path(__file__).resolve().parents[2]
assert root==Path.home()/'gpu-site/scratch-trellis/r8-wmma-core'
assert not (root/'runs').exists()
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
proof=json.loads((root/'build/proof.json').read_text())
assert proof['status']=='PASS_CPU_BUILD_ISA_ONLY' and not proof['uncovered_HIGH_conditions']
assert json.loads((root/'build/host-domain.json').read_text())['status']=='PASS_CPU_ONLY'
assert json.loads((root/'build/isa-audit.json').read_text())['status']=='PASS'
assert json.loads((root/'build/data-proof.json').read_text())['status']=='PASS_CPU_ONLY'
for g in range(4):
    d=root/'data'/str(g);record=json.loads((d/'manifest.json').read_text())
    for name,entry in record['files'].items():
        p=d/name;assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],p
with tarfile.open(root/'source-actual.tar','w') as t:
    for name in ('src','include','python','tools','tests','source-inputs.json'):
        t.add(root/name,arcname=name)
files=[]
for name in ('src','include','python','tools','tests','control','data','reference','build'):
    for p in sorted((root/name).rglob('*')):
        assert not p.is_symlink(),p
        if name=='build' and (p.name.startswith('controller-') or p.name=='peer-freeze.log'):continue
        if p.is_file() and '__pycache__' not in p.parts:files.append(p)
files.extend(root/name for name in ('source-actual.tar','source-inputs.json'))
out=root/'frozen.sha256'
with out.open('x') as f:
    for p in files:f.write(sha(p)+'  '+str(p.relative_to(root))+'\n')
print('FROZEN',len(files),sha(out))
