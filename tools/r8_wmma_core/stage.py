"""Stage matching V13 sources and the unchanged qualified GPU bodies, CPU only."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--runtime', type=Path, required=True)
p.add_argument('--ours', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
a.out.mkdir(exist_ok=False)
records = {}
def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    records[str(dst.relative_to(a.out))] = dict(source=str(src), sha256=hashlib.sha256(src.read_bytes()).hexdigest())
for folder in ('src', 'include', 'python'):
    for src in (a.runtime/folder).rglob('*'):
        if src.is_file() and '__pycache__' not in src.parts:
            copy(src, a.out/src.relative_to(a.runtime))
for name in ('wmma_decoder.hip', 'wmma_decoder.hpp'):
    copy(root/'src'/name, a.out/'src'/name)
# Only host domain changes. Every device function remains literal source.
path = a.out/'src/wmma_decoder.hip'
s = path.read_text()
old = 'w.input<128 || w.input>8192 || w.input%128 || w.output<1 || w.output>32768 ||'
new = '''w.input<128 || w.input>8192 || w.input%128 || w.output<1 ||
       !(w.output<=32768 || (r.rows==8 && r.total_rows==8 && r.splits==4 &&
          w.input==5120 && w.output==129280 && !r.diagnostic &&
          r.layout==Layout::m16_a16_plain)) ||'''
assert s.count(old) == 1
s=s.replace(old, new)
needle='out.reset();const auto& w=r.weights;'
assert s.count(needle)==1
s=s.replace(needle,needle+'''
    if(r.rows!=8 || r.total_rows!=8 || r.splits!=4 || r.diagnostic ||
       r.layout!=Layout::m16_a16_plain ||
       !((w.input==1280 && w.output==32768 && w.bits==5) ||
         (w.input==8192 && w.output==5120 && w.bits==5) ||
         (w.input==5120 && w.output==129280 && (w.bits==5 || w.bits==6))))
        return hipErrorInvalidValue;''')
path.write_text(s, newline='\n')
for src in Path(__file__).parent.glob('*'):
    if src.is_file():
        copy(src, a.out/'tools/r8_wmma_core'/src.name)
copy(a.ours/'bench/head_fixture.hpp', a.out/'bench/head_fixture.hpp')
# Import only our existing CPU decoder and its own dependencies.
copy(a.ours/'tools/audit_ours_packs.py', a.out/'tools/audit_ours_packs.py')
copy(a.ours/'tests/test_reference.py', a.out/'tests/test_reference.py')
(a.out/'source-inputs.json').write_text(json.dumps(records, indent=2)+'\n')
print(a.out)
