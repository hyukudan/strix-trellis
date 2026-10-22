"""CPU-only selective fixture extraction; never imports HIP or expands Q."""
from pathlib import Path
import hashlib
import json
import shutil
import struct

own = Path.home() / 'gpu-site/scratch-trellis/fused-both-core'
root = own.parent
assert Path.cwd().resolve() == own.resolve()
out = own / 'fixtures'
out.mkdir(exist_ok=True)
cases = {
    'q_b': ('layers.0.attn.wq_b.stgpu', 1280, 32768),
    'wo_b': ('layers.0.attn.wo_b.stgpu', 8192, 5120),
    'inverse': ('layers.0.ffn.shared_experts.w1.stgpu', 2304, 5120),
    'shared': ('layers.0.ffn.shared_experts.w2.stgpu', 5120, 2304),
}
# Select the existing inverse fixture by explicit metadata if w1 has another shape.
available = sorted((root / 'gpu-dense-prefill-tests').glob('*.stgpu'))
def sections(path):
    result = {}
    with path.open('rb') as f:
        magic, version, count = struct.unpack('<8sII', f.read(16))
        assert magic == b'STGPU001' and version == 1 and count <= 64
        for _ in range(count):
            tag, size = struct.unpack('<8sQ', f.read(16))
            tag = tag.rstrip(b'\0').decode()
            assert tag not in result
            result[tag] = (f.tell(), size)
            f.seek(size, 1)
        assert f.tell() == path.stat().st_size
    return result
def metadata(path):
    sec = sections(path)
    with path.open('rb') as f:
        f.seek(sec['meta'][0])
        return struct.unpack('<8I', f.read(32))
records = []
for name, (_, i, n) in cases.items():
    matches = [p for p in available if metadata(p)[:4] == (i, n, 5, 2)]
    assert matches, (name, i, n)
    source = matches[0]
    sec = sections(source)
    meta = metadata(source)
    assert meta[5] >= 8
    target = out / name
    target.mkdir(exist_ok=True)
    lengths = {'trellis': i*n*5//8, 'suh': i*2, 'svh': n*2,
               'bias': n*4 if meta[6] else 0, 'x': 8*i*4, 'ref': 8*n*4}
    names = {'trellis':'packed', 'suh':'u', 'svh':'v', 'bias':'b', 'x':'x', 'ref':'ref'}
    with source.open('rb') as f:
        for tag, size in lengths.items():
            assert sec[tag][1] >= size
            f.seek(sec[tag][0])
            with (target / (names[tag]+'.bin')).open('wb') as dst:
                left=size
                while left:
                    data=f.read(min(left, 1024*1024)); assert data
                    dst.write(data); left-=len(data)
    records.append({'name': name, 'source': str(source), 'meta': meta})
head = out / 'head'
head.mkdir(exist_ok=True)
for name in ('head.meta', 'head.blob', 'input.f32'):
    source=root / 'gpu-head-eaq-ref-v2' / name
    expected={'head.blob':'685f135a2a0c46f293969f563683a0d26984ffa240ca0fd455fc7d501ea2fd64','head.meta':'4bb8ed2f44cd0f3a28b9e2ae81cbd51918ea58187b44d53a249dddc068f9fbbd','input.f32':'a25e17a36d0bda262c29f6f055fa75d4fb1016b1dbae8dc067d56f33bc6aea20'}
    assert hashlib.file_digest(source.open('rb'),'sha256').hexdigest()==expected[name]
    shutil.copyfile(source, head / name)
(out / 'sources.json').write_text(json.dumps(records, indent=2)+'\n')
for name,digest in {'cuda-r1-a.f16':'9767475a0204e02d6c81831288ebde42716630e24a1115c9c5371571319159b6','cuda-r1-y.f32':'e76721e2572c3da947619b3d6f5ac98db9dff3fe4192b3f4bbcf326fe0c856e3'}.items():
    assert hashlib.file_digest((head/name).open('rb'),'sha256').hexdigest()==digest
files = sorted(p for p in out.rglob('*') if p.is_file())
with (own / 'build/fixtures.sha256').open('w') as f:
    for p in files:
        h=hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
        f.write(f'{h}  {p.relative_to(own).as_posix()}\n')
print('CPU_PREPARED selective existing fixtures; no dense head Q')
