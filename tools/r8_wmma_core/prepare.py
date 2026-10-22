"""Prepare R8 S4 FP64 dot/sumabs with bounded CPU decoding; never touches HIP."""
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
from pathlib import Path
import hashlib
import json
import resource
import shutil
import struct
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'python')]
from audit_ours_packs import decode_block, independent_lut
from strix_trellis.linear import input_stages_fp32

resource.setrlimit(resource.RLIMIT_AS, (1900*1024**2, 1900*1024**2))
base = Path.home()/'gpu-site/scratch-trellis'
assert ROOT == base/'r8-wmma-core'
def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
def copy_range(src, start, length, dst):
    with src.open('rb') as f, dst.open('xb') as g:
        f.seek(start)
        while length:
            b = f.read(min(length, 1<<20)); assert b
            g.write(b); length -= len(b)
def parts(path):
    result = {}
    with path.open('rb') as f:
        magic, version, count = struct.unpack('<8sII', f.read(16))
        assert magic == b'STGPU001' and version == 1 and count <= 64
        for _ in range(count):
            tag, size = struct.unpack('<8sQ', f.read(16))
            tag = tag.rstrip(b'\0').decode(); assert tag not in result
            result[tag] = (f.tell(), size); f.seek(size, 1)
        assert f.tell() == path.stat().st_size
    return result

lut = independent_lut('mul1')
assert hashlib.sha256(lut.tobytes()).hexdigest() == 'bc48d02cb1c14939dc47b90f870dd689d63e3b1ab69a157e65538db68677a6c8'
geoms = [(1280,32768,5),(8192,5120,5),(5120,129280,6),(5120,129280,5)]
records = []
for index,(i,n,b) in enumerate(geoms):
    out = ROOT/'data'/str(index); out.mkdir(parents=True, exist_ok=False)
    packed_bytes = i*n*b//8
    source_files = []
    if index < 2:
        source = base/'gpu-dense-prefill-tests'/(['layers.0.attn.wq_b.stgpu','layers.0.attn.wo_b.stgpu'][index])
        section = parts(source)
        with source.open('rb') as f:
            f.seek(section['meta'][0]); meta = struct.unpack('<8I', f.read(32))
        assert meta[:5] == (i,n,b,2,0x83dcd12d) and meta[5] >= 256
        mapping = [('trellis','packed',packed_bytes),('suh','u',2*i),('svh','v',2*n),
                   ('x','x',8*i*4),('a','a',8*i*2),('ref','y',8*n*4)]
        if meta[6]: mapping.append(('bias','bias',4*n))
        for tag,name,length in mapping:
            assert section[tag][1] >= length
            copy_range(source, section[tag][0], length, out/(name+'.bin'))
        source_files.append(dict(path=str(source), sha256=sha(source)))
        q_source = np.memmap(source, mode='r', dtype='<f2', offset=section['q'][0], shape=(i,n))
        assert section['q'][1] == 2*i*n
    else:
        directory = base/('gpu-head-eaq-ref-v2' if index == 2 else 'ours-head-v12/data/k5')
        reference = ROOT/'reference'/str(index)
        cpu_manifest = json.loads((reference/'manifest.json').read_text())
        assert cpu_manifest['passed'] and cpu_manifest['rows'] == 8 and cpu_manifest['bits'] == b
        for name,entry in cpu_manifest['files'].items():
            src=reference/name
            assert src.stat().st_size == entry['bytes'] and sha(src) == entry['sha256']
        source_files.append(dict(path=str(reference/'manifest.json'),sha256=sha(reference/'manifest.json')))
        manifest = json.loads((directory/'manifest.json').read_text())
        for name, entry in manifest['files'].items():
            src = directory/name
            assert src.stat().st_size == entry['bytes'] and sha(src) == entry['sha256']
            source_files.append(dict(path=str(src), sha256=sha(src)))
        for start,length,name in [(0,packed_bytes,'packed'),(packed_bytes,2*i,'u'),(packed_bytes+2*i,2*n,'v')]:
            copy_range(directory/'head.blob',start,length,out/(name+'.bin'))
        shutil.copyfile(directory/'input.f32',out/'x.bin')
        shutil.copyfile(reference/'r8-a.f16',out/'a.bin')
        shutil.copyfile(reference/'r8-y-ordered.f32',out/'y.bin')
        shutil.copyfile(reference/'r8-y-fp64-core.f32',out/'r8-y-fp64-core.f32')
        shutil.copyfile(reference/'manifest.json',out/'cpu-reference-manifest.json')
        if index == 2:
            shutil.copyfile(base/'runtime-v13-qa/reference/r8-y.f32',out/'cuda-y.bin')
        q_source = None
    (out/'meta.txt').write_text(f'{i} {n} {b} {int((out/"bias.bin").exists())}\n')
    a = np.fromfile(out/'a.bin',dtype='<f2').reshape(8,i)
    x = np.fromfile(out/'x.bin',dtype='<f4').reshape(8,i)
    u = np.fromfile(out/'u.bin',dtype='<f2')
    assert input_stages_fp32(x,u)['core_input_fp16'].tobytes() == a.tobytes()
    t = np.memmap(out/'packed.bin',mode='r',dtype='<u2',shape=(i//16,n//16,16*b))
    dots = np.memmap(out/'partial-fp64.bin',mode='w+',dtype='<f8',shape=(4,2,8,n))
    decoded = 0
    for col in range(0,n,512):
        end = min(n,col+512)
        q = decode_block(t[:,col//16:end//16],b,lut)
        assert q.shape == (i,end-col)
        if q_source is not None: assert q.tobytes() == q_source[:,col:end].tobytes()
        for z in range(4):
            lo=(i//16)*z//4*16; hi=(i//16)*(z+1)//4*16
            aa=a[:,lo:hi].astype(np.float64); qq=q[lo:hi].astype(np.float64)
            dots[z,0,:,col:end]=aa@qq
            dots[z,1,:,col:end]=np.abs(aa)@np.abs(qq)
        decoded += q.size
    dots.flush(); del dots,t,q,q_source
    assert decoded == i*n
    for item in source_files:
        assert sha(Path(item['path'])) == item['sha256'], 'Source changed during preparation'
    record=dict(geometry=[i,n,b], rows=8, splits=4, source=source_files,
                all_A_bytes=True, all_Q_bytes_STGPU=index<2, decoded_weights=decoded,
                max_columns=512, partial_values=4*8*n,
                files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in out.iterdir() if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    records.append(record); print('PASS_CPU_GEOMETRY',index,flush=True)
(ROOT/'build/data-proof.json').write_text(json.dumps(dict(status='PASS_CPU_ONLY',records=records,
    max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2)+'\n')
