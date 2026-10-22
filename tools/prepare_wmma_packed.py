"""CPU-only large split references from immutable independent fixture checkpoints."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import hashlib
import json
import resource
import struct
import numpy as np

resource.setrlimit(resource.RLIMIT_AS, (1900*1024**2, 1900*1024**2))
assert all(os.environ.get(k, '').startswith(str(Path.cwd()/'.tmp')) for k in
           ('TMPDIR', 'XDG_CACHE_HOME', 'AMD_COMGR_CACHE_DIR'))
names = ['layers.0.attn.wq_b', 'layers.0.attn.wo_b',
         'layers.0.ffn.shared_experts.w1', 'layers.0.ffn.shared_experts.w2',
         'layers.0.attn.wq_a', 'layers.0.attn.wo_a.slice.0']
geometries = [(1280,32768,5),(8192,5120,5),(5120,2304,5),
              (2304,5120,5),(5120,1280,6),(4096,1024,5)]
splits = [(1,1,1),(4,2,1),(8,4,2),(4,2,1),(16,8,4),(16,8,4)]
records=[]
for shape, name in enumerate(names):
    path=Path('fixtures',name+'.stgpu')
    offsets={}
    with path.open('rb') as f:
        magic,version,count=struct.unpack('<8sII',f.read(16))
        assert magic==b'STGPU001' and version==1
        for _ in range(count):
            tag,size=struct.unpack('<8sQ',f.read(16))
            tag=tag.rstrip(b'\0').decode()
            assert tag not in offsets
            offsets[tag]=(f.tell(),size)
            f.seek(size,1)
        assert f.tell()==path.stat().st_size
        f.seek(offsets['meta'][0]);i,n,b,cb,marker,rows,bias,seed=struct.unpack('<8I',f.read(32))
    assert (i,n,b)==geometries[shape] and cb==2 and marker==0x83dcd12d and rows>=256
    assert offsets['trellis'][1]==i*n*b//8
    def array(tag,dtype,dims):
        assert offsets[tag][1]==np.prod(dims)*np.dtype(dtype).itemsize
        return np.memmap(path,dtype=dtype,mode='r',offset=offsets[tag][0],shape=dims)
    a=array('a','<f2',(rows,i));q=array('q','<f2',(i,n))
    packed=array('trellis','<u2',(i//16,n//16,16*b))
    # Audit little-endian packed pairing and first/last tile against the bit-ring oracle.
    import sys
    sys.path.insert(0,str(Path('python').resolve()))
    from strix_trellis.codec import decode_tile
    for k,t in [(0,0),(i//16-1,n//16-1)]:
        assert np.array_equal(decode_tile(packed[k,t],b),q[k*16:k*16+16,t*16:t*16+16])
    for m,s in zip((64,128,256),splits[shape]):
        target=Path(f'build/oracles/perf-{shape}-{m}.bin')
        with target.open('wb') as f:
            for z in range(s):
                lo=(i//16)*z//s*16;hi=(i//16)*(z+1)//s*16
                aa=a[:m,lo:hi].astype(np.float64)
                # Bounded columns; every output and every partial, never sampling.
                for absolute in (False,True):
                    result=np.empty((m,n),dtype='<f8')
                    for col in range(0,n,256):
                        qq=q[lo:hi,col:col+256].astype(np.float64)
                        result[:,col:col+256]=(np.abs(aa) if absolute else aa) @ (np.abs(qq) if absolute else qq)
                    result.tofile(f)
        records.append(dict(shape=shape,rows=m,splits=s,inputs=i,outputs=n,bits=b,
                            file=str(target),bytes=target.stat().st_size,
                            sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        print(json.dumps(records[-1]),flush=True)
    del a,q,packed
Path('build/large-oracle-proof.json').write_text(json.dumps(dict(
    status='PASS_CPU_ONLY',endian='little',marker='0x83dcd12d',
    reference='FP64 dot and sumabs from independent full A/Q fixture checkpoints',
    max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,records=records),indent=2)+'\n')
