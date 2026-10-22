"""Independent CPU full-column references; bounded column tiles, no dense full Q."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import hashlib
import json
import resource
import time
import numpy as np

resource.setrlimit(resource.RLIMIT_AS, (1900 << 20, 1900 << 20))
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
root = Path('fixtures')

def h128(x):
    x = x.copy().reshape(-1, 128)
    for d in (1, 2, 4, 8, 16, 32, 64):
        blocks = x.reshape(-1, 2*d)
        a, b = blocks[:, :d].copy(), blocks[:, d:].copy()
        blocks[:, :d] = a + b
        blocks[:, d:] = a - b
    return (x * np.float32(float.fromhex('0x1.6a09e6p-4'))).ravel()

# Integer hash plus exact binary64 dyadic arithmetic, one half rounding.
states = np.arange(65536, dtype=np.uint64)
hashed = (states * 0x83dcd12d) & 0xffffffff
total = sum((hashed >> s) & 255 for s in (0, 8, 16, 24))
lut = (((1024 + total.astype(np.int64))*1774 - 1329*2048) * 2.**-18).astype('<f2')
assert np.isfinite(lut).all()
records = []
for name, i, n, bits in [('q_b',1280,32768,5), ('wo_b',8192,5120,5),
                         ('inverse',2304,5120,5), ('shared',5120,2304,5),
                         ('head',5120,129280,6)]:
    start = time.monotonic()
    folder = root / name
    pbytes = i*n*bits//8
    head = name == 'head'
    packed = np.memmap(folder/('head.blob' if head else 'packed.bin'), dtype='<u4',
                       mode='r', shape=(i//16,n//16,bits*8))
    x = np.fromfile(folder/('input.f32' if head else 'x.bin'),'<f4',count=i)
    u = np.fromfile(folder/'head.blob' if head else folder/'u.bin', '<f2',
                    count=i, offset=pbytes if head else 0)
    v = np.fromfile(folder/'head.blob' if head else folder/'v.bin', '<f2',
                    count=n, offset=pbytes+i*2 if head else 0)
    b = np.fromfile(folder/'b.bin','<f4') if (folder/'b.bin').exists() else np.array([],dtype='<f4')
    scaled = (x*u.astype('<f4')).astype('<f2').astype('<f4')
    h = h128(scaled)
    a = h.astype('<f2')
    assert all(np.isfinite(z).all() for z in (x,u,v,scaled,h,a,b))
    y = np.empty(n,dtype='<f4')
    partial = np.empty((4,n),dtype='<f4')
    pos = np.arange(256)
    lane, fragment = pos//8, pos%8
    row = 2*(lane%4)+fragment%2+8*((fragment//2)%2)
    col = lane//4+8*(fragment//4)
    for first in range(0,n,128):
        words = np.asarray(packed[:,first//16:(first+128)//16,:])
        state = np.zeros((i//16,8,256),dtype=np.uint32)
        for j in range(16):
            bit = ((pos+1)*bits-16+j) % (256*bits)
            state = (state << 1) | ((words[:,:,bit//32] >> (31-bit%32)) & 1)
        q = np.empty((i//16,8,16,16),dtype='<f2')
        q[:,:,row,col] = lut[state]
        tile = q.transpose(0,2,1,3).reshape(i,128).astype(np.float64)
        for g in range(4):
            begin, end = (i//128)*g//4*128, (i//128)*(g+1)//4*128
            partial[g,first:first+128] = a[begin:end].astype(np.float64) @ tile[begin:end]
        core = (a.astype(np.float64) @ tile).astype('<f4')
        y[first:first+128] = h128(core)*v[first:first+128].astype('<f4')
        if b.size:
            y[first:first+128] += b[first:first+128]
    assert np.isfinite(y).all() and np.isfinite(partial).all()
    for file, values in [('cpu-y.f32',y),('cpu-p.f32',partial),('cpu-a.f16',a),('cpu-h.f32',h)]:
        values.tofile(folder/file)
    records.append(dict(name=name, input=i, output=n, bits=bits, full_Y=n, full_P=4*n,
                        seconds=time.monotonic()-start,
                        y_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                        p_sha256=hashlib.sha256(partial.tobytes()).hexdigest()))
    print(json.dumps(records[-1]),flush=True)
    del packed, tile, q
print(json.dumps(dict(cases=records, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                     numpy=np.__version__, scope='Independent FP64 core/H4 partial; F32 staged transforms; no GPU')))
