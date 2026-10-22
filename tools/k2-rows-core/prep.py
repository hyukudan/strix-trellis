"""Independent, bounded H128 FP64 oracles. No project numerical imports or HIP."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time
import resource
os.nice(10)
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS,(1536*1024**2,1536*1024**2))
import numpy as np


def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def sections(path):
    result={}
    with open(path,'rb') as f:
        assert f.read(8)==b'STGPU001'
        version,count=struct.unpack('<II',f.read(8));assert version==1 and count<64
        for _ in range(count):
            tag=f.read(8).rstrip(b'\0').decode();size,=struct.unpack('<Q',f.read(8))
            assert tag not in result and size<256*1024**2
            result[tag]=(f.tell(),size);f.seek(size,1)
        assert f.tell()==path.stat().st_size
    return result


def view(path,sec,name,dtype,shape):
    offset,size=sec[name];assert size==np.prod(shape)*np.dtype(dtype).itemsize
    return np.memmap(path,mode='r',offset=offset,dtype=dtype,shape=shape) if size else np.empty(shape,dtype=dtype)


def h32(values):
    x=np.array(values,dtype='<f4',copy=True);blocks=x.reshape(-1,128)
    for stride in (1,2,4,8,16,32,64):
        z=blocks.reshape(-1,2,stride);a=z[:,0].copy();b=z[:,1].copy()
        z[:,0]=a+b;z[:,1]=a-b
    x*=np.float32(1/np.sqrt(128));return x


def lut():
    # Exact dyadic rational, rounded once by CPython's IEEE half encoder.
    raw=bytearray()
    for s in range(65536):
        h=(s*0x83dcd12d)&0xffffffff
        total=sum((h>>(8*j))&255 for j in range(4))
        raw+=struct.pack('<e',((1024+total)*1774-1329*2048)/262144)
    assert hashlib.sha256(raw).hexdigest()=='bc48d02cb1c14939dc47b90f870dd689d63e3b1ab69a157e65538db68677a6c8'
    return np.frombuffer(raw,dtype='<f2')


def decode(words,table):
    # Each state assembled from its sixteen individual circular stream bits.
    # No two-word window or lane reader from the kernel is reused.
    kt,nt,_=words.shape
    q=np.empty((kt,16,nt,16),dtype='<f2')
    for i in range(16):
        for c in range(16):
            pos=32*(c%8)+4*(i&6)+(i&1)+(i&8)//4+(c&8)//2
            state=np.zeros((kt,nt),dtype=np.uint16)
            for bit in range(2*(pos+1)-16,2*(pos+1)):
                b=bit%512
                state=(state<<1)|((words[:,:,b//32]>>(31-b%32))&1).astype(np.uint16)
            q[:,i,:,c]=table[state]
    return q.reshape(kt*16,nt*16)


def write_fresh(path,data):
    digest=hashlib.sha256(data).hexdigest()
    if path.exists():assert sha(path)==digest,('stale derived file',str(path))
    else:
        with open(path,'xb') as f:f.write(data)
    return digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fixtures',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    table=lut();records=[]
    assert len(list(Path('/proc/self/task').iterdir()))==1
    for case in ('wo_b','q_b'):
        start=time.monotonic();path=args.fixtures/(case+'.stgpu');sec=sections(path)
        meta=view(path,sec,'meta','<u4',(8,));I,N=map(int,meta[:2]);assert tuple(meta[2:4])==(2,2) and meta[5]==8
        packed=view(path,sec,'trellis','<u4',(I//16,N//16,16))
        A=view(path,sec,'a','<f2',(8,I));X=view(path,sec,'x','<f4',(8,I));su=view(path,sec,'suh','<f2',(I,))
        scaled=(X*su.astype('<f4')).astype('<f2');inh=h32(scaled);freshA=inh.astype('<f2')
        for tag,actual in (('scaled',scaled),('inh',inh),('a',freshA)):
            expected=view(path,sec,tag,actual.dtype,actual.shape)
            assert actual.tobytes()==expected.tobytes(),('input stage bytes',case,tag)
        Q=view(path,sec,'q','<f2',(I,N));p64=np.zeros((4,8,N),dtype='<f8');pabs=np.zeros_like(p64)
        for col in range(0,N,128):
            q=decode(packed[:,col//16:col//16+8,:],table)
            assert q.nbytes<=2*1024**2
            assert q.tobytes()==Q[:,col:col+128].tobytes(),('all Q bytes',case,col)
            for s in range(4):
                lo=(I//16)*s//4*16;hi=(I//16)*(s+1)//4*16
                b=q[lo:hi].astype('<f8')
                for r in range(8):
                    products=A[r,lo:hi].astype('<f8')[:,None]*b
                    p64[s,r,col:col+128]=products.sum(axis=0,dtype=np.float64)
                    pabs[s,r,col:col+128]=np.abs(products).sum(axis=0,dtype=np.float64)
            del q
        data=b'K2RFP641'+struct.pack('<4I',I,N,4,8)+p64.tobytes()+pabs.tobytes()
        digest=write_fresh(args.out/(case+'.fp64'),data)
        rec=dict(case=case,input_sha256=sha(path),reference_sha256=digest,I=I,N=N,S=4,rows=8,
                 full_Q_bytes_compared=I*N*2,A_exact=True,p64_shape=[4,8,N],seconds=time.monotonic()-start)
        records.append(rec);print(json.dumps(rec),flush=True)
    manifest=dict(producer_sha256=sha(Path(__file__)),reader='bit serial circular 16 bits; logical i,c mapping',
                  lut_sha256=hashlib.sha256(table.tobytes()).hexdigest(),numpy=np.__version__,records=records,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threads=len(list(Path('/proc/self/task').iterdir())),
                  policy='nice +10; affinity one CPU; RLIMIT_AS 1536 MiB; BLAS/OMP 1; Q slice <=2 MiB; no Rows,I,N product')
    assert manifest['threads']==1
    # Timing is observational, so preserve an existing manifest only if all immutable identities agree.
    dest=args.out/'provenance.json'
    if dest.exists():
        old=json.loads(dest.read_text());assert old['producer_sha256']==manifest['producer_sha256'] and old['lut_sha256']==manifest['lut_sha256']
        for a,b in zip(old['records'],records):
            for key in ('case','input_sha256','reference_sha256','I','N','S','rows','full_Q_bytes_compared','A_exact','p64_shape'):assert a[key]==b[key]
    else:dest.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
