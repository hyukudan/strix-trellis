"""Bounded synthetic K2 STGPU fixtures; no HIP initialization or downloads."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from strix_trellis.codec import decode_states, decode_tile
from strix_trellis.linear import input_stages_fp32, core_dot_fp32, output_stages_fp32


def decode_block(t):
    # Independent vectorized circular window, logical coordinate map.
    w=t.astype('<u2').view('<u4')
    q=np.empty((t.shape[0],16,t.shape[1],16),dtype='<f2')
    for r in range(16):
        for c in range(16):
            pos=((c&7)<<5)|((r&6)<<2)|(r&1)|((r&8)>>2)|((c&8)>>1)
            start=(2*(pos+1)-16)%512
            idx,shift=divmod(start,32)
            wide=(w[...,idx].astype(np.uint64)<<32)|w[...,(idx+1)%16]
            state=((wide >> (48-shift))&65535).astype(np.uint16)
            q[:,r,:,c]=decode_states(state)
    return q.reshape(t.shape[0]*16,t.shape[1]*16)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path('external-data/gpu-k2-production-tests'))
    a=ap.parse_args()
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    else:
        os.nice(10)
    a.out.mkdir(parents=True,exist_ok=True)
    records=[]
    for case,I,N,bias_on in [('wo_b',8192,5120,True),('q_b',1280,32768,False)]:
        started=time.perf_counter(); seed=20261021+I; R=8
        rng=np.random.Generator(np.random.PCG64(seed))
        t=rng.integers(0,65536,(I//16,N//16,32),dtype=np.uint16)
        np.testing.assert_array_equal(decode_block(t[:2,:2]),decode_tile(t[:2,:2],2).transpose(0,2,1,3).reshape(32,32))
        choices=np.array([-1.5,-.5,-.125,-0.,0.,.0625,.75,1.25],dtype='<f2')
        su=rng.choice(choices,I); sv=rng.choice(choices,N)
        x=rng.normal(0,.125,(R,I)).astype('<f4')
        x[0,:8]=[0.,-0.,2**-24,-2**-24,2**-14,-2**-14,.5,-.25]
        old=rng.normal(0,.2,(R,N)).astype('<f4')
        bias=rng.normal(0,.05,N).astype('<f4') if bias_on else np.empty(0,dtype='<f4')
        stages=input_stages_fp32(x,su); A=stages['core_input_fp16']
        # Only the required 84 MB Q storage is mapped. Decode temporaries <= I*128.
        qpath=a.out/(case+'.q.tmp')
        q=np.memmap(qpath,mode='w+',dtype='<f2',shape=(I,N))
        core=np.empty((R,N),dtype='<f4'); ref=core.copy()
        exact=np.empty((4,N),dtype='<f8'); absolute=exact.copy()
        max_sample=0.
        for col in range(0,N,128):
            b=decode_block(t[:,col//16:col//16+8]); q[:,col:col+128]=b
            core[:,col:col+128]=core_dot_fp32(A,b)
            ref[:,col:col+128]=output_stages_fp32(core[:,col:col+128],sv[col:col+128],bias[col:col+128] if bias_on else None)['output_fp32']
            for s in range(4):
                begin,end=I//16*s//4*16,I//16*(s+1)//4*16
                prod=A[0,begin:end,None].astype(np.float64)*b[begin:end].astype(np.float64)
                exact[s,col:col+128]=prod.sum(axis=0)
                absolute[s,col:col+128]=np.abs(prod).sum(axis=0)
            # Independent FP64 selected-column core check against fixed FP32 stage.
            truth=(A.astype(np.float64)*b[:,0].astype(np.float64)).sum(axis=1)
            err=np.abs(core[:,col].astype(np.float64)-truth)
            bound=(2*I*2**-24)/(1-2*I*2**-24)*(np.abs(A.astype(np.float64))*np.abs(b[:,0].astype(np.float64))).sum(axis=1)
            assert np.all(err<=bound+1e-12)
            max_sample=max(max_sample,float(err.max()))
        sections={'meta':struct.pack('<8I',I,N,2,2,0x83dcd12d,R,int(bias_on),seed),
                  'trellis':t,'suh':su,'svh':sv,'bias':bias,'x':x,'old':old,
                  'scaled':stages['scaled_fp16'],'inh':stages['input_hadamard_fp32'],
                  'a':A,'q':q,'core':core,'ref':ref,'p64':exact,'pabs':absolute}
        path=a.out/(case+'.stgpu')
        with path.open('wb') as f:
            f.write(b'STGPU001'+struct.pack('<II',1,len(sections)))
            for tag,v in sections.items():
                length=len(v) if isinstance(v,bytes) else v.nbytes
                f.write(tag.encode().ljust(8,b'\0')+struct.pack('<Q',length))
                if isinstance(v,bytes): f.write(v)
                elif v.size:
                    # Stream row slices, never duplicate the whole fixture.
                    for chunk in v.reshape(-1, v.shape[-1] if v.ndim>1 else v.size):
                        f.write(chunk.tobytes())
        del sections,q
        qpath.unlink()
        digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
        rec=dict(file=path.name,bytes=path.stat().st_size,sha256=digest,I=I,N=N,rows=R,seed=seed,bias=bias_on,seconds=time.perf_counter()-started,fp64_sample_max_error=max_sample)
        assert rec['bytes']<256*1024**2
        records.append(rec);print(json.dumps(rec),flush=True)
    manifest=dict(format='STGPU001',contract='NUMERICS RN32 -> RN16 -> H32 -> RN16; fixed separate FP32 core; H128 column blocks',numpy=np.__version__,decode_block_columns=128,host_budget_bytes=2*1024**3,fixtures=records)
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__': main()

