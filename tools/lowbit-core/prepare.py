"""Bounded single-thread CPU fixtures, using our independent Python codec.

Never materializes Q larger than I*128. No network or model-cache reads.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
from strix_trellis.codec import dequantize_rotated, unpack_states, decode_states
from strix_trellis.linear import input_stages_fp32, output_stages_fp32

def priority():
    if os.name=='nt':
        kernel=ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=ctypes.c_void_p
        kernel.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
        assert kernel.SetPriorityClass(kernel.GetCurrentProcess(),0x4000)
    else: os.nice(10)

def write(path,parts):
    with path.open('wb') as f:
        f.write(b'LOWBIT01'+struct.pack('<I',len(parts)))
        for key,value in parts.items():
            payload=value if isinstance(value,bytes) else value.tobytes()
            f.write(key.encode().ljust(16,b'\0')+struct.pack('<Q',len(payload)));f.write(payload)

def fixture(out,i,n,k,s,full):
    rng=np.random.default_rng(210000+k*10000+i*8+n+s)
    packed=rng.integers(0,65536,(i//16,n//16,16*k),dtype='<u2')
    suh=rng.choice(np.array([-1.5,-.5,.125,.75,2],dtype='<f2'),i)
    svh=rng.choice(np.array([-2,-.25,.125,.75,1.5],dtype='<f2'),n)
    bias=rng.normal(0,.125,n).astype('<f4');old=rng.normal(0,.25,n).astype('<f4')
    x=rng.normal(0,.2,i).astype('<f4')
    # Exact RN16 ties and adjacent RN32 values, plus signed zeros and subnormals.
    x[:8]=[0.,-0.,2**-24,-2**-24,1+2**-11,np.nextafter(np.float32(1+2**-11),np.float32(2)),.125,-.5]
    if full:a=input_stages_fp32(x,suh)['core_input_fp16'].astype('<f2')
    else:
        a=rng.normal(0,.25,i).astype('<f2')
        a[:8]=np.array([0.,-0.,2**-24,-2**-24,65504,-65504,2**-14,-2**-14],dtype='<f2')
    ref=np.zeros((s,n),dtype='<f4');bound=np.zeros((s,n),dtype='<f8')
    core=np.zeros(n,dtype='<f4');totalabs=np.zeros(n,dtype='<f8')
    for col in range(0,n,128):
        width=min(128,n-col)
        q=dequantize_rotated(packed[:,col//16:(col+width)//16],k,'mul1').astype('<f4')
        target=slice(col,col+width)
        for row in range(i):
            product=np.float32(a[row])*q[row]
            core[target]+=product;totalabs[target]+=abs(product.astype('f8'))
        for split in range(s):
            begin=(i//16)*split//s*16;end=(i//16)*(split+1)//s*16
            for row in range(begin,end):
                product=np.float32(a[row])*q[row]
                ref[split,target]+=product;bound[split,target]+=abs(product.astype('f8'))
        del q
    parts={'meta':struct.pack('<5I',i,n,k,s,int(full)), 'packed':packed,'a':a,
           'partial':ref,'sumabs':bound,'core':core,'coreabs':totalabs,
           'suh':suh,'svh':svh,'bias':bias,'old':old,'x':x}
    if full:parts['output']=output_stages_fp32(core,svh,bias)['output_fp32'].astype('<f4')
    name=f'k{k}-i{i}-n{n}-s{s}.lbc';write(out/name,parts)
    print(name,flush=True)
    return name

def exhaustive(out):
    for k in (3,4):
        rng=np.random.default_rng(20261021+k)
        words=8*k;count=256*k+4+257
        rings=np.zeros((count,words),dtype='<u4')
        for bit in range(256*k):rings[bit,bit//32]=1<<(bit%32)
        rings[256*k:256*k+4]=np.array([0,0xffffffff,0xaaaaaaaa,0x55555555],dtype='<u4')[:,None]
        rings[256*k+4:]=rng.integers(0,2**32,(257,words),dtype='<u4')
        states=unpack_states(rings.view('<u2'),k).astype('<u4')
        write(out/f'reader{k}.lbc',{'rings':rings,'states':states})
    lut=decode_states(np.arange(65536,dtype='<u4'),'mul1').view('<u2')
    write(out/'hash.lbc',{'lut':lut})

def main():
    priority();p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=args.out;out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    assert not (out/'manifest.json').exists(),'Never overwrite a frozen fixture set'
    small=[];large=[]
    for k in (3,4):
        for i in (16,80,128,176,272):
            for n in (16,128):
                for s in (1,3,11):small.append(fixture(out,i,n,k,s,False))
        for i,n,s in ((5120,2304,16),(2304,5120,4),(5120,2304,4)):
            large.append(fixture(out,i,n,k,s,True))
    exhaustive(out)
    (out/'cases.txt').write_bytes(('\n'.join(small+large)+'\n').encode('ascii'))
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
    manifest=dict(files=files,small=len(small),large=len(large),seconds=time.monotonic()-started,
                  numpy=np.__version__,python=sys.version,threads=1,q_max_bytes=5120*128*4)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
