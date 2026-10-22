"""Independent read-only full-Q validation and S4 FP64 derivative, CPU only.

Does not import the candidate or predict native fdot2 P. Never writes fixture sources.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse
import hashlib
import json
import struct
import time
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def sections(path):
    out={}
    with path.open('rb') as f:
        magic,version,count=struct.unpack('<8sII',f.read(16))
        assert (magic,version)==(b'STGPU001',1) and count==13
        for _ in range(count):
            tag,length=struct.unpack('<8sQ',f.read(16));tag=tag.rstrip(b'\0').decode()
            assert tag not in out and length<=path.stat().st_size-f.tell()
            out[tag]=(f.tell(),length);f.seek(length,1)
        assert f.tell()==path.stat().st_size
    return out

def h128(a):
    v=np.asarray(a,dtype='<f4').copy().reshape(-1,128);idx=np.arange(128)
    for bit in (1,2,4,8,16,32,64):
        old=v.copy();peer=old[:,idx^bit]
        v[:,(idx&bit)==0]=(old+peer)[:,(idx&bit)==0]
        v[:,(idx&bit)!=0]=(peer-old)[:,(idx&bit)!=0]
    return np.float32(v*np.float32(float.fromhex('0x1.6a09e6p-4'))).reshape(a.shape)

def lut():
    # Exact integer numerator followed by IEEE half RNE. Independent of GPU hash ISA.
    table=[]
    for state in range(65536):
        mixed=(state*0x83dcd12d)&0xffffffff
        total=sum(mixed.to_bytes(4,'little'))
        table.append(struct.pack('<e',((1024+total)*1774-1329*2048)/2**18))
    return np.frombuffer(b''.join(table),dtype='<f2')

def full_q(words,table):
    # Circular MSB-first bit positions; u32 storage is little endian.
    idx=np.arange(256);start=(5*(idx+1)-16)%1280
    states=np.zeros((*words.shape[:2],256),dtype='<u2')
    for bit in range(16):
        pos=(start+bit)%1280
        states=(states<<1)|((words[...,pos//32]>>(31-pos%32))&1).astype('<u2')
    tiles=np.empty((*words.shape[:2],16,16),dtype='<f2')
    for lane in range(32):
        for p in range(8):tiles[...,2*(lane%4)+(p%2)+8*((p//2)%2),lane//4+8*(p//4)]=table[states[...,8*lane+p]]
    return tiles.transpose(0,2,1,3).reshape(words.shape[0]*16,words.shape[1]*16)

def build(source,out):
    assert source.resolve()!=out.resolve() and not out.exists()
    out.mkdir(parents=True);table=lut();records=[];started=time.time()
    validation=source/'validation-primary.json';receipt=json.loads(validation.read_text(encoding='utf-8'))
    assert receipt['passed'] and len(receipt['fixtures'])==4 and all(v['passed'] for v in receipt['fixtures'])
    for layer in (0,21,39):
        for site in ('q_b','o_b'):
            for rank in (0,1):
                name=f'L{layer}.{site}.rank{rank}';directory=source/name
                manifest=directory/'manifest.json';m=json.loads(manifest.read_text(encoding='utf-8'))
                i,n=m['input'],m['output'];assert (i,n)==({'q_b':(1280,16384),'o_b':(4096,5120)}[site])
                assert m['seed']==202610216451+layer*100+(10 if site=='o_b' else 0)+rank
                for tag,info in m['files'].items():assert sha(directory/tag)==info['sha256'] and (directory/tag).stat().st_size==info['bytes']
                path=directory/'fixture.stgpu';parts=sections(path)
                def arr(tag,dtype,shape):
                    assert np.dtype(dtype).itemsize*np.prod(shape)==parts[tag][1]
                    return np.memmap(path,mode='r',dtype=dtype,offset=parts[tag][0],shape=shape)
                for tag,(offset,length) in parts.items():
                    with path.open('rb') as f:f.seek(offset);raw=f.read(length)
                    assert hashlib.sha256(raw).hexdigest()==m['files']['fixture.stgpu']['section_sha256'][tag]
                w=arr('trellis','<u4',(i//16,n//16,40));u=arr('suh','<f2',(i,));v=arr('svh','<f2',(n,))
                x=arr('x','<f4',(8,i));a=arr('a','<f2',(8,i));diskq=arr('q','<f2',(i,n))
                expected_x=np.random.Generator(np.random.PCG64(m['seed'])).normal(0,.125,(8,i)).astype('<f4')
                assert expected_x.tobytes()==x.tobytes()
                scaled=(x.astype('f8')*u.astype('f8')).astype('<f4')
                # struct half provides a second half-rounding implementation.
                rounded=np.frombuffer(b''.join(struct.pack('<e',float(z)) for z in scaled.flat),dtype='<f2').reshape(8,i)
                a_check=h128(rounded.astype('<f4')).astype('<f2');assert a_check.tobytes()==a.tobytes()
                p=np.empty((4,8,n),dtype='<f8');mass=np.empty_like(p)
                bounds=[16*((i//16)*s//4) for s in range(5)]
                for c in range(0,n,128):
                    q=full_q(w[:,c//16:c//16+8],table)
                    assert q.tobytes()==np.ascontiguousarray(diskq[:,c:c+128]).tobytes()
                    for s,(b,e) in enumerate(zip(bounds,bounds[1:])):
                        left=a[:,b:e].astype('f8');right=q[b:e].astype('f8')
                        p[s,:,c:c+128]=left@right;mass[s,:,c:c+128]=np.abs(left)@np.abs(right)
                # Ordered epilogue on FP64 split dots rounded to F32, not native expected P.
                summed=np.zeros((8,n),dtype='<f4')
                for s in range(4):summed=np.float32(summed+p[s].astype('<f4'))
                y=h128(summed)*v.astype('<f4')
                bias=arr('bias','<f4',(n,)) if m['bias'] else np.empty(0,dtype='<f4')
                if m['bias']:y=np.float32(y+bias)
                target=out/(name+'.tp')
                with target.open('xb') as f:
                    f.write(struct.pack('<8s6I',b'TPDECV01',i,n,8,4,int(m['bias']),5))
                    for value in (w,u,v,bias,x,a,p,mass,y):np.ascontiguousarray(value).tofile(f)
                record=dict(name=name,source_manifest_sha256=sha(manifest),source_fixture_sha256=sha(path),
                    seed=m['seed'],source=m['source'],source_prefix=m['source_prefix'],bounds=m['local_bounds_I_N'],
                    file=target.name,bytes=target.stat().st_size,sha256=sha(target),full_Q_values=i*n,
                    independent_Q_PASS=True,independent_A_PASS=True,K_boundaries=bounds,
                    P_contract='staged-half FP64 per logical split, not native expected P')
                records.append(record);print('PASS_REF',name,flush=True)
    report=dict(cpu_only=True,passed=True,fixtures=records,primary_validation_sha256=sha(validation),
                primary_validation=receipt,elapsed_seconds=time.time()-started,
                candidate_native_byte_identity='OPEN',reference_source_sha256=sha(Path(__file__)))
    (out/'manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('PASS all 12 full Q/A references and S4 derivatives; no native claims',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();build(args.source,args.out)
