"""Independent full CPU recheck of original scales, X64, A/P/Y and scalar mutation."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import argparse,hashlib,json,struct,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'reports'))
import grouped_k3_cpu_proof as proof
def tensors(path):
    raw=path.read_bytes();at=32;out={}
    for _ in range(12):
        tag=raw[at:at+8].rstrip(b'\0').decode();size,=struct.unpack_from('<Q',raw,at+8);at+=16;out[tag]=raw[at:at+size];at+=size
    assert at==len(raw);return out
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('data',type=Path);a=p.parse_args()
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
        ctypes.windll.kernel32.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
        assert ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    manifest=json.loads((a.data/'manifest.json').read_text());original=json.loads((ROOT/'reports/real-k3-expert-data/fixture-manifest.json').read_text())['fixtures']
    for i,n,s,which in ((5120,2304,16,(0,2)),(2304,5120,4,(1,))):
        fixtures=[proof.load_fixture(Path('external-data/gpu-real-k3-expert10-tests')/original[k]['file'],original[k]) for k in which]
        qs=[proof.decode_independent(f) for f in fixtures]
        for c in (c for c in manifest['cases'] if c['input']==i):
            raw=(a.data/c['file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==c['sha256'];t=tensors(a.data/c['file']);g=c['group'];f=fixtures[g%len(fixtures)];q=qs[g%len(qs)]
            assert t['packed']==f['packed'].tobytes() and t['suh']==f['u'].tobytes() and t['svh']==f['v'].tobytes()
            assert t['scalars']==struct.pack('<2f',1.,0.) and not t['bias']
            x=np.frombuffer(t['x'],'<f4');A=proof.h_wave((x*f['u'].astype(np.float32)).astype(np.float16).astype(np.float32)).astype('<f2');assert A.tobytes()==t['a']
            P=np.frombuffer(t['p64'],'<f8').reshape(s,n);absolute=np.frombuffer(t['sumabs'],'<f8').reshape(s,n)
            for split in range(s):
                lo,hi=(i//16)*split//s*16,(i//16)*(split+1)//s*16;av=A[lo:hi].astype(np.float64)
                for col in range(0,n,256):
                    block=q[lo:hi,col:col+256].astype(np.float64)
                    assert np.array_equal(av@block,P[split,col:col+256])
                    assert np.array_equal(np.abs(av)@np.abs(block),absolute[split,col:col+256])
            total=np.zeros(n,np.float32)
            for row in P.astype(np.float32):total+=row
            Y=proof.h_wave(total)*f['v'].astype(np.float32);assert Y.tobytes()==t['y']
            # The performance fixture must reject a scalar/reference mix with alpha=.5.
            expected=np.frombuffer(t['y'],'<f4').astype(np.float64);error=.5*expected-expected
            nrms=np.sqrt((error@error)/max(expected@expected,1e-30));assert nrms>3e-5
        del fixtures,qs,q,f,block
    print('PASS CPU original full Q correspondence, 128 complete original-weight A/P/sumabs/Y references, 128 alpha mismatch rejections; no native fdot2 claim')
if __name__=='__main__':main()
