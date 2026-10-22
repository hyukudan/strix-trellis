"""Performance references: original weights/scales, deterministic X64, alpha1/beta0. CPU only."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from pathlib import Path
import argparse
import hashlib
import json
import platform
import struct
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'reports'))
import grouped_k3_cpu_proof as proof


def sha(b):
    return hashlib.sha256(b).hexdigest()


def h64(x):
    a = np.array(x, dtype=np.float64, copy=True)
    for step in (1, 2, 4, 8, 16, 32, 64):
        v = a.reshape(-1, 2, step)
        l, r = v[:, 0].copy(), v[:, 1].copy()
        v[:, 0], v[:, 1] = l+r, l-r
    return a / np.sqrt(128.)


def write_case(path, i, n, s, g, arrays):
    with path.open('wb') as f:
        f.write(b'GK3CPU01'+struct.pack('<6I', i, n, s, g, 3, len(arrays)))
        for tag, array in arrays.items():
            raw = array.tobytes() if isinstance(array, np.ndarray) else array
            assert len(tag) <= 8
            f.write(tag.encode().ljust(8, b'\0')+struct.pack('<Q', len(raw)))
            f.write(raw)
    return dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path.read_bytes()),
                tensors={k:dict(bytes=len(v.tobytes() if isinstance(v,np.ndarray) else v),
                    sha256=sha(v.tobytes() if isinstance(v,np.ndarray) else v)) for k,v in arrays.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures',type=Path,default=Path('external-data/gpu-real-k3-expert10-tests'))
    p.add_argument('--payloads',type=Path,default=Path('external-data/eaq-k3-expert10-ref'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
        ctypes.windll.kernel32.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
        assert ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    assert not args.output.exists(),'fresh output only'
    args.output.mkdir(parents=True)
    fixture_manifest=ROOT/'reports/real-k3-expert-data/fixture-manifest.json'
    source_manifest=ROOT/'reports/real-k3-expert-data/source-manifest.json'
    assert sha(fixture_manifest.read_bytes())=='d8d823bcad8c5e44963a85cef36b692dedb33062a82560cbe961be48959dfdd7'
    assert sha(source_manifest.read_bytes())=='bdb5d7d45fa3722d9da569fbf54829530d4d09648a8393ac14d654f7b1095bbc'
    records=json.loads(fixture_manifest.read_text())['fixtures'];pins={};cases=[]
    # Only one geometry's decoded Q and reference buffers are active at a time.
    for i,n,s,which in ((5120,2304,16,(0,2)),(2304,5120,4,(1,))):
        fixtures=[];qs=[]
        for index in which:
            r=records[index];f=proof.load_fixture(args.fixtures/r['file'],r)
            for tensor,info in r['source_tensors'].items():
                raw=(args.payloads/info['file']).read_bytes()
                assert len(raw)==info['bytes'] and sha(raw)==info['sha256']
                field=tensor.rsplit('.',1)[1]
                assert raw==(struct.pack('<I',0x83dcd12d) if field=='mul1' else f[{'trellis':'packed','suh':'u','svh':'v'}[field]].tobytes())
                pins[info['file']]=info['sha256']
            fixtures.append(f);qs.append(proof.decode_independent(f))
        for g in range(64):
            f=fixtures[g%len(fixtures)];q=qs[g%len(qs)]
            rng=np.random.default_rng(202610216400+i+g)
            x=rng.uniform(-.12,.12,i).astype('<f4');u=f['u'];v=f['v']
            assert np.isfinite(x).all() and np.isfinite(u).all() and np.isfinite(v).all()
            a=proof.h_reference((x*u.astype(np.float32)).astype(np.float16).astype(np.float32)).astype('<f2')
            assert proof.same(a,proof.h_wave((x*u.astype(np.float32)).astype(np.float16).astype(np.float32)).astype('<f2'))
            p64=np.empty((s,n),'<f8');absolute=np.empty((s,n),'<f8')
            for split in range(s):
                lo,hi=(i//16)*split//s*16,(i//16)*(split+1)//s*16;av=a[lo:hi].astype(np.float64)
                for col in range(0,n,512):
                    block=q[lo:hi,col:col+512].astype(np.float64)
                    p64[split,col:col+512]=av@block;absolute[split,col:col+512]=np.abs(av)@np.abs(block)
            total=np.zeros(n,np.float32)
            for row in p64.astype(np.float32):total+=row
            y=proof.h_reference(total)*v.astype(np.float32)
            ah=h64(x.astype(np.float64)*u.astype(np.float64));core=np.empty(n,np.float64)
            for col in range(0,n,512):core[col:col+512]=ah@q[:,col:col+512].astype(np.float64)
            y64=h64(core)*v.astype(np.float64)
            arrays=dict(packed=f['packed'],x=x,suh=u,svh=v,bias=b'',old=np.full(n,np.nan,'<f4'),scalars=np.array([1.,0.],'<f4'),a=a,p64=p64,sumabs=absolute,y=y,y64=y64)
            assert all(np.isfinite(z).all() for z in (a,p64,absolute,y,y64))
            record=write_case(args.output/f'i{i}-g{g:02d}.gk3',i,n,s,g,arrays)
            record.update(input=i,output=n,splits=s,group=g,weights=f['record']['file'],derived_packed=False)
            cases.append(record);print('CPU_CASE',i,g,record['sha256'],flush=True)
        del fixtures,qs,f,q,block
    for r in records:assert sha((args.fixtures/r['file']).read_bytes())==r['sha256']
    for name,digest in pins.items():assert sha((args.payloads/name).read_bytes())==digest
    if os.name=='nt':
        class Counters(ctypes.Structure):
            _fields_=[('cb',ctypes.c_uint32),('faults',ctypes.c_uint32)]+[(name,ctypes.c_size_t) for name in ('peak_ws','ws','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
        counters=Counters();counters.cb=ctypes.sizeof(counters)
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint32]
        assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(counters),counters.cb)
        peak=counters.peak_ws
    else:
        import resource
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    assert peak<2*1024**3
    manifest=dict(schema=1,cases=cases,groups=[4,8,16,32],copies=64,seed=202610216400,alpha=1,beta=0,bias=False,
        original_scales=True,original_packed=True,cloned_real_expert_not_distinct_trained_experts=True,
        q_correspondence='full independent circular-bit and V1 comparison against full hashed STGPU Q',
        payload_pins=pins,fixture_pins={r['file']:r['sha256'] for r in records},threads=1,peak_host_bytes=peak,
        tolerances=dict(maxabs_scale=5e-4,nrms=3e-5),native_fdot2_identity=False,command=sys.argv,
        python=sys.version,numpy=np.__version__,platform=platform.platform(),gpu_executed=False)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('PASS CPU references only; native gates OPEN; peak_host_bytes',peak)

if __name__=='__main__':main()
