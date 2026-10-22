"""Before-queue validation of every saved large reference, without GPU access."""
from pathlib import Path
import json
import numpy as np

records=[]
for name,i,n in [('q_b',1280,32768),('wo_b',8192,5120),('inverse',2304,5120),('shared',5120,2304),('head',5120,129280)]:
    folder=Path('fixtures')/name
    y=np.fromfile(folder/'cpu-y.f32','<f4')
    p=np.fromfile(folder/'cpu-p.f32','<f4')
    a=np.fromfile(folder/'cpu-a.f16','<f2')
    h=np.fromfile(folder/'cpu-h.f32','<f4')
    assert len(y)==n and len(p)==4*n and len(a)==len(h)==i
    assert all(np.isfinite(z).all() for z in (y,p,a,h))
    ref=np.fromfile(folder/('cuda-r1-y.f32' if name=='head' else 'ref.bin'),'<f4',count=n)
    e=y.astype('d')-ref.astype('d')
    maxabs=float(abs(e).max());nrms=float(np.linalg.norm(e)/max(np.linalg.norm(ref.astype('d')),1e-15))
    assert maxabs<=5e-4*max(1,float(abs(ref).max())) and nrms<=3e-5
    records.append(dict(name=name,all_A=i,all_H=i,all_P=4*n,all_Y=n,maxabs=maxabs,nrms=nrms,
                        reference='Saved standalone CUDA R1' if name=='head' else 'Existing original-F32 fixture reference'))
print(json.dumps(dict(passed=True,cases=records,gpu_executed=False),indent=2))
