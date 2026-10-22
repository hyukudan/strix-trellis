"""Postprocess logged P errors to prove the original tolerance against unrounded FP64.

The GPU driver logs the complete-P error against RN32(FP64), plus its direct
FP64 error/sumabs. Triangle bounds below cover the unrounded FP64 reference;
they do not change a tolerance, regenerate a GPU output, or execute HIP.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from pathlib import Path
import hashlib
import json
import re
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('evidence',type=Path)
p.add_argument('oracles',type=Path)
a=p.parse_args()
proof=json.loads((a.evidence/'data-proof.json').read_text())
records=[]
for campaign in ('first','confirm'):
    log=a.evidence/campaign/'native.log'
    if not log.exists():continue
    text=log.read_text()
    errors=re.findall(r'NUMERIC label=S4_PARTIAL_CPU values=(\d+) maxabs=(\S+) worst_row_nrms=(\S+) PASS',text)
    assert len(errors)==16
    for g,geometry in enumerate(proof['records']):
        i,n,b=geometry['geometry'];path=a.oracles/f'partial-{g}.bin'
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        entry=geometry['files']['partial-fp64.bin']
        assert path.stat().st_size==entry['bytes'] and digest==entry['sha256']
        ref=np.memmap(path,mode='r',dtype='<f8',shape=(4,2,8,n))
        for split in range(4):
            values,ma,nrms=errors[g*4+split];assert int(values)==8*n
            # Twelve significant decimal digits were printed. This guard is
            # conservative for their rounding and CPU norm arithmetic.
            ma=np.nextafter(float(ma)*(1+1e-10),np.inf)
            nrms=np.nextafter(float(nrms)*(1+1e-10),np.inf)
            for row in range(8):
                exact=ref[split,0,row];rounded=exact.astype('<f4').astype(np.float64)
                quant=rounded-exact
                exact_norm=np.sqrt(np.sum(exact*exact))
                rounded_norm=np.sqrt(np.sum(rounded*rounded))
                quant_norm=np.sqrt(np.sum(quant*quant))
                norm_lower=max(exact_norm,1e-15)
                nrms_bound=(nrms*max(rounded_norm,1e-15)+quant_norm)/norm_lower*(1+1e-10)
                abs_bound=(ma+float(np.max(np.abs(quant))))*(1+1e-10)
                abs_limit=5e-4*max(1.,float(np.max(np.abs(exact))))
                assert np.isfinite(exact).all() and nrms_bound<=3e-5 and abs_bound<=abs_limit
                records.append(dict(campaign=campaign,geometry=g,split=split,row=row,
                    values=n,NRMS_FP64_upper_bound=float(nrms_bound),
                    maxabs_FP64_upper_bound=float(abs_bound),maxabs_limit=abs_limit))
        del ref
result=dict(status='PASS_CPU_ONLY',GPU_executed=False,tolerance_unchanged=True,
    proof='||P-FP64|| <= ||P-RN32(FP64)|| + ||RN32(FP64)-FP64||; full-row logged maxima conservatively cover every value, with decimal/norm rounding guards',
    checked_rows=len(records),records=records,
    max_NRMS_bound=max(r['NRMS_FP64_upper_bound'] for r in records),
    max_absolute_bound=max(r['maxabs_FP64_upper_bound'] for r in records))
(a.evidence/'fp64-tolerance-proof.json').write_text(json.dumps(result,indent=2)+'\n',newline='\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
