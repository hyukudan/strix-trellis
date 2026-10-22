"""CPU comparison of full saved tensors; never initializes HIP/CUDA."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

def compare(actual, reference):
    assert len(actual) == len(reference) and all(map(math.isfinite, actual + reference))
    error = max(abs(a-b) for a,b in zip(actual,reference))
    scale = max(map(abs,reference))
    nrms = math.sqrt(math.fsum((a-b)**2 for a,b in zip(actual,reference))/max(math.fsum(b*b for b in reference),1e-30))
    assert error <= 5e-4*max(1,scale) and nrms <= 3e-5, (error, scale, nrms)
    return dict(values=len(actual),maxabs=error,maxref=scale,nrms=nrms,passed=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate',type=Path);p.add_argument('cuda',type=Path)
    p.add_argument('--cpu',type=Path,help='Strict full 8x129280 original-input CPU oracle, little endian f32')
    a=p.parse_args();n=129280;count=8*n
    raw=(a.candidate/'r8-y.f32').read_bytes();assert len(raw)==count*4
    actual=list(struct.unpack('<%df'%count,raw));records=[]
    for kind,path in [('CUDA',a.cuda/'r8-y.f32')]+([('CPU',a.cpu)] if a.cpu else []):
        data=path.read_bytes();assert len(data)==count*4
        ref=list(struct.unpack('<%df'%count,data))
        records.append(dict(source=kind,path=str(path),sha256=hashlib.sha256(data).hexdigest(),
            aggregate=compare(actual,ref),rows=[compare(actual[r*n:(r+1)*n],ref[r*n:(r+1)*n]) for r in range(8)]))
    assert (a.candidate/'r8-a.f16').read_bytes()==(a.cuda/'r8-a.f16').read_bytes()
    print(json.dumps(dict(candidate_sha256=hashlib.sha256(raw).hexdigest(),full_CPU_gate='PASS' if a.cpu else 'PENDING_ROOT_ORACLE',references=records),indent=2))

if __name__=='__main__':main()
