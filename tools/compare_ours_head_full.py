"""Compare every GPU head output to both frozen complete CPU references."""
from array import array
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def floats(path, count):
    raw = path.read_bytes()
    if len(raw) != 4*count:
        raise ValueError('Unexpected full-reference length: '+str(path))
    values = array('f');values.frombytes(raw)
    if not all(math.isfinite(x) for x in values):
        raise ValueError('Nonfinite full-reference values')
    return values


def compare(a, b):
    error = max(abs(x-y) for x,y in zip(a,b))
    scale = max(abs(x) for x in b)
    nrms = math.sqrt(math.fsum((x-y)**2 for x,y in zip(a,b))/max(math.fsum(x*x for x in b),1e-30))
    if error > 5e-4*max(1.,scale) or nrms > 3e-5:
        raise ValueError('Full head numerical gate failed')
    return dict(values=len(a),maxabs=error,maxref=scale,nrms=nrms)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('candidate',type=Path);p.add_argument('reference',type=Path)
    args=p.parse_args()
    if sys.byteorder!='little':raise ValueError('Little endian required')
    manifest=json.loads((args.reference/'manifest.json').read_text())
    if not manifest['passed'] or manifest['bits'] not in (4,5):raise ValueError('Expected frozen OUR K4/K5 CPU reference')
    for name, info in manifest['files'].items():
        path=args.reference/name
        if path.stat().st_size!=info['bytes'] or digest(path)!=info['sha256']:raise ValueError('CPU reference integrity failure')
    expected_a=(args.reference/'r8-a.f16').read_bytes();results=[];n=129280;i=5120
    refs={name:floats(args.reference/name,8*n) for name in ['r8-y-ordered.f32','r8-y-fp64-core.f32']}
    for rows in range(1,9):
        if (args.candidate/f'r{rows}-a.f16').read_bytes()!=expected_a[:rows*i*2]:raise ValueError('Input stage differs by bytes')
        output_path=args.candidate/f'r{rows}-y.f32';actual=floats(output_path,rows*n)
        for name, ref in refs.items():
            results.append(dict(rows=rows,reference=name,output_sha256=digest(output_path),
                aggregate=compare(actual,ref[:rows*n]),
                per_row=[compare(actual[row*n:(row+1)*n],ref[row*n:(row+1)*n]) for row in range(rows)]))
    print(json.dumps(dict(passed=True,bits=manifest['bits'],A_R1_R8_all_bytes_equal=True,
        unique_output_values=36*n,reference_comparisons=results),indent=2))


if __name__=='__main__':main()
