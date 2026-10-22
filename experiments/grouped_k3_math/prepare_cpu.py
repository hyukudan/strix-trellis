"""New complete group-specific CPU references. Originals are read-only, no network."""
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


def scalar_pair(g):
    # Exact dyadic F32 pairs; signed-zero alpha/beta retain distinct byte patterns.
    alpha = (0., -0.)[g] if g < 2 else (-1. if g % 2 == 0 else 1.) * (64+g)/128
    beta = (0., -0., .25, -.125)[g % 4]
    return np.array([alpha, beta], dtype='<f4')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixtures', type=Path, default=Path('external-data/gpu-real-k3-expert10-tests'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--payloads', type=Path, default=Path('external-data/eaq-k3-expert10-ref'))
    args = p.parse_args()
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        ctypes.windll.kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        assert ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    assert not args.output.exists(), 'new output only'
    args.output.mkdir(parents=True)
    started = time.monotonic()
    manifest_path = ROOT / 'reports/real-k3-expert-data/fixture-manifest.json'
    source_path = ROOT / 'reports/real-k3-expert-data/source-manifest.json'
    assert sha(manifest_path.read_bytes()) == 'd8d823bcad8c5e44963a85cef36b692dedb33062a82560cbe961be48959dfdd7'
    assert sha(source_path.read_bytes()) == 'bdb5d7d45fa3722d9da569fbf54829530d4d09648a8393ac14d654f7b1095bbc'
    original = json.loads(manifest_path.read_text())
    fixtures = [proof.load_fixture(args.fixtures/r['file'], r) for r in original['fixtures']]
    payload_pins = {}
    for record, f in zip(original['fixtures'], fixtures):
        for tensor, info in record['source_tensors'].items():
            raw = (args.payloads/info['file']).read_bytes()
            assert len(raw)==info['bytes'] and sha(raw)==info['sha256']
            field = tensor.rsplit('.',1)[1]
            if field != 'mul1':
                assert raw == f[{'trellis':'packed','suh':'u','svh':'v'}[field]].tobytes()
            else:
                assert raw == struct.pack('<I',0x83dcd12d)
            payload_pins[info['file']] = info['sha256']
    qs = [proof.decode_independent(f) for f in fixtures]
    # QA-only w2 pattern: rotate complete N16 packed tiles, preserving the format.
    # Decode ALL derived Q again by independent circular bit gathering AND V1.
    f2 = dict(fixtures[1])
    f2['packed'] = np.roll(f2['packed'], 1, axis=1).copy()
    f2['q'] = np.roll(qs[1], 16, axis=1).copy()
    f2['record'] = dict(file='derived-w2-N16-rotation', sha256=sha(f2['packed'].tobytes()))
    q2 = proof.decode_independent(f2)
    pairs = [scalar_pair(g) for g in range(32)]
    assert all(np.isfinite(pair).all() for pair in pairs)
    assert len({pair.tobytes() for pair in pairs}) == 32
    scalar_proof = dict(pair_bytes=[pair.tobytes().hex() for pair in pairs],
        unique_pairs_per_geometry=32, finite=True, dyadic=True, mutation='scalar-only g->(g+4)%32', cases=[])
    cases = []
    for i, n, s in ((5120, 2304, 16), (2304, 5120, 4)):
        for g in range(32):
            f = fixtures[0 if g%2 == 0 else 2] if i == 5120 else (fixtures[1] if g%2 == 0 else f2)
            q = qs[0 if g%2 == 0 else 2] if i == 5120 else (qs[1] if g%2 == 0 else q2)
            rng = np.random.default_rng(202610210000 + i + g)
            x = (rng.uniform(-.12, .12, i) + (g+1)*1e-5).astype('<f4')
            u, v = f['u'].copy(), f['v'].copy()
            # Every group has distinct scale/bias/X data, with independent refs.
            u = (u.astype(np.float32)*np.float32(1+g/128)).astype('<f2')
            v = (v.astype(np.float32)*np.float32(1-g/256)).astype('<f2')
            u[g::127] *= np.float16(-1)
            v[g::131] *= np.float16(-1)
            bias = rng.uniform(-.002, .002, n).astype('<f4') if g%3 else np.empty(0, '<f4')
            alpha, beta = pairs[g]
            old = rng.uniform(-.01, .01, n).astype('<f4')
            if beta == 0:
                old.fill(np.nan)
            # Selected signs, subnormals and the RN32/RN16 tie discriminator.
            x[:5] = np.array([0., -0., np.nextafter(np.float32(0), np.float32(1)),
                              -np.float32(2**-24), np.uint32(0x3f7033f3).view(np.float32)])
            u[4] = np.uint16(0x3c01).view(np.float16)
            if g==31:
                x[5:7]=np.array([1024.,-512.],np.float32)
                u[5:7]=np.float16(16.) # large finite RN16 operands, no overflow
            scaled = np.multiply(x, u.astype(np.float32), dtype=np.float32)
            half = scaled.astype(np.float16)
            a = proof.h_reference(half.astype(np.float32)).astype('<f2')
            assert proof.same(a, proof.h_wave(half.astype(np.float32)).astype('<f2'))
            p64, absolute = np.empty((s,n),'<f8'), np.empty((s,n),'<f8')
            # Bound temporary decoded/FP64 blocks to <=512 output columns.
            for split in range(s):
                lo, hi = (i//16)*split//s*16, (i//16)*(split+1)//s*16
                av = a[lo:hi].astype(np.float64)
                for col in range(0,n,512):
                    block = q[lo:hi,col:col+512].astype(np.float64)
                    p64[split,col:col+512] = av @ block
                    absolute[split,col:col+512] = np.abs(av) @ np.abs(block)
            # FP32 ordered output from full split references, not a native fdot2 claim.
            p32 = p64.astype(np.float32)
            total = np.zeros(n,np.float32)
            for row in p32:
                total += row
            y = proof.h_reference(total)*v.astype(np.float32)
            if bias.size:
                y += bias
            # Independent CPU epilogue input, before either scalar is selected.
            base_y = y.copy()
            y *= alpha
            if beta != 0:
                y += beta*old
            rotated_alpha, rotated_beta = pairs[(g+4)%32]
            mutated = base_y * rotated_alpha
            if rotated_beta != 0:
                mutated += rotated_beta*old
            assert not np.array_equal(mutated.view(np.uint32), y.view(np.uint32))
            error = mutated.astype(np.float64)-y.astype(np.float64)
            maxabs = float(np.max(np.abs(error)))
            maxref = float(np.max(np.abs(y)))
            nrms = float(np.sqrt((error@error)/max(float(y.astype(np.float64)@y.astype(np.float64)),1e-30)))
            rejected = not (maxabs <= 5e-4*max(1., maxref) and nrms <= 3e-5)
            assert rejected, (i,g,'scalar rotation gate unexpectedly accepted')
            scalar_proof['cases'].append(dict(input=i,group=g,maxabs=maxabs,maxref=maxref,nrms=nrms,
                bitwise_changed=True,frozen_numeric_gate_rejects=rejected))
            # Independent FP64 full operator from original X, before RN16 stages.
            ah = h64(x.astype(np.float64)*u.astype(np.float64))
            fullcore = np.empty(n,np.float64)
            for col in range(0,n,512):
                fullcore[col:col+512] = ah @ q[:,col:col+512].astype(np.float64)
            y64 = h64(fullcore)*v.astype(np.float64)
            if bias.size:
                y64 += bias
            y64 *= float(alpha)
            if beta != 0:
                y64 += float(beta)*old.astype(np.float64)
            arrays = dict(packed=f['packed'], x=x, suh=u, svh=v, bias=bias, old=old,
                          scalars=np.array([alpha,beta],'<f4'), a=a, p64=p64,
                          sumabs=absolute, y=y, y64=y64)
            record = write_case(args.output/f'i{i}-g{g:02d}.gk3',i,n,s,g,arrays)
            record.update(input=i,output=n,splits=s,group=g,
                          weights=f['record']['file'],derived_packed=(i==2304 and g%2==1))
            cases.append(record)
            print('CPU_CASE', i,n,s,g,record['sha256'],flush=True)
    # Hash original complete fixtures again after all calculations.
    for r in original['fixtures']:
        assert sha((args.fixtures/r['file']).read_bytes()) == r['sha256']
    for name, digest in payload_pins.items():
        assert sha((args.payloads/name).read_bytes()) == digest
    if os.name == 'nt':
        class Counters(ctypes.Structure):
            _fields_ = [('cb',ctypes.c_uint32),('faults',ctypes.c_uint32)] + [(name,ctypes.c_size_t) for name in
                ('peak_ws','ws','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]
        counters = Counters();counters.cb=ctypes.sizeof(counters)
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint32]
        assert ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(counters),counters.cb)
        peak_host_bytes = counters.peak_ws
    else:
        import resource
        peak_host_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    assert peak_host_bytes < 2*1024**3
    result = dict(schema=1, cases=cases, groups=[1,2,4,8,16,31,32],prefix_references=True,
        command=sys.argv,platform=platform.platform(),python=sys.version,numpy=np.__version__,
        threads=1,priority='BelowNormal on Windows; nice10 required remotely',
        revision=original['revision'],source_manifest_sha256=sha(source_path.read_bytes()),
        fixture_manifest_sha256=sha(manifest_path.read_bytes()),full_original_q_values=35389440,
        full_derived_q_values=int(q2.size),max_n_block=512,elapsed_seconds=time.monotonic()-started,
        native_fdot2_identity=False, tolerances=dict(maxabs_scale=5e-4,nrms=3e-5),
        new_algorithm=False, gpu_executed=False, payload_pins=payload_pins,
        peak_host_bytes=peak_host_bytes,host_limit_bytes=2*1024**3)
    (args.output/'unique-scalar-proof.json').write_text(json.dumps(scalar_proof,indent=2)+'\n',encoding='utf-8',newline='\n')
    result['unique_scalar_proof_sha256']=sha((args.output/'unique-scalar-proof.json').read_bytes())
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n',newline='\n')
    print('PASS complete CPU references; native exact A/P/Y and fdot2 gates OPEN',flush=True)


if __name__ == '__main__':
    main()
