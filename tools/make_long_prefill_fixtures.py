#!/usr/bin/env python3
"""Prepare two real long-prefill fixtures without fetching any weights.

Uses this project's make_dense_gpu_fixtures load/check/atomic writer helpers
and its ordered staged FP32 oracle. All matrix products run on CPU here.
"""
import argparse
import json
from pathlib import Path
import struct
import time

import numpy as np
from make_dense_gpu_fixtures import (ROOT, SEED, digest, load_group,
                                     check_stages, write_stream,
                                     matmul_fp32_stages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path,
                        default=Path('external-data/gpu-long-prefill-tests'))
    parser.add_argument('--real-directory', type=Path,
                        default=Path('external-data/fixtures'))
    parser.add_argument('--metadata', type=Path, default=ROOT/'tests/fixtures/eaq_dense.json')
    parser.add_argument('--rows', type=int, choices=(512, 1024), default=512)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'manifest.json').exists():
        raise FileExistsError('Finished manifest already exists; use a fresh directory')
    metadata = json.loads(args.metadata.read_text(encoding='utf-8'))
    records, start = [], time.perf_counter()
    for index, prefix in enumerate(('layers.0.attn.wq_a', 'layers.0.attn.wo_a.slice.0')):
        t, u, v, provenance = load_group(metadata, args.real_directory, prefix)
        path = args.out/f'{prefix}.m{args.rows}.stgpu'
        if path.exists():
            raise FileExistsError(path)
        seed = SEED + 30000 + index
        rng = np.random.Generator(np.random.PCG64(seed))
        x = rng.normal(0, .25, (args.rows, len(u))).astype('<f4')
        # Channel impulses and distinct boundary rows expose row/column mistakes.
        channels = [0, 1, 31, 32, 63, 64, 127, len(u)-1]
        probes = [0, 254, 255, 256, 511]
        if args.rows == 1024:
            probes += [1023]
        for row in probes:
            x[row] = 0
            x[row, channels] = np.array([1, -.3, .7, -1, .2, .9, -.4, .6]) * (1+row/args.rows)
        old = rng.normal(0, .3, (args.rows, len(v))).astype('<f4')
        stages = matmul_fp32_stages(x, t, u, v, t.shape[-1]//16, 'mul1')
        if not all(np.isfinite(value).all() for value in stages.values()):
            raise ValueError('Nonfinite CPU checkpoint')
        checks = check_stages(x, t, u, v, stages)
        sections = dict(meta=np.frombuffer(struct.pack('<8I', len(u), len(v), t.shape[-1]//16,
                        2, 0x83DCD12D, args.rows, 0, seed), dtype='u1'),
                        trellis=t, suh=u, svh=v, bias=np.empty(0, dtype='<f4'), x=x, old=old)
        for tag, name in (('scaled', 'scaled_fp16'), ('inh', 'input_hadamard_fp32'),
                          ('a', 'core_input_fp16'), ('q', 'rotated_fp16'),
                          ('core', 'core_fp32'), ('ref', 'output_fp32')):
            sections[tag] = stages[name]
        size, sha = write_stream(path, sections)
        records.append(dict(file=path.name, bytes=size, sha256=sha, prefix=prefix,
                       input=len(u), output=len(v), bits=t.shape[-1]//16,
                       rows=args.rows, seed=seed, rng='PCG64', codebook='mul1',
                       row_probes=probes, cpu_checks=checks, source_tensors=provenance))
        print(json.dumps(records[-1]), flush=True)
        del stages, sections, x, old
    manifest = dict(format='STGPU001', fixtures=records, numpy=np.__version__,
                    generator_sha256=digest(Path(__file__)),
                    helper_sha256=digest(ROOT/'tools/make_dense_gpu_fixtures.py'),
                    oracle_sha256=digest(ROOT/'python/strix_trellis/linear.py'),
                    metadata_sha256=digest(args.metadata), model=metadata['model'],
                    revision=metadata['revision'], cpu_seconds=time.perf_counter()-start,
                    reference='ordered matmul_fp32_stages; no BLAS',
                    total_bytes=sum(r['bytes'] for r in records), gpu_validation='not executed')
    temporary = args.out/'manifest.json.partial'
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    temporary.replace(args.out/'manifest.json')
    print(f"PASS CPU fixtures seconds={manifest['cpu_seconds']:.3f}", flush=True)


if __name__ == '__main__':
    main()
