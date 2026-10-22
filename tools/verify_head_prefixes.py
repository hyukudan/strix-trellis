"""Check archived R1..R8 head outputs against a complete CPU row prefix.

The complete A prefix must match bit for bit before a row-independent CPU Y
prefix can be used. This program does not execute a GPU or decode new weights.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from make_full_head_cpu_reference import compare


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check_a(actual, expected, count):
    if len(actual) != count or actual != expected[:count]:
        raise ValueError("Complete A differs from the CPU prefix")


def expect_rejection(operation):
    try:
        operation()
    except ValueError:
        return
    raise ValueError("A deliberately corrupted candidate was accepted")


def verify(reference, candidate, self_test=False, candidate_hashes=None):
    manifest_raw = (reference / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if not manifest['passed'] or manifest['gpu_executed'] or manifest['rows'] != 8:
        raise ValueError("Expected the completed eight-row CPU reference")
    i, n = int(manifest['input']), int(manifest['output'])
    reference_files = {}
    for name in ['r8-a.f16', 'r8-y-ordered.f32', 'r8-y-fp64-core.f32']:
        raw = (reference / name).read_bytes()
        recorded = manifest['files'][name]
        if len(raw) != recorded['bytes'] or digest(raw) != recorded['sha256']:
            raise ValueError("CPU reference hash or length changed: " + name)
        reference_files[name] = raw
    if len(reference_files['r8-a.f16']) != 8 * i * 2:
        raise ValueError("CPU A shape")
    outputs = {}
    for name in ['r8-y-ordered.f32', 'r8-y-fp64-core.f32']:
        raw = reference_files[name]
        if len(raw) != 8 * n * 4:
            raise ValueError("CPU Y shape")
        outputs[name] = np.frombuffer(raw, dtype='<f4').reshape(8, n)
    archived = {}
    if candidate_hashes:
        for line in candidate_hashes.read_text().splitlines():
            expected, name = line.split(maxsplit=1)
            name = Path(name.lstrip('*')).name
            if name in archived or len(expected) != 64:
                raise ValueError("Ambiguous candidate hash manifest")
            archived[name] = expected
    results = []
    for rows in range(1, 9):
        a_raw = (candidate / f'r{rows}-a.f16').read_bytes()
        check_a(a_raw, reference_files['r8-a.f16'], rows * i * 2)
        y_raw = (candidate / f'r{rows}-y.f32').read_bytes()
        if candidate_hashes:
            for name, raw in [(f'r{rows}-a.f16', a_raw), (f'r{rows}-y.f32', y_raw)]:
                if digest(raw) != archived.get(name):
                    raise ValueError("Candidate does not match its archived manifest: " + name)
        if len(y_raw) != rows * n * 4:
            raise ValueError("Candidate Y length")
        y = np.frombuffer(y_raw, dtype='<f4').reshape(rows, n)
        metrics = {name: compare(y, ref[:rows]) for name, ref in outputs.items()}
        results.append(dict(rows=rows, A_bytes=len(a_raw), Y_values=int(y.size),
                            A_all_bytes_equal_CPU_prefix=True,
                            A_sha256=digest(a_raw), Y_sha256=digest(y_raw), metrics=metrics))
    negative_checks = 0
    if self_test:
        bad_a = bytearray(reference_files['r8-a.f16'])
        bad_a[-1] ^= 1
        expect_rejection(lambda: check_a(bytes(bad_a), reference_files['r8-a.f16'], len(bad_a)))
        negative_checks += 1
        good = outputs['r8-y-fp64-core.f32']
        delta = max(1., float(np.max(np.abs(good))))
        for value in [float('nan'), float('inf'), float(good[-1, -1]) + delta]:
            bad_y = good.copy()
            bad_y[-1, -1] = value
            expect_rejection(lambda: compare(bad_y, good))
            negative_checks += 1
    return dict(passed=True, gpu_executed=False,
                candidate_hashes_checked=bool(candidate_hashes),
                candidate_manifest_sha256=None if not candidate_hashes else digest(candidate_hashes.read_bytes()),
                CPU_manifest_sha256=digest(manifest_raw), input=i, output=n,
                bits=int(manifest['bits']), candidate_Y_elements=sum(x['Y_values'] for x in results),
                reference_comparisons=2 * sum(x['Y_values'] for x in results),
                negative_checks=negative_checks, results=results,
                contract='All A bytes equal the complete CPU prefix before comparing every Y value. '
                         'Rows are independent; no new GPU run or standalone CUDA R1 identity is claimed.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--candidate-hashes', type=Path)
    a = p.parse_args()
    result = verify(a.reference, a.candidate, a.self_test, a.candidate_hashes)
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'results'}))


if __name__ == '__main__':
    main()
