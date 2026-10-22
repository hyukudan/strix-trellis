"""Verify all future QA source/reference bytes without importing HIP or NumPy."""
from pathlib import Path
import hashlib
import json
import struct
import math

root=Path('data/cpu')
manifest=json.loads((root/'manifest.json').read_text())
assert len(manifest['cases'])==64 and manifest['full_original_q_values']==35389440
assert manifest['max_n_block']<=512 and manifest['peak_host_bytes']<2*1024**3
assert manifest['full_derived_q_values']==11796480
total=0
seen=set()
pairs={5120:set(),2304:set()}
proof=json.loads((root/'unique-scalar-proof.json').read_text())
assert hashlib.sha256((root/'unique-scalar-proof.json').read_bytes()).hexdigest()==manifest['unique_scalar_proof_sha256']
assert len(proof['cases'])==64 and all(c['bitwise_changed'] and c['frozen_numeric_gate_rejects'] for c in proof['cases'])
for case in manifest['cases']:
    raw=(root/case['file']).read_bytes()
    assert len(raw)==case['bytes'] and hashlib.sha256(raw).hexdigest()==case['sha256']
    assert raw[:8]==b'GK3CPU01'
    i,n,s,g,bits,count=struct.unpack_from('<6I',raw,8)
    assert (i,n,s) in ((5120,2304,16),(2304,5120,4)) and bits==3 and 0<=g<32 and count==12
    assert (i,g) not in seen;seen.add((i,g))
    p=32
    for _ in range(count):
        tag=raw[p:p+8].rstrip(b'\0').decode();size,=struct.unpack_from('<Q',raw,p+8);p+=16
        t=raw[p:p+size];p+=size
        if tag=='scalars':
            assert len(t)==8 and all(math.isfinite(v) for v in struct.unpack('<2f',t))
            assert t.hex()==proof['pair_bytes'][g]
            assert t not in pairs[i];pairs[i].add(t)
        expected=case['tensors'][tag]
        assert len(t)==size==expected['bytes'] and hashlib.sha256(t).hexdigest()==expected['sha256']
    assert p==len(raw)
    total+=len(raw)
assert all(len(v)==32 for v in pairs.values())
print('PASS 32 finite byte-distinct scalar pairs per geometry; scalar-only rotation gate rejects all64 CPU references')
print('PASS CPU data/reference full bytes cases=64 total_bytes='+str(total)+'; GPU gates OPEN')
original=json.loads(Path('reports/real-k3-expert-data/fixture-manifest.json').read_text())
for record in original['fixtures']:
    raw=(Path('data/originals')/record['file']).read_bytes()
    assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256']
for name,pin in manifest['payload_pins'].items():
    assert hashlib.sha256((Path('data/payloads')/name).read_bytes()).hexdigest()==pin
print('PASS original fixtures and all 12 original range payloads pinned and frozen')
