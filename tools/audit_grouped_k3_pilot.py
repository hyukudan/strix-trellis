"""Read-only local provenance/log/ELF audit; optional existing CPU guard.

Uses installed msgpack to decode the actual AMDGPU note, not the text summary.
Never executes a HIP binary, accesses the network or writes old evidence.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import struct
import tarfile
import msgpack

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'reports/grouped-k3-abi-data'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def manifest(text):
    return dict((name, digest) for digest, name in
                (line.split('  ', 1) for line in text.splitlines()))


def sections(blob):
    assert blob[:6] == b'\x7fELF\x02\x01', 'expected ELF64 little endian'
    offset, = struct.unpack_from('<Q', blob, 40)
    size, count, names_index = struct.unpack_from('<HHH', blob, 58)
    rows = [struct.unpack_from('<IIQQQQIIQQ', blob, offset+i*size) for i in range(count)]
    row = rows[names_index]
    names = blob[row[4]:row[4]+row[5]]
    return {names[row[0]:].split(b'\0', 1)[0].decode(): blob[row[4]:row[4]+row[5]]
            for row in rows if row[1] != 8}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guard', action='store_true')
    args = parser.parse_args()
    print('CPU_ENV ' + platform.platform() + ' Python=' + platform.python_version())
    source = manifest((DATA/'source.sha256').read_text())
    with tarfile.open(DATA/'source.tar') as archive:
        assert set(archive.getnames()) == set(source)
        for name, digest in source.items():
            assert sha(archive.extractfile(name).read()) == digest, name
            assert sha((ROOT/name).read_bytes()) == digest, name
    print(f'PROVENANCE source.tar={sha((DATA/"source.tar").read_bytes())} members={len(source)} archive/worktree=PASS')
    for node in ('node1', 'node2'):
        frozen = DATA/node/'frozen.sha256'
        assert sha(frozen.read_bytes()) == '51077fea93271fb7a0a91312e69761b8619b855ae7d072c966e6a3840f8383b6'
        checked, absent = 0, []
        for name, digest in manifest(frozen.read_text()).items():
            p = DATA/node/name
            if name == 'source.tar':
                p = DATA/name
            elif name in source:
                p = ROOT/name
            elif name == 'bin/abi_pilot':
                p = DATA/node/'abi_pilot'
            if not p.exists():
                absent.append(name)
                continue
            assert sha(p.read_bytes()) == digest, (node, name)
            checked += 1
        expected = ['bin/cpu_selftest', 'bin/high_guard']
        if node == 'node2':
            expected.insert(0, 'bin/abi_pilot')
        assert absent == expected, absent
        print(f'FREEZE {node} verified_files={checked} not_collected={absent}')
        post = (DATA/(node+'-postflight.log')).read_text()
        assert '51077fea93271fb7a0a91312e69761b8619b855ae7d072c966e6a3840f8383b6' in post
        assert 'source.tar: OK' in post and 'bin/abi_pilot: OK' in post
    binary = (DATA/'node1/abi_pilot').read_bytes()
    assert sha(binary) == 'ff705c7a2c4663ce064f767af2af0f8744019545806d9a1a72da951328e54fd2'
    fat = sections(binary)['.hip_fatbin']
    assert fat == (DATA/'node1/build/linked.fatbin').read_bytes()
    marker = b'__CLANG_OFFLOAD_BUNDLE__'
    p = fat.index(marker)
    q = p+len(marker)
    count, = struct.unpack_from('<Q', fat, q)
    q += 8
    images = []
    for _ in range(count):
        offset, size, length = struct.unpack_from('<QQQ', fat, q)
        q += 24
        target = fat[q:q+length].decode()
        q += length
        if 'gfx1151' in target:
            images.append(fat[p+offset:p+offset+size])
    assert len(images) == 1
    elf = images[0]
    assert elf == (DATA/'node1/build/linked.elf').read_bytes()
    assert sha(elf) == '8d8e46ad7cd73f8a3cb286cbaf56abcaf987959f52da7dcd3e2280fe75c66a95'
    note = sections(elf)['.note']
    p, metadata = 0, []
    while p < len(note):
        ns, ds, kind = struct.unpack_from('<III', note, p)
        p += 12
        name = note[p:p+ns]
        p += (ns+3)&~3
        desc = note[p:p+ds]
        p += (ds+3)&~3
        if name == b'AMDGPU\0' and kind == 32:
            metadata.append(msgpack.unpackb(desc, raw=False))
    assert len(metadata) == 1
    kernels = metadata[0]['amdhsa.kernels']
    saved = json.JSONDecoder().raw_decode((DATA/'node1/build/resources.json').read_text())[0]
    assert [k['.name'] for k in kernels] == ['trace_inputs', 'trace_core', 'trace_outputs']
    for k, old in zip(kernels, saved['resources']):
        for key, value in old.items():
            if '.'+key in k and key != 'args':
                assert k['.'+key] == value, key
        assert k['.args'] == [{'.'+key: int(v) if key in ('offset', 'size') else v
                              for key, v in arg.items()} for arg in old['args']]
        assert k['.args'][0] == {'.offset': 0, '.size': 2320, '.value_kind': 'by_value'}
        assert k['.args'][1]['.offset'] == 2320 and k['.args'][1]['.size'] == 8
        assert k['.kernarg_segment_size'] == 2584 and k['.kernarg_segment_align'] == 8
        assert max(a['.offset']+a['.size'] for a in k['.args']) == 2394
        assert k['.wavefront_size'] == 32 and k['.max_flat_workgroup_size'] == 128
        assert all(k['.'+key] == 0 for key in ('group_segment_fixed_size', 'private_segment_fixed_size', 'sgpr_spill_count', 'vgpr_spill_count'))
        print(f'ELF {k[".name"]} kernarg=2584 explicit=2328 hidden/reserved=256 last_arg_end=2394 wave32 block128 LDS/private/spills=0 SGPR={k[".sgpr_count"]} VGPR={k[".vgpr_count"]}')
    isa = (DATA/'node1/build/linked-isa.txt').read_text()
    assert not re.search(r'\bscratch_|\bs_barrier\b|\bds_(?:load|store)', isa)
    for symbol in ('trace_inputs', 'trace_core', 'trace_outputs'):
        assert '<'+symbol+'>:' in isa
    print('ELF_CHAIN executable -> .hip_fatbin -> gfx1151 ELF -> decoded AMDGPU note=PASS')
    log = (DATA/'node1/runs/native.log').read_text()
    replays = re.findall(r'^geometry=(\d) G=(\d+) replay=(\d) twoStreams=PASS', log, re.M)
    assert set(replays) == {(str(g), str(n), str(r)) for g in range(2)
                            for n in (1, 2, 4, 8, 16, 31, 32) for r in range(2)}
    assert len(replays) == 28
    assert log.count('graph kernels=3 edges=2 functions/grids/Batch2320/TraceArg=PASS') == 28
    assert log.count('capture_storage_reused checksum=') == 28
    assert log.count('nodes=0 fullTrace/allBuffers=unchanged PASS') == 10
    owners = re.findall(r'owner_destroy generation=(\d+) owned_peak_bytes=(\d+)', log)
    assert [int(g) for g, _ in owners] == list(range(1, 15))
    assert max(int(b) for _, b in owners) == 29632400
    records = sum(4*n*(10+36*16+5 if g == 0 else 5+80*4+10)
                  for g in range(2) for n in (1, 2, 4, 8, 16, 31, 32))
    assert records == 348176
    assert 'PASS native container trace only wall_seconds=0.652996' in log
    assert 'MPs=20 runtime=71626385 driver=71626385' in log
    controller = (DATA/'node1-postflight.log').read_text()
    assert len(re.findall(r' start strix-trellis-grouped-k3-abi ', controller)) == 1
    assert len(re.findall(r' end   strix-trellis-grouped-k3-abi ', controller)) == 1
    assert 'payload_exit=0 integrity_exit=0' in (DATA/'node1/runs/closed.txt').read_text()
    assert 'controller_exit=0' in (DATA/'node1/attempts/controller-closed.txt').read_text()
    stderr = (DATA/'node1/runs/native.stderr').read_text()
    assert 'Maximum resident set size (kbytes): 276136' in stderr and 'Exit status: 0' in stderr
    summary = json.loads((DATA/'summary.json').read_text())
    for key, value in dict(geometry_group_configurations=14, graphs=28, kernel_nodes=84,
                           dependency_edges=56, replay_requests=56, owner_generations=14,
                           replay_cta_records_checked=records, full_trace_comparisons=80,
                           full_trace_bytes_compared=80*2143536).items():
        assert summary[key] == value, key
    print('LOGS configs=14 graphs=28 nodes=84 edges=56 replays=56 negatives=10 owner_generations=14 PASS')
    print('LOGS fullTrace=80 comparisons (14 direct + 56 replay + 10 negative); trace_writes are not math')
    if args.guard:
        spec = importlib.util.spec_from_file_location('old_cpu_proof', ROOT/'reports/grouped_k3_cpu_proof.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.compile_guard()  # Only CPU C++, not sources()/numeric historic sweep.
    print('PASS CPU-only pilot evidence audit; no new native execution')


if __name__ == '__main__':
    main()
