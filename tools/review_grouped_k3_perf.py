"""Read frozen ELF/data and exercise CPU receipt/statistics counterexamples only.

Never executes, loads, or extracts a HIP binary. Writes only new review outputs.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import importlib.util
import json
import re
import struct
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PERF = ROOT / 'experiments/grouped_k3_perf'
HEAP = ('_Znwm', '_Znam', '_ZdlPv', '_ZdaPv', '_ZdlPvm', '_ZdaPvm')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(name):
    spec = importlib.util.spec_from_file_location('review_perf_' + name, PERF / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def members(path):
    with tarfile.open(path) as archive:
        return {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}


def manifest(raw):
    return {line[66:]: line[:64] for line in raw.decode().splitlines()}


def elf_heap(raw):
    assert raw[:6] == b'\x7fELF\x02\x01'
    offset, = struct.unpack_from('<Q', raw, 40)
    stride, count, names_index = struct.unpack_from('<HHH', raw, 58)
    headers = [struct.unpack_from('<IIQQQQIIQQ', raw, offset + j * stride) for j in range(count)]
    names_row = headers[names_index]
    names = raw[names_row[4]:names_row[4] + names_row[5]]
    sections = {names[r[0]:].split(b'\0', 1)[0].decode(): r for r in headers}
    sym = sections['.dynsym']
    strings_row = headers[sym[6]]
    strings = raw[strings_row[4]:strings_row[4] + strings_row[5]]
    version = sections['.gnu.version']
    result = []
    for index, pos in enumerate(range(sym[4], sym[4] + sym[5], sym[9])):
        name, info, other, section, address, size = struct.unpack_from('<IBBHQQ', raw, pos)
        name = strings[name:].split(b'\0', 1)[0].decode()
        if name in HEAP:
            ver, = struct.unpack_from('<H', raw, version[4] + 2 * index)
            result.append(dict(name=name, binding=info >> 4, type=info & 15,
                               visibility=other & 3, section=section, address=hex(address),
                               size=size, version_index=ver))
    dynamic = sections['.dynamic']
    needed, flags = [], []
    for pos in range(dynamic[4], dynamic[4] + dynamic[5], dynamic[9]):
        tag, value = struct.unpack_from('<QQ', raw, pos)
        if tag == 1:
            needed.append(strings[value:].split(b'\0', 1)[0].decode())
        if tag in (16, 30, 0x6ffffffb):
            flags.append(dict(tag=hex(tag), value=hex(value)))
    return dict(operators=result, needed=needed, flags=flags)


def reject(fn):
    try:
        fn()
    except (AssertionError, KeyError, FileNotFoundError):
        return True
    return False


def save_gzip(path, raw):
    with path.open('wb') as f:
        with gzip.GzipFile(filename='', fileobj=f, mode='wb', mtime=0) as g:
            g.write(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, required=True)
    parser.add_argument('--cpu-data', type=Path, required=True)
    parser.add_argument('--native-qa', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    handoff = json.loads((ROOT / 'reports/grouped-k3-perf-data/handoffCPU_ONLY_NATIVE_QA_PASS_REQUIRED.json').read_text())
    source = members(args.evidence_dir / 'source.tar')
    pins = manifest((args.evidence_dir / 'source.sha256').read_bytes())
    assert source.keys() == pins.keys()
    for name, raw in source.items():
        assert sha(raw) == pins[name] == sha((ROOT / name).read_bytes()), name
    nodes = {}
    for node in ('node1', 'node2'):
        expected = handoff['nodes'][node]
        path = args.evidence_dir / (node + '-evidence.tar.gz')
        assert sha(path.read_bytes()) == expected['evidence_archive_sha256']
        a = members(path)
        frozen = manifest(a['frozen.sha256'])
        assert sha(a['frozen.sha256']) == expected['frozen_manifest_sha256']
        checked = 0
        for name, raw in a.items():
            if name in frozen:
                assert sha(raw) == frozen[name], (node, name)
                checked += 1
        assert a['source.tar'] == (args.evidence_dir / 'source.tar').read_bytes()
        binary = a['bin/perf_native']
        assert sha(binary) == expected['binary_sha256']
        info = elf_heap(binary)
        assert {x['name'] for x in info['operators']} == set(HEAP)
        assert all(x['binding'] == 1 and x['visibility'] == 0 and x['section'] > 0 and
                   x['version_index'] == 1 for x in info['operators'])
        imports = [line.strip() for line in a['build/real-HIP-exports.txt'].decode().splitlines()
                   if ' UND ' in line and any(re.search(r'\b' + re.escape(n) + '@', line) for n in HEAP)]
        assert len(imports) == 6
        diagnostic = a['build/host-symbols-relocations.txt'].decode()
        lines = [line for line in diagnostic.splitlines()
                 if any(x in line for x in ('(NEEDED)', '(FLAGS', '(SYMBOLIC)', *HEAP))]
        (args.output / (node + '-elf-excerpts.txt')).write_text(
            'ARCHIVED ACTUAL HOST ELF\n' + '\n'.join(lines) +
            '\nARCHIVED ACTUAL HIP DYNSYM IMPORTS\n' + '\n'.join(imports) +
            '\nARCHIVED HIP LIBRARY HASH\n' + a['build/real-HIP-library.sha256'].decode(),
            encoding='utf-8', newline='\n')
        nodes[node] = dict(binary_sha256=sha(binary), archive_sha256=sha(path.read_bytes()),
                           frozen_manifest_sha256=sha(a['frozen.sha256']), frozen_members_checked=checked,
                           ELF=info, HIP_imports=imports,
                           HIP_library_hash_record=a['build/real-HIP-library.sha256'].decode().strip(),
                           linkmap_sha256=sha(a['build/link.map']),
                           effective_source_sha256=sha(a['build/effective-source.ii']))
    # Reuse read-only identity helpers on the actual archived images, no re-decode.
    sys.path.insert(0, str(ROOT / 'tools'))
    from audit_grouped_k3_math import sections, metadata
    from audit_grouped_k3_math_review import archive_images
    reference = json.loads((ROOT / 'reports/grouped-k3-math-r1-data/handoff.json').read_text())['nodes']['node1']
    identity = load('identity')
    library = a['lib/libstrix_trellis.a']
    assert sha(library) == reference['library_sha256']
    control_hashes = {sha(b) for b in archive_images(library)}
    for j in range(1, 8):
        assert sha(a[f'build/linked-{j:02d}.elf']) in control_hashes
    math_image = a['build/linked-00.elf']
    functions = {n: v for n, v in identity.function_bytes(math_image).items() if n.startswith('_ZN13strix_trellis')}
    assert functions == reference['math_function_hashes']
    assert sha(sections(math_image)['.text']) == handoff['nodes']['node2']['math_text_sha256']
    assert sha(json.dumps(metadata(math_image), sort_keys=True).encode()) == reference['math_metadata_sha256']
    data_manifest = (args.cpu_data / 'manifest.json').read_bytes()
    assert sha(data_manifest) == handoff['CPU_manifest_sha256']
    load('verify_data').verify(args.cpu_data)
    m = json.loads(data_manifest)
    for c in m['cases']:
        assert frozen['data/cpu/' + c['file']] == c['sha256']
    bounds = load('resource_bounds').bounds(m)
    # Existing CPU suite is imported and executed without writing its old reports.
    load('cpu_audit').main()
    stats, admission = load('statistics'), load('admission')
    rows = []
    for i in (5120, 2304):
        for g in (4, 8, 16, 32):
            for mode in ('direct', 'graph'):
                for v in ('V13', 'new'):
                    for sweep in range(11):
                        for clone in range(64 // g):
                            value = (10 + sweep + clone) * (1 if v == 'V13' else .8 if mode == 'direct' else 1.1)
                            rows.append(dict(input=i, G=g, mode=mode, variant=v, sweep=sweep,
                                             clone=clone, event_ms=value, wall_ms=2 * value))
    def raw_log(values, extra=''):
        return ('CPU SYNTHETIC ONLY; NOT A PHYSICAL CAMPAIGN\n' + extra +
                '\n'.join('SAMPLE ' + json.dumps(r, sort_keys=True) for r in values) +
                '\nPASS physical performance campaign all fixed cells; contexts=2 calls=14\n').encode()
    with tempfile.TemporaryDirectory(prefix='grouped-k3-review-') as temporary:
        tmp = Path(temporary)
        base = tmp / 'synthetic.log'
        raw = raw_log(rows)
        base.write_bytes(raw)
        save_gzip(args.output / 'synthetic-complete.log.gz', raw)
        result = stats.summarize(base)
        for g in (4, 8, 16, 32):
            cell = result['cells'][f'5120/G{g}/direct']
            offset = (64 // g - 1) / 2
            assert cell['event_ms_V13']['min'] == 10 + offset
            assert cell['event_ms_V13']['median'] == 15 + offset
            assert cell['event_ms_V13']['p90'] == 19 + offset
            assert abs(cell['event_ms_gain'] - .2) < 1e-12
        negatives = {}
        cases = dict(missing=rows[:-1], duplicate=rows + [rows[0]])
        cases['nonfinite'] = [dict(r, event_ms=float('nan')) if j == 0 else r for j, r in enumerate(rows)]
        for name, values in cases.items():
            raw = raw_log(values)
            base.write_bytes(raw)
            save_gzip(args.output / ('synthetic-' + name + '.log.gz'), raw)
            negatives[name] = reject(lambda: stats.summarize(base))
            assert negatives[name]
        # Full-pass proposal accepts receipts that have no QA, freeze or binary pins.
        log_paths, receipt_paths = [], []
        for j in (1, 2):
            path = tmp / f'campaign{j}.log'
            raw = raw_log(rows, extra=f'CPU synthetic physical_id={j}\n')
            path.write_bytes(raw)
            save_gzip(args.output / (f'synthetic-campaign{j}.log.gz'), raw)
            receipt = dict(ROOT_physical_full_PASS=True, physical_id=f'CPU-SYNTHETIC-{j}',
                           log_sha256=sha(raw), lease_seconds=300, exit=0)
            receipt_path = tmp / f'receipt{j}.json'
            receipt_path.write_text(json.dumps(receipt))
            (args.output / f'synthetic-receipt{j}.json').write_text(json.dumps(receipt, indent=2) + '\n')
            log_paths.append(str(path)); receipt_paths.append(str(receipt_path))
        cli = subprocess.run([sys.executable, '-I', '-S', '-B', str(PERF / 'statistics.py'),
                              *log_paths, '--root-full-pass-receipts', *receipt_paths],
                             capture_output=True, text=True, check=True)
        proposal = json.loads(cli.stdout)['proposal']
        assert proposal['performance_eligible'] is True
        # The admission QA log validator also accepts a three-token synthetic log.
        manifest_path = tmp / 'frozen.sha256'
        manifest_path.write_text('CPU SYNTHETIC MANIFEST ONLY\n')
        qa_raw = b'PASS native mathematical QA only; no performance claim\nfailed=0 overflow=0\n'
        (tmp / 'math-native.log').write_bytes(qa_raw)
        receipt = dict(schema=1, actual_native_QA='PASS', QA_binary_sha256=admission.QA,
                       ROOT_review='APPROVED_HOST_POOL_TIMING', perf_frozen_manifest_sha256=sha(manifest_path.read_bytes()),
                       physical_limit_seconds=300, inner_limit_seconds=270, campaign=1,
                       label='ROOT-grouped-k3-perf-CPU_SYNTHETIC', GPU_lease_obtained_by_ROOT=True,
                       actual_QA_log_sha256=sha(qa_raw), QA_gates={k: 'PASS' for k in
                           ('P1_01', 'P1_02', 'P1_03', 'all_A_P_Y', 'ownership_LAST', 'actual_attributes', 'peak_RSS_and_ledger')})
        receipt_path = tmp / 'admission.json'
        receipt_path.write_text(json.dumps(receipt))
        admission.check(receipt_path, manifest_path)
        (args.output / 'synthetic-admission.json').write_text(json.dumps(receipt, indent=2) + '\n')
        (args.output / 'synthetic-math-native.log').write_bytes(qa_raw)
    qa_files = {}
    for name in ('summary.json', 'native.log', 'native.stderr', 'meta.txt', 'closed.txt', 'parent-passed.txt'):
        raw = (args.native_qa / name).read_bytes()
        qa_files[name] = dict(sha256=sha(raw), bytes=len(raw))
    native_summary = json.loads((args.native_qa / 'summary.json').read_text())
    assert native_summary['passed'] and native_summary['configs'] == 14
    assert reference['binary_sha256'] in (args.native_qa / 'meta.txt').read_text()
    output = dict(scope='CPU_REVIEW_ONLY_NO_HIP_EXECUTION', source_files_checked=len(source), nodes=nodes,
                  actual_R1_QA_files=qa_files, CPU_data_manifest_sha256=sha(data_manifest),
                  CPU_cases_hashed=128, resource_bounds=bounds,
                  fixed_samples_per_campaign=len(rows), full_verifications_per_campaign=216,
                  minimum_packed_D2H_bytes_from_verification=216*64*4423680,
                  statistics_negative_rejections=negatives,
                  nonuniform_clone_stats='PASS mean per Batch G, min/median/nearest-rank-p90',
                  counterexamples=dict(proposal_without_QA_or_freeze_pins=proposal,
                                       synthetic_short_QA_log_accepted=True),
                  note='Counterexamples are synthetic CPU inputs, never admission receipts for a real run.')
    (args.output / 'review.json').write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(scope=output['scope'], output=str(args.output), samples=len(rows),
                          counterexamples=output['counterexamples']), indent=2))


if __name__ == '__main__':
    main()
