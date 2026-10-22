"""Read-only HIGH review evidence. CPU only; never invokes HIP, SSH or a compiler.

Archived instruction words are matched to ELF .text, not merely trusted as text.
The optional local CPU cases are hashed completely; only six cases have small
independent decoded column checks. This does not repeat the full Q decode.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import re
import struct
import sys
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'reports/grouped-k3-math-data'
sys.path.insert(0, str(ROOT / 'tools'))
from audit_grouped_k3_math import images, metadata, sections, manifest


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def emit(label, value):
    print(label, json.dumps(value, sort_keys=True), flush=True)


def members(path):
    with tarfile.open(path) as archive:
        return {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}


def functions(text):
    matches = list(re.finditer(r'^(?:[0-9a-f]+ )?<([^>]+)>:', text, re.M))
    return {m[1]: text[m.start():matches[j+1].start() if j+1 < len(matches) else len(text)]
            for j, m in enumerate(matches)}


def instructions(text):
    return [(m[1].strip(), int(m[2], 16), bytes().join(struct.pack('<I', int(w, 16))
            for w in m[3].split())) for m in re.finditer(
                r'^\s*(.*?)\s*// ([0-9A-Fa-f]+): ((?:[0-9A-F]{8} ?)+)\s*$', text, re.M)]


def verify_words(elf, text):
    offset, = struct.unpack_from('<Q', elf, 40)
    size, count, names_index = struct.unpack_from('<HHH', elf, 58)
    rows = [struct.unpack_from('<IIQQQQIIQQ', elf, offset+j*size) for j in range(count)]
    nr = rows[names_index]
    names = elf[nr[4]:nr[4]+nr[5]]
    row, = [r for r in rows if names[r[0]:].split(b'\0', 1)[0] == b'.text']
    ins = instructions(text)
    assert ins
    for _, at, raw in ins:
        start = at-row[3]
        assert 0 <= start <= row[5]-len(raw)
        assert elf[row[4]+start:row[4]+start+len(raw)] == raw
    return len(ins)


def archive_images(raw):
    assert raw[:8] == b'!<arch>\n'
    pos, result = 8, []
    while pos < len(raw):
        header = raw[pos:pos+60]
        assert header[-2:] == b'`\n'
        size = int(header[48:58])
        body = raw[pos+60:pos+60+size]
        pos += 60+size+(size % 2)
        if body[:4] == b'\x7fELF':
            result.extend(images(sections(body).get('.hip_fatbin', b'')))
    assert pos == len(raw)
    return result


def audit_archives():
    source = manifest((DATA/'source.sha256').read_text())
    packed_source = members(DATA/'source.tar')
    assert set(packed_source) == set(source)
    normalized_reports = []
    for name, pin in source.items():
        assert sha(packed_source[name]) == pin, name
        local = (ROOT/name).read_bytes()
        if sha(local) != pin:
            assert name in {'reports/grouped-k3-math-data/'+f for f in
                            ('resource-bounds.json', 'cpu-prepare-frozen.log',
                             'pilot-audit-cpu.log', 'lexical-cpu.log')}
            assert packed_source[name].replace(b'\r\n', b'\n') == local
            normalized_reports.append(dict(file=name, frozen_sha256=pin,
                                           worktree_sha256=sha(local), only_CRLF_to_LF=True))
    emit('SOURCE', dict(files=len(source), tar_sha256=sha((DATA/'source.tar').read_bytes()),
                        source_code_exact=True, report_byte_differences=normalized_reports))
    handoff = json.loads((DATA/'handoff.json').read_text())
    first = None
    for node in ('node1', 'node2'):
        a = members(DATA/(node+'-evidence.tar.gz'))
        frozen = manifest(a['frozen.sha256'].decode())
        assert a['frozen.sha256'] == (DATA/(node+'-frozen.sha256')).read_bytes()
        checked = [name for name in a if name in frozen]
        for name in checked:
            assert sha(a[name]) == frozen[name], (node, name)
        for name in source:
            if name in frozen:
                assert frozen[name] == source[name]
            else:
                assert name == 'tools/verify_grouped_k3_adapter.py'
        assert a['source.tar'] == (DATA/'source.tar').read_bytes()
        assert a['data/cpu/manifest.json'] == (DATA/'cpu-manifest.json').read_bytes()
        assert sha(a['lib/libstrix_trellis.a']) == handoff['baseline_archive_sha256']
        fat = sections(a['bin/math_native_qa'])['.hip_fatbin']
        assert fat == a['build/linked.fatbin']
        elfs = images(fat)
        assert len(elfs) == 8
        old_images = {sha(e) for e in archive_images(a['lib/libstrix_trellis.a'])}
        assert all(sha(e) in old_images for e in elfs[1:])
        assert sha(elfs[0]) not in old_images
        all_functions = {}
        word_count = 0
        for j, elf in enumerate(elfs):
            assert elf == a[f'build/linked-{j:02d}.elf']
            text = a[f'build/linked-{j:02d}.isa'].decode()
            word_count += verify_words(elf, text)
            for symbol, body in functions(text).items():
                if symbol in all_functions:
                    assert symbol in ('__cxa_pure_virtual', '__cxa_deleted_virtual')
                    continue
                all_functions[symbol] = body
        assert [len(metadata(e)) for e in elfs] == [3, 2, 72, 252, 1, 32, 2, 2]
        resources = []
        for k in metadata(elfs[0]):
            assert k['.args'] == [{'.offset': 0, '.size': 2320, '.value_kind': 'by_value'}]
            assert (k['.kernarg_segment_size'], k['.kernarg_segment_align'], k['.wavefront_size'],
                    k['.max_flat_workgroup_size']) == (2320, 8, 32, 128)
            assert all(k['.'+f] == 0 for f in ('group_segment_fixed_size', 'private_segment_fixed_size',
                                              'sgpr_spill_count', 'vgpr_spill_count'))
            resources.append(dict(name=k['.name'], sgpr=k['.sgpr_count'], vgpr=k['.vgpr_count']))
        headers = json.loads(a['build/effective-header-hashes.json'])
        for name, raw in packed_source.items():
            for h in headers:
                if h['resolved'].endswith('/'+name):
                    assert h['sha256'] == sha(raw), name
        effective = a['build/effective-source.ii'].decode()
        # Inspect effective binding definitions, not a second helper implementation.
        for symbol in ('materialize_fp32', 'hadamard128', 'mul1_half2', 'grouped_baseline_core'):
            assert symbol in effective
        assert sha(a['build/effective-source.ii']) == json.loads(a['build/scalar-selection.json'])['effective_preprocessed_sha256']
        command = a['build/hip-compile.stderr'].decode().splitlines()[0]
        assert '-O2 -ffp-contract=off --offload-arch=gfx1151' in command
        assert '-ffast-math' not in command and '-Llib -lstrix_trellis' in command
        emit('ARCHIVE', dict(node=node, frozen_sha256=sha(a['frozen.sha256']),
             binary_sha256=sha(a['bin/math_native_qa']), math_elf_sha256=sha(elfs[0]),
             archived_frozen_members=len(checked), instruction_rows_matched_to_ELF=word_count,
             images=8, kernels=366, math=resources, effective_header_records=len(headers),
             all_seven_control_ELFs_literal_members_of_pinned_archive=True,
             data_bytes_not_in_archive=True))
        if first is None:
            first = (a, all_functions)
        else:
            assert a['bin/math_native_qa'] == first[0]['bin/math_native_qa']
    a, fs = first
    stages = [('grouped_input_transform', '15input_transform'),
              ('grouped_baseline_core', 'gemv_baseline_kernelILi3ELb1ELi1ELb0E'),
              ('grouped_output_transform', '16output_transform')]
    for stage, old in stages:
        new_symbol, = [s for s in fs if stage in s]
        old_symbol, = [s for s in fs if old in s]
        new, control = fs[new_symbol], fs[old_symbol]
        assert instructions(new) == instructions(a['build/'+stage+'.isa'].decode())
        assert not re.search(r'\bscratch_|\bs_barrier\b|\bds_(?:load|store)', new)
        critical = r'\b(v_dot2acc_f32_f16|v_pk_fma_f16|v_fma_mix_f32|v_cvt_f16_f32_e32|v_cvt_f32_f16_e32|ds_bpermute_b32|v_add_f32_dpp)\b'
        nc, oc = Counter(re.findall(critical, new)), Counter(re.findall(critical, control))
        assert nc == oc, (stage, nc, oc)
        if stage == 'grouped_baseline_core':
            assert nc['v_dot2acc_f32_f16'] == nc['v_pk_fma_f16'] == 4
            assert '0x83dcd12d' in new and '0x1eee' in new and '0xc931' in new
            assert new.count('quad_perm:[2,3,2,3]') == control.count('quad_perm:[2,3,2,3]') == 2
            assert 'clamp' not in new
        else:
            assert nc['ds_bpermute_b32'] == 20 and '0x3db504f3' in new
            assert all('neg(0)' in i and 'op_sel_hi:[1,0,0]' in i
                       for i, _, _ in instructions(new) if i.startswith('v_fma_mix_f32'))
        emit('ISA_FINITE', dict(stage=stage, original_symbol=old_symbol,
             critical_counts=dict(nc), whole_instruction_identity=False,
             note='Counts supplement manual operand/order/branch review; not a semantic proof'))


def h_ordered(x):
    v = np.asarray(x, dtype=np.float32).copy().reshape(-1, 128)
    for bit in range(7):
        low = np.array([j for j in range(128) if not j & (1 << bit)])
        high = low + (1 << bit)
        l, r = v[:, low].copy(), v[:, high].copy()
        v[:, low], v[:, high] = np.add(l, r), np.subtract(l, r)
    return (v*np.float32(float.fromhex('0x1.6a09e6p-4'))).reshape(np.shape(x))


def h_matrix(x):
    matrix = np.array([[1 if (r & c).bit_count() % 2 == 0 else -1
                        for c in range(128)] for r in range(128)], dtype=np.float64)
    return ((np.asarray(x, np.float64).reshape(-1, 128) @ matrix)/np.sqrt(128.)).reshape(np.shape(x))


def decode_columns(packed, i, n, columns):
    words = np.frombuffer(packed, '<u4').reshape(i//16, n//16, 24)
    out = np.empty((i, len(columns)), np.float64)
    # Gather individual MSB-first circular bits, without V1 shifts or helpers.
    for index, column in enumerate(columns):
        c = column % 16
        for row in range(16):
            order = (c % 8)*32 + ((row//2) % 4)*8 + (row % 2) + (row//8)*2 + (c//8)*4
            state = np.zeros(i//16, np.uint32)
            for bit in range(16):
                position = ((order+1)*3-16+bit) % 768
                state = (state << 1) | ((words[:, column//16, position//32] >> (31-position % 32)) & 1)
            mixed = state*np.uint32(0x83dcd12d)
            byte_sum = sum((mixed >> shift) & 255 for shift in range(0, 32, 8))
            value = ((1024+byte_sum.astype(np.int64))*1774-1329*2048)/262144.
            out[row::16, index] = value.astype(np.float16).astype(np.float64)
    return out


def read_case(raw):
    assert raw[:8] == b'GK3CPU01'
    header = struct.unpack_from('<6I', raw, 8)
    result, pos = {}, 32
    for _ in range(header[-1]):
        tag = raw[pos:pos+8].rstrip(b'\0').decode()
        size, = struct.unpack_from('<Q', raw, pos+8)
        assert tag not in result
        result[tag] = raw[pos+16:pos+16+size]
        pos += 16+size
    assert pos == len(raw)
    return header, result


def audit_cpu(path):
    manifest_cpu = json.loads((DATA/'cpu-manifest.json').read_text())
    checked_bytes, selected = 0, []
    for c in manifest_cpu['cases']:
        raw = (path/c['file']).read_bytes()
        assert sha(raw) == c['sha256'] and len(raw) == c['bytes']
        (i, n, s, g, bits, fields), t = read_case(raw)
        assert (i, n, s, g, bits, fields) == (c['input'], c['output'], c['splits'], c['group'], 3, 12)
        for name, payload in t.items():
            assert sha(payload) == c['tensors'][name]['sha256']
        checked_bytes += len(raw)
        view = lambda name, dtype: np.frombuffer(t[name], dtype)
        x, u, v = view('x', '<f4'), view('suh', '<f2'), view('svh', '<f2')
        half = np.multiply(x, u.astype(np.float32), dtype=np.float32).astype(np.float16)
        a = h_ordered(half.astype(np.float32)).astype(np.float16)
        assert a.tobytes() == t['a']
        p64 = view('p64', '<f8').reshape(s, n)
        total = np.zeros(n, np.float32)
        for partial in p64.astype(np.float32):
            total = np.add(total, partial, dtype=np.float32)
        y = np.multiply(h_ordered(total), v.astype(np.float32), dtype=np.float32)
        if t['bias']:
            y = np.add(y, view('bias', '<f4'), dtype=np.float32)
        alpha, beta = view('scalars', '<f4')
        y = np.multiply(y, alpha, dtype=np.float32)
        if beta != 0:
            y = np.add(y, np.multiply(beta, view('old', '<f4'), dtype=np.float32), dtype=np.float32)
        assert y.tobytes() == t['y']
        if g not in (0, 1, 31):
            continue
        columns = [0, 7, 8, 15, 16, 31, n-1]
        q = decode_columns(t['packed'], i, n, columns)
        absolute = view('sumabs', '<f8').reshape(s, n)
        error, abs_error = 0., 0.
        for split in range(s):
            lo, hi = i*split//s, i*(split+1)//s
            got = a[lo:hi].astype(np.float64) @ q[lo:hi]
            ab = np.abs(a[lo:hi].astype(np.float64)) @ np.abs(q[lo:hi])
            assert np.array_equal(got, p64[split, columns]), (i, g, split, got-p64[split, columns])
            assert np.allclose(ab, absolute[split, columns], rtol=4e-15, atol=0), (i, g, split, ab-absolute[split, columns])
            error = max(error, float(np.max(np.abs(got-p64[split, columns]))))
            abs_error = max(abs_error, float(np.max(np.abs(ab-absolute[split, columns]))))
        # One whole H128 output block independently validates original-basis
        # FP64 scaling, using a parity matrix rather than butterfly routines.
        qlast = decode_columns(t['packed'], i, n, list(range(n-128, n)))
        ah = h_matrix(x.astype(np.float64)*u.astype(np.float64))
        y64 = h_matrix(ah @ qlast)*v[-128:].astype(np.float64)
        if t['bias']:
            y64 += view('bias', '<f4')[-128:]
        y64 *= float(alpha)
        if beta != 0:
            y64 += float(beta)*view('old', '<f4')[-128:].astype(np.float64)
        target = view('y64', '<f8')[-128:]
        assert np.allclose(y64, target, rtol=2e-12, atol=2e-12)
        selected.append(dict(input=i, group=g, all_split_columns=columns,
                             p64_max_error=error, sumabs_max_error=abs_error,
                             y64_last128_max_error=float(np.max(np.abs(y64-target)))))
    emit('CPU_REFERENCES', dict(cases=64, hashed_bytes=checked_bytes, all_A_Y_ordered_exact=True,
         independent_selected_checks=selected, repeated_full_Q_decode=False,
         native_fdot2_proof=False))
    counts = {i: {k: len({c['tensors'][k]['sha256'] for c in manifest_cpu['cases'] if c['input'] == i})
                  for k in ('packed', 'x', 'suh', 'svh', 'bias', 'scalars')}
              for i in (5120, 2304)}
    assert all(counts[i]['scalars'] == 4 for i in counts)
    assert all(counts[i]['x'] == counts[i]['suh'] == counts[i]['svh'] == 32 for i in counts)
    for i in counts:
        rows = sorted((c for c in manifest_cpu['cases'] if c['input'] == i), key=lambda c: c['group'])
        assert all(rows[g]['tensors']['scalars']['sha256'] ==
                   rows[(g+4) % 32]['tensors']['scalars']['sha256'] for g in range(32))
    emit('COVERAGE', dict(distinct_tensor_hashes_per_geometry=counts,
                         scalar_g0_equals_g4=True, scalar_g3_equals_g31=True,
                         scalar_neighbor_plus_four_mutation_is_invisible=True))


def audit_resources():
    cases = json.loads((DATA/'cpu-manifest.json').read_text())['cases']
    weights = persistent = host = initial = max_snapshot = 0
    for c in cases:
        i, n, s = c['input'], c['output'], c['splits']
        sizes = {k: v['bytes'] for k, v in c['tensors'].items()}
        assert sizes['packed'] == i*n*3//8
        assert sizes['suh'] == 2*i and sizes['svh'] == 2*n
        assert sizes['a'] == 2*i and sizes['p64'] == sizes['sumabs'] == 8*s*n
        assert sizes['y'] == sizes['old'] == 4*n and sizes['y64'] == 8*n
        w = 0
        for k in ('packed', 'suh', 'svh', 'bias'):
            w = ((w+255)//256)*256 + sizes[k]
        weights += w
        persistent += w+(4*i+512)+2*((4*n+512)+(2*i+512)+(4*s*n+512))
        host += sum(sizes.values())
        initial += 4*i+2*4*n
        max_snapshot = max(max_snapshot, 32*(2*i+4*s*n+4*n))
    transient = 157696
    host_bound = host+initial+10*max_snapshot+64*1024**2
    recorded = json.loads((DATA/'resource-bounds.json').read_text())
    assert persistent == recorded['persistent_explicit_gpu_bytes_planned'] == 303399936
    assert weights == recorded['actual_v13_weight_allocation_bytes_planned'] == 284689408
    assert persistent+transient == recorded['maximum_explicit_gpu_bytes_planned'] < 1024**3
    assert host_bound == recorded['host_payload_bound_planned'] == 442648064 < 2*1024**3
    emit('PLANNED_RESOURCES', dict(GPU_persistent=persistent, GPU_with_transient=persistent+transient,
         host_payload_bound=host_bound, runtime_driver_RSS_not_measured=True,
         actual_allocation_ledger_not_measured=True))


def audit_gates():
    owner = (ROOT/'experiments/grouped_k3_math/owner.hpp').read_text()
    qa = (ROOT/'experiments/grouped_k3_math/native_qa.hip').read_text()
    payload = (ROOT/'experiments/grouped_k3_math/native_payload.sh').read_text()
    submit = owner[owner.index('inline hipError_t submit(const PreparedPlan& p) {'):]
    launches = owner[owner.index('inline hipError_t launch_three('):owner.index('hipError_t submit(const PreparedPlan& p);')]
    assert launches.count('hipLaunchKernelGGL') == 3
    assert not re.search(r'hip(?:Malloc|Memcpy|StreamSynchronize|PointerGet|FuncGet)|\bnew\b', submit+launches)
    assert 'ONLY graph leases retain everything' in qa
    assert 'timeout --signal=TERM --kill-after=3s 145s' in payload
    # The literal public context invokes the four-image V13 probe; the private
    # loop invokes three more. Frozen harness prints only anonymous math attrs.
    assert 'strix_context_create(&d,&o->context_)' in owner
    assert owner.count('hipFuncGetAttributes(') == 1
    assert 'math_resource maxthreads=' in owner
    emit('ADMISSION_FINDING', dict(id='P1-R1', admission='STOP_PENDING_QUERY_ACCOUNTING_AND_REFREEZE',
         expected_successful_context_queries=dict(V13=4, grouped_math=3, total=7),
         frozen_harness_actual_query_counter=False, CPU_expected_count_is_not_native_evidence=True,
         math_source_defect_observed=False))
    emit('COVERAGE_FINDING', dict(id='P1-R2', gate='BEFORE_COMPLETE_NUMERICAL_QUALIFICATION',
         cause='Only four alpha/beta byte patterns for 32 groups',
         remedy='Distinct finite per-group scalar pairs preserving required zero/sign cases; regenerate references and freeze'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu-data', type=Path, required=True)
    args = p.parse_args()
    emit('ENV', dict(platform=platform.platform(), python=sys.version, numpy=np.__version__,
                     command=sys.argv, CPU_ONLY=True))
    audit_archives()
    audit_cpu(args.cpu_data)
    audit_resources()
    audit_gates()
    print('PASS CPU review audit; admission STOP P1-R1; no HIP/native execution')


if __name__ == '__main__':
    main()
