#!/usr/bin/env python3
"""Independent stdlib CPU proof of the minimal WMMA decoder contract.

No project imports, compiler, runtime, device enumeration, network or subprocess.
This models source expressions; it does not execute LLVM, ISA, or WMMA.
"""
import ctypes
from ctypes import wintypes as wt
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import sys
import threading
import time

MASK = (1 << 32) - 1
MULT = 0x83DCD12D
ROOT = Path(__file__).resolve().parents[1]


def restrict_process():
    """Bound this process to one CPU and 512 MiB, with reduced priority."""
    if os.name == 'posix':
        import resource
        os.nice(10)
        selected = min(os.sched_getaffinity(0))
        os.sched_setaffinity(0, {selected})
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        assert len(os.listdir('/proc/self/task')) == 1
        return None, None, None, selected
    assert os.name == 'nt', 'This audited invocation requires Windows limits'
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wt.HANDLE
    handle = kernel.GetCurrentProcess()
    kernel.SetPriorityClass.argtypes = [wt.HANDLE, wt.DWORD]
    assert kernel.SetPriorityClass(handle, 0x4000)
    affinity, system = ctypes.c_size_t(), ctypes.c_size_t()
    kernel.GetProcessAffinityMask.argtypes = [wt.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    assert kernel.GetProcessAffinityMask(handle, ctypes.byref(affinity), ctypes.byref(system))
    selected = affinity.value & -affinity.value
    kernel.SetProcessAffinityMask.argtypes = [wt.HANDLE, ctypes.c_size_t]
    assert kernel.SetProcessAffinityMask(handle, selected)

    class Basic(ctypes.Structure):
        _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                    ('flags', wt.DWORD), ('minimum', ctypes.c_size_t),
                    ('maximum', ctypes.c_size_t), ('active', wt.DWORD),
                    ('affinity', ctypes.c_size_t), ('priority', wt.DWORD), ('scheduling', wt.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]

    class Extended(ctypes.Structure):
        _fields_ = [('basic', Basic), ('io', IO), ('process_memory', ctypes.c_size_t),
                    ('job_memory', ctypes.c_size_t), ('peak_process', ctypes.c_size_t),
                    ('peak_job', ctypes.c_size_t)]

    kernel.CreateJobObjectW.restype = wt.HANDLE
    job = kernel.CreateJobObjectW(None, None)
    assert job
    limits = Extended()
    limits.basic.flags = 0x100 | 0x8  # process memory and active process limits
    limits.basic.active = 1
    limits.process_memory = 512 * 1024 * 1024
    kernel.SetInformationJobObject.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
    assert kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits))
    kernel.AssignProcessToJobObject.argtypes = [wt.HANDLE, wt.HANDLE]
    assert kernel.AssignProcessToJobObject(job, handle)
    # Keep the job handle alive until process termination; no other process joins.
    return kernel, handle, job, selected


def process_stats(kernel, handle, require_single=True):
    if os.name == 'posix':
        import resource
        count = len(os.listdir('/proc/self/task'))
        assert threading.active_count() == count == 1
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        assert peak < 2**31
        return dict(os_threads=count, peak_working_set_bytes=peak,
                    nice=os.nice(0), cpu_affinity=sorted(os.sched_getaffinity(0)))
    class Memory(ctypes.Structure):
        _fields_ = [('cb', wt.DWORD), ('faults', wt.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('peak_ws', 'ws', 'peak_paged',
                                                'paged', 'peak_nonpaged', 'nonpaged',
                                                'pagefile', 'peak_pagefile', 'private')]
    mem = Memory()
    mem.cb = ctypes.sizeof(mem)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD]
    assert psapi.GetProcessMemoryInfo(handle, ctypes.byref(mem), mem.cb)
    class Thread(ctypes.Structure):
        _fields_ = [('size', wt.DWORD), ('usage', wt.DWORD), ('id', wt.DWORD),
                    ('owner', wt.DWORD), ('base', wt.LONG), ('delta', wt.LONG), ('flags', wt.DWORD)]
    kernel.CreateToolhelp32Snapshot.restype = wt.HANDLE
    snapshot = kernel.CreateToolhelp32Snapshot(4, 0)
    assert snapshot != ctypes.c_void_p(-1).value
    item = Thread()
    item.size = ctypes.sizeof(item)
    kernel.Thread32First.argtypes = [wt.HANDLE, ctypes.POINTER(Thread)]
    kernel.Thread32Next.argtypes = [wt.HANDLE, ctypes.POINTER(Thread)]
    valid = kernel.Thread32First(snapshot, ctypes.byref(item))
    count = 0
    while valid:
        count += item.owner == os.getpid()
        valid = kernel.Thread32Next(snapshot, ctypes.byref(item))
    kernel.CloseHandle.argtypes = [wt.HANDLE]
    assert kernel.CloseHandle(snapshot)
    assert threading.active_count() == 1
    if require_single:
        assert count == 1, count
    assert mem.peak_ws < 2**31 and mem.peak_pagefile < 2**31
    return dict(os_threads=count, peak_working_set_bytes=mem.peak_ws,
                peak_pagefile_bytes=mem.peak_pagefile, private_bytes=mem.private)


def oracle(words, bits, position):
    # Walk the published circular stream one bit at a time, MSB first.
    result = 0
    for bit in range((position + 1) * bits - 16, (position + 1) * bits):
        physical = bit % (256 * bits)
        result = (result << 1) | ((words[physical // 32] >> (31 - physical % 32)) & 1)
    return result


def cooperative(words, bits, position):
    # Model the two shuffle banks, including all 32 participating source lanes.
    lo = words[:32]
    hi = words[32:] + [0] * (64 - len(words))
    def word(index):
        return lo[index & 31] if index < 32 else hi[index & 31]
    end = (position + 1) * bits + 256 * bits
    pair = (word(((end - 16) // 32) % len(words)) << 32) | word(((end - 1) // 32) % len(words))
    return (pair >> ((32 - (end & 31)) & 31)) & 65535


def lane_reader(words, bits, lane, mode):
    start = (lane * 8 + 1) * bits + 256 * bits - 16
    index, shift = divmod(start, 32)
    assert 0 < shift < 32
    if mode == 'aligned_modulo':
        indices = [(index + j) % len(words) for j in range(3)]
    else:
        index -= len(words)
        indices = [index + len(words) if index < 0 else index,
                   index + 1 - len(words) if index + 1 >= len(words) else index + 1,
                   index + 2 - len(words) if index + 2 >= len(words) else index + 2]
    assert all(0 <= x < len(words) and (4 * x) % 4 == 0 for x in indices)
    a, b, c = [words[x] for x in indices]
    high = ((a << shift) | (b >> (32 - shift))) & MASK
    low = ((b << shift) | (c >> (32 - shift))) & MASK
    tail = (c << shift) & MASK
    states = []
    for j in range(8):
        offset = j * bits
        if offset <= 16:
            value = high >> (16 - offset)
        elif offset < 32:
            value = (high << (offset - 16)) | (low >> (48 - offset))
        elif offset <= 48:
            value = low >> (48 - offset)
        else:
            value = (low << (offset - 48)) | (tail >> (80 - offset))
        states.append(value & 65535)
    return states


def readers():
    rng = random.Random(20261021)
    result = []
    for bits in (5, 6):
        states_checked = 0
        length = 256 * bits
        # Zero, every physical basis vector, all ones, and eight dense patterns.
        for pattern in range(length + 10):
            words = [0] * (8 * bits)
            if 1 <= pattern <= length:
                physical = pattern - 1
                words[physical // 32] = 1 << (31 - physical % 32)
            elif pattern == length + 1:
                words = [MASK] * len(words)
            elif pattern > length + 1:
                words = [rng.getrandbits(32) for _ in words]
            for lane in range(32):
                expected = [oracle(words, bits, lane * 8 + j) for j in range(8)]
                assert expected == [cooperative(words, bits, lane * 8 + j) for j in range(8)]
                for mode in ('aligned_modulo', 'aligned_fastwrap', 'saddr'):
                    assert lane_reader(words, bits, lane, mode) == expected
                states_checked += 8
        shifts = sorted({((lane * 8 + 1) * bits - 16) % 32 for lane in range(32)})
        result.append(dict(bits=bits, basis_tiles=length, total_tiles=length + 10,
                           state_coordinates_per_reader=states_checked, shifts=shifts))
    return result


def coordinate(lane, j):
    return 2 * (lane % 4) + j % 2 + 8 * ((j // 2) % 2), lane // 4 + 8 * (j // 4)


def maps():
    maps_checked = 0
    for swizzle in (False, True):
        owners = {}
        for lane in range(32):
            for pair in range(4):
                r = 2 * (lane % 4) + 8 * (pair % 2)
                c = lane // 4 + 8 * (pair // 2)
                for half in (0, 1):
                    assert coordinate(lane, 2 * pair + half) == (r + half, c)
                    physical = (r + half) * 16 + (c ^ (((r + half) & 7) * 2 if swizzle else 0))
                    assert physical not in owners
                    owners[physical] = lane * 8 + pair * 2 + half
        assert set(owners) == set(range(256))
        for lane in range(32):
            for k in range(16):
                offset = k * 16 + ((lane % 16) ^ ((k & 7) * 2 if swizzle else 0))
                assert coordinate(*divmod(owners[offset], 8)) == (k, lane % 16)
        assert len({wave * 256 + offset for wave in range(4) for offset in owners}) == 1024
        maps_checked += 1
    store_cases = 0
    for tile_m, stride, swizzle in ((16, 16, False), (64, 16, True), (64, 18, True)):
        writes = [j // 16 * stride + j % 16 for thread in range(128)
                  for j in range(thread, tile_m * 16, 128)]
        assert len(set(writes)) == tile_m * 16
        for lane in range(32):
            for m in range(tile_m // 16):
                for k in range(16):
                    assert (m * 16 + lane % 16) * stride + k in writes
        for rows in range(1, 513):
            stores = [(first + m * 16 + 2 * j + lane // 16, lane % 16)
                      for first in range(0, rows, tile_m) for m in range(tile_m // 16)
                      for lane in range(32) for j in range(8)
                      if first + m * 16 + 2 * j + lane // 16 < rows]
            assert len(stores) == len(set(stores)) == rows * 16
            store_cases += 1
    return dict(q_maps=maps_checked, q_coordinates_per_wave=256,
                replicated_right_operands=512, output_store_cases=store_cases,
                row_domain=[1, 512], conserved_layouts=[[16, 16, False], [64, 16, True], [64, 18, True]])


def product_m1(container):
    state = container & 65535
    adjusted = MULT - 1
    packed = state | (((state * (adjusted >> 16)) & 65535) << 16)
    return (state * (adjusted & 65535) + packed) & MASK


def books():
    half = lambda bits: struct.unpack('<e', struct.pack('<H', bits))[0]
    rn16 = lambda value: struct.unpack('<H', struct.pack('<e', value))[0]
    rn32 = lambda value: struct.unpack('<f', struct.pack('<f', value))[0]
    assert half(0x1EEE) == 1774 / 2**18 and half(0xC931) == -1329 / 128
    expected, original, m1 = [], [], []
    max_numerator = 0
    checks = 0
    for state in range(65536):
        product = (state * MULT) & MASK
        total = sum(product.to_bytes(4, 'little'))
        numerator = (1024 + total) * 1774 - 1329 * 2048
        exact = numerator / 2**18
        max_numerator = max(max_numerator, abs(numerator))
        assert rn32(exact) == exact  # no intermediate FP32 rounding here
        assert half(0x6400 + total) == 1024 + total
        expected.append(rn16(exact))
        original.append(rn16(half(0x6400 + total) * half(0x1EEE) + half(0xC931)))
        for high in (0, 1, 0x8000, 0xFFFF, state ^ 0xA5A5):
            hashed = product_m1(state | (high << 16))
            assert hashed == product
            value = rn16(half(0x6400 + sum(hashed.to_bytes(4, 'little'))) * half(0x1EEE) + half(0xC931))
            assert value == expected[-1]
            checks += 1
        m1.append(value)
    assert expected == original == m1
    digest = hashlib.sha256(b''.join(struct.pack('<H', value) for value in expected)).hexdigest()
    assert digest == 'bc48d02cb1c14939dc47b90f870dd689d63e3b1ab69a157e65538db68677a6c8'
    pairs = 0
    # All states in both components, two independent partner permutations.
    for state in range(65536):
        for partner in (65535 - state, (state * 40503 + 17) & 65535):
            for first, second in ((state, partner), (partner, state)):
                first_sum = sum(((first * MULT) & MASK).to_bytes(4, 'little'))
                second_sum = sum(product_m1(second | 0xFFFF0000).to_bytes(4, 'little'))
                packed_input = (0x6400 + first_sum) | ((0x6400 + second_sum) << 16)
                lo = rn16(half(packed_input & 65535) * half(0x1EEE) + half(0xC931))
                hi = rn16(half(packed_input >> 16) * half(0x1EEE) + half(0xC931))
                raw = struct.pack('<I', lo | (hi << 16))
                assert raw[:2] == struct.pack('<H', expected[first])
                assert raw[2:] == struct.pack('<H', expected[second])
                pairs += 1
    # Original optimized helper needs u16 input; unlike M-1 it is not high-invariant.
    assert ((0x10000 * MULT) & MASK) != 0
    return dict(states=65536, high_container_products=checks, packed_pair_checks=pairs,
                maximum_absolute_numerator=max_numerator, fp32_rounding_mismatches=0,
                sha256=digest, physical_hfma2_executed=False,
                original_helper_requires_u16=True)


def policy(bits, inputs, outputs, rows):
    medium = rows >= 32 and ((rows >= 256 and ((bits == 6 and (inputs, outputs) == (5120, 1280)) or
                (bits == 5 and (inputs, outputs) == (4096, 1024)))) or
                (rows < 256 and ((bits == 6 and (inputs, outputs) == (5120, 1280)) or
                 (bits == 5 and (inputs, outputs) in ((1280, 32768), (8192, 5120), (5120, 2304), (2304, 5120), (4096, 1024))))))
    tile = 64 if medium or (rows >= 256 and outputs >= 2304) or (rows >= 64 and outputs >= 32768 and bits == 5) else 16
    groups = ((rows + tile - 1) // tile) * ((outputs + 63) // 64)
    splits = 1
    if medium and outputs >= 32768:
        splits = 4 if rows < 64 else 1
    else:
        target = (512 if rows > 64 and rows % 64 else 256) if medium else 160
        cap = 16 if medium else 32
        while groups * splits < target and splits < cap:
            splits *= 2
    pad = 2 if tile == 64 and (bits == 5 or medium) else 0
    return [tile, 16 + pad, tile == 64, min(splits, inputs // 16), groups]


def partitions_and_bounds():
    partitions = empty = 0
    for tiles in range(8, 513, 8):
        for splits in range(1, 33):
            spans = [(tiles * s // splits, tiles * (s + 1) // splits) for s in range(splits)]
            assert [k for begin, end in spans for k in range(begin, end)] == list(range(tiles))
            assert all(0 <= begin <= end <= tiles for begin, end in spans)
            empty += sum(begin == end for begin, end in spans)
            partitions += 1
    sizes = dict(packed=8192 * 32768 * 6 // 8, a=512 * 8192 * 2,
                 partial=32 * 512 * 32768 * 4, x=512 * 8192 * 4,
                 y=512 * 32768 * 4, suh=8192 * 2, svh=32768 * 2, bias=32768 * 4)
    assert max(sizes.values()) < 2**64 and sizes['partial'] == 2**31
    # Compare subtraction-based containment/disjointness against unbounded math.
    interval_cases = 0
    for base in (0, 4, 2**64 - 1024):
        for capacity in (0, 128, 1024):
            if base + capacity > 2**64:
                continue
            for offset in (0, 2, 4, 127, 128, 1024, 1025):
                for needed in (0, 2, 128, 1024):
                    valid = offset <= capacity and needed <= capacity - offset
                    assert valid == (0 <= offset <= capacity and offset + needed <= capacity)
                    interval_cases += 1
    for x in (0, 4, 128, 2**64 - 1024):
        for y in (0, 4, 128, 2**64 - 1024):
            for nx in (4, 128):
                for ny in (4, 128):
                    assert (nx <= y - x if x < y else ny <= x - y) == (x + nx <= y or y + ny <= x)
    table = {}
    for name, bits, inputs, outputs in (('q_b', 5, 1280, 32768), ('wo_b', 5, 8192, 5120),
              ('shared_w1_w3', 5, 5120, 2304), ('shared_w2', 5, 2304, 5120),
              ('q_a', 6, 5120, 1280), ('wo_a', 5, 4096, 1024)):
        table[name] = {str(rows): policy(bits, inputs, outputs, rows) for rows in (64, 128, 256)}
    return dict(partitions=partitions, empty_intervals=empty, interval_cases=interval_cases,
                maximum_bytes=sizes, maximum_partition_product=512 * 32,
                policy_columns=['TileM', 'AStride', 'SwizzleQ', 'S', 'groups'], baseline=table)


def source_hashes():
    paths = ['AGENTS.md', 'docs/WMMA_DECODER.md', 'docs/NUMERICS.md', 'src/gemm.hip',
             'src/backend.hip', 'src/launch_policy.hpp', 'src/packed_decoder.hpp',
             'src/trellis_optimized.hpp', 'src/trellis_device.hpp', 'src/transforms.hpp',
             'src/gemv_lds.hip', 'src/gemv_static_combined.hip', 'src/wmma_probe.hip',
             'reports/wmma-blocked-submit.md', 'reports/wmma-debug.md',
             'reports/gemv-static-combined-data/source-proof.json',
             'reports/gemv-static-combined-data/provenance.json',
             'reports/runtime-v10-data/frozen.sha256',
             'reports/wmma-blocked-submit-data/compiled.sha256']
    hashes = {}
    for path in paths:
        raw = (ROOT / path).read_bytes()
        hashes[path] = dict(raw=hashlib.sha256(raw).hexdigest(),
                            lf=hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest())
    old = json.loads((ROOT / 'reports/gemv-static-combined-data/provenance.json').read_text())['files']
    reuse = {path: hashes[path]['lf'] == old[path] for path in
             ('src/packed_decoder.hpp', 'src/gemv_static_combined.hip')}
    assert all(reuse.values())
    frozen = {line.split(maxsplit=1)[1]: line.split()[0] for line in
              (ROOT / 'reports/runtime-v10-data/frozen.sha256').read_text().splitlines()}
    integrated = {path: hashes[path]['lf'] == frozen[path] for path in
                  ('src/gemm.hip', 'src/gemv_lds.hip', 'src/backend.hip', 'src/launch_policy.hpp',
                   'src/trellis_optimized.hpp', 'src/trellis_device.hpp')}
    assert all(match for path, match in integrated.items() if path != 'src/trellis_device.hpp')
    old_baseline = {line.split(maxsplit=1)[1]: line.split()[0] for line in
                    (ROOT / 'reports/wmma-blocked-submit-data/compiled.sha256').read_text().splitlines()}
    assert hashes['src/trellis_device.hpp']['lf'] == old_baseline['src/trellis_device.hpp']
    return dict(inputs=hashes, historic_source_hash_match_lf=reuse,
                runtime_v10_source_hash_match_lf=integrated,
                baseline_device_header_matches_blocked_submit=True,
                native_probe_matches_historic_final=hashes['src/wmma_probe.hip']['lf'] ==
                    '0c16b4159f873eef9b2819fd0d02712d6e61516c1c5fdcb1628c0f30cc66ca86',
                proof_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    assert not sys.flags.optimize, 'Assertions must be enabled'
    kernel, handle, job, cpu = restrict_process()
    # Windows loader workers may exist briefly after process/DLL initialization.
    # Wait for their normal retirement before starting any proof computation.
    quiesce = time.perf_counter()
    while process_stats(kernel, handle, require_single=False)['os_threads'] != 1:
        assert time.perf_counter() - quiesce < 50, 'Loader workers did not retire'
        time.sleep(1)
    quiescence_seconds = time.perf_counter() - quiesce
    start = time.perf_counter()
    before = process_stats(kernel, handle)
    processor = os.environ.get('PROCESSOR_IDENTIFIER')
    if os.name == 'posix':
        processor = next(line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                         if line.startswith('model name'))
    result = dict(status='PASS_CPU_ONLY', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  python=sys.version, platform=list(os.uname()) if os.name == 'posix' else sys.platform,
                  processor=processor,
                  limits=dict(priority='nice 10' if os.name == 'posix' else 'BelowNormal',
                              cpu_affinity_selection=cpu, no_child_processes=True,
                              hard_process_memory_bytes=512 * 1024 * 1024), initial_process=before,
                  loader_quiescence_seconds=quiescence_seconds)
    result['sources'] = source_hashes()
    result['readers'] = readers()
    result['maps'] = maps()
    result['mul1'] = books()
    result['bounds'] = partitions_and_bounds()
    result['elapsed_cpu_proof_seconds'] = time.perf_counter() - start
    result['final_process'] = process_stats(kernel, handle)
    result['limits_of_proof'] = ['source-expression model only', 'no physical LLVM/half2/WMMA gate',
                                 'no GPU correctness or performance claim', 'high-container patterns are finite; algebra proves arbitrary high16']
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
