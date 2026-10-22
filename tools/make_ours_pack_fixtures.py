"""Four full CPU-only STGPU001 fixtures from stable OUR layer-zero packs."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import time
import unittest

import numpy as np
from audit_ours_packs import (ROOT, Header, MARKERS, CODEBOOKS, independent_lut,
                             decode_block, check_tile, low_priority, sha_file)
from strix_trellis.container import SafetensorsFile, load_exl3
from strix_trellis.codec import decode_tile
from strix_trellis.linear import input_stages_fp32, core_dot_fp32, output_stages_fp32

LIMIT = 256 * 1024**2
HOST_LIMIT = 2 * 1024**3


def peak_working_set():
    if os.name != 'nt':
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('peak_ws','ws','peak_pool','pool','peak_nonpaged',
                                              'nonpaged','pagefile','peak_pagefile')]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    api = ctypes.WinDLL('psapi', use_last_error=True)
    api.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
    if not api.GetProcessMemoryInfo(wintypes.HANDLE(-1), ctypes.byref(counters), counters.cb):
        raise OSError('Cannot measure CPU working set')
    if counters.peak_ws >= HOST_LIMIT:
        raise ValueError('Host memory budget exceeded')
    return counters.peak_ws


def strict_parse(path):
    """CPU structural equivalent of runtime Fixture, with finite payload checks."""
    size = path.stat().st_size
    if not 16 <= size <= LIMIT:
        raise ValueError('Invalid fixture size')
    parts = {}
    with path.open('rb') as f:
        header = f.read(16)
        magic, version, count = struct.unpack('<8sII', header)
        if magic != b'STGPU001' or version != 1 or not 1 <= count <= 64:
            raise ValueError('Invalid fixture header')
        for _ in range(count):
            raw = f.read(16)
            if len(raw) != 16:
                raise ValueError('Truncated section')
            tag_raw, length = struct.unpack('<8sQ', raw)
            tag = tag_raw.rstrip(b'\0').decode('ascii')
            if not tag or b'\0' in tag_raw.rstrip(b'\0') or tag in parts or length > size - f.tell():
                raise ValueError('Invalid/duplicate section')
            parts[tag] = (f.tell(), length)
            f.seek(length, 1)
        if f.tell() != size:
            raise ValueError('Trailing bytes')
        if parts.get('meta', (0, 0))[1] != 32:
            raise ValueError('Bad metadata section')
        f.seek(parts['meta'][0])
        i, n, bits, cb, marker, rows, bias, seed = struct.unpack('<8I', f.read(32))
        if (not i or not n or i % 128 or n % 128 or i > 8192 or n > 131072 or
                not 1 <= rows <= 512 or not 1 <= bits <= 8 or cb > 2 or bias > 1 or marker != MARKERS[cb]):
            raise ValueError('Invalid metadata/marker')
        expected = dict(meta=32, trellis=i*n*bits//8, suh=2*i, svh=2*n, bias=4*n*bias,
                        x=4*rows*i, old=4*rows*n, scaled=2*rows*i, inh=4*rows*i,
                        a=2*rows*i, q=2*i*n, core=4*rows*n, ref=4*rows*n)
        if set(parts) != set(expected) or any(parts[k][1] != v for k,v in expected.items()):
            raise ValueError('Bad required sections')
        for tag, (offset, length) in parts.items():
            if tag in ('trellis', 'meta'):
                continue
            dtype = '<f2' if tag in ('suh', 'svh', 'scaled', 'a', 'q') else '<f4'
            f.seek(offset)
            remaining = length
            while remaining:
                block = f.read(min(1024**2, remaining))
                if not block or not np.isfinite(np.frombuffer(block, dtype=dtype)).all():
                    raise ValueError('Nonfinite/truncated section: ' + tag)
                remaining -= len(block)
    return parts


def write_sections(path, sections):
    size = 16 + sum(16 + (len(v) if isinstance(v, bytes) else v.nbytes) for v in sections.values())
    if size > LIMIT:
        raise ValueError('Fixture size limit exceeded')
    hashes = {}
    with path.open('xb') as f:
        f.write(b'STGPU001' + struct.pack('<II', 1, len(sections)))
        for tag, value in sections.items():
            length = len(value) if isinstance(value, bytes) else value.nbytes
            f.write(tag.encode().ljust(8, b'\0') + struct.pack('<Q', length))
            h = hashlib.sha256()
            if isinstance(value, bytes):
                f.write(value)
                h.update(value)
            elif value.size:
                # All arrays are contiguous or row-strided; never copy a whole large section.
                for row in value.reshape(-1, value.shape[-1]):
                    view = memoryview(np.ascontiguousarray(row)).cast('B')
                    for start in range(0, len(view), 1024**2):
                        chunk = view[start:start+1024**2]
                        f.write(chunk)
                        h.update(chunk)
            hashes[tag] = h.hexdigest()
    parts = strict_parse(path)
    with path.open('rb') as f:
        for tag, (offset, length) in parts.items():
            h = hashlib.sha256()
            f.seek(offset)
            while length:
                block = f.read(min(1024**2, length))
                h.update(block)
                length -= len(block)
            if h.hexdigest() != hashes[tag]:
                raise ValueError('Serialized checkpoint SHA differs')
    return dict(bytes=size, sha256=sha_file(path), section_sha256=hashes)


def publish(temporary, target):
    """Create only; an existing identical fixture is verified without rewriting."""
    if target.exists():
        if target.stat().st_size != temporary.stat().st_size or sha_file(target) != sha_file(temporary):
            raise ValueError('Refusing to overwrite differing file: ' + str(target))
        return 'verified existing identical file'
    # Hard-link creation fails if another process created the target in between.
    os.link(temporary, target)
    return 'created'


def tie_inputs(x, u):
    """Place actual exact halfway FP32 scale products in original-basis input."""
    candidates = np.arange(0x3f000000, 0x3f010000, dtype=np.uint32).view(np.float32)
    records = []
    for channel, scale in enumerate(u):
        if not scale or not np.isfinite(scale):
            continue
        exact = candidates.astype(np.float64) * float(scale)
        rounded = exact.astype(np.float32)
        direction = np.where(exact > rounded.astype(np.float64), np.inf, -np.inf).astype(np.float32)
        adjacent = np.nextafter(rounded, direction)
        mask = (exact != rounded.astype(np.float64)) & (exact == (rounded.astype(np.float64)+adjacent.astype(np.float64))/2)
        indices = np.flatnonzero(mask)
        if len(indices):
            index = int(indices[0])
            x[7, channel] = candidates[index]
            records.append(dict(row=7, channel=channel, x_hex=hex(int(candidates[index:index+1].view(np.uint32)[0])),
                                suh_hex=hex(int(np.array(scale).view(np.uint16))),
                                exact_product=float(exact[index]), rounded_product=float(rounded[index])))
            if len(records) == 4:
                return records
    raise ValueError('Could not construct RN32 ties using supplied scales')


def generate(base, out, bits, site, table):
    started = time.perf_counter()
    source = base/f'ours-K{bits}/dense.L0.safetensors'
    prefix = 'layers.0.attn.'+site
    h = Header(source)
    g = load_exl3(SafetensorsFile(source), prefix)
    if g.bits != bits or g.codebook != 'mul1':
        raise ValueError('Unexpected actual pack bitrate/codebook')
    source_hashes = {n:dict(**info, sha256=h.tensor_sha(n)) for n,info in h.header.items() if n.startswith(prefix+'.')}
    i, n, rows, seed = g.input_size, g.output_size, 8, 20261021+bits*100+(1 if site == 'wq_b' else 0)
    rng = np.random.Generator(np.random.PCG64(seed))
    x = rng.normal(0, .125, (rows, i)).astype('<f4')
    x[0, :8] = [0, -0., 2**-24, -2**-24, 2**-14, -2**-14, .5, -.25]
    ties = tie_inputs(x, g.suh)
    stages = input_stages_fp32(x, g.suh)
    a = stages['core_input_fp16']
    old = rng.normal(0, .2, (rows, n)).astype('<f4')
    core, ref = np.empty((rows, n), dtype='<f4'), np.empty((rows, n), dtype='<f4')
    max_error, max_bound_ratio, canonical_weights = 0., 0., 0
    name = f'ours-k{bits}-L0-{site}-r8.stgpu'
    # Unique files live ONLY in our output directory and are removed on failure.
    with tempfile.TemporaryDirectory(prefix='.ours-audit-', dir=out) as work:
        qpath = Path(work)/'q.tmp'
        q = np.memmap(qpath, mode='w+', dtype='<f2', shape=(i, n))
        try:
            for col in range(0, n, 128):
                h.guard()
                t = g.trellis[:, col//16:col//16+8]
                block = decode_block(t, bits, table)
                canonical = decode_tile(t, bits, g.codebook).transpose(0, 2, 1, 3).reshape(i, 128)
                if block.tobytes() != canonical.tobytes():
                    raise ValueError('Full canonical Q mismatch')
                canonical_weights += block.size
                q[:, col:col+128] = block
                core[:, col:col+128] = core_dot_fp32(a, block)
                ref[:, col:col+128] = output_stages_fp32(core[:, col:col+128], g.svh[col:col+128],
                    None if g.bias is None else g.bias[col:col+128])['output_fp32']
                # FP64 all rows/all columns bound; distinct reduction, no BLAS or FMA.
                exact = np.zeros((rows, 128), dtype=np.float64)
                mass = exact.copy()
                for input_index in range(i):
                    products = a[:, input_index, None].astype(np.float64)*block[input_index].astype(np.float64)
                    exact += products
                    mass += np.abs(products)
                error = np.abs(core[:, col:col+128].astype(np.float64)-exact)
                gamma = (2*i*2**-24)/(1-2*i*2**-24)
                bound = gamma*mass
                if not np.all(error <= bound + 1e-30):
                    raise ValueError('All-output FP64 error bound exceeded')
                max_error = max(max_error, float(error.max()))
                max_bound_ratio = max(max_bound_ratio, float(np.max(error / np.maximum(bound, 1e-300))))
                if col % 4096 == 0:
                    print(f'{name}: columns {col+128}/{n}; peak {peak_working_set()} bytes', flush=True)
            for ti,tj in sorted({(0,0), (g.trellis.shape[0]-1,g.trellis.shape[1]-1)}):
                check_tile(g.trellis[ti,tj], bits, table, g.codebook)
            q.flush()
            sections = dict(meta=struct.pack('<8I', i,n,bits,2,MARKERS[2],rows,int(g.bias is not None),seed),
                            trellis=g.trellis, suh=g.suh, svh=g.svh,
                            bias=np.empty(0,dtype='<f4') if g.bias is None else g.bias,
                            x=x, old=old, scaled=stages['scaled_fp16'], inh=stages['input_hadamard_fp32'],
                            a=a, q=q, core=core, ref=ref)
            temporary = Path(work)/name
            record = write_sections(temporary, sections)
            # Rehash EVERY selected source tensor after full decode AND serialization.
            h.verify()
            record['publication'] = publish(temporary, out/name)
            del sections
        finally:
            q._mmap.close()
            del q
    record.update(file=name, sourcepath=str(source), tensorprefix=prefix, source_stat=h.before,
                  header_sha256=h.header_sha256, header_and_length_sha256=h.ranges[(0,h.data_offset)],
                  source_tensors=source_hashes, seed=seed, input=i, output=n, rows=rows, bits=bits,
                  codebook=g.codebook, marker=hex(MARKERS[2]), bias=g.bias is not None,
                  decoder_lut_sha256=hashlib.sha256(table.tobytes()).hexdigest(), rn32_ties=ties,
                  canonical_q_weights_bit_equal=canonical_weights, fp64_checked_outputs=rows*n,
                  fp64_max_abs_error=max_error, fp64_max_error_bound_ratio=max_bound_ratio,
                  strict_parse_and_all_serialized_section_sha=True,
                  peak_process_working_set_bytes=peak_working_set(), elapsed_cpu_seconds=time.perf_counter()-started)
    print(json.dumps({k:record[k] for k in ('file','bytes','sha256','elapsed_cpu_seconds')}), flush=True)
    return record


class FixtureTests(unittest.TestCase):
    def test_strict_parser_and_create_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'case'
            i=n=128; rows=8
            sections = dict(meta=struct.pack('<8I',i,n,4,2,MARKERS[2],rows,0,1),
                trellis=np.zeros((8,8,64),dtype='<i2'), suh=np.ones(i,dtype='<f2'),
                svh=np.ones(n,dtype='<f2'), bias=np.empty(0,dtype='<f4'))
            for tag,shape,dtype in [('x',(rows,i),'<f4'),('old',(rows,n),'<f4'),
                ('scaled',(rows,i),'<f2'),('inh',(rows,i),'<f4'),('a',(rows,i),'<f2'),
                ('q',(i,n),'<f2'),('core',(rows,n),'<f4'),('ref',(rows,n),'<f4')]:
                sections[tag] = np.zeros(shape,dtype=dtype)
            write_sections(path, sections)
            target = Path(folder)/'published'
            self.assertEqual(publish(path,target),'created')
            self.assertIn('identical',publish(path,target))
            original = path.read_bytes()
            for data in (original[:-1], original+b'x', b'BAD'+original[3:],
                         original[:12]+struct.pack('<I',65)+original[16:]):
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    strict_parse(path)
            path.write_bytes(original)
            parts = strict_parse(path)
            duplicate = bytearray(original)
            duplicate[parts['trellis'][0]-16:parts['trellis'][0]-8] = b'meta\0\0\0\0'
            path.write_bytes(duplicate)
            with self.assertRaises(ValueError):
                strict_parse(path)
            for tag, replacement in [('meta',struct.pack('<8I',i,n,4,2,17,rows,0,1)),
                                      ('suh',np.array([np.inf],dtype='<f2').tobytes())]:
                data=bytearray(original); start=parts[tag][0]
                data[start:start+len(replacement)]=replacement
                path.write_bytes(data)
                with self.assertRaises(ValueError):
                    strict_parse(path)
            # Break the hard link only by using a distinct temporary source.
            other=Path(folder)/'different'; other.write_bytes(b'different')
            with self.assertRaises(ValueError):
                publish(other,target)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=Path('external-data'))
    p.add_argument('--out',type=Path,default=Path('external-data/gpu-ours-pack-tests'))
    p.add_argument('--metadata',type=Path,default=ROOT/'reports/ours-pack-audit-data/fixtures.json')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    low_priority()
    if args.self_test:
        unittest.main(argv=[__file__],exit=True)
    args.out.mkdir(parents=True,exist_ok=True)
    table=independent_lut('mul1')
    records=[generate(args.base,args.out,bits,site,table) for bits in (4,5) for site in ('wkv','wq_b')]
    manifest=dict(date_utc=datetime.now(timezone.utc).isoformat(),format='STGPU001',version=1,
        generator_sha256=sha_file(Path(__file__)),audit_script_sha256=sha_file(ROOT/'tools/audit_ours_packs.py'),
        numpy=np.__version__,output_directory=str(args.out),fixtures=records,
        host_limit_bytes=HOST_LIMIT,decode_block_columns=128,threads=1,priority='BelowNormal on Windows; nice +10 elsewhere',
        total_bytes=sum(r['bytes'] for r in records),
        contract='Original-basis finite synthetic FP32; supplied signed/nonunit folded scales preserved; RN32 scale -> RN16 -> ascending H128 FP32 normalized once -> RN16 -> separate ordered FP32 core -> H128 FP32 -> svh -> optional bias',
        semantics='CPU transport/numerics evidence only. No GPU/API gate or model/base-weight quality claim.')
    # Portable deterministic provenance accompanies the external fixtures. Timings
    # and process high-water measurements stay in the report, not artifact identity.
    external = {k:v for k,v in manifest.items() if k not in ('date_utc','fixtures')}
    external['fixtures'] = [{k:v for k,v in r.items() if k not in
                            ('publication','elapsed_cpu_seconds','peak_process_working_set_bytes')} for r in records]
    with tempfile.TemporaryDirectory(prefix='.ours-manifest-',dir=args.out) as work:
        temporary=Path(work)/'manifest.json'
        temporary.write_text(json.dumps(external,indent=2)+'\n',encoding='utf-8')
        manifest['external_manifest_publication']=publish(temporary,args.out/'manifest.json')
        manifest['external_manifest_sha256']=sha_file(args.out/'manifest.json')
    args.metadata.parent.mkdir(parents=True,exist_ok=True)
    args.metadata.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
