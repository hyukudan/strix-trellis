"""Read-only, bounded CPU audit of independently produced OUR EXL3 snapshots."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
sys.path.insert(0, str(ROOT / 'tests'))
from strix_trellis.container import SafetensorsFile, load_exl3
from strix_trellis.codec import decode_states, decode_tile, unpack_states
from test_reference import bit_oracle

DTYPE_BYTES = dict(BOOL=1, U8=1, I8=1, I16=2, U16=2, I32=4, U32=4,
                   I64=8, U64=8, F16=2, BF16=2, F32=4, F64=8)
MARKERS = (0, 0xCBAC1FED, 0x83DCD12D)
CODEBOOKS = ('3inst', 'mcg', 'mul1')


def low_priority():
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        if not kernel.SetPriorityClass(wintypes.HANDLE(-1), 0x4000):
            raise OSError('Cannot set BelowNormal process priority')
    else:
        os.nice(10)


def sha_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unique(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def stamp(path):
    s = Path(path).stat()
    return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns, inode=s.st_ino,
                mtime_utc=datetime.fromtimestamp(s.st_mtime, timezone.utc).isoformat())


class StableFile:
    """Stat guard plus repeated hashes of exactly the bytes used, no source locks."""
    def __init__(self, path):
        self.path = Path(path)
        self.before = stamp(path)
        self.ranges = {}

    def guard(self):
        if stamp(self.path) != self.before:
            raise ValueError('Source modified: ' + str(self.path))

    def digest_range(self, offset, length):
        self.guard()
        h = hashlib.sha256()
        with self.path.open('rb') as stream:
            stream.seek(offset)
            while length:
                block = stream.read(min(length, 1024**2))
                if not block:
                    raise ValueError('Truncated range')
                h.update(block)
                length -= len(block)
        self.guard()
        return h.hexdigest()

    def remember(self, offset, length):
        key = (offset, length)
        value = self.digest_range(*key)
        if key in self.ranges and self.ranges[key] != value:
            raise ValueError('Source bytes modified')
        self.ranges[key] = value
        return value

    def verify(self):
        self.guard()
        for key, value in self.ranges.items():
            if self.digest_range(*key) != value:
                raise ValueError('Source bytes modified: ' + str(self.path))
        self.guard()


class Header(StableFile):
    """Independent safetensors parser validating ALL header spans and shapes."""
    def __init__(self, path):
        super().__init__(path)
        with self.path.open('rb') as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError('Truncated header length')
            length, = struct.unpack('<Q', prefix)
            if not 2 <= length <= min(8_000_000, self.before['bytes'] - 8):
                raise ValueError('Invalid header length')
            raw = stream.read(length)
        self.header_sha256 = hashlib.sha256(raw).hexdigest()
        self.data_offset = 8 + length
        self.header = json.loads(raw, object_pairs_hook=unique)
        if not isinstance(self.header, dict):
            raise ValueError('Header must be an object')
        spans = []
        for name, info in self.header.items():
            if name == '__metadata__':
                if not isinstance(info, dict) or any(not isinstance(v, str) for v in info.values()):
                    raise ValueError('Invalid metadata')
                continue
            if not isinstance(info, dict) or info.get('dtype') not in DTYPE_BYTES:
                raise ValueError('Unsupported tensor dtype: ' + name)
            shape, offsets = info.get('shape'), info.get('data_offsets')
            if (not isinstance(shape, list) or len(shape) > 8 or
                    any(type(x) is not int or x < 0 for x in shape) or
                    not isinstance(offsets, list) or len(offsets) != 2 or
                    any(type(x) is not int for x in offsets)):
                raise ValueError('Invalid shape/span: ' + name)
            begin, end = offsets
            if not 0 <= begin <= end <= self.before['bytes'] - self.data_offset:
                raise ValueError('Truncated/out-of-range tensor: ' + name)
            if end - begin != math.prod(shape) * DTYPE_BYTES[info['dtype']]:
                raise ValueError('Shape/dtype/span mismatch: ' + name)
            if end > begin:
                spans.append((begin, end, name))
        spans.sort()
        cursor = 0
        for begin, end, name in spans:
            if begin != cursor:
                raise ValueError('Overlap or gap before: ' + name)
            cursor = end
        if cursor != self.before['bytes'] - self.data_offset:
            raise ValueError('Unreferenced/truncated payload')
        if self.remember(0, self.data_offset) != hashlib.sha256(prefix + raw).hexdigest():
            raise ValueError('Header changed during parsing')
        self.guard()

    def tensor_sha(self, name):
        begin, end = self.header[name]['data_offsets']
        return self.remember(self.data_offset + begin, end - begin)


def independent_lut(codebook):
    """Exact scalar arithmetic and struct RN16; independent of codec LUT builder."""
    out = bytearray()
    for state in range(65536):
        if codebook == 'mul1':
            p = state * 0x83DCD12D & 0xffffffff
            b = sum((p >> shift) & 255 for shift in (0, 8, 16, 24))
            value = ((1024 + b) * 1774 - 1329 * 2048) / 262144
        else:
            p = (state * (89226354 if codebook == '3inst' else 0xCBAC1FED)
                 + (64248484 if codebook == '3inst' else 0)) & 0xffffffff
            h = (p & 0x8FFF8FFF) ^ 0x3B603B60
            value = sum(struct.unpack('<e', struct.pack('<H', (h >> shift) & 65535))[0]
                        for shift in (0, 16))
        out.extend(struct.pack('<e', value))
    table = np.frombuffer(out, dtype='<f2')
    canonical = decode_states(np.arange(65536, dtype=np.uint16), codebook)
    if table.tobytes() != canonical.tobytes():
        raise ValueError('Canonical state LUT differs')
    return table


def decode_block(trellis, bits, table):
    """Extract circular 16-bit windows from u32 pairs, bounded output columns."""
    if type(bits) is not int or bits not in range(1, 9):
        raise ValueError('Integer rate required')
    w = np.ascontiguousarray(trellis, dtype='<u2').view('<u4')
    q = np.empty((trellis.shape[0], 16, trellis.shape[1], 16), dtype='<f2')
    size, count = 256 * bits, 8 * bits
    for row in range(16):
        for col in range(16):
            pos = ((col & 7) << 5) | ((row & 6) << 2) | (row & 1) | ((row & 8) >> 2) | ((col & 8) >> 1)
            idx, shift = divmod((bits * (pos + 1) - 16) % size, 32)
            wide = (w[..., idx].astype(np.uint64) << 32) | w[..., (idx + 1) % count]
            states = ((wide >> (48 - shift)) & 65535).astype(np.uint16)
            q[:, row, :, col] = table[states]
    return q.reshape(trellis.shape[0] * 16, trellis.shape[1] * 16)


def check_tile(tile, bits, table, codebook):
    states = bit_oracle(tile, bits)
    np.testing.assert_array_equal(states, unpack_states(tile, bits))
    expected = np.empty((16, 16), dtype='<f2')
    # Inverse lane/fragment coordinate mapping, distinct from decoder mapping.
    for lane in range(32):
        for fragment in range(8):
            r = 2*(lane % 4) + fragment % 2 + 8*((fragment//2) % 2)
            c = lane//4 + 8*(fragment//4)
            expected[r, c] = table[states[8*lane + fragment]]
    got = decode_block(tile[None, None], bits, table)
    if got.tobytes() != expected.tobytes() or got.tobytes() != decode_tile(tile, bits, codebook).tobytes():
        raise ValueError('All-256-weight tile bits differ')


def required(layer):
    p = f'layers.{layer}.'
    return [p+'attn.'+s for s in ('wq_a', 'wq_b', 'wkv', 'wo_b')] + [
        p+f'attn.wo_a.slice.{s}' for s in range(8)] + [
        p+'ffn.shared_experts.'+s for s in ('w1', 'w2', 'w3')]


def audit_directory(directory, tables):
    paths = sorted(directory.glob('*.safetensors'))
    records, actual = [], {}
    for path in paths:
        h = Header(path)
        file = SafetensorsFile(path)
        groups = []
        for prefix in sorted(n[:-8] for n in h.header if n.endswith('.trellis')):
            g = load_exl3(file, prefix)
            suffixes = sorted(n[len(prefix)+1:] for n in h.header if n.startswith(prefix+'.'))
            if set(suffixes) - {'trellis', 'suh', 'svh', 'mul1', 'mcg', 'bias'}:
                raise ValueError('Unsupported group suffix/marker: ' + prefix)
            scale_info = {}
            for suffix in suffixes:
                if suffix != 'trellis':
                    a = file.tensor(prefix+'.'+suffix)
                    rec = dict(sha256=h.tensor_sha(prefix+'.'+suffix))
                    if suffix in ('suh', 'svh'):
                        rec.update(negative=int(np.count_nonzero(np.signbit(a))),
                                   nonunit=int(np.count_nonzero(np.abs(a) != 1)),
                                   zeros=int(np.count_nonzero(a == 0)), min=float(a.min()), max=float(a.max()))
                    if suffix in ('mul1', 'mcg'):
                        rec['value'] = None if not a.size else hex(int(a.item()) & 0xffffffff)
                    scale_info[suffix] = rec
            selected = sorted({(0, 0), (g.trellis.shape[0]//2, g.trellis.shape[1]//2),
                               (g.trellis.shape[0]-1, g.trellis.shape[1]-1)})
            samples = []
            for i, j in selected:
                start = h.data_offset + h.header[prefix+'.trellis']['data_offsets'][0]
                offset = start + (i*g.trellis.shape[1] + j)*g.trellis.shape[2]*2
                digest = h.remember(offset, g.trellis.shape[2]*2)
                check_tile(g.trellis[i, j], int(g.bits), tables[g.codebook], g.codebook)
                samples.append(dict(tile=[i, j], sha256=digest, all_256_weights_bit_equal=True))
            groups.append(dict(prefix=prefix, bits=g.bits, codebook=g.codebook, suffixes=suffixes,
                               scales_and_markers=scale_info, tiles=samples, loader_accepted=True))
        for name in h.header:
            if name != '__metadata__':
                if name in actual:
                    raise ValueError('Duplicate tensor across files: ' + name)
                actual[name] = path.name
        layers = sorted({int(n.split('.')[1]) for n in h.header if n.startswith('layers.')})
        present = {g['prefix'] for g in groups}
        h.verify()
        records.append(dict(file=path.name, stat=h.before, header_sha256=h.header_sha256,
                            header_and_length_sha256=h.ranges[(0, h.data_offset)],
                            metadata=h.header.get('__metadata__'), tensors=h.header, groups=groups,
                            layers=layers, required_sites_missing=[p for l in layers for p in required(l) if p not in present],
                            stability='stat and relevant byte SHA repeated; full trellis not hashed'))
        print(f'{directory.name}/{path.name}: {len(groups)} groups accepted', flush=True)
    index = StableFile(directory/'index.json')
    index_sha = index.remember(0, index.before['bytes'])
    content = json.loads(index.path.read_bytes(), object_pairs_hook=unique)
    wm = content.get('weight_map')
    if not isinstance(wm, dict) or any(not isinstance(v, str) for v in wm.values()):
        raise ValueError('Invalid index weight_map')
    index.verify()
    layers = sorted({l for r in records for l in r['layers']})
    return dict(directory=str(directory), directory_entries=sorted(p.name for p in directory.iterdir()),
                filenames=[p.name for p in paths], files=records, layers=layers,
                layers_0_to_39_absent=[l for l in range(40) if l not in layers],
                head_tensor_names=[n for n in actual if 'head' in n.lower()],
                index=dict(stat=index.before, sha256=index_sha, content=content,
                           filenames=sorted(set(wm.values())),
                           layers=sorted({int(n.split('.')[1]) for n in wm if n.startswith('layers.')}),
                           actual_tensors_not_indexed=sorted(set(actual)-set(wm)),
                           index_entries_missing_or_wrong_file={n:f for n,f in wm.items() if actual.get(n) != f}),
                source_semantics='OUR pack local snapshot; header names/shapes only establish sites. No base-weight or model-quality inference.')


class AuditTests(unittest.TestCase):
    def test_malformed_headers(self):
        cases = [({'x':dict(dtype='F16', shape=[2], data_offsets=[0, 2])}, b'12'),
                 ({'x':dict(dtype='F16', shape=[2], data_offsets=[0, 4])}, b'12'),
                 ({'x':dict(dtype='BAD', shape=[1], data_offsets=[0, 2])}, b'12'),
                 ({'x':dict(dtype='I16', shape=[True], data_offsets=[0, 2])}, b'12'),
                 ({n:dict(dtype='I16', shape=[1], data_offsets=[0, 2]) for n in ('x','y')}, b'12')]
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'bad.safetensors'
            for header, body in cases:
                raw = json.dumps(header).encode()
                p.write_bytes(struct.pack('<Q', len(raw))+raw+body)
                with self.assertRaises(ValueError):
                    Header(p)
            p.write_bytes(b'\xff'*8)
            with self.assertRaises(ValueError):
                Header(p)
            raw = b'{"x":{},"x":{}}'
            p.write_bytes(struct.pack('<Q', len(raw))+raw)
            with self.assertRaises(ValueError):
                Header(p)

    def test_changed_bytes_even_with_restored_mtime(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'bytes'
            p.write_bytes(b'abcd')
            s = StableFile(p)
            s.remember(0, 4)
            p.write_bytes(b'abce')
            os.utime(p, ns=(s.before['mtime_ns'], s.before['mtime_ns']))
            with self.assertRaises(ValueError):
                s.verify()
            p.write_bytes(b'longer')
            with self.assertRaises(ValueError):
                s.guard()

    def test_loader_rejects_nonfinite_and_bad_marker(self):
        from test_container import tensors, write_file
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'source'
            for suffix in ('suh', 'svh'):
                a = tensors()
                a['projection.'+suffix][3] = np.inf
                write_file(p, a)
                with self.assertRaises(ValueError):
                    load_exl3(p, 'projection')
            a = tensors()
            a['projection.mul1'] = np.array(17, dtype='<i4')
            write_file(p, a)
            with self.assertRaises(ValueError):
                load_exl3(p, 'projection')

    def test_all_integer_decoders_and_codebooks(self):
        rng = np.random.default_rng(20261021)
        for cb in CODEBOOKS:
            table = independent_lut(cb)
            for bits in range(1, 9):
                tile = rng.integers(0, 65536, 16*bits, dtype=np.uint16)
                check_tile(tile, bits, table, cb)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=Path('external-data'))
    p.add_argument('--out', type=Path, default=ROOT/'reports/ours-pack-audit-data/inventory.json')
    p.add_argument('--self-test', action='store_true')
    a = p.parse_args()
    low_priority()
    if a.self_test:
        unittest.main(argv=[sys.argv[0]], exit=True)
    tables = {cb:independent_lut(cb) for cb in CODEBOOKS}
    result = dict(date_utc=datetime.now(timezone.utc).isoformat(), generator_sha256=sha_file(Path(__file__)),
                  base_commit='2e12c3cf117de8c2d243a6c4aaaa8931e8ec346b',
                  lut_sha256={cb:hashlib.sha256(t.tobytes()).hexdigest() for cb,t in tables.items()},
                  packs=[audit_directory(a.base/f'ours-K{k}', tables) for k in (4, 5)])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
