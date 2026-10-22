"""CPU-only byte identity gate; operates solely on owned evidence copies."""
import hashlib
import json
import struct
from pathlib import Path


def sections(path):
    data = path if isinstance(path, bytes) else Path(path).read_bytes()
    assert data[:6] == b'\x7fELF\x02\x01'
    off = struct.unpack_from('<Q', data, 40)[0]
    size, count, names = struct.unpack_from('<HHH', data, 58)
    headers = [struct.unpack_from('<IIQQQQIIQQ', data, off+i*size) for i in range(count)]
    strings = data[headers[names][4]:headers[names][4]+headers[names][5]]
    result = {}
    for h in headers:
        name = strings[h[0]:].split(b'\0', 1)[0].decode()
        result[name] = data[h[4]:h[4]+h[5]]
    return result


def images(bundle):
    magic = b'__CLANG_OFFLOAD_BUNDLE__'
    cursor = 0
    result = []
    while True:
        start = bundle.find(magic, cursor)
        if start < 0:
            break
        pos = start+len(magic)
        count = struct.unpack_from('<Q', bundle, pos)[0]
        pos += 8
        end = pos
        for _ in range(count):
            off, size, length = struct.unpack_from('<QQQ', bundle, pos)
            pos += 24
            target = bundle[pos:pos+length]
            pos += length
            end = max(end, start+off+size)
            if target.startswith(b'hip-'):
                result.append(bundle[start+off:start+off+size])
        cursor = end
    assert result
    return result


def instructions(bundle, prefix):
    result = {}
    for i, data in enumerate(images(bundle)):
        Path(f'build/{prefix}-image-{i}.elf').write_bytes(data)
        sec = sections(data)
        # Symbols define the executed instruction byte extents, ignoring ELF
        # source-file strings and build IDs. Record whole text and metadata too.
        off = struct.unpack_from('<Q', data, 40)[0]
        size, count, names = struct.unpack_from('<HHH', data, 58)
        headers = [struct.unpack_from('<IIQQQQIIQQ', data, off+j*size) for j in range(count)]
        strings = sec['.strtab']
        for pos in range(0, len(sec['.symtab']), 24):
            name, info, other, index, value, length = struct.unpack_from('<IBBHQQ', sec['.symtab'], pos)
            if not index or index >= count or not length or (info & 15) != 2:
                continue
            h = headers[index]
            name = strings[name:].split(b'\0', 1)[0].decode()
            if h[2] & 4:  # executable section
                begin = h[4]+value-h[3]
                result[name] = sha(data[begin:begin+length])
    return result


def sha(data):
    return hashlib.sha256(data).hexdigest()


old = sections('build/old-evidence/wmma_blocked.o')['.hip_fatbin']
new = sections('build/wmma_blocked_submit.o')['.hip_fatbin']
Path('build/old-core-fatbin.bin').write_bytes(old)
Path('build/new-core-fatbin.bin').write_bytes(new)
record = dict(old_core_fatbin_sha256=sha(old), new_core_fatbin_sha256=sha(new),
              complete_core_image_identical=old == new)
# The old source and benchmark snapshot are unmodified; its gemm translation
# unit is compiled with the same flags. Check machine code in linked images
# separately below if the benchmark's larger fatbin changes due to host edits.
old_bench = sections('build/old-evidence/wmma_blocked_bench')['.hip_fatbin']
new_bench = sections('build/wmma_blocked_submit_bench')['.hip_fatbin']
record.update(old_bench_fatbin_sha256=sha(old_bench), new_bench_fatbin_sha256=sha(new_bench),
              complete_bench_image_identical=old_bench == new_bench)
Path('build/old-bench-fatbin.bin').write_bytes(old_bench)
Path('build/new-bench-fatbin.bin').write_bytes(new_bench)
old_code, new_code = instructions(old, 'old-core'), instructions(new, 'new-core')
old_all, new_all = instructions(old_bench, 'old-bench'), instructions(new_bench, 'new-bench')
record.update(core_symbols=old_code, core_instruction_bytes_identical=old_code == new_code,
              bench_symbols=old_all, bench_instruction_bytes_identical=old_all == new_all,
              core_symbol_count=len(old_code), bench_symbol_count=len(old_all))
old_metadata = [sha(sections(data)['.note']) for data in images(old)]
new_metadata = [sha(sections(data)['.note']) for data in images(new)]
record.update(old_core_metadata=old_metadata,new_core_metadata=new_metadata,
              core_metadata_identical=old_metadata==new_metadata)
Path('build/kernel-identity.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record, indent=2))
assert old_code and old_code == new_code, 'Core instruction bytes changed: cannot reuse old validation'
assert old_metadata == new_metadata, 'Kernel resource/argument metadata changed'
assert old_all and old_all == new_all, 'Baseline/diagnostic instructions changed'
