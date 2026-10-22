"""Optional read-only SSH CPU inspector of two fixed paths from the linkmap.

Run with `ssh -o BatchMode=yes node2 python3 - < read_sdk.py` (stdin).
Does not execute a library, create a file, compile, or query a GPU.
"""
from pathlib import Path
import hashlib
import json
import struct

names = {'_Znwm', '_Znam', '_ZdlPv', '_ZdaPv', '_ZdlPvm', '_ZdaPvm'}
result = []
for path in (Path('/opt/rocm/core-10.1/lib/libamdhip64.so'),
             Path('/usr/lib/gcc/x86_64-linux-gnu/16/libstdc++.so')):
    raw = path.read_bytes()
    assert raw[:6] == b'\x7fELF\x02\x01'
    off, = struct.unpack_from('<Q', raw, 40)
    stride, count, si = struct.unpack_from('<HHH', raw, 58)
    rows = [struct.unpack_from('<IIQQQQIIQQ', raw, off+j*stride) for j in range(count)]
    s = rows[si]
    strings = raw[s[4]:s[4]+s[5]]
    sections = {strings[r[0]:].split(b'\0', 1)[0].decode(): r for r in rows}
    s = sections['.dynsym']
    strings_row = rows[s[6]]
    strings = raw[strings_row[4]:strings_row[4]+strings_row[5]]
    symbols, heap = [], []
    for p in range(s[4], s[4]+s[5], s[9]):
        n, info, other, index, value, size = struct.unpack_from('<IBBHQQ', raw, p)
        name = strings[n:].split(b'\0', 1)[0].decode()
        symbols.append(name)
        if name in names:
            heap.append(dict(name=name, binding=info >> 4, visibility=other & 3,
                             section=index, address=hex(value), size=size))
    relocations = []
    for key, s in sections.items():
        if s[1] != 4 or rows[s[6]] != sections['.dynsym']:
            continue
        for p in range(s[4], s[4]+s[5], s[9]):
            address, info, addend = struct.unpack_from('<QQq', raw, p)
            name = symbols[info >> 32]
            if name in names:
                relocations.append(dict(section=key, address=hex(address), type=info & 0xffffffff, symbol=name))
    s = sections['.dynamic']
    needed, flags = [], []
    for p in range(s[4], s[4]+s[5], s[9]):
        tag, value = struct.unpack_from('<QQ', raw, p)
        if tag == 1:
            needed.append(strings[value:].split(b'\0', 1)[0].decode())
        if tag in (16, 30, 0x6ffffffb):
            flags.append(dict(tag=hex(tag), value=hex(value)))
    result.append(dict(path=str(path), resolved_path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                       heap_symbols=heap, heap_relocations=relocations, needed=needed, flags=flags))
assert result[0]['sha256'] == '5ad5a94ea0c3a4152b83925330c6c9a59739d4239b3c5e048aa89e1dad609a0a'
print(json.dumps(dict(scope='READ_ONLY_SSH_CPU_ELF_INSPECTION', libraries=result), indent=2))
