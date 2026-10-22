"""CPU-only check of every actual V13 attribute relocation and linked target."""
from pathlib import Path
import hashlib
import json
import re
import struct


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def functions(text):
    rows=list(re.finditer(r'^([0-9a-f]+) <(.+)>:$',text,re.M))
    return {m[2]:text[m.start():rows[j+1].start() if j+1<len(rows) else len(text)]
            for j,m in enumerate(rows)}


binary=Path('bin/math_native_qa').read_bytes()
shoff,=struct.unpack_from('<Q',binary,40)
shsize,shcount,shnames=struct.unpack_from('<HHH',binary,58)
rows=[struct.unpack_from('<IIQQQQIIQQ',binary,shoff+j*shsize) for j in range(shcount)]
r=rows[shnames];strings=binary[r[4]:r[4]+r[5]]
sections={strings[r[0]:].split(b'\0',1)[0].decode():r for r in rows}
text=sections['.text']

def verify_call(line):
    m=re.match(r'\s*([a-f0-9]+): ((?:[a-f0-9]{2} )+)\s*callq?\s+0x([a-f0-9]+)',line)
    assert m,line
    address=int(m[1],16);code=bytes.fromhex(m[2]);target=int(m[3],16)
    offset=text[4]+address-text[3]
    assert binary[offset:offset+len(code)]==code
    assert code[0]==0xe8 and len(code)==5
    assert address+5+struct.unpack_from('<i',code,1)[0]==target
    return address,target

archive=Path('build/archive-relocations.txt').read_text()
relocations=[line for line in archive.splitlines() if re.search(r'\bhipFuncGetAttributes(?:-|\s|$)',line)]
assert len(relocations)==4,relocations
assert all('R_X86_64_PLT32' in line for line in relocations)
host=Path('build/host-disassembly.txt').read_text()
funcs=functions(host)
query,=[body for name,body in funcs.items() if name.startswith('strix_trellis::query_fused_output_resources(')]
linked_relocations=[line for line in query.splitlines() if 'R_X86_64_PLT32' in line and 'FuncGetAttributes' in line]
assert len(linked_relocations)==len(relocations)
assert all('__wrap_hipFuncGetAttributes' in line for line in linked_relocations)
calls=[line for line in query.splitlines() if re.search(r'\bcallq?\s',line) and 'FuncGetAttributes' in line]
assert len(calls)==4 and all('<__wrap_hipFuncGetAttributes>' in line for line in calls),calls
verified_calls=[verify_call(line) for line in calls]
assert len({target for _,target in verified_calls})==1
# Every frozen archive relocation has the same function-relative site in the linked control.
query_address=int(query.split()[0],16)
assert [at+1-query_address for at,_ in verified_calls]==[int(line.split()[0],16) for line in relocations]
hook=funcs['__wrap_hipFuncGetAttributes']
forward=[line for line in hook.splitlines() if re.search(r'\bcallq?\s',line) and 'hipFuncGetAttributes' in line]
assert len(forward)==1 and '<hipFuncGetAttributes@plt>' in forward[0],forward
verify_call(forward[0])
symbols=Path('build/host-symbols-relocations.txt').read_text()
assert re.search(r'\bUND\s+hipFuncGetAttributes\b',symbols)
assert re.search(r'\bGLOBAL\s+DEFAULT\s+\d+\s+hipFuncGetAttributes\b',Path('build/real-HIP-exports.txt').read_text())
assert 'libamdhip64.so' in symbols
assert '__wrap_hipFuncGetAttributes' in Path('build/link.map').read_text()
print(json.dumps(dict(status='PASS_BUILD_ONLY',scope='user/static-library link boundary; not every driver internal query',
    archive_attribute_relocations=relocations,actual_linked_V13_relocations=linked_relocations,
    actual_linked_V13_calls=calls,real_HIP_forward_call=forward,
    original_HIP_symbol_dynamically_undefined=True,
    archive_sha256=sha(Path('lib/libstrix_trellis.a').read_bytes()),
    binary_sha256=sha(Path('bin/math_native_qa').read_bytes()),
    linkmap_sha256=sha(Path('build/link.map').read_bytes())),indent=2))
