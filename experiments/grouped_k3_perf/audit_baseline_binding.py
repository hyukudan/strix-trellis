"""Read-only x86 call-site proof for the actual archive launch_linear binding."""
from pathlib import Path
import hashlib,json,re,struct
raw=Path('bin/perf_native').read_bytes();offset,=struct.unpack_from('<Q',raw,40);size,count,names=struct.unpack_from('<HHH',raw,58)
rows=[struct.unpack_from('<IIQQQQIIQQ',raw,offset+j*size) for j in range(count)]
name_section=rows[names];strings=raw[name_section[4]:name_section[4]+name_section[5]]
text=next(r for r in rows if strings[r[0]:].split(b'\0',1)[0]==b'.text')
disassembly=Path('build/host-disassembly.txt').read_text();calls=[]
for line in disassembly.splitlines():
    if not re.search(r'callq?\s+0x[0-9a-f]+ <strix_trellis::launch_linear\(',line):continue
    match=re.match(r'\s*([a-f0-9]+): ((?:[a-f0-9]{2} )+)\s*callq?\s+0x([a-f0-9]+)',line);assert match
    address=int(match[1],16);code=bytes.fromhex(match[2]);target=int(match[3],16)
    at=text[4]+address-text[3];assert raw[at:at+len(code)]==code and code[0]==0xe8 and len(code)==5
    assert address+5+struct.unpack_from('<i',code,1)[0]==target
    calls.append(dict(address=address,target=target,instruction_bytes=code.hex(),disassembly=line.strip()))
assert calls and len({c['target'] for c in calls})==1
assert hashlib.sha256(Path('lib/libstrix_trellis.a').read_bytes()).hexdigest()=='0b1453a3c9c25e3f7e4f12251d310a3a9f2c3b53c76d78c314e9624ba763e3b3'
print(json.dumps(dict(state='PASS_BUILD_ONLY',baseline='actual V13 archive launch_linear function calls',calls=calls),indent=2))
