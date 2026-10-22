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



import subprocess
import re
import difflib
root=Path('.')
out={}
def sha(b): return hashlib.sha256(b).hexdigest()
def body(s,needle):
    begin=s.index(needle);start=s.index('{',begin);depth=1;end=start+1
    while depth:
        depth+=(s[end]=='{')-(s[end]=='}');end+=1
    return s[begin:end]
def remove_q(s):
    needle='if (tile_n < tiles_n) {'
    begin=s.index(needle);start=s.index('{',begin);depth=1;end=start+1
    while depth:
        depth+=(s[end]=='{')-(s[end]=='}');end+=1
    return s[:begin]+'APPROVED_Q_BLOCK'+s[end:]
base=Path('src/gemm.hip').read_text()
new=Path('src/wmma_decoder.hip').read_text()
old_body=body(base,'template<int Bits, int Codebook')
new_body=body(new,'template<int Bits, int Mode,')
a=remove_q(old_body).replace('int Codebook,','int Mode,').replace('void wmma_core(','void decoder_core(')
b=remove_q(new_body)
assert a==b, ''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True)))
out['non_Q_kernel_exact_text_identity']=True
# Direct text diff against the frozen runtime V11 sources, not a manifest inference.
out['runtime_v11_source_comparison']={}
for name in ('gemm.hip','trellis_device.hpp','trellis_optimized.hpp','launch_policy.hpp','backend.hip'):
    local=Path('src',name).read_bytes();frozen=Path('control/src',name).read_bytes()
    ln=local.replace(b'\r\n',b'\n');fn=frozen.replace(b'\r\n',b'\n')
    record=dict(actual_raw=sha(local),actual_lf=sha(ln),V11_raw=sha(frozen),V11_lf=sha(fn),raw_identical=local==frozen,LF_identical=ln==fn)
    diff=''.join(difflib.unified_diff(fn.decode().splitlines(True),ln.decode().splitlines(True),fromfile='V11/'+name,tofile='local/'+name))
    Path('build',name+'.diff').write_text(diff)
    assert not diff, name+' semantic text discrepancy'
    out['runtime_v11_source_comparison'][name]=record
lib=Path('control/libstrix_trellis.a').read_bytes()
assert sha(lib)=='201d596e469c3d81a91aaeffa4588801fad5ed2d3b8d8cc1279458b78f6b0597'
out['immutable_V11_archive_sha256']=sha(lib)
binary=Path('build/wmma_decoder_probe').read_bytes()
out['linked_executable_sha256']=sha(binary)
linked=images(sections(binary)['.hip_fatbin'])
out['linked_images']=[]
for i,data in enumerate(linked):
    path=Path(f'build/linked-{i}.elf');path.write_bytes(data)
    dump=subprocess.check_output(['/opt/rocm/core-10.1/llvm/bin/llvm-objdump','-d','--mcpu=gfx1151',str(path)],text=True)
    Path(f'build/linked-{i}.isa').write_text(dump)
    notes=subprocess.check_output(['/opt/rocm/core-10.1/llvm/bin/llvm-readobj','--notes',str(path)],text=True)
    Path(f'build/linked-{i}.notes').write_text(notes)
    out['linked_images'].append(dict(sha256=sha(data),bytes=len(data),isa_sha256=sha(dump.encode()),notes_sha256=sha(notes.encode())))
Path('build/source-linked-proof.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
