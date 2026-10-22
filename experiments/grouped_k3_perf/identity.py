"""CPU-only linked ELF identity against frozen R1, never loads HIP."""
from pathlib import Path
import hashlib,json,struct

def sha(raw):return hashlib.sha256(raw).hexdigest()

def sections(blob):
    assert blob[:6] == b'\x7fELF\x02\x01', 'expected ELF64 little endian'
    offset, = struct.unpack_from('<Q', blob, 40)
    size, count, names_index = struct.unpack_from('<HHH', blob, 58)
    rows = [struct.unpack_from('<IIQQQQIIQQ', blob, offset+i*size) for i in range(count)]
    row = rows[names_index]
    names = blob[row[4]:row[4]+row[5]]
    return {names[row[0]:].split(b'\0', 1)[0].decode(): blob[row[4]:row[4]+row[5]]
            for row in rows if row[1] != 8}


def function_bytes(elf):
    off,=struct.unpack_from('<Q',elf,40)
    size,count,_=struct.unpack_from('<HHH',elf,58)
    rows=[struct.unpack_from('<IIQQQQIIQQ',elf,off+j*size) for j in range(count)]
    result={}
    for r in rows:
        if r[1]!=2:continue
        names=rows[r[6]];strings=elf[names[4]:names[4]+names[5]]
        for pos in range(r[4],r[4]+r[5],r[9]):
            name,info,_,index,address,length=struct.unpack_from('<IBBHQQ',elf,pos)
            if info&15!=2 or not index:continue
            s=rows[index];raw=elf[s[4]+address-s[3]:s[4]+address-s[3]+length]
            key=strings[name:].split(b'\0',1)[0].decode()
            result[key]=dict(bytes=length,sha256=sha(raw))
    return result


if __name__=='__main__':
    old=Path.home()/'gpu-site/scratch-trellis/grouped-k3-math-r1'
    handoff=json.loads(Path('reports/grouped-k3-math-r1-data/handoff.json').read_text())
    candidate=handoff['nodes']['node1']
    assert sha((old/'bin/math_native_qa').read_bytes())==candidate['binary_sha256']
    assert sha(Path('lib/libstrix_trellis.a').read_bytes())==candidate['library_sha256']
    images=sorted(Path('build').glob('linked-*.elf'));assert len(images)==8
    records=[]
    for index,path in enumerate(images):
        raw=path.read_bytes();prior=(old/path).read_bytes()
        if index:assert raw==prior
        else:
            assert function_bytes(raw)==function_bytes(prior)
            assert sections(raw)['.text']==sections(prior)['.text']
            assert sections(raw)['.note']==sections(prior)['.note']
            funcs={k:v for k,v in function_bytes(raw).items() if k.startswith('_ZN13strix_trellis')}
            assert funcs==candidate['math_function_hashes']
        records.append(dict(image=path.name,sha256=sha(raw),whole_image_identical=raw==prior,
            text_sha256=sha(sections(raw)['.text']),note_sha256=sha(sections(raw)['.note']),functions=function_bytes(raw)))
    result=dict(state='PASS_BUILD_ONLY_NATIVE_QA_PASS_REQUIRED',reference_binary=candidate['binary_sha256'],
        library_sha256=candidate['library_sha256'],binary_sha256=sha(Path('bin/perf_native').read_bytes()),images=records)
    print(json.dumps(result,indent=2))
