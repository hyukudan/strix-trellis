from pathlib import Path
import struct,hashlib,subprocess

def images(blob):
    marker=b'__CLANG_OFFLOAD_BUNDLE__';pos=0;out=[]
    while (p:=blob.find(marker,pos))>=0:
        q=p+len(marker);n,=struct.unpack_from('<Q',blob,q);q+=8;records=[]
        for _ in range(n):
            offset,size,ln=struct.unpack_from('<QQQ',blob,q);q+=24
            target=blob[q:q+ln].decode();q+=ln;records.append((offset,size,target))
        for offset,size,target in records:out.append((target,blob[p+offset:p+offset+size]))
        pos=p+max([q-p]+[a+b for a,b,_ in records])
    return out
for group,path in [('old','original/exe.fatbin'),('new','build/exe.fatbin')]:
    for i,(target,blob) in enumerate(images(Path(path).read_bytes())):
        print(group,i,target,len(blob),hashlib.sha256(blob).hexdigest())
        if blob[:4]==b'\x7fELF':Path(f'build/{group}-{i}.elf').write_bytes(blob)
