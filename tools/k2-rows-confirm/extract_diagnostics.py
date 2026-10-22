"""Extract exactly the original diagnostic bundle, without changing a device byte."""
from pathlib import Path
import struct,hashlib
blob=Path('original/exe.fatbin').read_bytes();marker=b'__CLANG_OFFLOAD_BUNDLE__'
p=blob.index(marker);q=p+len(marker);n,=struct.unpack_from('<Q',blob,q);q+=8;ends=[]
for _ in range(n):
    off,size,ln=struct.unpack_from('<QQQ',blob,q);q+=24+ln;ends.append(off+size)
first=blob[p:p+max(ends)]
Path('build/original-diagnostics.hipfb').write_bytes(first)
print('original diagnostic bundle',len(first),hashlib.sha256(first).hexdigest())
