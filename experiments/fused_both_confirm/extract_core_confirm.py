"""Extract the first original core bundle verbatim for a host-only repair."""
from pathlib import Path
import hashlib
import struct
import subprocess

old=Path.home()/'gpu-site/scratch-trellis/fused-both-core/bin/probe'
assert hashlib.sha256(old.read_bytes()).hexdigest()=='e4323d12987936c7c6503d7a697b3fa92d88b13e03cbdbb5263de1cd065f8436'
subprocess.run(['/opt/rocm/core-10.1/llvm/bin/llvm-objcopy',
                '--dump-section','.hip_fatbin=build/original-executable.fatbin',str(old),
                'build/original-read-copy.elf'],check=True)
blob=Path('build/original-executable.fatbin').read_bytes()
marker=b'__CLANG_OFFLOAD_BUNDLE__'
p=blob.index(marker); q=p+len(marker)
n,=struct.unpack_from('<Q',blob,q);q+=8
ends=[]
for _ in range(n):
    offset,size,length=struct.unpack_from('<QQQ',blob,q);q+=24+length
    ends.append(offset+size)
bundle=blob[p:p+max(ends)]
assert (old.parent/'probe.0.hip-amdgcn-amd-amdhsa--gfx1151').read_bytes() in bundle
Path('build/original-core.hipfb').write_bytes(bundle)
print('Original core bundle',len(bundle),hashlib.sha256(bundle).hexdigest())
