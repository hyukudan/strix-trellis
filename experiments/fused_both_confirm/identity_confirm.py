"""Fail closed on core-image or any of six instruction-byte differences."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

own=Path.cwd()
old=own.parent/'fused-both-core'
image='probe.0.hip-amdgcn-amd-amdhsa--gfx1151'
before=(old/'bin'/image).read_bytes()
after=(own/'bin'/image.replace('probe','probe_confirm')).read_bytes()
assert hashlib.sha256(before).hexdigest()=='a2616be9221bd9db77cc626b86d9437e499bfbb82f9138d36024de11744ef636'
assert before==after, 'Whole GPU core image differs from frozen b79f058'

def instructions(path):
    result=subprocess.check_output(['/opt/rocm/core-10.1/llvm/bin/llvm-objdump','-d','--mcpu=gfx1151',str(path)],text=True)
    parts=re.split(r'\n[0-9a-f]+ <([^>]+)>:\n',result)
    bodies={}
    for name,body in zip(parts[1::2],parts[2::2]):
        if 'fused_both' in name and 'bothILi' in name:
            words=re.findall(r'// [0-9A-F]+: ([0-9A-F ]+)',body)
            data=b''.join(int(w,16).to_bytes(4,'little') for line in words for w in line.split())
            assert data
            bodies[name]=data
    assert len(bodies)==6
    return bodies
a=instructions(old/'bin'/image)
b=instructions(own/'bin'/image.replace('probe','probe_confirm'))
assert a==b
print(json.dumps(dict(whole_core_bytes=len(before),whole_core_sha256=hashlib.sha256(before).hexdigest(),
                     old_commit='b79f058',six_functions=[dict(name=k,bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for k,v in a.items()],
                     whole_core_identical=True,instructions_identical=True),indent=2))
