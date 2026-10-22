"""Record ROOT CPU findings, sanitizer closures, and unchanged GPU function bytes."""
from pathlib import Path
import hashlib
import json
import struct
repo=Path(__file__).resolve().parents[1];data=repo/'reports/tp-decode-private-data';raw=Path('external-data/work/tp-decode-v1-build')
def sha(b):return hashlib.sha256(b).hexdigest()
def elf(path):
    b=path.read_bytes();assert b[:6]==b'\x7fELF\x02\x01'
    offset=struct.unpack_from('<Q',b,40)[0];entsize,count,names=struct.unpack_from('<HHH',b,58)
    sections=[struct.unpack_from('<IIQQQQIIQQ',b,offset+j*entsize) for j in range(count)]
    text=sections[names];strings=b[text[4]:text[4]+text[5]]
    def word(s,p):return s[p:s.index(b'\0',p)].decode()
    section_data={word(strings,s[0]):b[s[4]:s[4]+s[5]] for s in sections if s[1]!=8}
    functions={}
    for section in sections:
        if section[1]!=2:continue
        text=sections[section[6]];namestring=b[text[4]:text[4]+text[5]]
        for p in range(section[4],section[4]+section[5],section[9]):
            name,info,other,which,value,size=struct.unpack_from('<IBBHQQ',b,p)
            if info&15!=2 or not size or which>=len(sections):continue
            target=sections[which];begin=target[4]+value-target[3]
            functions[word(namestring,name)]=dict(bytes=size,sha256=sha(b[begin:begin+size]))
    return section_data,functions
parent=Path('external-data/work/tp-private-root-cpu-review');preserve=data/'ROOT-cpu-review';preserve.mkdir(exist_ok=True)
for name in ('summary.json','run.stderr','variant-summary.json','variant.cpp'):
    p=parent/name
    if p.exists():(preserve/name).write_bytes(p.read_bytes())
records={}
for node in ('node1','node2'):
    assert 'exit=0' in (data/(node+'-old-host-STOP.log')).read_text()
    old=raw/node/'sealed-cpu';new=raw/node/'host-r2'
    a,fa=elf(old/'linked-00.elf');b,fb=elf(new/'linked-00.elf');assert fa==fb and len(fa)==32
    resources=json.loads((new/'resources.json').read_text());assert resources['old11_byte_identity']=='PASS'
    sanitized=(new/'sanitized-cpu.log').read_text();assert 'PASS private CPU' in sanitized and 'runtime error:' not in sanitized and 'ERROR: AddressSanitizer' not in sanitized
    records[node]=dict(sanitized='PASS actual cpu_test.cpp ASan+UBSan -fno-sanitize-recover=all',
        sanitized_log_sha256=sha((new/'sanitized-cpu.log').read_bytes()),
        old11_whole_ELFs='PASS unchanged qualified V16',candidate_GPU_FUN_bytes='PASS all30 kernels and2 non-launchable C++ sentinels identical to preserved sealed-cpu',functions=fb,
        whole_ELF_equal=sha((old/'linked-00.elf').read_bytes())==sha((new/'linked-00.elf').read_bytes()),
        differing_sections=[k for k in a if a[k]!=b.get(k)],
        candidate_whole_ELF_sha256=sha((new/'linked-00.elf').read_bytes()))
result=dict(state='HOST_R2_CPU_FIXED_NATIVE_OPEN',root_findings=['ROOT-P1-private-enum','ROOT-P1-private-variant'],
    fixes=['memcpy C enum object representations into ABI-sized uint32 before comparison',
           'raw invalid numerics/dtype bytes via memcpy in actual CPU/native-negative tests',
           'Variant whitelist before any enqueue; CPU and future zero-node capture negative'],
    old_freeze_preserved=True,old_payload_state='STOP for CPU enum failure; no physical run',new_scope='~/gpu-site/scratch-trellis/tp-decode-v1/host-r2',
    GPU_math_matrix_order_changed=False,nodes=records,native_capture='OPEN; no HIP binary run',
    root_note='ROOT_REVIEW_REQUIRED.json acknowledged and intentionally left untracked, owned by ROOT')
(data/'host-r2-proof.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print('PASS ROOT host findings fixed; ASan/UBSan both nodes; all30 candidate GPU FUN bytes unchanged')
print({node:r['differing_sections'] for node,r in records.items()})
