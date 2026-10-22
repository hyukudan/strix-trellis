"""Audit actual new/old ELF function bytes and the repaired packed probe, CPU only."""
from pathlib import Path
import hashlib
import json
import re
import struct

def sha(data):
    return hashlib.sha256(data).hexdigest()

def functions(path):
    data=path.read_bytes()
    assert data[:6]==b'\x7fELF\x02\x01'
    offset=struct.unpack_from('<Q',data,40)[0]
    size,count,_=struct.unpack_from('<HHH',data,58)
    headers=[struct.unpack_from('<IIQQQQIIQQ',data,offset+i*size) for i in range(count)]
    result={}
    for h in headers:
        if h[1]!=2:
            continue
        strings_header=headers[h[6]]
        strings=data[strings_header[4]:strings_header[4]+strings_header[5]]
        for pos in range(h[4],h[4]+h[5],h[9]):
            name,info,other,index,value,length=struct.unpack_from('<IBBHQQ',data,pos)
            if info&15!=2 or not length or not 0<index<count:
                continue
            name=strings[name:].split(b'\0',1)[0].decode()
            section=headers[index]
            start=section[4]+value-section[3]
            result[name]=data[start:start+length]
    return result

old=Path.home()/'gpu-site/scratch-trellis/wmma-decoder-core'
assert sha((old/'build/wmma_decoder_probe').read_bytes())=='eb4cd9732fb4e0f1fbb85333ab7c0dc30e6b146a84af04faec4f1e2c413997c5'
oracle_manifest=json.loads(Path('build/old-oracle-manifest.json').read_text())
for path,digest in oracle_manifest.items():
    assert sha(Path(path).read_bytes())==digest,path
lut=Path('build/oracles/mul1.bin').read_bytes()
for state in range(65536):
    product=state*0x83dcd12d&0xffffffff
    value=((1024+sum(product.to_bytes(4,'little')))*1774-1329*2048)/2**18
    assert lut[state*2:state*2+2]==struct.pack('<e',value)
small=[]
for bits in (5,6):
    for inputs,outputs in ((128,256),(256,128),(384,256)):
        path=Path(f'build/oracles/core-k{bits}-i{inputs}-n{outputs}.stgpu')
        data=path.read_bytes()
        assert data[:8]==b'STGPU001'
        version,count=struct.unpack_from('<II',data,8)
        assert version==1
        cursor=16;parts={}
        for _ in range(count):
            tag,length=struct.unpack_from('<8sQ',data,cursor);cursor+=16
            parts[tag.rstrip(b'\0').decode()]=data[cursor:cursor+length];cursor+=length
        assert cursor==len(data)
        i,n,b,cb,marker,rows,bias,seed=struct.unpack('<8I',parts['meta'])
        assert (i,n,b,cb,marker,rows)==(inputs,outputs,bits,2,0x83dcd12d,512)
        assert len(parts['a'])==512*i*2 and len(parts['q'])==i*n*2
        assert len(parts['trellis'])==i*n*b//8
        assert path.with_suffix('.stgpu.partial').stat().st_size==2*56*512*n*2*8
        assert path.with_suffix('.stgpu.impulse').stat().st_size==512*i*2
        small.append(dict(path=str(path),inputs=i,outputs=n,bits=b,marker=hex(marker),rows=rows))
source_identity={}
for name in ('wmma_decoder.hip','wmma_decoder.hpp','gemm.hip','gemm.hpp','backend.hpp','transforms.hpp','packed_decoder.hpp','trellis_optimized.hpp','trellis_device.hpp'):
    a=Path('src',name).read_bytes();b=(old/'src'/name).read_bytes()
    assert a==b,name
    source_identity[name]=sha(a)
old_functions=functions(old/'build/linked-0.elf')
new_functions=functions(Path('build/linked-0.elf'))
records=[]
for name,a in old_functions.items():
    if 'decoder_core' not in name and 'staging_dump' not in name:
        continue
    b=new_functions[name]
    assert a==b,name
    records.append(dict(symbol=name,bytes=len(a),old_sha256=sha(a),new_sha256=sha(b),identical=True))
assert sum('decoder_core' in r['symbol'] for r in records)==18
assert sum('staging_dump' in r['symbol'] for r in records)==18
isas='\n'.join(p.read_text() for p in Path('build').glob('linked-*.isa'))
probe=[(name,body) for name,body in re.findall(r'<([^>]+)>:\n(.*?)(?=\n[0-9a-f]+ <|\Z)',isas,re.S) if 'hash_half_gate' in name]
assert len(probe)==1
name,body=probe[0]
assert body.count('v_pk_fma_f16')==2
assert len(re.findall(r'v_lshrrev_b32[^\n]*16',body))>=2
Path('build/probe-isa.txt').write_text(name+'\n'+body)
source=Path('bench/wmma_decoder_probe.hip').read_text()
assert not re.search(r'__builtin_bit_cast\(uint16_t,\s*[ab]\[',source)
for token in ('out[index]=a&65535u','out[index+1]=a>>16','out[index+2]=b&65535u',
              'out[index+3]=b>>16','out[index+6]=a;out[index+7]=b'):
    assert token in source
abi=Path('src/backend.hpp').read_text().split('struct DeviceWeights {')[1].split('};')[0]
assert abi.count(';')==8 and 'version' not in abi
proof=dict(status='PASS_CPU_ISA_ONLY',old_commit='c05931f',frozen_sources=source_identity,
           immutable_V11_sha256=sha(Path('control/libstrix_trellis.a').read_bytes()),
           device_weights_fields=8,functions=records,probe_symbol=name,
           probe_isa_sha256=sha(body.encode()),
           probe_contract='Whole __half2 to uint32; low mask and high integer shift; both complete words stored',
           physical_qualification=False,old_oracle_hashes_verified=len(oracle_manifest),
           independent_half_LUT_states=65536,small_fixture_metadata=small,
           core_image_byte_identical=(old/'build/linked-0.elf').read_bytes()==Path('build/linked-0.elf').read_bytes())
proof['immutable_control_image_byte_identical']=(old/'build/linked-2.elf').read_bytes()==Path('build/linked-2.elf').read_bytes()
assert proof['core_image_byte_identical'] and proof['immutable_control_image_byte_identical']
assert proof['immutable_V11_sha256']=='201d596e469c3d81a91aaeffa4588801fad5ed2d3b8d8cc1279458b78f6b0597'
Path('build/packed-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print('PASS CPU/ISA only: 18 cores and 18 staging functions byte-identical; packed extraction audited')
