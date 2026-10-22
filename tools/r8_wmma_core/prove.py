"""CPU-only extended-domain and linked FUNC byte-identity gates."""
from pathlib import Path
import hashlib
import json
import re
import struct
import subprocess

root=Path(__file__).resolve().parents[2]
old=Path.home()/'gpu-site/scratch-trellis/wmma-packed-probe'
def sha(b): return hashlib.sha256(b).hexdigest()
def sections(data):
    assert data[:6]==b'\x7fELF\x02\x01'
    off=struct.unpack_from('<Q',data,40)[0]
    size,count,names=struct.unpack_from('<HHH',data,58)
    headers=[struct.unpack_from('<IIQQQQIIQQ',data,off+j*size) for j in range(count)]
    s=headers[names];strings=data[s[4]:s[4]+s[5]]
    return headers,{strings[h[0]:].split(b'\0',1)[0].decode():data[h[4]:h[4]+h[5]] for h in headers}
def funcs(data):
    headers,_=sections(data);out={}
    for h in headers:
        if h[1]!=2: continue
        s=headers[h[6]];strings=data[s[4]:s[4]+s[5]]
        for at in range(h[4],h[4]+h[5],h[9]):
            name,info,other,index,value,length=struct.unpack_from('<IBBHQQ',data,at)
            if info&15==2 and length and 0<index<len(headers):
                s=headers[index];start=s[4]+value-s[3]
                out[strings[name:].split(b'\0',1)[0].decode()]=data[start:start+length]
    return out
def images(data):
    _,s=sections(data);data=s['.hip_fatbin'];magic=b'__CLANG_OFFLOAD_BUNDLE__';cursor=0;out=[]
    while (start:=data.find(magic,cursor))>=0:
        pos=start+len(magic);count=struct.unpack_from('<Q',data,pos)[0];pos+=8;end=pos
        for _ in range(count):
            off,size,length=struct.unpack_from('<QQQ',data,pos);pos+=24;target=data[pos:pos+length];pos+=length;end=max(end,start+off+size)
            if target.startswith(b'hip-'):out.append(data[start+off:start+off+size])
        cursor=end
    assert out;return out
def body(s,needle):
    begin=s.index(needle);at=s.index('{',begin);depth=1;end=at+1
    while depth: depth+=(s[end]=='{')-(s[end]=='}');end+=1
    return s[begin:end]
current=(root/'src/wmma_decoder.hip').read_text();frozen=(old/'src/wmma_decoder.hip').read_text()
submit=body((root/'tools/r8_wmma_core/driver.hip').read_text(),'void submit(bool full,')
for forbidden in ('require(', 'hipMemcpy', 'hipMalloc', 'hipStreamSynchronize', 'hipEvent', 'hipGetDevice', 'hipPointerGetAttributes', 'hipMemGetAddressRange', 'std::vector', 'new '):
    assert forbidden not in submit, 'Forbidden timed submit operation: '+forbidden
for needle in ('template<int Bits, int Mode,','template<int Bits,int Mode,int TileM'):
    assert body(current,needle)==body(frozen,needle),'GPU source body changed'
assert (root/'src/wmma_decoder.hpp').read_bytes()==(old/'src/wmma_decoder.hpp').read_bytes()
assert (root/'src/backend.hpp').read_text().split('struct DeviceWeights {')[1].split('};')[0].count(';')==9
assert (root/'src/backend.hpp').read_bytes()==(root/'control/src/backend.hpp').read_bytes()
matched_sources={}
for p in (root/'control/src').iterdir():
    q=root/'src'/p.name
    if p.name in ('wmma_decoder.hip','wmma_decoder.hpp') or not q.exists():continue
    assert p.read_text()==q.read_text(),p.name+' actual V13 source differs'
    matched_sources[p.name]=dict(actual=sha(q.read_bytes()),frozen=sha(p.read_bytes()),LF_identical=True)
assert sha((root/'control/libstrix_trellis.a').read_bytes())=='0b1453a3c9c25e3f7e4f12251d310a3a9f2c3b53c76d78c314e9624ba763e3b3'
for name in ('trellis_device.hpp','trellis_optimized.hpp','packed_decoder.hpp','hadamard.hpp','transforms.hpp'):
    assert (root/'src'/name).read_text()==(old/'src'/name).read_text(),name
oldfunc={}
for p in (old/'build').glob('linked-*.elf'):
    oldfunc.update(funcs(p.read_bytes()))
reused={};other_reused={};new_funcs={}
for index,data in enumerate(images((root/'build/r8-driver').read_bytes())):
    p=root/f'build/linked-{index}.elf';p.write_bytes(data)
    dis=subprocess.check_output(['/opt/rocm/core-10.1/llvm/bin/llvm-objdump','-d','--mcpu=gfx1151',str(p)],text=True)
    p.with_suffix('.isa').write_text(dis)
    notes=subprocess.check_output(['/opt/rocm/core-10.1/llvm/bin/llvm-readobj','--notes',str(p)],text=True);p.with_suffix('.notes').write_text(notes)
    for name,code in funcs(data).items():
        new_funcs[name]=code
        if 'decoder_core' in name or 'staging_dump' in name:
            assert code==oldfunc[name],name
            reused[name]=dict(bytes=len(code),sha256=sha(code),identical=True)
        elif name in oldfunc:
            assert code==oldfunc[name],name+' reused function instructions differ'
            other_reused[name]=dict(bytes=len(code),sha256=sha(code),identical=True)
assert sum('decoder_core' in n for n in reused)==18
assert sum('staging_dump' in n for n in reused)==18
for new,previous in [('r8_input_transform','prefill_test_input_transform'),('r8_output_transform','prefill_test_output_transform')]:
    a=[(n,c) for n,c in new_funcs.items() if new in n]
    b=[(n,c) for n,c in oldfunc.items() if previous in n]
    assert len(a)==len(b)==1 and a[0][1]==b[0][1],new
    other_reused[a[0][0]]=dict(frozen_symbol=b[0][0],bytes=len(a[0][1]),sha256=sha(a[0][1]),identical=True)
# Explicit maximum addresses for the four exact R8/S4 shapes; no Q allocation.
bounds=[]
for i,n,b in [(1280,32768,5),(8192,5120,5),(5120,129280,6),(5120,129280,5)]:
    tk,tn=i//16,n//16;packed=i*n*b//8;a=8*i*2;p=4*8*n*4
    assert n%128==i%128==0 and tn%4==0
    last_word=((tk-1)*tn+(tn-1))*(b*8)+(b*8-1)
    assert (last_word+1)*4==packed
    last_a=7*i+(tk-1)*16+15;assert (last_a+1)*2==a
    last_p=(3*8+7)*n+(tn-1)*16+15;assert (last_p+1)*4==p
    # int locals and products used before explicit size_t promotions.
    for value in (tn,tk,(tn//4-1)*4+3,(tn-1)*16+15,tk*4,16*16,4*256):assert 0<=value<2**31
    for s in range(4):assert 0<=tk*s//4<=tk*(s+1)//4<=tk
    # M16 plain LDS: 256 halves of A, 1024 halves Q, fixed independent of N.
    for lane in range(32):
        for j in range(8):
            row=2*j+lane//16
            if row<8:assert 0<=row<8 and 0<=lane%16<16
    bounds.append(dict(I=i,N=n,bits=b,rows=8,total_rows=8,S=4,layout='M16/A16/plain',
        packed_bytes=packed,A_bytes=a,P_bytes=p,Y_bytes=8*n*4,grid=[tn//4,1,4],
        last_packed_word=last_word,last_A_half=last_a,last_P_float=last_p,LDS_bytes=2560,
        int_local_bounds=True,size_t_global_bounds=True,all_waves_active=True))
report=dict(status='PASS_CPU_BUILD_ISA_ONLY',physical=False,device_weights_fields=9,
    reused_functions=reused,other_reused_functions=other_reused,matching_actual_V13_sources=matched_sources,domain=bounds,uncovered_HIGH_conditions=[],
    extension='Only host N domain for I5120/N129280/R8/total_rows8/S4/M16 plain; no instruction change',
    native_envelope=16*2**-24,NRMS=3e-5,maxabs=5e-4,
    native_probes='Inherited exact source and physical 2268-case campaigns, including Rows8; not rerun here')
(root/'build/proof.json').write_text(json.dumps(report,indent=2)+'\n')
print('PASS_CPU_BUILD_ISA_ONLY 18 cores + 18 staging functions, head index/resource bounds')
