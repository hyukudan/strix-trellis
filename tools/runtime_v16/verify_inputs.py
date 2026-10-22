"""Revalidate every compact R8 CPU reference against its actual frozen source."""
from pathlib import Path
import hashlib
import json
import resource
import struct
import sys

root=Path.home()/'gpu-site/scratch-trellis';own=root/'runtime-v16';qa=root/'runtime-v16-qa/direct'
assert Path.cwd()==own and not (own/'frozen.sha256').exists()
resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30))
sys.path.insert(0,str(own/'python'))
import numpy as np
from strix_trellis.codec import dequantize_rotated
from strix_trellis.linear import input_stages_fp32,hadamard

def sha(raw):return hashlib.sha256(raw).hexdigest()
def parse(raw):
    assert len(raw)>=16
    magic,version,count=struct.unpack_from('<8sII',raw);assert magic==b'STGPU001' and version==1 and count<=64
    parts={};pos=16
    for _ in range(count):
        assert pos+16<=len(raw)
        tag,n=struct.unpack_from('<8sQ',raw,pos);pos+=16;tag=tag.rstrip(b'\0').decode()
        assert tag not in parts and pos+n<=len(raw)
        parts[tag]=raw[pos:pos+n];pos+=n
    assert pos==len(raw)
    return parts

records=[]
for index,name in enumerate(('layers.0.attn.wo_b.stgpu','layers.0.ffn.shared_experts.w2.stgpu','layers.0.attn.wq_b.stgpu')):
    path=root/'gpu-dense-prefill-tests'/name;raw=path.read_bytes();original_sha=sha(raw);parts=parse(raw)
    compact=parse((qa/'inputs'/f'{index}.stgpu').read_bytes())
    i,n,bits,cb,marker,rows,bias,seed=struct.unpack('<8I',parts['meta'])
    assert (i,n,bits,cb)==((8192,5120,5,2),(2304,5120,5,2),(1280,32768,5,2))[index] and rows>=8
    for key,width in [('x',i),('old',n),('ref',n)]:assert compact[key]==parts[key][:8*width*4],key
    for key in ('trellis','suh','svh'):assert compact[key]==parts[key]
    x=np.frombuffer(parts['x'],dtype='<f4').reshape(rows,i)[:8]
    a=input_stages_fp32(x,np.frombuffer(parts['suh'],dtype='<f2'))['core_input_fp16']
    assert a.astype('<f2').tobytes()==parts['a'][:8*i*2]
    packed=np.frombuffer(parts['trellis'],dtype='<u2').reshape(i//16,n//16,bits*16)
    q=np.frombuffer(parts['q'],dtype='<f2').reshape(i,n)
    core=np.empty((8,n),np.float64);p=np.empty((4,8,n),np.float64);ph=np.empty_like(p)
    # Existing fallback partitions K16; Direct partitions full H128 groups.
    for col in range(0,n,128):
        decoded=dequantize_rotated(packed[:,col//16:(col+128)//16],bits,'mul1')
        assert np.array_equal(decoded.view('u2'),q[:,col:col+128].view('u2'))
        d=decoded.astype(np.float64);af=a.astype(np.float64)
        core[:,col:col+128]=af@d
        for group in range(4):
            begin=(i//16)*group//4*16;end=(i//16)*(group+1)//4*16
            p[group,:,col:col+128]=af[:,begin:end]@d[begin:end]
            hb=(i//128)*group//4*128;he=(i//128)*(group+1)//4*128
            ph[group,:,col:col+128]=af[:,hb:he]@d[hb:he]
    oracle=hadamard(core)*np.frombuffer(parts['svh'],dtype='<f2').astype(np.float64)
    assert oracle.astype('<f8').tobytes()==(qa/'inputs'/f'{index}.f64').read_bytes()
    reference=np.frombuffer(parts['ref'],dtype='<f4').reshape(rows,n)[:8].astype(np.float64)
    if bias:reference-=np.frombuffer(parts['bias'],dtype='<f4')
    numeric=[]
    for row in range(8):
        delta=reference[row]-oracle[row];ma=float(np.max(np.abs(delta)));nr=float(np.linalg.norm(delta)/max(np.linalg.norm(oracle[row]),1e-30))
        assert np.isfinite(reference[row]).all() and ma<=5e-4*max(1,float(np.max(np.abs(oracle[row])))) and nr<=3e-5
        numeric.append(dict(row=row,values=n,maxabs=ma,NRMS=nr))
    p.astype('<f8').tofile(qa/'inputs'/f'{index}.p-f64')
    ph.astype('<f8').tofile(qa/'inputs'/f'{index}.ph-f64')
    records.append(dict(source=str(path),source_sha256=original_sha,metadata=list(struct.unpack('<8I',parts['meta'])),source_parts={k:sha(v) for k,v in parts.items()},X_first_eight_F32_rows_exact=True,A_RN32_RN16_exact=True,full_Q_decoded_values=i*n,full_Y_FP64_values=8*n,full_P_FP64_values=4*8*n,P_FP64_sha256=sha((qa/'inputs'/f'{index}.p-f64').read_bytes()),Y_FP64_exact_inherited=True,CPU_rows=numeric))
    del raw,parts,compact,q,x,a,af,decoded,d,packed,core,p,ph,oracle,reference
result=dict(state='CPU_ALL_THREE_FULL_SOURCE_REFERENCES_PASS',records=records,host_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
assert result['host_peak_bytes']<2<<30
(own/'build/reference-validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
