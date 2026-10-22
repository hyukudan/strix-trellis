"""CPU-only bounded fixtures/oracles. No device queries, downloads or shards."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.nice(10)
import hashlib
import json
from pathlib import Path
import struct
import sys
import numpy as np
from make_gpu_fixtures import write_fixture
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python'))
from strix_trellis.codec import decode_states, dequantize_rotated
from strix_trellis.linear import input_stages_fp32

out = Path('build/oracles')
out.mkdir(parents=True, exist_ok=True)
records = []
for bits in (5, 6):
    for inputs, outputs in ((128, 256), (256, 128), (384, 256)):
        seed = 20261021 + bits + inputs
        rng = np.random.default_rng(seed)
        t = rng.integers(0, 65536, (inputs//16, outputs//16, 16*bits), dtype=np.uint16)
        su = rng.choice(np.array([-1, -.5, 0, .125, 1], dtype='<f2'), inputs)
        su[:128] = 1
        sv = rng.choice(np.array([-1, -.5, 0, .125, 1], dtype='<f2'), outputs)
        x = rng.normal(0, .125, (512, inputs)).astype('<f4')
        x[:8] = 0
        x[0, :128] = 4e-6
        x[1, :128] = np.where(np.arange(128)&1, -4e-6, 4e-6)
        x[2, :128] = np.linspace(-3e-5, 3e-5, 128)
        x[3, ::2] = 2**-24
        x[4, :128] = 4000
        x[5, :128] = np.where(np.arange(128)&1, -4000, 4000)
        x[6, :128] = np.linspace(-6000, 6000, 128)
        x[7, ::2] = -0.
        # RN16 ties, zero signs and range in subsequent rows.
        x[8, :8] = [1+2**-11, 1+3*2**-11, -1-2**-11, -0., 0., 2**-24, -2**-24, 65504/16]
        bias = rng.normal(0, .02, outputs).astype('<f4') if bits==5 else None
        path = out / f'core-k{bits}-i{inputs}-n{outputs}.stgpu'
        records.append(write_fixture(path,t,su,sv,bias,'mul1',seed,512,x=x))
        a = input_stages_fp32(x,su)['core_input_fp16']
        q = dequantize_rotated(t,bits,'mul1').astype(np.float64)
        if inputs == 128:
            for n in (16,32,48):
                short = t.reshape(-1,16*bits)[:(inputs//16)*(n//16)].reshape(inputs//16,n//16,16*bits)
                dq = dequantize_rotated(short,bits,'mul1')
                dq[:16].astype('<f2').tofile(str(path)+'.diagq'+str(n))
                with open(str(path)+'.diag'+str(n),'wb') as target:
                    for split in range(3):
                        begin=(inputs//16)*split//3*16
                        end=(inputs//16)*(split+1)//3*16
                        (a[:17,begin:end].astype(np.float64) @ dq[begin:end].astype(np.float64)).astype('<f8').tofile(target)
        impulse = np.zeros_like(a)
        for row in range(512):
            impulse[row,(row*17)%inputs] = (-1 if row&1 else 1)*(.5+(row%4)*.25)
        impulse.astype('<f2').tofile(str(path)+'.impulse')
        # Full independent FP64 partials and sumabs, same split endpoints.
        with open(str(path)+'.partial','wb') as f:
            for source in (a,impulse):
                af = source.astype(np.float64)
                for splits in (1,3,4,16,32):
                    for split in range(splits):
                        begin=(inputs//16)*split//splits*16
                        end=(inputs//16)*(split+1)//splits*16
                        for value in (af[:,begin:end] @ q[begin:end], np.abs(af[:,begin:end]) @ np.abs(q[begin:end])):
                            value.astype('<f8').tofile(f)
lut = decode_states(np.arange(65536,dtype=np.uint16), 'mul1').astype('<f2')
lut.tofile(out/'mul1.bin')
native = np.zeros((2,16,16),dtype='<f8')
for r in range(16):
    for n in range(16):
        for k in range(16):
            product=(((r*16+k)*7%13)-6)*(((k*16+n)*11%17)-8)
            native[0,r,n]+=product
            native[1,r,n]+=abs(product)
native.tofile(out/'native.bin')
for bits in (5,6):
    count=bits*256
    rings=np.zeros((count+2,bits*8),dtype='<u4')
    for bit in range(count):
        rings[bit+1,bit//32]=1 << (31-bit%32)
    rings[-1]=0xffffffff
    rings.tofile(out/f'basis{bits}.bin')
    states=np.empty((len(rings),256),dtype='<u2')
    for tile,words in enumerate(rings):
        for state in range(256):
            start=((state+1)*bits-16)%count
            value=0
            for b in range(16):
                pos=(start+b)%count
                value=(value<<1)|((int(words[pos//32])>>(31-pos%32))&1)
            states[tile,state]=value
    states.tofile(out/f'states{bits}.bin')
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
(Path('build')/'oracle-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(records,indent=2))
