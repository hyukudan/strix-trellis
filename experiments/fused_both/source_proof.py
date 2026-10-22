"""CPU bridge from the normative ownership proof to the private source."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[2]
source = (root/'experiments/fused_both/kernel.hip').read_text()
required = [
    '__launch_bounds__(1024)',
    'extern __shared__ _Float16 cache[] __attribute__((aligned(16)))',
    '__shared__ float partial[4][128]',
    'Hbegin=H*g/4, Hend=H*(g+1)/4',
    'kbegin=8*Hbegin, kend=8*Hend',
    'for(int h=Hbegin+nw;h<Hend;h+=8)',
    'cache[g*stride+local]=_Float16(v[j])',
    'materialize_fp32(original[c]*__half2float(__ushort_as_half(suh[c])))',
    'v[j]=float(_Float16(scaled))',
    '__builtin_memcpy(&av[0],cache+g*stride+local,4)',
    '__builtin_memcpy(&av[1],cache+g*stride+local+8,4)',
    'local=(k-kbegin)*16+2*(lane&3)',
    'acc[p/2]=__builtin_amdgcn_fdot2(weight,av[p&1],acc[p/2],false)',
    'v+=__shfl_down(v,2,4);v+=__shfl_down(v,1,4)',
    'c=16*nw+lane/4+8*h', 'partial[g][c]=v',
    'if(wave!=0)return', 'float v[4]={}',
    'for(int split=0;split<4;++split)',
    'v[j]+=partial[split][lane+32*j]',
    'if(beta!=0)value+=beta*y[c]',
    'hipStreamGetDevice(q.stream,&stream_device)',
    'stream_device!=id', 'hipMemGetAddressRange',
    'dynamic>size_t(a.maxDynamicSharedSizeBytes)',
    'dynamic>p.sharedMemPerBlock-a.sharedSizeBytes',
]
for expression in required:
    assert expression in source, expression
kernel = source[source.index('__global__'):source.index('static size_t dynamic_bytes')]
assert kernel.count('__syncthreads()') == 2
assert kernel.index('// B1') < kernel.index('for(int k=')
assert kernel.index('// B2') > kernel.index('partial[g][c]=v')
assert kernel.index('return') > kernel.index('// B2')
assert 'reinterpret_cast' not in kernel
for name in ('wave', 'g', 'nw', 'tile'):
    assert f'const int {name}=__builtin_amdgcn_readfirstlane' in kernel
submit = source[source.index('hipError_t submit('):]
for forbidden in ('hipGetDevice', 'hipMalloc', 'hipMemcpy', 'hipStream', 'hipFunc', 'Synchronize'):
    assert forbidden not in submit

loads = arrivals = 0
for H in range(1,65):
    stride=(H+3)//4*128
    initialized={}
    for wave in range(32):
        g,nw=divmod(wave,8);begin=H*g//4;end=H*(g+1)//4
        for h in range(begin+nw,end,8):
            for lane in range(32):
                for j in range(4):
                    c=h*128+lane+32*j
                    off=g*stride+(h-begin)*128+lane+32*j
                    assert off not in initialized
                    initialized[off]=c
    assert len(initialized)==H*128
    for wave in range(32):
        g,nw=divmod(wave,8);begin=H*g//4;end=H*(g+1)//4
        for k in range(8*begin,8*end):
            for lane in range(32):
                for t in range(2):
                    local=(k-8*begin)*16+2*(lane&3)+8*t
                    off=g*stride+local
                    assert off%2==0 and 0<=local<local+1<(end-begin)*128
                    assert initialized[off]==k*16+2*(lane&3)+8*t
                    assert initialized[off+1]==initialized[off]+1
                    loads+=1
    arrivals+=1024*2
    assert 4*stride*2<=16384
writers=[(g,16*nw+lane//4+8*h) for g in range(4)
         for nw in range(8) for lane in range(0,32,4) for h in range(2)]
assert len(set(writers))==512
print(json.dumps(dict(expressions=len(required), H_domain=64, pair_reads=loads,
    barrier_thread_arrivals=arrivals, P_writers=len(writers),
    source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    limitation='Source/model bridge; actual ELF and physical checks remain mandatory'),indent=2))
