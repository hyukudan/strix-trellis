"""Compile the exact pure-host guard with a minimal HIP type shim, no HIP runtime."""
from pathlib import Path
import hashlib
import json
import subprocess

root=Path(__file__).resolve().parents[2]
build=root/'build/host-check'
(build/'hip').mkdir(parents=True,exist_ok=True)
(build/'hip/hip_runtime.h').write_text('using hipStream_t=void*; enum hipError_t { hipSuccess=0, hipErrorInvalidValue=1 };\n')
source=(root/'experiments/fused_both/kernel.hip').read_text()
guard=source[source.index('hipError_t guard('):source.index('template<int B,Mode M> static hipError_t resource_one')]
main=r'''
#include "kernel.hpp"
#include <cmath>
#include <limits>
#include <cassert>
#include <iostream>
namespace strix_trellis::fused_both {
GUARD
}
using namespace strix_trellis;
namespace fb=strix_trellis::fused_both;
static void pointer(fb::Request& q,int k,uintptr_t p) {
    switch(k) {
    case 0:q.weights.trellis=(const uint32_t*)p;break;
    case 1:q.x=(const float*)p;break;
    case 2:q.weights.suh=(const uint16_t*)p;break;
    case 3:q.weights.svh=(const uint16_t*)p;break;
    case 4:q.weights.bias=(const float*)p;break;
    case 5:q.y=(float*)p;break;
    case 6:q.partial=(float*)p;break;
    case 7:q.debug_a=(_Float16*)p;break;
    case 8:q.debug_h=(float*)p;break;
    }
    q.regions[k].data=(const void*)p;
}
static fb::Request make(int i,int n,int bits) {
    fb::Request q;q.weights.input=i;q.weights.output=n;q.weights.bits=bits;q.weights.codebook=2;
    q.mode=fb::Mode::Diagnostic;
    size_t bytes[]={size_t(i)*n*bits/8,size_t(i)*4,size_t(i)*2,size_t(n)*2,size_t(n)*4,
        size_t(n)*4,size_t(4)*n*4,size_t(i)*2,size_t(i)*4};
    for(int k=0;k<9;++k){pointer(q,k,(uintptr_t(k)+1)<<33);q.regions[k].bytes=bytes[k];}
    return q;
}
int main(){
    size_t valid=0,negative=0;
    auto reject=[&](const fb::Request& q){assert(fb::guard(q)!=hipSuccess);++negative;};
    for(int i=128;i<=8192;i+=128)for(int n=128;n<=131072;n+=128)for(int bits:{5,6}) {
        auto q=make(i,n,bits);assert(fb::guard(q)==hipSuccess);++valid;
    }
    auto q=make(2304,5120,5);
    for(int k=0;k<9;++k) {
        auto b=q;--b.regions[k].bytes;reject(b);
        b=q;pointer(b,k,0);reject(b);
        b=q;pointer(b,k,uintptr_t(b.regions[k].data)+1);reject(b);
        b=q;b.regions[k].bytes=std::numeric_limits<size_t>::max();reject(b);
        for(int j=0;j<k;++j){b=q;pointer(b,k,uintptr_t(b.regions[j].data));reject(b);}
    }
    for(int x:{-128,0,1,127,129,8320}){auto b=q;b.weights.input=x;reject(b);}
    for(int x:{-128,0,1,127,129,131200}){auto b=q;b.weights.output=x;reject(b);}
    for(int x:{-1,0,2,8}){auto b=q;b.rows=x;reject(b);}
    for(int x:{-1,0,1,2,8}){auto b=q;b.groups=x;reject(b);}
    for(int x:{-1,0,1,4,7,8}){auto b=q;b.weights.bits=x;reject(b);}
    for(int x:{-1,0,1,3}){auto b=q;b.weights.codebook=x;reject(b);}
    for(float x:{NAN,INFINITY,-INFINITY}){auto b=q;b.alpha=x;reject(b);b=q;b.beta=x;reject(b);}
    {auto b=q;b.mode=fb::Mode(99);reject(b);}
    // All optional debug/bias ranges may be null; no external A input exists.
    for(int k:{4,7,8}){pointer(q,k,0);q.regions[k].bytes=0;}
    assert(fb::guard(q)==hipSuccess);
    q.mode=fb::Mode::Direct;pointer(q,6,0);q.regions[6].bytes=0;assert(fb::guard(q)==hipSuccess);
    std::cout<<"PASS actual_guard valid="<<valid<<" negative="<<negative<<" optional_null=PASS\n";
}
'''.replace('GUARD',guard)
(build/'check.cpp').write_text(main)
command=['nice','-n','10','c++','-O2','-std=c++17','-I'+str(build),
         '-I'+str(root/'src'),'-I'+str(root/'experiments/fused_both'),str(build/'check.cpp'),'-o',str(build/'check')]
subprocess.run(command,check=True)
result=subprocess.run([str(build/'check')],check=True,capture_output=True,text=True)
print(json.dumps(dict(command=command,guard_sha256=hashlib.sha256(guard.encode()).hexdigest(),
    output=result.stdout,scope='Exact guard only; HIP device/stream/resource checks remain native and normative models'),indent=2))
