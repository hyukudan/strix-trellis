"""Execute the actual new host factory with CPU storage mocks, no HIP linkage."""
from pathlib import Path
import subprocess
import hashlib
import json

root=Path(__file__).resolve().parents[2]
s=(root/'src/wmma_decoder.hip').read_text()
def body(needle):
    begin=s.index(needle);at=s.index('{',begin);end=at+1;depth=1
    while depth:depth+=(s[end]=='{')-(s[end]=='}');end+=1
    return s[begin:end]
h=(root/'src/wmma_decoder.hpp').read_text()
h=h[h.index('namespace strix_trellis::decoder {'):]
prefix=r'''
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cstring>
using hipStream_t=void*;using hipError_t=int;
constexpr int hipSuccess=0,hipErrorInvalidValue=1,hipErrorNotSupported=2,hipMemoryTypeDevice=3;
auto hipStreamLegacy=reinterpret_cast<void*>(1);auto hipStreamPerThread=reinterpret_cast<void*>(2);
struct hipPointerAttribute_t{int type=3,isManaged=0,device=0;};
struct hipDeviceProp_t{char gcnArchName[32]="gfx1151";int warpSize=32;};
unsigned calls=0,checks=0,rejections=0;int current_device=0;
struct Allocation{uintptr_t base;size_t bytes;};std::vector<Allocation> allocations;
int hipGetDevice(int* d){++calls;*d=current_device;return 0;}
int hipGetDeviceProperties(hipDeviceProp_t* p,int){++calls;*p={};return 0;}
int hipStreamGetFlags(void* s,unsigned* f){++calls;*f=0;return s==reinterpret_cast<void*>(0x1000)?0:1;}
int hipStreamGetDevice(void*,int* d){++calls;*d=0;return 0;}
int hipPointerGetAttributes(hipPointerAttribute_t* p,const void* at){++calls;for(auto a:allocations)if(uintptr_t(at)>=a.base&&uintptr_t(at)-a.base<a.bytes){*p={};return 0;}return 1;}
int hipMemGetAddressRange(void** p,size_t* bytes,void* at){++calls;for(auto a:allocations)if(uintptr_t(at)>=a.base&&uintptr_t(at)-a.base<a.bytes){*p=reinterpret_cast<void*>(a.base);*bytes=a.bytes;return 0;}return 1;}
int hipGetLastError(){return 0;}
namespace strix_trellis {struct DeviceWeights{const uint32_t* trellis;const uint16_t* suh;const uint16_t* svh;const float* bias;int input,output,bits,codebook;bool fused_output_capable=false;};}
'''
source=prefix+h+'\nnamespace strix_trellis::decoder {\n'+body('bool storage(')+'\n'+body('bool disjoint(')+'\n'+body('hipError_t prepare(')+r'''
}
namespace d=strix_trellis::decoder;
void check(bool b){++checks;if(!b)throw std::runtime_error("CPU host factory");}
int main(){
    for(auto shape:std::vector<std::vector<int>>{{1280,32768,5},{8192,5120,5},{5120,129280,6},{5120,129280,5}}){
        size_t packed=size_t(shape[0])*shape[1]*shape[2]/8,ab=16ull*shape[0],pb=128ull*shape[1];
        allocations={{0x100000000ull,packed},{0x200000000ull,ab},{0x300000000ull,pb}};
        d::Request good{{reinterpret_cast<const uint32_t*>(allocations[0].base),nullptr,nullptr,nullptr,shape[0],shape[1],shape[2],2,false},
            {reinterpret_cast<void*>(allocations[0].base),packed},{reinterpret_cast<void*>(allocations[1].base),ab},{reinterpret_cast<void*>(allocations[2].base),pb},
            8,4,shape[0],shape[1],8,d::Layout::m16_a16_plain,d::Mode::aligned,reinterpret_cast<void*>(0x1000)};
        std::unique_ptr<const d::Token> token;
        for(auto mode:{d::Mode::baseline,d::Mode::aligned,d::Mode::saddr}){good.mode=mode;check(d::prepare(good,token)==0&&token);}
        auto reject=[&](d::Request bad){check(d::prepare(good,token)==0&&token);check(d::prepare(bad,token)!=0&&!token);++rejections;};
        for(int value:{-1,0,1,7,9,16,512,513}){auto b=good;b.rows=value;reject(b);b=good;b.total_rows=value;reject(b);}
        for(int value:{-1,0,1,2,3,5,16,32,33}){auto b=good;b.splits=value;reject(b);}
        for(int value:{-1,0,127,129,8193,2147483647}){auto b=good;b.weights.input=value;reject(b);b=good;b.weights.output=value;reject(b);}
        for(int value:{-1,0,1,3,4,7,8,9}){auto b=good;b.weights.bits=value;reject(b);}
        for(int value:{-1,0,1,3}){auto b=good;b.weights.codebook=value;reject(b);}
        for(int value:{-1,1,2,3}){auto b=good;b.layout=d::Layout(value);reject(b);}
        for(int value:{-1,3,4}){auto b=good;b.mode=d::Mode(value);reject(b);}
        {auto b=good;b.diagnostic=true;reject(b);}
        for(int field=0;field<3;++field){
            for(size_t cap:{size_t(0),size_t(1),SIZE_MAX}){auto b=good;auto* r=field==0?&b.packed:field==1?&b.a:&b.partial;r->capacity=cap;reject(b);}
            {auto b=good;auto* r=field==0?&b.packed:field==1?&b.a:&b.partial;--r->capacity;reject(b);}
            for(uintptr_t ptr:{uintptr_t(0),uintptr_t(1),UINTPTR_MAX-1}){auto b=good;auto* r=field==0?&b.packed:field==1?&b.a:&b.partial;r->pointer=reinterpret_cast<void*>(ptr);reject(b);}
        }
        {auto b=good;b.a_stride++;reject(b);b=good;b.partial_stride++;reject(b);b=good;b.a=b.partial;reject(b);b=good;b.packed.pointer=b.a.pointer;reject(b);}
        for(auto stream:{static_cast<void*>(nullptr),hipStreamLegacy,hipStreamPerThread,reinterpret_cast<void*>(0x2000)}){auto b=good;b.stream=stream;reject(b);}
        current_device=1;check(d::prepare(good,token)!=0&&!token);current_device=0;
        check(d::prepare(good,token)==0&&token);
        // Offsets into a live allocation must fit the entire advertised region.
        {auto b=good;b.a.pointer=reinterpret_cast<void*>(allocations[1].base+2);reject(b);}
    }
    std::cout<<"PASS_CPU_ACTUAL_HOST_FACTORY checks="<<checks<<" rejected="<<rejections<<" mock_queries="<<calls<<" GPU_enqueues=0\n";
}
'''
p=root/'build/host-domain.cpp';p.write_text(source)
subprocess.run(['g++','-O2','-std=c++17','-Wall','-Wextra',str(p),'-o',str(root/'build/host-domain-cpu')],check=True)
r=subprocess.check_output([str(root/'build/host-domain-cpu')],text=True)
(root/'build/host-domain.json').write_text(json.dumps(dict(status='PASS_CPU_ONLY',result=r.strip(),actual_factory_sha256=hashlib.sha256(s.encode()).hexdigest(),generated_cpu_sha256=hashlib.sha256(source.encode()).hexdigest(),HIP_linked=False),indent=2)+'\n')
print(r,end='')
