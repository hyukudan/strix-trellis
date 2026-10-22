// CPU-only contract tests: no HIP headers, runtime, or synthetic native-dot claims.
#include "contract.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
using namespace tp_decode;
static void check(bool b){if(!b)std::abort();}
static tp_desc good() {
    tp_desc d{};d.struct_size=sizeof(d);d.abi_version=1;d.numerics=TP_LOGICAL_K16_V1;d.logical_splits=4;
    auto& m=d.linear;m.struct_size=sizeof(m);m.abi_version=1;m.alpha=1;
    m.workspace=reinterpret_cast<void*>(0x1000000);m.workspace_bytes=alignof(tp_desc); // replaced below
    auto tensor=[](int width,uintptr_t address){strix_tensor_desc t{};t.struct_size=sizeof(t);t.abi_version=1;t.dtype=STRIX_DTYPE_F32;t.rank=2;
        t.shape[0]=8;t.shape[1]=width;t.strides[0]=width+13;t.strides[1]=1;t.data=reinterpret_cast<void*>(address);t.buffer_bytes=4*(7*(width+13)+width);return t;};
    m.input=tensor(1280,0x10000);m.output=tensor(16384,0x100000);
    m.workspace_bytes=((2*8*1280+255)&~uint64_t(255))+16*8*16384;return d;
}
int main(){
    tp_desc d=good();Region owners[4]={{uintptr_t(d.linear.input.data),d.linear.input.buffer_bytes},{uintptr_t(d.linear.output.data),d.linear.output.buffer_bytes},
        {uintptr_t(d.linear.workspace),d.linear.workspace_bytes},{0x2000000,13107200+2*1280+2*16384}};
    Layout l;check(validate(d,1280,16384,Variant::A,owners,l)==Error::Ok);
    check(validate(d,1280,16384,Variant(99),owners,l)==Error::Shape);
    check(validate(d,8192,5120,Variant::A,owners,l)==Error::Shape);
    auto reject=[&](tp_desc bad,Error expected){std::array<unsigned char,256> bytes{};auto before=bytes;unsigned nodes=0;
        auto err=validate(bad,1280,16384,Variant::A,owners,l);if(err==Error::Ok)++nodes;
        check(err==expected&&nodes==0&&bytes==before&&l.workspace==0);};
    auto b=d;b.struct_size--;reject(b,Error::Header);
    b=d;uint32_t invalid=99;std::memcpy(&b.numerics,&invalid,4);reject(b,Error::Enum);
    b=d;invalid=UINT32_MAX;std::memcpy(&b.numerics,&invalid,4);reject(b,Error::Enum);
    b=d;std::memcpy(&b.linear.input.dtype,&invalid,4);reject(b,Error::Shape);
    b=d;std::memcpy(&b.linear.output.dtype,&invalid,4);reject(b,Error::Shape);
    b=d;b.logical_splits=3;reject(b,Error::Splits);
    b=d;b.linear.input.shape[0]=0;reject(b,Error::Rows);
    b=d;b.linear.input.shape[0]=9;reject(b,Error::Rows);
    b=d;b.linear.input.strides[0]=1279;reject(b,Error::Stride);
    b=d;b.linear.input.strides[0]=UINT64_MAX;reject(b,Error::Overflow);
    b=d;b.linear.input.buffer_bytes--;reject(b,Error::DeclaredExtent);
    b=d;b.linear.workspace_bytes--;reject(b,Error::DeclaredExtent);
    b=d;b.linear.workspace=static_cast<char*>(b.linear.workspace)+4;reject(b,Error::Alignment);
    b=d;b.linear.output.data=reinterpret_cast<void*>(UINTPTR_MAX-3);reject(b,Error::Overflow);
    b=d;b.linear.output.reserved[3]=1;reject(b,Error::Header);
    b=d;b.linear.alpha=std::numeric_limits<float>::infinity();reject(b,Error::Shape);
    b=d;b.linear.beta=std::numeric_limits<float>::quiet_NaN();reject(b,Error::Shape);
    // LAST invalid: declared size is sufficient; actual factory allocation is short.
    owners[2].bytes--;reject(d,Error::OwnedExtent);owners[2].bytes++;
    check(owned({100,20},{100,19})==Error::OwnedExtent);
    check(owned({120,20},{100,40})==Error::Ok);
    check(owned({UINTPTR_MAX-3,8},{0,100})==Error::Overflow);
    b=d;b.linear.output.data=reinterpret_cast<void*>(uintptr_t(d.linear.input.data)+4);
    b.linear.output.buffer_bytes=d.linear.output.buffer_bytes;
    Region shared[4]={owners[0],{uintptr_t(b.linear.output.data),b.linear.output.buffer_bytes},owners[2],owners[3]};
    check(validate(b,1280,16384,Variant::A,shared,l)==Error::Alias);
    // An output placed only into the X row-gap still aliases the conservative
    // declared X envelope. No unsafe exception for gaps is made privately.
    b=d;b.linear.output.data=reinterpret_cast<void*>(uintptr_t(d.linear.input.data)+1280*4);
    shared[1]={uintptr_t(b.linear.output.data),b.linear.output.buffer_bytes};
    check(validate(b,1280,16384,Variant::A,shared,l)==Error::Alias);
    // Exact adjacency of full declared envelopes is allowed.
    b=d;b.linear.output.data=reinterpret_cast<void*>(uintptr_t(d.linear.input.data)+d.linear.input.buffer_bytes);
    shared[1]={uintptr_t(b.linear.output.data),b.linear.output.buffer_bytes};
    check(validate(b,1280,16384,Variant::A,shared,l)==Error::Ok);
    b=d;b.linear.input.data=reinterpret_cast<void*>(uintptr_t(b.linear.input.data)+2);reject(b,Error::Alignment);
    b=d;b.linear.input.dtype=STRIX_DTYPE_F16;reject(b,Error::Shape);
    b=d;b.linear.input.strides[0]=(UINT64_MAX-1280)/7;reject(b,Error::Overflow);
    b=d;b.linear.weights=reinterpret_cast<strix_weights>(1);reject(b,Error::Shape);
    b=d;b.struct_size=0;b.logical_splits=0;reject(b,Error::Header); // deterministic first error
    b=d;b.linear.input.shape[0]=b.linear.output.shape[0]=1;b.linear.input.strides[0]=b.linear.output.strides[0]=UINT64_MAX;
    check(validate(b,1280,16384,Variant::A,owners,l)==Error::Ok);
    // Real host backing and canaries: exact final extent, distinct row strides.
    for(int rows=1;rows<=8;++rows){int width=16384,stride=width+17;
        std::vector<uint32_t> v((rows-1)*stride+width+11,0x7fc01234),before=v;
        for(int r=0;r<rows;++r)for(int c=0;c<width;++c)v[r*stride+c]=uint32_t(r+c);
        for(size_t j=0;j<v.size();++j)if(j/stride>=size_t(rows)||j%stride>=size_t(width))check(v[j]==before[j]);
    }
    // Raw circular word/state mapping against a literal bit reader, all lanes/states.
    uint32_t words[40];for(int j=0;j<40;++j)words[j]=uint32_t(j*0x1234567u)^0x87654321u;
    for(int lane=0;lane<32;++lane)for(int j=0;j<8;++j){int start=(lane*8+1)*5+1280-16,idx=start/32,shift=start&31;
        uint32_t a=words[idx%40],b0=words[(idx+1)%40],c=words[(idx+2)%40];
        uint32_t hi=(a<<shift)|(b0>>(32-shift)),lo=(b0<<shift)|(c>>(32-shift));
        int off=j*5;uint32_t fast=off<=16?(hi>>(16-off))&65535u:off<32?((hi<<(off-16))|(lo>>(48-off)))&65535u:(lo>>(48-off))&65535u;
        uint32_t scalar=0;for(int z=0;z<16;++z){int bit=(start+j*5+z)%1280;scalar=(scalar<<1)|((words[bit/32]>>(31-bit%32))&1);}
        check(fast==scalar);
    }
    for(uint32_t s=0;s<65536;++s){uint64_t product=uint64_t(s)*0x83dcd12du;
        uint32_t lo=uint32_t(s*uint32_t(0xd12d)),high=uint32_t(s*uint32_t(0x83dc))<<16;
        check(uint32_t(product)==lo+high);}
    check(!forbidden(0));{SubmitScope scope;for(unsigned k=0;k<6;++k)check(forbidden(k));}check(!clean_submit());
    // Trusted CPU allocation factory mock: the descriptor cannot enlarge records.
    struct MockAllocation {std::unique_ptr<unsigned char[]> bytes;size_t size;
        explicit MockAllocation(size_t n):bytes(new unsigned char[n]),size(n){}
        Region region()const{return {uintptr_t(bytes.get()),size};}};
    auto factory=[](size_t n){return std::make_shared<MockAllocation>(n);};
    auto allocation=factory(1024);auto retained=allocation;
    Region actual=allocation->region();check(owned({actual.address,1024},actual)==Error::Ok);
    check(owned({actual.address,1025},actual)==Error::OwnedExtent);
    // A retained command snapshot survives a descriptor/owner-variable swap.
    actual=retained->region();allocation=factory(32);retained->bytes[1023]=71;
    check(actual.address==uintptr_t(retained->bytes.get())&&retained->bytes[1023]==71&&allocation->size==32);
    std::puts("PASS private CPU layout/overflow/owned-extent/first-error/LAST/canary/raw-ring/hash/guard; no native fdot2 proof");
}
