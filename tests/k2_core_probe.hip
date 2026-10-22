// Private qualification only. No C API or production dispatch changes.
#include "../src/gemv_k2.hpp"
#include "../src/gemv_lds.hpp"
#define input_transform k2_probe_input_transform
#define output_transform k2_probe_output_transform
#include "../src/transforms.hpp"
#undef input_transform
#undef output_transform
// Recompile only private helpers for exhaustive physical arithmetic probes.
#define k2_private k2_probe_helpers
#define launch_gemv_k2_core k2_probe_unused_submit
#include "../src/gemv_k2.hip"
#undef launch_gemv_k2_core
#undef k2_private
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>
using namespace strix_trellis;
static void ck(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
static void require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
struct Buffer {
    unsigned char *base,*p; size_t size;
    explicit Buffer(size_t n):size(n){ck(hipMalloc(&base,n+512));p=base+256;ck(hipMemset(base,0xa5,n+512));}
    ~Buffer(){(void)hipFree(base);}
    void upload(const void* src,size_t n){require(n<=size,"upload size");ck(hipMemcpy(p,src,n,hipMemcpyHostToDevice));}
    std::vector<unsigned char> get(){std::vector<unsigned char> v(size);ck(hipMemcpy(v.data(),p,size,hipMemcpyDeviceToHost));return v;}
    void guards(){unsigned char v[256];for(auto* ptr:{base,p+size}){ck(hipMemcpy(v,ptr,256,hipMemcpyDeviceToHost));for(auto c:v)require(c==0xa5,"outer canary");}}
};
static uint32_t random_word(){static uint32_t s=0x732ad031;s^=s<<13;s^=s>>17;s^=s<<5;return s;}
static double host_half(uint16_t b){_Float16 h;memcpy(&h,&b,2);return double(h);}
static uint16_t book(uint32_t s){uint32_t h=s*0x83dcd12du;int sum=0;for(int j=0;j<4;++j)sum+=(h>>(8*j))&255;_Float16 v=_Float16(((1024+sum)*1774-1329*2048)/262144.0);uint16_t b;memcpy(&b,&v,2);return b;}
static uint16_t state(const uint32_t* p,int pos){uint16_t s=0;for(int j=0;j<16;++j){int b=(2*(pos+1)-16+j+512)%512;s=uint16_t((s<<1)|((p[b/32]>>(31-b%32))&1));}return s;}
static double weight(const uint32_t* p,int I,int N,int i,int col){(void)I;int r=i&15,c=col&15;int pos=((c&7)<<5)|((r&6)<<2)|(r&1)|((r&8)>>2)|((c&8)>>1);return host_half(book(state(p+(size_t(i/16)*(N/16)+col/16)*16,pos)));}
static hipError_t shim(const DeviceWeights& w,const _Float16* a,float* p,int total,int S){if(reinterpret_cast<uintptr_t>(a)&3u)return launch_gemv_lds_core(w,a,p,1,total,S,nullptr);return launch_gemv_k2_core(w,a,p,total,S,nullptr);}
__global__ void books_probe(uint32_t* out){uint32_t s=blockIdx.x*256+threadIdx.x,v[8],h[8];for(int j=0;j<8;++j)v[j]=((j&1)?65535-s:s)|(uint32_t(0xa55a+j)<<16);k2_probe_helpers::staged(v,h);for(int j=0;j<8;++j)out[s*12+j]=h[j];for(int j=0;j<4;++j)out[s*12+8+j]=__builtin_bit_cast(uint32_t,k2_probe_helpers::sad_pair(h[2*j],h[2*j+1]));}
__global__ void reader_probe(const uint32_t* p,uint16_t* out){int l=threadIdx.x,index=(498+16*l)/32-16;auto* b=reinterpret_cast<const unsigned char*>(p+blockIdx.x*16);k2_probe_helpers::Raw r{k2_probe_helpers::word_load(b,index<0?index+16:index),k2_probe_helpers::word_load(b,index+1>=16?index+1-16:index+1),0,0};uint32_t s[8];k2_probe_helpers::states(r,s);for(int j=0;j<8;++j)out[blockIdx.x*256+l*8+j]=uint16_t(s[j]);}
static void exhaustive(){Buffer h(65536*12*4);books_probe<<<256,256>>>((uint32_t*)h.p);ck(hipGetLastError());auto raw=h.get();auto* v=(uint32_t*)raw.data();for(uint32_t s=0;s<65536;++s){for(int j=0;j<8;++j)require(v[s*12+j]==((j&1)?65535-s:s)*0x83dcd12du,"hash contaminated");for(int j=0;j<4;++j)require(v[s*12+8+j]==(uint32_t(book(s))|(uint32_t(book(65535-s))<<16)),"paired half bits");}h.guards();std::vector<uint32_t> p(769*16);for(int b=0;b<512;++b)p[b*16+b/32]=1u<<(b%32);for(size_t j=512*16;j<p.size();++j)p[j]=random_word();Buffer dp(p.size()*4),dy(769*256*2);dp.upload(p.data(),dp.size);reader_probe<<<769,32>>>((uint32_t*)dp.p,(uint16_t*)dy.p);ck(hipGetLastError());auto y=dy.get();auto* s=(uint16_t*)y.data();for(int t=0;t<769;++t)for(int j=0;j<256;++j)require(s[t*256+j]==state(p.data()+t*16,j),"reader basis/random");dp.guards();dy.guards();puts("EXHAUSTIVE PASS hashes=524288 halves=524288 reader_states=196864");}
static void small(){for(int I:{16,80,128,176,272})for(int N:{16,128}){
    std::vector<uint32_t> p(size_t(I)*N/16);for(auto& x:p)x=random_word();std::vector<uint16_t> a(I+1);const uint16_t special[]={0,0x8000,1,0x8001,0x3ff,0x83ff,0x7bff,0xfbff,0x3800,0xb400};for(int j=0;j<I+1;++j)a[j]=special[j%10];
    Buffer dp(p.size()*4),da(a.size()*2);dp.upload(p.data(),dp.size);da.upload(a.data(),da.size);DeviceWeights w{(uint32_t*)dp.p,nullptr,nullptr,nullptr,I,N,2,2};
    for(int S:{1,3,11})for(int offset:{0,1}){Buffer dy(size_t(S)*3*N*4+256);std::vector<unsigned char> ref;
        for(int mode=0;mode<2;++mode){ck(hipMemset(dy.p,0xa5,dy.size));auto* x=(_Float16*)da.p+offset;auto* dst=(float*)dy.p+N;
            if(mode==0)ck(launch_gemv_lds_core(w,x,dst,1,3,S,nullptr));else ck(shim(w,x,dst,3,S));auto raw=dy.get();
            for(size_t b=0;b<raw.size();++b){size_t elem=b/4;bool written=elem<size_t(S)*3*N && elem%(3*N)>=size_t(N)&&elem%(3*N)<size_t(2*N);if(!written)require(raw[b]==0xa5,"all holes/padding");}
            if(mode==0)ref=raw;else require(raw==ref,"small whole partial bytes");
            for(int s=0;s<S;++s)for(int c=0;c<N;++c){double sum=0,abs=0;for(int i=(I/16)*s/S*16;i<(I/16)*(s+1)/S*16;++i){double v=host_half(a[i+offset])*weight(p.data(),I,N,i,c);sum+=v;abs+=std::abs(v);}float got;memcpy(&got,raw.data()+((size_t(s)*3+1)*N+c)*4,4);require(std::isfinite(got)&&std::abs(double(got)-sum)<=2e-6*std::max(1.,abs),"all small FP64 partial bound");}dy.guards();
        }
        if(offset)require(launch_gemv_k2_core(w,(_Float16*)da.p+1,(float*)dy.p,3,S,nullptr)==hipErrorInvalidValue,"offset2 direct reject");
    }dp.guards();da.guards();printf("SMALL PASS I=%d N=%d S=1,3,11 total_rows=3 row_offset=1 offset2=1 all_cpu=1\n",I,N);
}}
struct Fixture {
    std::map<std::string, std::vector<unsigned char>> v;
    int I = 0, N = 0;
    explicit Fixture(const char* path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        require(bool(f), "fixture file");
        const auto length = f.tellg();
        require(length >= 16 && length < 256ll * 1024 * 1024, "fixture file size");
        f.seekg(0);
        char magic[8]; uint32_t ver = 0, count = 0;
        f.read(magic, 8); f.read(reinterpret_cast<char*>(&ver), 4);
        f.read(reinterpret_cast<char*>(&count), 4);
        require(f && std::memcmp(magic, "STGPU001", 8) == 0 && ver == 1 && count < 64, "fixture header");
        for (uint32_t j = 0; j < count; ++j) {
            char tag[9] = {}; uint64_t n = 0;
            f.read(tag, 8); f.read(reinterpret_cast<char*>(&n), 8);
            require(bool(f), "section header");
            const auto position = f.tellg();
            require(position >= 0 && position <= length && n <= uint64_t(length - position), "section length");
            require(!v.count(tag), "duplicate fixture section");
            auto& payload = v[tag]; payload.resize(static_cast<size_t>(n));
            f.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(n));
            require(bool(f), "section read");
        }
        require(f.tellg() == length, "trailing fixture bytes");
        const auto metadata = v.find("meta");
        require(metadata != v.end() && metadata->second.size() == 32, "fixture metadata size");
        uint32_t meta[8]; std::memcpy(meta, metadata->second.data(), sizeof(meta));
        require(meta[0] > 0 && meta[0] <= 8192 && meta[0] % 128 == 0 &&
            meta[1] > 0 && meta[1] <= 131072 && meta[1] % 128 == 0 &&
            meta[2] == 2 && meta[3] == 2 && meta[4] == 0x83dcd12du && meta[5] == 8 && meta[6] <= 1,
            "fixture contract");
        I = static_cast<int>(meta[0]); N = static_cast<int>(meta[1]);
        const auto extent = [&](const char* tag, size_t bytes) {
            const auto section = v.find(tag);
            require(section != v.end() && section->second.size() == bytes, "fixture logical section size");
        };
        // These are the exact R8 layouts emitted by make_k2_fixtures.py. Check
        // every section consumed by large() before any device upload or oracle
        // access; a structurally valid file may still truncate a logical array.
        extent("trellis", size_t(I) * N / 4);
        extent("a", size_t(meta[5]) * I * 2);
        extent("x", size_t(meta[5]) * I * 4);
        extent("suh", size_t(I) * 2); extent("svh", size_t(N) * 2);
        extent("old", size_t(meta[5]) * N * 4);
        extent("ref", size_t(meta[5]) * N * 4);
        extent("p64", size_t(4) * N * 8); extent("pabs", size_t(4) * N * 8);
        extent("bias", meta[6] ? size_t(N) * 4 : 0);
    }
};

static void large(const char* path){Fixture f(path);int I=f.I,N=f.N;size_t B=size_t(I)*N/4,C=((size_t(256)<<20)+B-1)/B;require(C*B+(size_t(8)<<20)<(size_t(1)<<30),"GPU budget");
    Buffer packed(C*B),a(I*2+4),x(I*4),su(I*2),sv(N*2),bias(f.v.at("bias").size()),p(4*N*4),y(N*4);
    for(size_t c=0;c<C;++c)ck(hipMemcpy(packed.p+c*B,f.v.at("trellis").data(),B,hipMemcpyHostToDevice));a.upload(f.v.at("a").data(),I*2);x.upload(f.v.at("x").data(),I*4);su.upload(f.v.at("suh").data(),I*2);sv.upload(f.v.at("svh").data(),N*2);bias.upload(f.v.at("bias").data(),bias.size);
    DeviceWeights w{(uint32_t*)packed.p,(uint16_t*)su.p,(uint16_t*)sv.p,bias.size?(float*)bias.p:nullptr,I,N,2,2};
    auto submit=[&](int mode,size_t clone,bool full,float alpha=1,float beta=0){w.trellis=(uint32_t*)(packed.p+clone*B);if(full){k2_probe_input_transform<<<(I/128+3)/4,128>>>((float*)x.p,w.suh,(_Float16*)a.p,I,1);ck(hipGetLastError());}if(mode)ck(launch_gemv_k2_core(w,(_Float16*)a.p,(float*)p.p,1,4,nullptr));else ck(launch_gemv_lds_core(w,(_Float16*)a.p,(float*)p.p,1,1,4,nullptr));if(full){k2_probe_output_transform<<<(N/128+3)/4,128>>>((float*)p.p,w.svh,w.bias,(float*)y.p,N,1,4,alpha,beta);ck(hipGetLastError());}};
    submit(0,0,false);auto core_ref=p.get();auto* exact=(double*)f.v.at("p64").data();auto* abs=(double*)f.v.at("pabs").data();for(int j=0;j<4*N;++j){float got;memcpy(&got,core_ref.data()+4*j,4);require(std::isfinite(got)&&std::abs(double(got)-exact[j])<=2e-6*std::max(1.,abs[j]),"all large FP64 partials");}
    for(auto ab:std::vector<std::array<float,2>>{{1,0},{.75f,0},{1,.5f},{-.5f,-.25f}}){std::vector<unsigned char> oldref;for(int m=0;m<2;++m){if(ab[1]==0)ck(hipMemset(y.p,0xff,y.size));else y.upload(f.v.at("old").data(),y.size);submit(m,0,true,ab[0],ab[1]);auto raw=y.get();if(m==0)oldref=raw;else require(raw==oldref,"operator bytes all alpha beta");auto* cpu=(float*)f.v.at("ref").data();auto* old=(float*)f.v.at("old").data();double maxerr=0,maxref=1;for(int c=0;c<N;++c){float got;memcpy(&got,raw.data()+c*4,4);float want=cpu[c]*ab[0];if(ab[1]!=0)want+=old[c]*ab[1];require(std::isfinite(got),"finite operator beta0 poison");maxerr=std::max(maxerr,std::abs(double(got)-want));maxref=std::max(maxref,std::abs(double(want)));}require(maxerr<=5e-4*maxref,"all CPU operator columns");printf("CPU_OPERATOR I=%d N=%d mode=%d alpha=%g beta=%g maxabs=%.9g scale=%.9g\n",I,N,m,ab[0],ab[1],maxerr,maxref);}}
    // Both target shapes physically exercise offset2 fallback and direct rejection.
    ck(hipMemcpy(a.p+2,f.v.at("a").data(),I*2,hipMemcpyHostToDevice));w.trellis=(uint32_t*)packed.p;ck(launch_gemv_lds_core(w,(_Float16*)(a.p+2),(float*)p.p,1,1,4,nullptr));auto off=p.get();ck(shim(w,(_Float16*)(a.p+2),(float*)p.p,1,4));require(off==p.get(),"large offset2 bytes");require(launch_gemv_k2_core(w,(_Float16*)(a.p+2),(float*)p.p,1,4,nullptr)==hipErrorInvalidValue,"large offset2 reject");a.upload(f.v.at("a").data(),I*2);
    for(bool full:{false,true}){submit(0,0,full);auto ref=full?y.get():p.get();auto check=[&](){require((full?y.get():p.get())==ref,"all clones byte compare");};
        for(int phase=0;phase<2;++phase){if(phase==0){for(size_t c=0;c<C;++c)for(int m=0;m<2;++m){submit(m,c,full);check();}}else{for(size_t c=0;c<C;++c)for(int m=0;m<2;++m){submit(m,c,full);check();}}if(phase)break;
            hipEvent_t begin,end;ck(hipEventCreate(&begin));ck(hipEventCreate(&end));std::array<std::vector<double>,2> samples;
            for(int sweep=-1;sweep<11;++sweep)for(int order=0;order<2;++order){int m=(sweep&1)?1-order:order;ck(hipEventRecord(begin));for(size_t c=0;c<C;++c)submit(m,c,full);ck(hipEventRecord(end));ck(hipEventSynchronize(end));float ms;ck(hipEventElapsedTime(&ms,begin,end));if(sweep>=0){double us=ms*1000./C;samples[m].push_back(us);printf("SAMPLE I=%d N=%d protocol=%s mode=%d sweep=%d us=%.9g\n",I,N,full?"operator":"core",m,sweep,us);}if(sweep==10)check();}
            for(int m=0;m<2;++m){auto s=samples[m];std::sort(s.begin(),s.end());printf("RESULT I=%d N=%d R=1 S=4 protocol=%s mode=%d clones=%zu resident_bytes=%zu min_us=%.9g median_us=%.9g p90_us=%.9g\n",I,N,full?"operator":"core",m,C,C*B,s[0],s[5],s[9]);}ck(hipEventDestroy(begin));ck(hipEventDestroy(end));
        }
    }for(Buffer* b:{&packed,&a,&x,&su,&sv,&bias,&p,&y})b->guards();printf("LARGE PASS I=%d N=%d all_partials=1 all_operator_columns=1 all_clones_before_after_last=1 canaries=1\n",I,N);
}
int main(int argc,char** argv){try{setvbuf(stdout,nullptr,_IOLBF,0);require(argc==3,"two fixture paths required");hipDeviceProp_t prop{};ck(hipGetDeviceProperties(&prop,0));require(prop.warpSize==32&&!strncmp(prop.gcnArchName,"gfx1151",7),"gfx1151 wave32");printf("DEVICE name=%s arch=%s wave=%d\n",prop.name,prop.gcnArchName,prop.warpSize);exhaustive();small();large(argv[1]);large(argv[2]);puts("PASS k2_core_probe");return 0;}catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
