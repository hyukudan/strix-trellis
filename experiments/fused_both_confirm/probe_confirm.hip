// Private physical qualification. Never execute (including options) outside a lease.
#include "../fused_both/kernel.hpp"
#include "gemv_fused.hpp"
#include "gemv_lds.hpp"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// Recompiled consumer of old stage symbols from the matching immutable V12 archive.
namespace strix_trellis {
__global__ void input_transform(const float*,const uint16_t*,_Float16*,int,int);
__global__ void output_transform(const float*,const uint16_t*,const float*,float*,int,int,int,float,float);
}
using namespace strix_trellis;
namespace fo=strix_trellis::fused_both;
using Clock=std::chrono::steady_clock;
static Clock::time_point epoch;
static hipStream_t stream;
static size_t live=0,peak=0;
static void require(bool ok,const std::string& what){if(!ok)throw std::runtime_error(what);}
static void ck(hipError_t e){require(e==hipSuccess,hipGetErrorString(e));}
static void budget(){
    require(std::chrono::duration<double>(Clock::now()-epoch).count()<140,"Internal 140s work limit");
    std::string maintenance=std::string(std::getenv("HOME"))+"/gpu-site/.gpu-turn/maintenance";
    require(access(maintenance.c_str(),F_OK)!=0,"Maintenance veto");
    std::time_t t=std::time(nullptr);std::tm local{};localtime_r(&t,&local);int m=60*local.tm_hour+local.tm_min;
    require(m<1260||m>=1410,"Local 21:00-23:30 veto");
}
static void drain(){ck(hipStreamSynchronize(stream));budget();}
struct Buffer {
    unsigned char* allocation=nullptr;unsigned char* p=nullptr;size_t bytes;
    explicit Buffer(size_t n):bytes(n) {
        require(live+n+512<(size_t(1)<<30),"GPU budget <1GiB");
        ck(hipMalloc(reinterpret_cast<void**>(&allocation),n+512));p=allocation+256;
        live+=n+512;peak=std::max(peak,live);ck(hipMemsetAsync(allocation,0xa5,n+512,stream));
    }
    ~Buffer(){if(allocation)(void)hipFree(allocation);live-=bytes+512;}
    Buffer(const Buffer&)=delete;
    void put(const void* src){if(bytes)ck(hipMemcpyAsync(p,src,bytes,hipMemcpyHostToDevice,stream));}
    std::vector<unsigned char> read() const {
        std::vector<unsigned char> v(bytes);if(bytes)ck(hipMemcpyAsync(v.data(),p,bytes,hipMemcpyDeviceToHost,stream));drain();return v;
    }
    void guards() const {
        unsigned char v[512];ck(hipMemcpyAsync(v,allocation,256,hipMemcpyDeviceToHost,stream));
        ck(hipMemcpyAsync(v+256,p+bytes,256,hipMemcpyDeviceToHost,stream));drain();
        for(auto x:v)require(x==0xa5,"Canary changed");
    }
    void unchanged(const void* src)const {auto v=read();require(!std::memcmp(v.data(),src,bytes),"Immutable input changed");guards();}
};
template<class T> static T* as(Buffer& b){return reinterpret_cast<T*>(b.p);}
template<class T> static std::vector<T> typed(const Buffer& b){auto raw=b.read();std::vector<T> v(raw.size()/sizeof(T));std::memcpy(v.data(),raw.data(),raw.size());return v;}
template<class T> static std::vector<unsigned char> raw(const std::vector<T>& v){std::vector<unsigned char> b(v.size()*sizeof(T));std::memcpy(b.data(),v.data(),b.size());return b;}
static void exact(const std::vector<unsigned char>& a,const std::vector<unsigned char>& b,const char* tag){require(a==b,std::string("Byte identity: ")+tag);}
static void finite(const std::vector<unsigned char>& v){for(size_t i=0;i<v.size();i+=4){float x;std::memcpy(&x,v.data()+i,4);require(std::isfinite(x),"Nonfinite compared value");}}
static void tolerance(const std::vector<unsigned char>& got,const std::vector<float>& ref) {
    require(got.size()==ref.size()*4,"Reference extent");
    double e2=0,r2=0,m=0,mr=0;
    for(size_t i=0;i<ref.size();++i){float x;std::memcpy(&x,got.data()+4*i,4);require(std::isfinite(x)&&std::isfinite(ref[i]),"Nonfinite CPU comparison");double e=double(x)-ref[i];m=std::max(m,std::abs(e));mr=std::max(mr,std::abs(double(ref[i])));e2+=e*e;r2+=double(ref[i])*ref[i];}
    double nrms=std::sqrt(e2/std::max(r2,1e-30));
    require(m<=5e-4*std::max(1.0,mr)&&nrms<=3e-5,"Unchanged CPU tolerance");
    if(ref.size()>256)std::cout<<"PRECISION full_values="<<ref.size()<<" maxabs="<<m<<" nrms="<<nrms<<" limit="<<5e-4*std::max(1.0,mr)<<" PASS\n";
}
static float half(uint16_t bits){_Float16 h;std::memcpy(&h,&bits,2);return float(h);}
static uint16_t halfbits(float x){_Float16 h=x;uint16_t b;std::memcpy(&b,&h,2);return b;}
static void cpu_h(float* x){for(int d=1;d<128;d*=2)for(int b=0;b<128;b+=2*d)for(int j=0;j<d;++j){float a=x[b+j],c=x[b+j+d];x[b+j]=a+c;x[b+j+d]=a-c;}for(int j=0;j<128;++j)x[j]*=0x1.6a09e6p-4f;}
static float weight(const std::vector<uint32_t>& packed,int bits,int input_k,int col,int n) {
    int r=input_k%16,c=col%16;
    int pos=(c%8)*32+((r/2)%4)*8+(r%2)+(r/8)*2+(c/8)*4;
    size_t offset=(size_t(input_k/16)*(n/16)+col/16)*bits*8;
    uint32_t state=0;int end=(pos+1)*bits,width=256*bits;
    for(int j=16;j>0;--j){int bit=(end-j+width)%width;state=2*state+((packed[offset+bit/32]>>(31-bit%32))&1);}
    uint32_t p=state*0x83dcd12du;unsigned sum=0;for(int b=0;b<4;++b)sum+=(p>>(8*b))&255;
    // Products and addition are exact in binary64 before a single RN16 rounding.
    return float(_Float16(double(1024+sum)*half(0x1eee)+half(0xc931)));
}
static std::vector<float> cpu_core(const std::vector<uint32_t>& p,const std::vector<uint16_t>& a,int i,int n,int bits) {
    std::vector<float> y(size_t(n),0);
    for(int k=0;k<i;++k)for(int c=0;c<n;++c){float w=weight(p,bits,k,c,n);for(int r=0;r<1;++r){float product=half(a[size_t(r)*i+k])*w;y[size_t(r)*n+c]+=product;}}
    for(float v:y)require(std::isfinite(v),"Finite scalar CPU core");return y;
}
static std::vector<float> cpu_partial(const std::vector<uint32_t>& p,const std::vector<uint16_t>& a,int i,int n,int bits) {
    std::vector<float> v(size_t(4)*n);
    for(int g=0;g<4;++g)for(int c=0;c<n;++c){double sum=0;
        for(int k=(i/128)*g/4*128;k<(i/128)*(g+1)/4*128;++k)
            sum+=double(half(a[k]))*weight(p,bits,k,c,n);
        v[size_t(g)*n+c]=float(sum);
    }return v;
}
static std::vector<float> cpu_output(std::vector<float> v,const std::vector<uint16_t>& scale,
    const std::vector<float>& bias,const std::vector<float>& old,float alpha,float beta,int n) {
    for(int r=0;r<1;++r)for(int c=0;c<n;c+=128)cpu_h(v.data()+size_t(r)*n+c);
    for(size_t k=0;k<v.size();++k){float x=v[k]*half(scale[k%n]);if(!bias.empty())x+=bias[k%n];x*=alpha;if(beta!=0){float p=beta*old[k];x+=p;}require(std::isfinite(x),"Finite CPU epilogue");v[k]=x;}return v;
}
static fo::Request request(DeviceWeights w,Buffer& x,Buffer& y,Buffer* p,fo::Mode mode,
    float alpha=1,float beta=0,Buffer* a=nullptr,Buffer* h=nullptr) {
    fo::Request q;q.weights=w;q.x=as<float>(x);q.y=as<float>(y);q.partial=p?as<float>(*p):nullptr;
    q.debug_a=a?as<_Float16>(*a):nullptr;q.debug_h=h?as<float>(*h):nullptr;
    q.stream=stream;q.mode=mode;q.alpha=alpha;q.beta=beta;
    q.regions={fo::Region{w.trellis,size_t(w.input)*w.output*w.bits/8},
        {x.p,x.bytes},{w.suh,size_t(w.input)*2},{w.svh,size_t(w.output)*2},
        {w.bias,w.bias?size_t(w.output)*4:0},{y.p,y.bytes},
        {p?p->p:nullptr,p?p->bytes:0},{a?a->p:nullptr,a?a->bytes:0},{h?h->p:nullptr,h?h->bytes:0}};
    return q;
}
static void old_output(DeviceWeights w,Buffer& p,Buffer& y,int s,float alpha=1,float beta=0){
    output_transform<<<((w.output/128)+3)/4,128,0,stream>>>(as<float>(p),w.svh,w.bias,as<float>(y),w.output,1,s,alpha,beta);ck(hipGetLastError());
}
static std::vector<uint16_t> cpu_input(const std::vector<float>& x,const std::vector<uint16_t>& u,std::vector<float>& h) {
    h.resize(x.size());std::vector<uint16_t>a(x.size());
    for(size_t k=0;k<x.size();++k){volatile float scaled=x[k]*half(u[k]);require(std::isfinite(scaled),"Finite RN32 product");h[k]=half(halfbits(scaled));require(std::isfinite(h[k]),"Finite scaled16");}
    for(size_t k=0;k<x.size();k+=128)cpu_h(h.data()+k);
    for(size_t k=0;k<x.size();++k){require(std::isfinite(h[k]),"Finite H");a[k]=halfbits(h[k]);require(std::isfinite(half(a[k])),"Finite A");}
    return a;
}
static void negatives(const fo::Request& q) {
    auto reject=[](const fo::Request& b){require(fo::guard(b)!=hipSuccess,"Missing host rejection");};
    for(int rows:{0,2,8}){auto b=q;b.rows=rows;reject(b);}
    for(int g:{0,1,2,8}){auto b=q;b.groups=g;reject(b);}
    for(int bits:{0,4,7}){auto b=q;b.weights.bits=bits;reject(b);}
    for(int n:{-128,0,127,129,131200}){auto b=q;b.weights.output=n;reject(b);}
    for(int i:{-128,0,127,129,8320}){auto b=q;b.weights.input=i;reject(b);}
    for(int j=0;j<9;++j)if(q.regions[j].bytes){auto b=q;--b.regions[j].bytes;reject(b);}
    auto b=q;b.weights.codebook=1;reject(b);b=q;b.alpha=NAN;reject(b);b=q;b.beta=INFINITY;reject(b);
    b=q;b.x=reinterpret_cast<const float*>(reinterpret_cast<uintptr_t>(q.x)+2);b.regions[1].data=b.x;reject(b);
    b=q;b.y=nullptr;reject(b);b=q;b.y=const_cast<float*>(q.x);b.regions[5].data=b.y;reject(b);
    b=q;b.x=reinterpret_cast<float*>(~uintptr_t(3));b.regions[1].data=b.x;reject(b);
    fo::Token token;require(fo::submit(token)!=hipSuccess,"Unprepared token accepted");
    b=q;std::vector<float> host(q.weights.input);b.x=host.data();b.regions[1]={b.x,host.size()*4};
    require(fo::prepare(b,token)!=hipSuccess,"Host storage accepted");
}
static void small() {
    size_t cases=0;
    for(int bits:{5,6})for(int i:{128,256,384,512,640,1280,2304,5120,8192})for(int n:{128,256}) {
        budget();std::vector<uint32_t> p(size_t(i)*n*bits/32);uint32_t rng=123+i+n+bits;
        for(auto& x:p){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;x=rng;}
        std::vector<float>x(i),b(n),old(n);std::vector<uint16_t>u(i),v(n);
        float patterns[]={0.f,-0.f,0x1p-24f,-0x1p-24f,.2001299113035202f,1.00048828125f,-.75f,.5f};
        const uint16_t scales[]={0x0000,0x8000,0x0001,0x8001,0x03ff,0x83ff,0x0400,0x8400,0x3a00,0xbc00,0x3800,0xb400};
        for(int k=0;k<i;++k){x[k]=patterns[(k*7+k/13)%8];u[k]=scales[k%12];}
        x[4]=.2001299113035202f;u[4]=halfbits(.02642822265625f);
        require(u[4]==0x26c4 && halfbits(x[4]*half(u[4]))==7530,"Explicit RN32/RN16 tie");
        x[17]=65504.f;u[17]=0x3c00; // Sparse high range: H/A stay finite.
        for(int c=0;c<n;++c){v[c]=scales[c%12];b[c]=float((c*7)%19-9)/16;old[c]=float(c*13%31-15)/8;}
        std::vector<float>h;auto a=cpu_input(x,u,h);auto core=cpu_core(p,a,i,n,bits);auto cpu_p=cpu_partial(p,a,i,n,bits);
        Buffer dp(p.size()*4),dx(i*4),du(i*2),dv(n*2),db(n*4),dy(n*4),bp(4*n*4),cp(4*n*4),diag(4*n*4),da(i*2),dh(i*4),ba(i*2),bh(i*4);
        dp.put(p.data());dx.put(x.data());du.put(u.data());dv.put(v.data());db.put(b.data());
        DeviceWeights w{as<uint32_t>(dp),as<uint16_t>(du),as<uint16_t>(dv),as<float>(db),i,n,bits,2,false};
        ck(launch_gemv_fused_input_debug(w,as<float>(dx),as<float>(bp),1,1,4,stream,as<_Float16>(ba),as<float>(bh)));
        auto pref=bp.read();finite(pref);tolerance(pref,cpu_p);exact(ba.read(),raw(a),"B all A vs CPU");exact(bh.read(),raw(h),"B all H vs CPU");
        auto q=request(w,dx,dy,&cp,fo::Mode::GlobalPartial);negatives(q);fo::Token c;ck(fo::prepare(q,c));ck(fo::submit(c));exact(cp.read(),pref,"small C P vs B H4");
        for(bool bias:{false,true})for(float alpha:{0.f,-0.f,1.f,-.75f,.5f})for(float beta:{0.f,-0.f,1.f,-.75f,.5f}) {
            w.bias=bias?as<float>(db):nullptr;
            auto init=[&]{if(beta==0)ck(hipMemsetAsync(dy.p,0xff,dy.bytes,stream));else dy.put(old.data());};
            init();old_output(w,bp,dy,4,alpha,beta);auto ref=dy.read();finite(ref);
            tolerance(ref,cpu_output(core,v,bias?b:std::vector<float>{},old,alpha,beta,n));
            for(auto mode:{fo::Mode::GlobalPartial,fo::Mode::Direct,fo::Mode::Diagnostic}) {
                fo::Token d;ck(fo::prepare(request(w,dx,dy,mode==fo::Mode::Direct?nullptr:(mode==fo::Mode::GlobalPartial?&cp:&diag),mode,alpha,beta,
                    mode==fo::Mode::Diagnostic?&da:nullptr,mode==fo::Mode::Diagnostic?&dh:nullptr),d));
                init();ck(fo::submit(d));if(mode==fo::Mode::GlobalPartial){exact(cp.read(),pref,"small epilogue C P");old_output(w,cp,dy,4,alpha,beta);}exact(dy.read(),ref,"small B vs D Y");
                if(mode==fo::Mode::Diagnostic){exact(diag.read(),pref,"small diagnostic P");exact(da.read(),raw(a),"All new A vs CPU");exact(dh.read(),raw(h),"All new H vs CPU");}
            }
            ++cases;
        }
        dp.unchanged(p.data());dx.unchanged(x.data());du.unchanged(u.data());dv.unchanged(v.data());db.unchanged(b.data());
        for(Buffer* z:{&dy,&bp,&cp,&diag,&da,&dh,&ba,&bh})z->guards();
        std::cout<<"SMALL PASS I="<<i<<" N="<<n<<" bits="<<bits<<" full_AH=1 full_P_bytes="<<bp.bytes<<" full_Y_values="<<n<<"\n";
    }
    std::cout<<"SMALL_ALL PASS epilogues="<<cases<<" canaries_not_race_proof=1\n";
}
struct Map {
    int fd;void* data;size_t bytes;
    explicit Map(const std::string& file){fd=open(file.c_str(),O_RDONLY|O_CLOEXEC);require(fd>=0,"Open fixture "+file);struct stat st{};require(fstat(fd,&st)==0,"Stat fixture");bytes=st.st_size;data=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);require(data!=MAP_FAILED,"Map fixture");}
    ~Map(){munmap(data,bytes);close(fd);}
};
template<class T> static std::vector<T> load(const std::string& file,size_t count){Map m(file);require(m.bytes>=count*sizeof(T),"Fixture extent "+file);std::vector<T> v(count);std::memcpy(v.data(),m.data,count*sizeof(T));return v;}
static void large(const std::string& dir,const std::string& name,int i,int n,int bits,int policy,bool head) {
    budget();std::cout<<"BEGIN name="<<name<<"\n";
    Map packed(dir+(head?"/head.blob":"/packed.bin"));size_t pbytes=size_t(i)*n*bits/8;
    require(packed.bytes==(head?pbytes+size_t(i+n)*2:pbytes),"Packed fixture extent");
    auto x=load<float>(dir+(head?"/input.f32":"/x.bin"),size_t(i));
    std::vector<uint16_t> u(i),v(n);std::vector<float> b;
    if(head){std::memcpy(u.data(),static_cast<char*>(packed.data)+pbytes,2*i);std::memcpy(v.data(),static_cast<char*>(packed.data)+pbytes+2*i,2*n);}
    else{u=load<uint16_t>(dir+"/u.bin",i);v=load<uint16_t>(dir+"/v.bin",n);std::ifstream f(dir+"/b.bin",std::ios::ate|std::ios::binary);if(f&&f.tellg()>0)b=load<float>(dir+"/b.bin",n);}
    size_t clones=((size_t(256)<<20)+pbytes-1)/pbytes;
    std::vector<std::unique_ptr<Buffer>> weights,ys,bps,cps,workspaces;
    for(size_t c=0;c<clones;++c){weights.emplace_back(new Buffer(pbytes));weights.back()->put(packed.data);}
    Buffer dx(x.size()*4),du(i*2),dv(n*2),db(b.size()*4),da(size_t(i)*2),dump(size_t(4)*n*4),dh(i*4),na(i*2),nh(i*4);
    DeviceWeights base{nullptr,as<uint16_t>(du),as<uint16_t>(dv),b.empty()?nullptr:as<float>(db),i,n,bits,2,true};
    dx.put(x.data());du.put(u.data());dv.put(v.data());if(!b.empty())db.put(b.data());
    // A uses the actual V12 backend and its actual split policy, including input stage.
    size_t wsbytes=linear_workspace_bytes(base,1);
    for(size_t c=0;c<clones;++c){ys.emplace_back(new Buffer(size_t(n)*4));bps.emplace_back(new Buffer(size_t(4)*n*4));
        cps.emplace_back(new Buffer(size_t(4)*n*4));workspaces.emplace_back(new Buffer(wsbytes));}
    std::vector<std::array<fo::Token,3>> tokens(clones);
    for(size_t c=0;c<clones;++c){auto& dy=*ys[c];auto& cp=*cps[c];auto& ws=*workspaces[c];auto w=base;w.trellis=as<uint32_t>(*weights[c]);
        ck(fo::prepare(request(w,dx,dy,&cp,fo::Mode::GlobalPartial),tokens[c][0]));
        ck(fo::prepare(request(w,dx,dy,nullptr,fo::Mode::Direct),tokens[c][1]));
        ck(fo::prepare(request(w,dx,dy,&dump,fo::Mode::Diagnostic,1,0,&na,&nh),tokens[c][2]));
        // All A/B storage is owned, device checked and disjoint before events too.
        int device=-1,stream_device=-1;ck(hipGetDevice(&device));ck(hipStreamGetDevice(stream,&stream_device));require(device==stream_device,"Control stream provenance");
        for(Buffer* z:{&dx,&du,&dv,&db,&ws,&dy,bps[c].get(),&cp,&da,&dh})if(z->bytes){
            hipPointerAttribute_t attr{};ck(hipPointerGetAttributes(&attr,z->p));require(attr.type==hipMemoryTypeDevice&&!attr.isManaged&&attr.device==device,"Control device storage");
            void* allocation=nullptr;size_t size=0;ck(hipMemGetAddressRange(&allocation,&size,z->p));
            auto offset=uintptr_t(z->p)-uintptr_t(allocation);require(offset<=size&&z->bytes<=size-offset,"Control full allocation extent");
        }
        require(wsbytes>=da.bytes+size_t(policy)*n*4,"Actual policy workspace");
    }
    bool validating=true;
    auto launch=[&](int path,size_t c){auto& dy=*ys[c];auto& bp=*bps[c];auto& cp=*cps[c];auto& ws=*workspaces[c];auto w=base;w.trellis=as<uint32_t>(*weights[c]);
        if(path==0){ck(launch_linear(w,as<float>(dx),as<float>(dy),1,ws.p,stream,1,0,false));return;}
        if(path==1){if(!validating)ck(launch_gemv_fused_input(w,as<float>(dx),as<float>(bp),1,1,4,stream));else ck(launch_gemv_fused_input_debug(w,as<float>(dx),as<float>(bp),1,1,4,stream,as<_Float16>(da),as<float>(dh)));old_output(w,bp,dy,4);}
        else if(path==2){ck(fo::submit(tokens[c][0]));old_output(w,cp,dy,4);}
        else ck(fo::submit(tokens[c][path==3?1:2]));
    };
    std::vector<float> cpu_h_all;auto cpu_a_all=cpu_input(x,u,cpu_h_all);
    std::array<std::vector<unsigned char>,4> ref;
    std::vector<unsigned char> pref,aref;
    auto weights_check=[&]{for(auto& p:weights)p->unchanged(packed.data);dx.unchanged(x.data());du.unchanged(u.data());dv.unchanged(v.data());if(!b.empty())db.unchanged(b.data());};
    auto guards=[&]{for(Buffer* z:{&dx,&du,&dv,&db,&da,&dump,&dh,&na,&nh})z->guards();
        for(size_t c=0;c<clones;++c)for(Buffer* z:{ys[c].get(),bps[c].get(),cps[c].get(),workspaces[c].get()})z->guards();};
    auto cpu=load<float>(dir+"/cpu-y.f32",n);
    auto cpu_p=load<float>(dir+"/cpu-p.f32",size_t(4)*n);
    if(head){auto prefix=load<float>(dir+"/r8-y-fp64-core.f32",n);tolerance(raw(cpu),prefix);}
    auto validate=[&](bool first){
        for(size_t c=0;c<clones;++c)for(int path=0;path<5;++path){auto& dy=*ys[c];auto& bp=*bps[c];auto& cp=*cps[c];ck(hipMemsetAsync(dy.p,0xff,dy.bytes,stream));launch(path,c);auto got=dy.read();finite(got);
            if(first&&c==0&&path<4)ref[path]=got;
            exact(got,ref[path<4?path:3],"large immutable path reference");
            if(path==0&&(name=="q_b"||head)){auto all=workspaces[c]->read();require(!std::memcmp(all.data(),cpu_a_all.data(),i*2),"Actual policy A full input CPU");}
            if(path==1){auto pp=bp.read();finite(pp);tolerance(pp,cpu_p);if(first&&c==0){pref=pp;aref=da.read();exact(aref,raw(cpu_a_all),"large B A CPU");exact(dh.read(),raw(cpu_h_all),"large B H CPU");}exact(pp,pref,"large B P");exact(da.read(),aref,"large A checkpoint");exact(dh.read(),raw(cpu_h_all),"Every clone B H CPU");}
            if(path==2)exact(cp.read(),pref,"large C P vs V12S4");
            if(path==4){exact(dump.read(),pref,"large diagnostic P vs B H4");exact(na.read(),raw(cpu_a_all),"large all new A vs CPU");exact(nh.read(),raw(cpu_h_all),"large all new H vs CPU");}
            if(path>=2)exact(got,ref[1],"large all Y vs V12S4");
            if((name=="wo_b"||name=="inverse")&&path>0)exact(got,ref[0],"large actual V12 matched H4 Y");
        }
        for(auto& y:ref)tolerance(y,cpu);
        if(head){auto cuda=load<float>(dir+"/cuda-r1-y.f32",n);auto ca=load<uint16_t>(dir+"/cuda-r1-a.f16",i);exact(raw(cpu_a_all),raw(ca),"head CUDA A complete");for(auto& y:ref)tolerance(y,cuda);}
        weights_check();guards();
        std::cout<<"VALID name="<<name<<" phase="<<(first?"before":"after")<<" clones="<<clones<<" full_P_bytes="<<pref.size()<<" full_Y_values="<<n<<" real_head="<<head<<" PASS\n";
    };
    validate(true);validating=false;
    hipEvent_t begin,end;ck(hipEventCreate(&begin));ck(hipEventCreate(&end));
    std::array<std::vector<double>,4> events,walls;
    std::array<std::vector<std::vector<unsigned char>>,4> last_measured;
    for(auto& outputs:last_measured)outputs.resize(clones);
    for(int sweep=-1;sweep<11;++sweep)for(int j=0;j<4;++j){budget();int path=(sweep%2==0)?3-j:j;
        auto start=Clock::now();ck(hipEventRecord(begin,stream));for(size_t c=0;c<clones;++c)launch(path,c);ck(hipEventRecord(end,stream));ck(hipEventSynchronize(end));
        double wall=std::chrono::duration<double,std::micro>(Clock::now()-start).count()/clones;
        float ms;ck(hipEventElapsedTime(&ms,begin,end));double event=double(ms)*1000/clones;
        // Validate every last timed output before another submission.
        for(size_t c=0;c<clones;++c){auto got=ys[c]->read();if(sweep==10)last_measured[path][c]=got;exact(got,ref[path],"Every clone last measured full Y");tolerance(got,cpu);
            if(path==1){auto gotp=bps[c]->read();exact(gotp,pref,"Every clone measured B P");tolerance(gotp,cpu_p);}
            if(path==2){auto gotp=cps[c]->read();exact(gotp,pref,"Every clone measured C P");tolerance(gotp,cpu_p);}
        }guards();
        if(sweep>=0){events[path].push_back(event);walls[path].push_back(wall);std::cout<<"SAMPLE name="<<name<<" sweep="<<sweep<<" path="<<char('A'+path)<<" event_us="<<event<<" wall_us="<<wall<<" all_clones_last_Y_exact=1 full_cpu=1\n";}
    }
    ck(hipEventDestroy(begin));ck(hipEventDestroy(end));
    // Retain all four modes for every clone after the entire last measured sweep.
    for(int path=0;path<4;++path)for(size_t c=0;c<clones;++c){exact(last_measured[path][c],ref[path],"Retained every mode/clone last measured Y");tolerance(last_measured[path][c],cpu);}
    for(size_t c=0;c<clones;++c){exact(bps[c]->read(),pref,"Retained last measured B P");exact(cps[c]->read(),pref,"Retained last measured C P");}
    std::cout<<"LAST_ALL_MODES name="<<name<<" clones="<<clones<<" modes=4 full_Y=1 full_P=1 CPU=1 PASS\n";
    validating=true;validate(false);
    for(int path=0;path<4;++path){std::sort(events[path].begin(),events[path].end());std::sort(walls[path].begin(),walls[path].end());std::cout<<"RESULT name="<<name<<" path="<<char('A'+path)<<" clones="<<clones<<" packed_resident="<<clones*pbytes<<" event_min_us="<<events[path][0]<<" event_median_us="<<events[path][5]<<" event_p90_us="<<events[path][9]<<" wall_min_us="<<walls[path][0]<<" wall_median_us="<<walls[path][5]<<" wall_p90_us="<<walls[path][9]<<"\n";}
}
int main(int argc,char** argv){
    std::cout<<std::unitbuf<<std::setprecision(10);epoch=Clock::now();
    try{require(argc==2,"Expected private fixtures root");ck(fo::resources());ck(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
        small();std::string root=argv[1];
        large(root+"/q_b","q_b",1280,32768,5,4,false);
        large(root+"/wo_b","wo_b",8192,5120,5,4,false);
        large(root+"/inverse","inverse",2304,5120,5,4,false);
        large(root+"/shared","shared",5120,2304,5,16,false);
        large(root+"/head","head",5120,129280,6,4,true);
        drain();ck(hipStreamDestroy(stream));std::cout<<"PASS private_G4_confirm gpu_peak_bytes="<<peak<<" seconds="<<std::chrono::duration<double>(Clock::now()-epoch).count()<<"\n";return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<" gpu_peak_bytes="<<peak<<"\n";return 1;}
}
