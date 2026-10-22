// Private physical qualification. Never execute (including options) outside a lease.
#include "../src/gemv_fused_output.hpp"
#include "../src/gemv_lds.hpp"
#include <algorithm>
#include <chrono>
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

// Launch the OLD stage symbols from the immutable v9 archive, without rebuilding them.
namespace strix_trellis {
__global__ void input_transform(const float*,const uint16_t*,_Float16*,int,int);
__global__ void output_transform(const float*,const uint16_t*,const float*,float*,int,int,int,float,float);
}
using namespace strix_trellis;
namespace fo=strix_trellis::fused_output;
using Clock=std::chrono::steady_clock;
static Clock::time_point epoch;
static hipStream_t stream;
static size_t live=0,peak=0;
static void require(bool ok,const std::string& what){if(!ok)throw std::runtime_error(what);}
static void ck(hipError_t e){require(e==hipSuccess,hipGetErrorString(e));}
static void budget(){require(std::chrono::duration<double>(Clock::now()-epoch).count()<145,"Internal 145s limit");}
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
    std::vector<float> y(size_t(8)*n,0);
    for(int k=0;k<i;++k)for(int c=0;c<n;++c){float w=weight(p,bits,k,c,n);for(int r=0;r<8;++r){float product=half(a[size_t(r)*i+k])*w;y[size_t(r)*n+c]+=product;}}
    return y;
}
static std::vector<float> cpu_output(std::vector<float> v,const std::vector<uint16_t>& scale,
    const std::vector<float>& bias,const std::vector<float>& old,float alpha,float beta,int n) {
    for(int r=0;r<8;++r)for(int c=0;c<n;c+=128)cpu_h(v.data()+size_t(r)*n+c);
    for(size_t k=0;k<v.size();++k){float x=v[k]*half(scale[k%n]);if(!bias.empty())x+=bias[k%n];x*=alpha;if(beta!=0){float p=beta*old[k];x+=p;}v[k]=x;}return v;
}
static fo::Request request(DeviceWeights w,Buffer& a,Buffer& y,Buffer* p,fo::Mode mode,float alpha=1,float beta=0) {
    fo::Request q;q.weights=w;q.a=as<_Float16>(a);q.y=as<float>(y);q.partial=p?as<float>(*p):nullptr;
    q.stream=stream;q.mode=mode;q.alpha=alpha;q.beta=beta;
    q.regions={fo::Region{const_cast<uint32_t*>(w.trellis),size_t(w.input)*w.output*w.bits/8},
        {a.p,a.bytes},{const_cast<uint16_t*>(w.svh),size_t(w.output)*2},
        {const_cast<float*>(w.bias),w.bias?size_t(w.output)*4:0},{y.p,y.bytes},{p?p->p:nullptr,p?p->bytes:0}};
    return q;
}
static void old_output(DeviceWeights w,Buffer& p,Buffer& y,int s,float alpha=1,float beta=0){
    output_transform<<<(8*(w.output/128)+3)/4,128,0,stream>>>(as<float>(p),w.svh,w.bias,as<float>(y),w.output,8,s,alpha,beta);ck(hipGetLastError());
}
static void negatives(const fo::Request& q) {
    auto reject=[](const fo::Request& b){require(fo::guard(b)!=hipSuccess,"Missing host rejection");};
    for(int rows:{0,1,7,9}){auto b=q;b.rows=rows;reject(b);}
    for(int g:{0,1,2,8}){auto b=q;b.groups=g;reject(b);}
    for(int bits:{0,4,7}){auto b=q;b.weights.bits=bits;reject(b);}
    for(int n:{0,127,129,131200}){auto b=q;b.weights.output=n;reject(b);}
    for(int i:{0,127,129,8320}){auto b=q;b.weights.input=i;reject(b);}
    for(int j=0;j<6;++j)if(q.regions[j].bytes){auto b=q;--b.regions[j].bytes;reject(b);}
    auto b=q;b.weights.codebook=1;reject(b);b=q;b.alpha=NAN;reject(b);b=q;b.beta=INFINITY;reject(b);
    b=q;b.a=reinterpret_cast<const _Float16*>(reinterpret_cast<uintptr_t>(q.a)+2);b.regions[1].data=const_cast<_Float16*>(b.a);reject(b);
    b=q;b.y=nullptr;reject(b);b=q;b.y=reinterpret_cast<float*>(const_cast<_Float16*>(q.a));b.regions[4].data=b.y;reject(b);
    fo::Token token;require(fo::submit(token)!=hipSuccess,"Unprepared token accepted");
}
static void small() {
    size_t cases=0;
    for(int bits:{5,6})for(int i:{128,256,384,512,640})for(int n:{128,256}) {
        budget();std::vector<uint32_t> p(size_t(i)*n*bits/32);uint32_t rng=123+i+n+bits;
        for(auto& x:p){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;x=rng;}
        std::vector<uint16_t> a(8*i),v(n);std::vector<float>b(n),old(8*n);
        // Nonsymmetric, high finite FP16, subnormals, signed zeros and half ties.
        uint16_t patterns[]={0x7bff,0xfbff,1,0x8001,0,0x8000,0x3555,0xb801,0x3c00,0xbc00,0x100,0x8400};
        for(size_t k=0;k<a.size();++k)a[k]=patterns[(k*7+k/13)%12];
        for(int c=0;c<n;++c){v[c]=halfbits((c%2?-1:1)*(0.25f+float(c%7)/8));b[c]=float((c*7)%19-9)/16;}
        for(size_t k=0;k<old.size();++k)old[k]=float(int(k*13%31)-15)/8;
        Buffer dp(p.size()*4),da(a.size()*2),dv(n*2),db(n*4),dy(8*n*4),bp(4*8*n*4),cp(4*8*n*4),diag(4*8*n*4);
        dp.put(p.data());da.put(a.data());dv.put(v.data());db.put(b.data());
        DeviceWeights w{as<uint32_t>(dp),nullptr,as<uint16_t>(dv),as<float>(db),i,n,bits,2};
        auto core=cpu_core(p,a,i,n,bits);
        ck(launch_gemv_lds_core(w,as<_Float16>(da),as<float>(bp),8,8,4,stream));auto pref=bp.read();finite(pref);
        auto q=request(w,da,dy,&cp,fo::Mode::GlobalPartial);negatives(q);fo::Token c;ck(fo::prepare(q,c));ck(fo::submit(c));exact(cp.read(),pref,"small C P vs v9 S4");
        for(bool bias:{false,true})for(float alpha:{0.f,-0.f,1.f,-.75f})for(float beta:{0.f,-0.f,1.f,-.5f}) {
            w.bias=bias?as<float>(db):nullptr;
            auto init=[&]{if(beta==0)ck(hipMemsetAsync(dy.p,0xff,dy.bytes,stream));else dy.put(old.data());};
            init();old_output(w,cp,dy,4,alpha,beta);auto ref=dy.read();finite(ref);
            tolerance(ref,cpu_output(core,v,bias?b:std::vector<float>{},old,alpha,beta,n));
            for(auto mode:{fo::Mode::Direct,fo::Mode::Diagnostic}) {
                fo::Token d;ck(fo::prepare(request(w,da,dy,mode==fo::Mode::Direct?nullptr:&diag,mode,alpha,beta),d));
                init();ck(fo::submit(d));exact(dy.read(),ref,"small C vs D Y");
                if(mode==fo::Mode::Diagnostic)exact(diag.read(),pref,"small diagnostic P");
            }
            ++cases;
        }
        dp.unchanged(p.data());da.unchanged(a.data());dv.unchanged(v.data());db.unchanged(b.data());
        dy.guards();bp.guards();cp.guards();diag.guards();
        std::cout<<"SMALL PASS I="<<i<<" N="<<n<<" bits="<<bits<<" full_P_bytes="<<bp.bytes<<" full_Y_values="<<8*n<<"\n";
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
    auto x=load<float>(dir+(head?"/input.f32":"/x.bin"),size_t(8)*i);
    std::vector<uint16_t> u(i),v(n);std::vector<float> b;
    if(head){std::memcpy(u.data(),static_cast<char*>(packed.data)+pbytes,2*i);std::memcpy(v.data(),static_cast<char*>(packed.data)+pbytes+2*i,2*n);}
    else{u=load<uint16_t>(dir+"/u.bin",i);v=load<uint16_t>(dir+"/v.bin",n);std::ifstream f(dir+"/b.bin",std::ios::ate|std::ios::binary);if(f&&f.tellg()>0)b=load<float>(dir+"/b.bin",n);}
    size_t clones=((size_t(256)<<20)+pbytes-1)/pbytes;
    std::vector<std::unique_ptr<Buffer>> weights;
    for(size_t c=0;c<clones;++c){weights.emplace_back(new Buffer(pbytes));weights.back()->put(packed.data);}
    Buffer dx(x.size()*4),du(i*2),dv(n*2),db(b.size()*4),da(size_t(8)*i*2),dy(size_t(8)*n*4),bp(size_t(4)*8*n*4),cp(bp.bytes),dump(bp.bytes);
    DeviceWeights base{nullptr,as<uint16_t>(du),as<uint16_t>(dv),b.empty()?nullptr:as<float>(db),i,n,bits,2};
    dx.put(x.data());du.put(u.data());dv.put(v.data());if(!b.empty())db.put(b.data());
    // A uses the actual v9 backend and its actual split policy, including input stage.
    size_t wsbytes=linear_workspace_bytes(base,8);Buffer ws(wsbytes);
    std::vector<std::array<fo::Token,3>> tokens(clones);
    for(size_t c=0;c<clones;++c){auto w=base;w.trellis=as<uint32_t>(*weights[c]);
        ck(fo::prepare(request(w,da,dy,&cp,fo::Mode::GlobalPartial),tokens[c][0]));
        ck(fo::prepare(request(w,da,dy,nullptr,fo::Mode::Direct),tokens[c][1]));
        ck(fo::prepare(request(w,da,dy,&dump,fo::Mode::Diagnostic),tokens[c][2]));
        // All A/B storage is owned, device checked and disjoint before events too.
        for(Buffer* z:{&dx,&du,&ws}){hipPointerAttribute_t attr{};ck(hipPointerGetAttributes(&attr,z->p));require(attr.type==hipMemoryTypeDevice&&!attr.isManaged,"Control device storage");}
        require(wsbytes>=da.bytes+size_t(policy)*8*n*4,"Actual policy workspace");
    }
    auto launch=[&](int path,size_t c){auto w=base;w.trellis=as<uint32_t>(*weights[c]);
        if(path==0){ck(launch_linear(w,as<float>(dx),as<float>(dy),8,ws.p,stream,1,0,false));return;}
        input_transform<<<(8*(i/128)+3)/4,128,0,stream>>>(as<float>(dx),w.suh,as<_Float16>(da),i,8);ck(hipGetLastError());
        if(path==1){ck(launch_gemv_lds_core(w,as<_Float16>(da),as<float>(bp),8,8,4,stream));old_output(w,bp,dy,4);}
        else if(path==2){ck(fo::submit(tokens[c][0]));old_output(w,cp,dy,4);}
        else ck(fo::submit(tokens[c][path==3?1:2]));
    };
    std::array<std::vector<unsigned char>,4> ref;
    std::vector<unsigned char> pref,aref;
    auto weights_check=[&]{for(auto& p:weights)p->unchanged(packed.data);dx.unchanged(x.data());du.unchanged(u.data());dv.unchanged(v.data());if(!b.empty())db.unchanged(b.data());};
    auto guards=[&]{for(Buffer* z:{&dx,&du,&dv,&db,&da,&dy,&bp,&cp,&dump,&ws})z->guards();};
    auto validate=[&](bool first){
        for(size_t c=0;c<clones;++c)for(int path=0;path<5;++path){ck(hipMemsetAsync(dy.p,0xff,dy.bytes,stream));launch(path,c);auto got=dy.read();finite(got);
            if(first&&c==0&&path<4)ref[path]=got;
            exact(got,ref[path<4?path:3],"large immutable path reference");
            if(path==1){auto pp=bp.read();finite(pp);if(first&&c==0){pref=pp;aref=da.read();}exact(pp,pref,"large B P");exact(da.read(),aref,"large A checkpoint");}
            if(path==2)exact(cp.read(),pref,"large C P vs v9S4");
            if(path==4)exact(dump.read(),pref,"large diagnostic P vs v9S4");
            if(path>=2)exact(got,ref[1],"large all Y vs v9S4");
            if(policy==4&&path>0)exact(got,ref[0],"large actual v9 full Y");
        }
        if(!head){auto cpu=load<float>(dir+"/ref.bin",size_t(8)*n);for(auto& y:ref)tolerance(y,cpu);}
        weights_check();guards();
        std::cout<<"VALID name="<<name<<" phase="<<(first?"before":"after")<<" clones="<<clones<<" full_P_bytes="<<pref.size()<<" full_Y_values="<<8*n<<" real_head="<<head<<" PASS\n";
    };
    validate(true);
    hipEvent_t begin,end;ck(hipEventCreate(&begin));ck(hipEventCreate(&end));
    std::array<std::vector<double>,4> events,walls;
    for(int sweep=-1;sweep<11;++sweep)for(int j=0;j<4;++j){budget();int path=(sweep%2==0)?3-j:j;
        auto start=Clock::now();ck(hipEventRecord(begin,stream));for(size_t c=0;c<clones;++c)launch(path,c);ck(hipEventRecord(end,stream));ck(hipEventSynchronize(end));
        double wall=std::chrono::duration<double,std::micro>(Clock::now()-start).count()/clones;
        float ms;ck(hipEventElapsedTime(&ms,begin,end));double event=double(ms)*1000/clones;
        // Validate every last timed output before another submission.
        exact(dy.read(),ref[path],"Last timed full Y");guards();
        if(sweep>=0){events[path].push_back(event);walls[path].push_back(wall);std::cout<<"SAMPLE name="<<name<<" sweep="<<sweep<<" path="<<char('A'+path)<<" event_us="<<event<<" wall_us="<<wall<<" last_Y_exact=1\n";}
    }
    ck(hipEventDestroy(begin));ck(hipEventDestroy(end));validate(false);
    for(int path=0;path<4;++path){std::sort(events[path].begin(),events[path].end());std::sort(walls[path].begin(),walls[path].end());std::cout<<"RESULT name="<<name<<" path="<<char('A'+path)<<" clones="<<clones<<" packed_resident="<<clones*pbytes<<" event_min_us="<<events[path][0]<<" event_median_us="<<events[path][5]<<" event_p90_us="<<events[path][9]<<" wall_min_us="<<walls[path][0]<<" wall_median_us="<<walls[path][5]<<" wall_p90_us="<<walls[path][9]<<"\n";}
}
int main(int argc,char** argv){
    std::cout<<std::unitbuf<<std::setprecision(10);epoch=Clock::now();
    try{require(argc==2,"Expected private fixtures root");ck(hipSetDevice(0));ck(fo::resources());ck(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
        small();std::string root=argv[1];
        large(root+"/q_b","q_b",1280,32768,5,4,false);
        large(root+"/wo_b","wo_b",8192,5120,5,4,false);
        large(root+"/inverse","inverse",2304,5120,5,4,false);
        large(root+"/shared","shared",5120,2304,5,16,false);
        large(root+"/head","head",5120,129280,6,4,true);
        drain();ck(hipStreamDestroy(stream));std::cout<<"PASS private_G4 gpu_peak_bytes="<<peak<<" seconds="<<std::chrono::duration<double>(Clock::now()-epoch).count()<<"\n";return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<" gpu_peak_bytes="<<peak<<"\n";return 1;}
}
