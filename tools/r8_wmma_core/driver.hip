// R8-only host driver. Device math is included verbatim from qualified sources.
#include "../../include/strix_trellis.h"
#include "../../src/backend.hpp"
#include "../../src/wmma_decoder.hpp"
#include "../../src/gemm.hpp"
#include "../../src/launch_policy.hpp"
#include "../../src/fused_output_policy.hpp"
#define input_transform r8_input_transform
#define output_transform r8_output_transform
#include "../../src/transforms.hpp"
#undef input_transform
#undef output_transform
#include "private_layout.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sys/resource.h>
#include <vector>

namespace fs=std::filesystem;
namespace st=strix_trellis;
namespace wd=st::decoder;
using Clock=std::chrono::steady_clock;
static size_t live=0,peak=0;
static Clock::time_point began;
void require(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
void hc(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
void ac(strix_status e){if(e!=STRIX_SUCCESS)throw std::runtime_error(strix_status_string(e));}
void budget(){require(std::chrono::duration<double>(Clock::now()-began).count()<140,"Inner budget");}
template<class T>std::vector<T> read(const fs::path& path,size_t count){
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    require(bool(f)&&f.tellg()==std::streamoff(count*sizeof(T)),"File extent "+path.string());
    std::vector<T> v(count);f.seekg(0);f.read(reinterpret_cast<char*>(v.data()),count*sizeof(T));
    require(bool(f),"File read "+path.string());return v;
}
struct Buffer{
    void* base=nullptr;unsigned char* p;size_t bytes;
    Buffer(size_t n,hipStream_t s):bytes(n){
        require(n+512<(1ull<<30)-live,"Explicit GPU budget");hc(hipMalloc(&base,n+512));
        p=static_cast<unsigned char*>(base)+256;live+=n+512;peak=std::max(peak,live);
        hc(hipMemsetAsync(base,0xa5,n+512,s));
    }
    ~Buffer(){if(base){(void)hipFree(base);live-=bytes+512;}}
    Buffer(const Buffer&)=delete;
    void check(hipStream_t s)const{
        unsigned char g[512];hc(hipMemcpyAsync(g,base,256,hipMemcpyDeviceToHost,s));
        hc(hipMemcpyAsync(g+256,p+bytes,256,hipMemcpyDeviceToHost,s));hc(hipStreamSynchronize(s));
        for(auto b:g)require(b==0xa5,"Canary changed");
    }
};
void upload(void* p,const void* x,size_t n,hipStream_t s){hc(hipMemcpyAsync(p,x,n,hipMemcpyHostToDevice,s));}
void exact(const void* p,const void* ref,size_t n,hipStream_t s,const std::string& label){
    const auto* r=static_cast<const unsigned char*>(ref);std::vector<unsigned char> tmp(std::min(n,size_t(4<<20)));
    for(size_t at=0;at<n;at+=tmp.size()){
        size_t length=std::min(tmp.size(),n-at);hc(hipMemcpyAsync(tmp.data(),static_cast<const char*>(p)+at,length,hipMemcpyDeviceToHost,s));
        hc(hipStreamSynchronize(s));require(!std::memcmp(tmp.data(),r+at,length),label+" byte mismatch at chunk "+std::to_string(at));
    }
}
std::vector<float> download(const Buffer& b,hipStream_t s){
    std::vector<float> v(b.bytes/4);hc(hipMemcpyAsync(v.data(),b.p,b.bytes,hipMemcpyDeviceToHost,s));hc(hipStreamSynchronize(s));return v;
}
void compare(const std::vector<float>& got,const std::vector<float>& ref,int n,const std::string& label){
    require(got.size()==ref.size(),"Compare dimensions");double worst=0,nrms=0;
    for(size_t row=0;row<got.size()/n;++row){double ma=0,mr=0,e2=0,r2=0;
        for(int c=0;c<n;++c){size_t k=row*n+c;double a=got[k],b=ref[k],e=a-b;
            require(std::isfinite(a)&&std::isfinite(b),label+" nonfinite");ma=std::max(ma,std::abs(e));mr=std::max(mr,std::abs(b));e2+=e*e;r2+=b*b;}
        double r=std::sqrt(e2/std::max(r2,1e-30));require(ma<=5e-4*std::max(1.,mr)&&r<=3e-5,label+" full CPU tolerance");
        worst=std::max(worst,ma);nrms=std::max(nrms,r);
    }
    std::cout<<"NUMERIC label="<<label<<" values="<<got.size()<<" maxabs="<<worst<<" worst_row_nrms="<<nrms<<" PASS\n";
}
struct Runtime{
    hipStream_t s=nullptr;strix_stream stream{};strix_context ctx=nullptr;
    Runtime(){hc(hipSetDevice(0));hc(hipStreamCreateWithFlags(&s,hipStreamNonBlocking));
        stream.struct_size=sizeof(stream);stream.abi_version=1;stream.hip_stream=s;
        strix_context_desc d{};d.struct_size=sizeof(d);d.abi_version=1;ac(strix_context_create(&d,&ctx));
        hipDeviceProp_t p{};hc(hipGetDeviceProperties(&p,0));require(std::string(p.gcnArchName).find("gfx1151")==0&&p.warpSize==32,"Hardware");
        std::cout<<"HARDWARE name="<<p.name<<" arch="<<p.gcnArchName<<" wave="<<p.warpSize<<" WGP="<<p.multiProcessorCount<<" memory="<<p.totalGlobalMem<<" nominal_clock_kHz="<<p.clockRate<<" nominal_mem_kHz="<<p.memoryClockRate<<"\n";
    }
    ~Runtime(){if(s)(void)hipStreamSynchronize(s);if(ctx)(void)strix_context_destroy(ctx);if(s)(void)hipStreamDestroy(s);}
};
struct Data{
    int i,n,b,bias;size_t wb,ab,pb,yb,ws;std::vector<unsigned char> blob,a;
    std::vector<float>x,y,old,cuda;std::vector<double> oracle;
    Data(const fs::path& dir){
        std::ifstream f(dir/"meta.txt");require(bool(f>>i>>n>>b>>bias),"Metadata");
        require((i==1280&&n==32768&&b==5)||(i==8192&&n==5120&&b==5)||(i==5120&&n==129280&&(b==5||b==6)),"R8 geometry");
        wb=size_t(i)*n*b/8;ab=16ull*i;pb=4ull*8*n*4;yb=8ull*n*4;ws=((ab+255)&~size_t(255))+pb;
        blob.resize(wb+2ull*(i+n)+(bias?4ull*n:0));size_t at=0;
        std::vector<std::pair<std::string,size_t>> fields={{"packed",wb},{"u",2ull*i},{"v",2ull*n}};
        if(bias)fields.emplace_back("bias",4ull*n);
        for(auto field:fields){std::ifstream part(dir/(field.first+".bin"),std::ios::binary|std::ios::ate);
            require(bool(part)&&part.tellg()==std::streamoff(field.second),"Canonical field extent");part.seekg(0);
            part.read(reinterpret_cast<char*>(blob.data()+at),field.second);require(bool(part),"Canonical field read");at+=field.second;}
        require(at==blob.size(),"Canonical blob extent");
        a=read<unsigned char>(dir/"a.bin",ab);x=read<float>(dir/"x.bin",8ull*i);y=read<float>(dir/"y.bin",8ull*n);
        oracle=read<double>(dir/"partial-fp64.bin",4ull*2*8*n);
        if(fs::exists(dir/"cuda-y.bin"))cuda=read<float>(dir/"cuda-y.bin",8ull*n);
        old.resize(y.size());for(size_t k=0;k<old.size();++k)old[k]=(int(k%41)-20)*.015625f;
    }
};
struct Weight{
    Runtime& r;strix_weights handle=nullptr;size_t bytes;
    Weight(Runtime& rt,const Data& f):r(rt),bytes(f.blob.size()){
        require(bytes<(1ull<<30)-live,"Weights GPU budget");
        strix_exl3_metadata_v1 m{};m.struct_size=sizeof(m);m.transport_version=1;m.rows=f.n;m.columns=f.i;
        m.bits_x2=2*f.b;m.codebook=STRIX_CODEBOOK_MUL1;m.marker=0x83dcd12d;m.trellis_bytes=f.wb;
        m.suh_offset=f.wb;m.suh_bytes=2ull*f.i;m.svh_offset=f.wb+2ull*f.i;m.svh_bytes=2ull*f.n;
        if(f.bias){m.bias_offset=f.wb+2ull*(f.i+f.n);m.bias_bytes=4ull*f.n;}
        strix_exl3_transport_desc d{};d.struct_size=sizeof(d);d.abi_version=1;d.format=STRIX_TRANSPORT_EXL3_CANONICAL;
        d.transport_version=1;d.rows=f.n;d.columns=f.i;d.metadata=&m;d.metadata_bytes=sizeof(m);d.data=f.blob.data();d.data_bytes=bytes;
        ac(strix_weights_prepare(r.ctx,&d,&r.stream,&handle));live+=handle->allocation_bytes;peak=std::max(peak,live);
        require(live<(1ull<<30),"Weights actual budget");require(handle->owner==r.ctx,"Actual owner");
        require(handle->device.fused_output_capable==r.ctx->fused_output_capable,"Actual owner capability");
    }
    ~Weight(){if(handle){(void)hipStreamSynchronize(r.s);live-=handle->allocation_bytes;(void)strix_weights_destroy(handle);}}
    void check(const Data& f){auto& w=handle->device;exact(w.trellis,f.blob.data(),f.wb,r.s,"Packed immutable");
        exact(w.suh,f.blob.data()+f.wb,2ull*f.i,r.s,"suh immutable");exact(w.svh,f.blob.data()+f.wb+2ull*f.i,2ull*f.n,r.s,"svh immutable");
        if(f.bias)exact(w.bias,f.blob.data()+f.wb+2ull*(f.i+f.n),4ull*f.n,r.s,"bias immutable");}
};
struct Region{const void* p;size_t bytes,alignment;};
bool regions(const std::vector<Region>& v,int device){
    for(size_t k=0;k<v.size();++k){auto r=v[k];uintptr_t at=reinterpret_cast<uintptr_t>(r.p);
        if(!at||at%r.alignment||r.bytes>UINTPTR_MAX-at)return false;
        for(size_t j=0;j<k;++j){uintptr_t q=reinterpret_cast<uintptr_t>(v[j].p);if(at<q?r.bytes>q-at:v[j].bytes>at-q)return false;}
        hipPointerAttribute_t a{};if(hipPointerGetAttributes(&a,r.p)!=hipSuccess){(void)hipGetLastError();return false;}
        if(a.type!=hipMemoryTypeDevice||a.isManaged||a.device!=device)return false;
        void* base=nullptr;size_t size=0;if(hipMemGetAddressRange(&base,&size,const_cast<void*>(r.p))!=hipSuccess){(void)hipGetLastError();return false;}
        uintptr_t origin=reinterpret_cast<uintptr_t>(base);if(at<origin||at-origin>size||r.bytes>size-(at-origin))return false;
    }return true;
}
struct Operator final{
    Runtime& r;const Data& f;Weight& w;const Buffer& x;Buffer y,ws;int mode;
    std::unique_ptr<const wd::Token> token;strix_matmul_desc desc{};
    Operator(Runtime& rt,const Data& data,Weight& weight,const Buffer& input,int which):r(rt),f(data),w(weight),x(input),y(f.yb,r.s),ws(f.ws,r.s),mode(which){
        const auto& dw=w.handle->device;int device=-1;hc(hipGetDevice(&device));
        require(st::wmma_tile_rows(dw,8)==16,"Production R8 M16");
        std::vector<Region> v={{x.p,x.bytes,4},{y.p,y.bytes,4},{ws.p,ws.bytes,256},{dw.trellis,f.wb,4},{dw.suh,2ull*f.i,2},{dw.svh,2ull*f.n,2}};
        if(f.bias)v.push_back({dw.bias,4ull*f.n,4});require(regions(v,device),"Operator storage");
        auto bad=v;bad[0].bytes=SIZE_MAX;require(!regions(bad,device),"Capacity overflow accepted");
        bad=v;bad[1]=bad[0];require(!regions(bad,device),"Alias accepted");bad=v;bad[0].p=x.p+1;require(!regions(bad,device),"Alignment accepted");
        bad=v;bad[0].bytes=x.bytes+257;require(!regions(bad,device),"Allocation extent accepted");
        require(!regions(v,device+1),"Foreign device accepted");
        int sd=-1;unsigned flags=0;hc(hipStreamGetDevice(r.s,&sd));hc(hipStreamGetFlags(r.s,&flags));require(sd==device,"Stream device");
        wd::Request request{dw,{dw.trellis,f.wb},{ws.p,f.ab},{ws.p+f.ab,f.pb},8,4,f.i,f.n,8,wd::Layout::m16_a16_plain,wd::Mode(std::max(0,mode-1)),r.s};
        auto reject=[&](wd::Request b){std::unique_ptr<const wd::Token> t;require(wd::prepare(b,t)!=hipSuccess&&!t,"Core rejected before enqueue");(void)hipGetLastError();};
        auto b=request;b.a.capacity--;reject(b);b=request;b.partial.capacity--;reject(b);b=request;b.packed.capacity--;reject(b);
        b=request;b.total_rows=7;reject(b);b=request;b.stream=nullptr;reject(b);b=request;b.a_stride++;reject(b);b=request;b.partial_stride++;reject(b);
        if(mode>=2)hc(wd::prepare(request,token));
        strix_matmul_desc_init(&desc);desc.weights=w.handle;
        auto tensor=[](void* p,size_t bytes,int width){strix_tensor_desc t{};t.struct_size=sizeof(t);t.abi_version=1;t.dtype=STRIX_DTYPE_F32;t.rank=2;t.data=p;t.buffer_bytes=bytes;t.shape[0]=8;t.shape[1]=width;t.strides[0]=width;t.strides[1]=1;return t;};
        desc.input=tensor(x.p,x.bytes,f.i);desc.output=tensor(y.p,y.bytes,f.n);desc.workspace=ws.p;desc.workspace_bytes=ws.bytes;
        uint64_t need=0;ac(strix_matmul_workspace_bytes(r.ctx,&desc,&need));require(need==ws.bytes,"Actual V13 workspace");reset();
    }
    void reset(float beta=0){hc(hipMemsetAsync(ws.p,0xff,ws.bytes,r.s));if(beta==0)hc(hipMemsetAsync(y.p,0xff,y.bytes,r.s));else upload(y.p,f.old.data(),y.bytes,r.s);}
    void submit(bool full,float alpha=1,float beta=0){
        const auto& dw=w.handle->device;
        if(mode==0){if(!full)throw std::runtime_error("No GEMV core timing");hc(st::launch_linear(dw,reinterpret_cast<const float*>(x.p),reinterpret_cast<float*>(y.p),8,ws.p,r.s,alpha,beta,false));return;}
        if(full){st::r8_input_transform<<<(8ull*(f.i/128)+3)/4,128,0,r.s>>>(reinterpret_cast<const float*>(x.p),dw.suh,reinterpret_cast<_Float16*>(ws.p),f.i,8);hc(hipGetLastError());}
        if(mode==1)hc(st::launch_wmma_core(dw,reinterpret_cast<_Float16*>(ws.p),reinterpret_cast<float*>(ws.p+f.ab),8,4,r.s));else hc(token->submit());
        if(full){st::r8_output_transform<<<(8ull*(f.n/128)+3)/4,128,0,r.s>>>(reinterpret_cast<const float*>(ws.p+f.ab),dw.svh,dw.bias,reinterpret_cast<float*>(y.p),f.n,8,4,alpha,beta);hc(hipGetLastError());}
    }
    void public_identity(float alpha=1,float beta=0){
        reset(beta);submit(true,alpha,beta);auto expected=download(y,r.s);reset(beta);
        desc.alpha=alpha;desc.beta=beta;ac(strix_gemv(r.ctx,&desc,&r.stream));exact(y.p,expected.data(),y.bytes,r.s,"Public actual V13 launch identity");
    }
    void immutable(){exact(ws.p,f.a.data(),f.ab,r.s,"A all bytes");x.check(r.s);y.check(r.s);ws.check(r.s);}
};
void partial_gate(const std::vector<float>& p,const Data& f){
    size_t plane=8ull*f.n;double ratio=0;
    for(int z=0;z<4;++z){std::vector<float> actual(plane),ref(plane);
        for(size_t k=0;k<plane;++k){double v=p[z*plane+k],expected=f.oracle[(z*2)*plane+k],mass=f.oracle[(z*2+1)*plane+k];
            require(std::isfinite(v)&&std::isfinite(expected)&&std::isfinite(mass)&&mass>=0,"FP64 partial finite");
            ratio=std::max(ratio,std::abs(v-expected)/std::max(mass,1e-30));actual[k]=v;ref[k]=expected;}
        compare(actual,ref,f.n,"S4_PARTIAL_CPU");
    }
    std::cout<<"GATE independent_FP64_sumabs values="<<p.size()<<" max_error_over_mass="<<ratio<<" PASS\n";
}
void print_stats(int geom,const char* scope,int mode,const char* clock,const std::vector<double>& raw,size_t clones,size_t packed){
    auto sorted=raw;std::sort(sorted.begin(),sorted.end());
    std::cout<<"RESULT geometry="<<geom<<" scope="<<scope<<" mode="<<mode<<" clock="<<clock<<" clones="<<clones<<" cold_bytes="<<clones*packed<<" min_us="<<sorted[0]<<" median_us="<<sorted[5]<<" p90_us="<<sorted[9]<<" raw=";
    for(auto t:raw)std::cout<<t<<',';std::cout<<'\n';
}
void run(Runtime& r,const fs::path& dir,int geom){
    budget();Data f(dir);Buffer x(f.x.size()*4,r.s);upload(x.p,f.x.data(),x.bytes,r.s);
    size_t clones=((256ull<<20)+f.wb-1)/f.wb;
    std::vector<std::unique_ptr<Weight>> weights;std::array<std::vector<std::unique_ptr<Operator>>,4> ops;
    for(size_t c=0;c<clones;++c){weights.emplace_back(new Weight(r,f));for(int mode=0;mode<4;++mode)ops[mode].emplace_back(new Operator(r,f,*weights.back(),x,mode));}
    auto& dw=weights[0]->handle->device;int groups=st::fused_output_groups(f.i,f.n,f.b,2,8,4,false,dw.fused_output_capable,true);
    require(groups==(geom==0?1:geom==1?4:geom==2?2:0),"Real V13 G1/G4/G2/generic selection");
    std::cout<<"GEOMETRY index="<<geom<<" I="<<f.i<<" N="<<f.n<<" bits="<<f.b<<" rows=8 S=4 layout=16/16/plain clones="<<clones<<" actual_owner_capable="<<dw.fused_output_capable<<" actual_groups="<<groups<<" peak_GPU="<<peak<<"\n";
    auto& control=*ops[1][0];control.submit(true);auto py=download(control.y,r.s);
    std::vector<float> pp(f.pb/4);hc(hipMemcpyAsync(pp.data(),control.ws.p+f.ab,f.pb,hipMemcpyDeviceToHost,r.s));hc(hipStreamSynchronize(r.s));
    partial_gate(pp,f);compare(py,f.y,f.n,"PRODUCTION_WMMA_CPU");
    std::vector<float> baseline_y;
    std::vector<unsigned char> direct_poison(groups?f.pb:0,0xff);
    auto check_all=[&](bool full,bool numeric,const char* phase){
        for(int m=full?0:1;m<4;++m)for(size_t c=0;c<clones;++c){auto& op=*ops[m][c];
            if(m){exact(op.ws.p+f.ab,pp.data(),f.pb,r.s,"All S4 P production WMMA");if(full)exact(op.y.p,py.data(),f.yb,r.s,"All Y production WMMA");}
            else if(full){exact(op.y.p,baseline_y.data(),f.yb,r.s,"All baseline Y deterministic bytes");if(groups)exact(op.ws.p+f.ab,direct_poison.data(),f.pb,r.s,"Actual direct baseline did not touch global P");}
            if(full&&numeric){auto got=download(op.y,r.s);compare(got,f.y,f.n,"ALL_Y_CPU");if(!f.cuda.empty())compare(got,f.cuda,f.n,"ALL_Y_CACHED_CUDA");}
            op.immutable();
        }
        std::cout<<"GATE phase="<<phase<<" all_clones="<<clones<<" full="<<full<<" ALL_P_Y_A_canaries=PASS\n";
    };
    for(int m=0;m<4;++m)for(auto& op:ops[m]){op->reset();op->submit(true);}
    for(auto& op:ops[0])op->public_identity();
    baseline_y=download(ops[0][0]->y,r.s);
    // Prove the actual launch stage count without timing graphs or C API checks.
    for(int mode=0;mode<4;++mode){
        hipGraph_t graph=nullptr;hc(hipStreamBeginCapture(r.s,hipStreamCaptureModeThreadLocal));
        ops[mode][0]->submit(true);hc(hipStreamEndCapture(r.s,&graph));size_t nodes=0;hc(hipGraphGetNodes(graph,nullptr,&nodes));
        require(nodes==size_t(mode==0&&groups?2:3),"Actual full operator stage count");
        std::cout<<"GATE actual_launch_stages mode="<<mode<<" nodes="<<nodes<<" PASS\n";hc(hipGraphDestroy(graph));
    }
    auto immutable=[&](){exact(x.p,f.x.data(),x.bytes,r.s,"Immutable original X");for(auto& w:weights)w->check(f);for(auto& set:ops)for(auto& op:set)op->immutable();};
    check_all(true,true,"before");immutable();
    // Alpha/beta gates are outside events, each starts from the same original Y.
    for(auto ab:std::array<std::pair<float,float>,5>{{{1,-0.f},{-0.f,0.f},{0.f,-.5f},{-.75f,.25f},{-.75f,-.5f}}}){
        std::vector<float> expected=f.y;for(size_t k=0;k<expected.size();++k){volatile float av=ab.first*f.y[k];if(ab.second!=0){volatile float bv=ab.second*f.old[k];expected[k]=av+bv;}else expected[k]=av;}
        std::vector<float> exact_y;
        for(int m=0;m<4;++m){auto& op=*ops[m][0];op.reset(ab.second);op.submit(true,ab.first,ab.second);auto got=download(op.y,r.s);compare(got,expected,f.n,"SCALARS_CPU");if(m==1)exact_y=got;if(m>1)require(!std::memcmp(got.data(),exact_y.data(),f.yb),"Scalar candidate parity");if(!m)op.public_identity(ab.first,ab.second);}
    }
    for(auto& set:ops)for(auto& op:set){op->reset();op->submit(true);}
    hipEvent_t start{},stop{};hc(hipEventCreate(&start));hc(hipEventCreate(&stop));
    for(bool full:{true,false}){
        std::array<std::vector<double>,4> event,host;
        int count=full?4:3,first=full?0:1;
        for(int m=first;m<4;++m)for(auto& op:ops[m])op->submit(full);hc(hipStreamSynchronize(r.s));
        for(int sweep=0;sweep<11;++sweep){budget();
            for(int pos=0;pos<count;++pos){int mode=first+(sweep+(sweep%2?-pos:pos)+count)%count;
                hc(hipEventRecord(start,r.s));auto begin=Clock::now();
                for(size_t c=0;c<clones;++c)ops[mode][(c+sweep)%clones]->submit(full);
                hc(hipEventRecord(stop,r.s));hc(hipEventSynchronize(stop));
                host[mode].push_back(std::chrono::duration<double,std::micro>(Clock::now()-begin).count()/clones);
                float ms=0;hc(hipEventElapsedTime(&ms,start,stop));event[mode].push_back(ms*1000./clones);
            }
            check_all(full,false,sweep==10?"LAST_MEASURED_SWEEP_ALL_CLONES":"sweep");
        }
        check_all(full,true,"after");immutable();
        for(int m=first;m<4;++m){print_stats(geom,full?"full":"core",m,"event",event[m],clones,f.wb);print_stats(geom,full?"full":"core",m,"host",host[m],clones,f.wb);}
    }
    for(auto& op:ops[0])op->public_identity();check_all(true,true,"last_replay");immutable();
    hc(hipEventDestroy(start));hc(hipEventDestroy(stop));
    rusage usage{};require(getrusage(RUSAGE_SELF,&usage)==0&&uint64_t(usage.ru_maxrss)*1024<(2ull<<30),"Host RSS budget");
    std::cout<<"PASS_GEOMETRY index="<<geom<<" peak_host_bytes="<<uint64_t(usage.ru_maxrss)*1024<<" peak_device_bytes="<<peak<<'\n';
}
int main(int argc,char** argv){try{
    std::cout<<std::unitbuf<<std::setprecision(12);require(argc==2,"Usage: r8-driver DATA_ROOT (GPU lease only)");began=Clock::now();
    static_assert(sizeof(st::DeviceWeights)==56&&offsetof(st::DeviceWeights,fused_output_capable)==48);
    Runtime r;for(int g=0;g<4;++g)run(r,fs::path(argv[1])/std::to_string(g),g);budget();
    std::cout<<"PASS_R8_CAMPAIGN four_geometries=1 sweeps=11 elapsed="<<std::chrono::duration<double>(Clock::now()-began).count()<<" peak_GPU="<<peak<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
