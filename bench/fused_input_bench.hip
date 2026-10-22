// Optional R1 fusion: staged fixtures, exact decode, and cold full-operator timing.
#include "../src/gemv_fused.hpp"
#include "../src/gemv_lds.hpp"
#include "../src/transforms.hpp"
#include "../src/trellis_optimized.hpp"
#include "../src/host_reference.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cstring>
using namespace strix_trellis;
#define HIP(call) do {hipError_t e=(call);if(e!=hipSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(2);}} while(0)
static uint32_t rng=0x723f281a;
static uint32_t random_word(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
__global__ void validate_decode(const uint32_t* p,uint16_t* states,uint16_t* values,int bits,int cb) {
    int t=threadIdx.x&31,base=blockIdx.x*256+t*8;
    #define DECODE(B) case B: {optimized::AlignedLane<B,true> tile(p+blockIdx.x*B*8); \
      states[base+0]=tile.template state<0>();states[base+1]=tile.template state<1>(); \
      states[base+2]=tile.template state<2>();states[base+3]=tile.template state<3>(); \
      states[base+4]=tile.template state<4>();states[base+5]=tile.template state<5>(); \
      states[base+6]=tile.template state<6>();states[base+7]=tile.template state<7>();break;}
    switch(bits){DECODE(1);DECODE(2);DECODE(3);DECODE(4);DECODE(5);DECODE(6);DECODE(7);DECODE(8);}
    #undef DECODE
    for(int j=0;j<8;j+=2){
        if(cb==2){__half2 h=optimized::mul1_half2(states[base+j],states[base+j+1]);uint32_t v=__builtin_bit_cast(uint32_t,h);values[base+j]=v;values[base+j+1]=v>>16;}
        else for(int d=0;d<2;++d)values[base+j+d]=__half_as_ushort(__float2half_rn(codebook_value(states[base+j+d],cb)));
    }
}
__global__ void exhaustive_books(uint16_t* out) {
    unsigned s=blockIdx.x*blockDim.x+threadIdx.x;
    for(int cb=0;cb<2;++cb)out[cb*65536+s]=__half_as_ushort(__float2half_rn(codebook_value(s,cb)));
    uint32_t v=__builtin_bit_cast(uint32_t,optimized::mul1_half2(s,65535-s));
    out[2*65536+s]=v;out[3*65536+s]=v>>16;
}
static void decode_checks(){
    uint16_t* d;HIP(hipMalloc(&d,4*65536*2));exhaustive_books<<<256,256>>>(d);
    std::vector<uint16_t> h(4*65536);HIP(hipMemcpy(h.data(),d,h.size()*2,hipMemcpyDeviceToHost));HIP(hipFree(d));
    for(int cb=0;cb<4;++cb)for(int s=0;s<65536;++s)
        if(h[cb*65536+s]!=host_codebook(cb==3?65535-s:s,std::min(cb,2))){fprintf(stderr,"codebook mismatch\n");exit(3);}
    for(int bits=1;bits<=8;++bits){
        const int tiles=257;std::vector<uint32_t> p(tiles*bits*8);for(auto& v:p)v=random_word();
        uint32_t* dp;uint16_t *ds,*dv;HIP(hipMalloc(&dp,p.size()*4));HIP(hipMalloc(&ds,tiles*256*2));HIP(hipMalloc(&dv,tiles*256*2));HIP(hipMemcpy(dp,p.data(),p.size()*4,hipMemcpyHostToDevice));
        std::vector<uint16_t> ss(tiles*256),vv(tiles*256);
        for(int cb=0;cb<3;++cb){validate_decode<<<tiles,32>>>(dp,ds,dv,bits,cb);HIP(hipMemcpy(ss.data(),ds,ss.size()*2,hipMemcpyDeviceToHost));HIP(hipMemcpy(vv.data(),dv,vv.size()*2,hipMemcpyDeviceToHost));
            for(int t=0;t<tiles;++t)for(int j=0;j<256;++j){auto s=host_state(p.data()+t*bits*8,bits,j);if(ss[t*256+j]!=s||vv[t*256+j]!=host_codebook(s,cb)){fprintf(stderr,"decode mismatch K%d cb%d\n",bits,cb);exit(3);}}
        }HIP(hipFree(dp));HIP(hipFree(ds));HIP(hipFree(dv));
    }
    puts("{\"validation\":\"all_65536_states_all_codebooks_both_mul1_halves_and_257_tiles_each_K1_to_8_cb0_to_2\",\"pass\":true}");
}

#include <fstream>
#include <map>
#include <string>
#include <stdexcept>
// All device buffers are scoped per fixture; expanded Q stays on the host.
struct Buffer {
    void* p=nullptr; size_t bytes;
    explicit Buffer(size_t n):bytes(n) { if(n) HIP(hipMalloc(&p,n)); }
    ~Buffer(){if(p)(void)hipFree(p);}
    template<class T> T* as(){return static_cast<T*>(p);}
    void upload(const void* v,size_t n){HIP(hipMemcpy(p,v,n,hipMemcpyHostToDevice));}
};
static uint64_t le(const unsigned char* p, int n) {
    uint64_t v = 0;
    for (int i = 0; i < n; ++i) v |= uint64_t(p[i]) << (8*i);
    return v;
}
struct Fixture {
    std::map<std::string, std::vector<unsigned char>> parts;
    uint32_t i, n, bits, cb, marker, rows, bias, seed;
    explicit Fixture(const std::string& path) {
        const uint16_t endian = 1;
        if (*reinterpret_cast<const unsigned char*>(&endian) != 1)
            throw std::runtime_error("Little endian host required");
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f || f.tellg() < 16 || f.tellg() > 256 * 1024 * 1024)
            throw std::runtime_error("Invalid fixture size/path: " + path);
        std::vector<unsigned char> bytes(static_cast<size_t>(f.tellg()));
        f.seekg(0); f.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
        if (!f || std::memcmp(bytes.data(), "STGPU001", 8) || le(bytes.data()+8,4)!=1)
            throw std::runtime_error("Invalid fixture header");
        size_t pos = 16;
        const uint64_t count = le(bytes.data()+12,4);
        if (count > 64) throw std::runtime_error("Too many sections");
        for (uint64_t k=0; k<count; ++k) {
            if (bytes.size()-pos < 16) throw std::runtime_error("Truncated section");
            std::string tag(reinterpret_cast<char*>(bytes.data()+pos), 8);
            tag.resize(tag.find('\0') == std::string::npos ? 8 : tag.find('\0'));
            const uint64_t len = le(bytes.data()+pos+8,8); pos += 16;
            if (len > bytes.size()-pos || parts.count(tag)) throw std::runtime_error("Invalid section");
            parts.emplace(tag, std::vector<unsigned char>(bytes.begin()+pos, bytes.begin()+pos+len));
            pos += static_cast<size_t>(len);
        }
        if (pos != bytes.size()) throw std::runtime_error("Trailing fixture bytes");
        require("meta",32);
        const auto* m = parts.at("meta").data();
        i=le(m,4); n=le(m+4,4); bits=le(m+8,4); cb=le(m+12,4);
        marker=le(m+16,4); rows=le(m+20,4); bias=le(m+24,4); seed=le(m+28,4);
        const uint32_t markers[] = {0,0xCBAC1FEDu,0x83DCD12Du};
        if (!i || !n || i%128 || n%128 || i>8192 || n>131072 || !rows || rows>256 ||
            bits<1 || bits>8 || cb>2 || bias>1 || marker!=markers[cb])
            throw std::runtime_error("Invalid fixture metadata/marker");
        require("trellis",size_t(i)*n*bits/8); require("suh",2*i); require("svh",2*n);
        require("bias",bias?4*n:0); require("x",size_t(rows)*i*4);
        require("old",size_t(rows)*n*4); require("ref",size_t(rows)*n*4);
        require("scaled",size_t(rows)*i*2); require("inh",size_t(rows)*i*4);
        require("a",size_t(rows)*i*2); require("q",size_t(i)*n*2);
        require("core",size_t(rows)*n*4);
    }
    void require(const std::string& key, size_t bytes) const {
        auto it=parts.find(key);
        if (it==parts.end() || it->second.size()!=bytes) throw std::runtime_error("Bad section: "+key);
    }
    std::vector<float> floats(const std::string& key) const {
        const auto& p=parts.at(key); std::vector<float> v(p.size()/4);
        std::memcpy(v.data(),p.data(),p.size()); return v;
    }
};
static void exact(Buffer& d,const Fixture& f,const char* tag,size_t bytes) {
    std::vector<unsigned char> v(bytes);HIP(hipMemcpy(v.data(),d.p,bytes,hipMemcpyDeviceToHost));
    if(memcmp(v.data(),f.parts.at(tag).data(),bytes)) {
        size_t k=0;while(k<bytes&&v[k]==f.parts.at(tag)[k])++k;
        fprintf(stderr,"checkpoint %s mismatch byte %zu\n",tag,k);exit(3);
    }
}
static void metrics(const std::vector<float>& got,const std::vector<float>& ref,int n,const char* label) {
    double e2=0,r2=0,ma=0,mr=0;
    for(int k=0;k<n;++k){double e=double(got[k])-ref[k];if(!std::isfinite(got[k]))exit(3);e2+=e*e;r2+=double(ref[k])*ref[k];ma=std::max(ma,std::abs(e));mr=std::max(mr,std::abs(double(ref[k])));}
    double nr=std::sqrt(e2/std::max(r2,1e-30));
    printf("metric=%s maxabs=%.9g maxref=%.9g nrmse=%.9g\n",label,ma,mr,nr);
    if(ma>5e-4*std::max(1.0,mr)||nr>3e-5){fprintf(stderr,"numerical tolerance failed\n");exit(3);}
}
static void fixture(const char* path,bool benchmark) {
    Fixture f(path); const int I=f.i,N=f.n;size_t bytes=f.parts.at("trellis").size();
    // Use random original row 1 for large fixtures; wkv includes the RN32/RN16 tie.
    if(I>256 && f.rows>1) {
        for(const char* tag:{"x","scaled","inh","a","old","core","ref"}) {
            size_t stride=(std::string(tag)=="x"||std::string(tag)=="inh")?size_t(I)*4:
                (std::string(tag)=="scaled"||std::string(tag)=="a")?size_t(I)*2:size_t(N)*4;
            auto& part=f.parts.at(tag);part.erase(part.begin(),part.begin()+stride);
        }
        --f.rows;
    }
    int clones=benchmark?int(((size_t(256)<<20)+bytes-1)/bytes):1;
    Buffer p(bytes*clones),u(I*2),v(N*2),bias(f.bias?N*4:0),x(I*4),a(I*2),h(I*4),da(I*2),partial(size_t(32)*3*N*4),out(N*4);
    p.upload(f.parts.at("trellis").data(),bytes);
    for(int c=1;c<clones;++c)HIP(hipMemcpy(static_cast<char*>(p.p)+size_t(c)*bytes,p.p,bytes,hipMemcpyDeviceToDevice));
    u.upload(f.parts.at("suh").data(),I*2);v.upload(f.parts.at("svh").data(),N*2);
    if(f.bias)bias.upload(f.parts.at("bias").data(),N*4);
    x.upload(f.parts.at("x").data(),I*4);
    DeviceWeights w{p.as<uint32_t>(),u.as<uint16_t>(),v.as<uint16_t>(),bias.as<float>(),I,N,int(f.bits),int(f.cb)};
    printf("fixture=%s I=%d N=%d K=%u cb=%u R=1 clones=%d resident_bytes=%zu pointer_validation=direct_backend_not_CAPI\n",path,I,N,f.bits,f.cb,clones,bytes*clones);
    // Validation must reject unsupported contracts before any kernel submission.
    if(launch_gemv_fused_input(w,nullptr,out.as<float>(),1,1,1,nullptr)!=hipErrorInvalidValue ||
       launch_gemv_fused_input(w,x.as<float>(),out.as<float>(),2,2,1,nullptr)!=hipErrorInvalidValue ||
       launch_gemv_fused_input(w,x.as<float>(),out.as<float>(),1,1,0,nullptr)!=hipErrorInvalidValue)exit(3);
    for(int sp:{1,3,4,16,32}) {
        std::vector<float> sentinel(size_t(sp)*3*N,1234567);
        partial.upload(sentinel.data(),sentinel.size()*4);
        HIP(launch_gemv_fused_input_debug(w,x.as<float>(),partial.as<float>()+N,1,3,sp,nullptr,da.as<_Float16>(),h.as<float>()));
        exact(da,f,"a",I*2);exact(h,f,"inh",I*4);
        HIP(hipMemcpy(sentinel.data(),partial.p,sentinel.size()*4,hipMemcpyDeviceToHost));
        // Each H-aligned partial checked independently against exact FP64 products.
        auto* av=reinterpret_cast<const uint16_t*>(f.parts.at("a").data());
        auto* qv=reinterpret_cast<const uint16_t*>(f.parts.at("q").data());
        for(int s=0;s<sp;++s)for(int n=0;n<N;++n)
            if(sentinel[size_t(s)*3*N+n]!=1234567||sentinel[(size_t(s)*3+2)*N+n]!=1234567)exit(3);
        double maxerr=0,maxnorm=0;
        for(int s=0;s<sp;++s)for(int sample=0;sample<(I>256?32:N);++sample) {
            int n=I>256?int((uint64_t(sample)*7919+11)%N):sample;
            double ref=0,ab=0;int begin=(I/128)*s/sp*128,end=(I/128)*(s+1)/sp*128;
            for(int k=begin;k<end;++k){double prod=double(host_half(av[k]))*host_half(qv[size_t(k)*N+n]);ref+=prod;ab+=std::abs(prod);}
            float got=sentinel[(size_t(s)*3+1)*N+n];
            double err=std::abs(double(got)-ref),gu=2.0*I*std::ldexp(1.0,-24),bound=gu/(1-gu)*ab+1e-6;
            if(!std::isfinite(got)||err>bound||(begin==end&&(got!=0||std::signbit(got)))){fprintf(stderr,"partial mismatch split%d n%d got%.9g ref%.17g\n",s,n,got,ref);exit(3);}
            maxerr=std::max(maxerr,err);maxnorm=std::max(maxnorm,err/std::max(ab,1e-300));
        }
        // Both reduction trees, exact same output transform, fixture alpha/beta.
        for(bool fused:{false,true})for(int ab=0;ab<2;++ab) {
            float alpha=ab?-.75f:1.f,beta=ab?.25f:0.f;
            out.upload(f.parts.at("old").data(),N*4);
            if(fused)HIP(launch_gemv_fused_input(w,x.as<float>(),partial.as<float>(),1,1,sp,nullptr));
            else { input_transform<<<(I/128+3)/4,128>>>(x.as<float>(),u.as<uint16_t>(),a.as<_Float16>(),I,1);HIP(hipGetLastError());HIP(launch_gemv_lds_core(w,a.as<_Float16>(),partial.as<float>(),1,1,sp,nullptr)); }
            output_transform<<<(N/128+3)/4,128>>>(partial.as<float>(),v.as<uint16_t>(),bias.as<float>(),out.as<float>(),N,1,sp,alpha,beta);HIP(hipGetLastError());
            std::vector<float> got(N),ref=f.floats("ref"),old=f.floats("old");
            HIP(hipMemcpy(got.data(),out.p,N*4,hipMemcpyDeviceToHost));
            for(int n=0;n<N;++n){volatile float scaled=ref[n]*alpha;volatile float prior=old[n]*beta;ref[n]=scaled+prior;}
            metrics(got,ref,N,fused?"fused_full":"baseline_full");
        }
        printf("PASS splits=%d checkpoint_A=exact checkpoint_H32=exact core_maxabs=%.9g core_maxnorm=%.9g sentinels=pass empty_positive_zero=pass\n",sp,maxerr,maxnorm);
    }
    // Explicit regression uses the real fixture values when present.
    if(I==5120&&N==512) {
        auto xx=f.floats("x");auto* uu=reinterpret_cast<const uint16_t*>(f.parts.at("suh").data());
        printf("tie_regression channel576 x=%.17g suh=%.17g\n",xx[576],host_half(uu[576]));
        if(xx[576]!=float(0.2001299113035202)||host_half(uu[576])!=float(0.02642822265625))exit(3);
    }
    if(!benchmark)return;
    // Same-turn paired whole rotations, alternating order to limit drift bias.
    for(int sp:{4,16,32}) {
        std::vector<float> times[2];hipEvent_t start,stop;HIP(hipEventCreate(&start));HIP(hipEventCreate(&stop));
        for(int rep=-1;rep<11;++rep)for(int order=0;order<2;++order) {
            int fused=order^((rep+1)&1);HIP(hipEventRecord(start));
            for(int c=0;c<clones;++c) {
                w.trellis=p.as<uint32_t>()+size_t(c)*bytes/4;
                if(fused)HIP(launch_gemv_fused_input(w,x.as<float>(),partial.as<float>(),1,1,sp,nullptr));
                else {input_transform<<<(I/128+3)/4,128>>>(x.as<float>(),u.as<uint16_t>(),a.as<_Float16>(),I,1);HIP(hipGetLastError());HIP(launch_gemv_lds_core(w,a.as<_Float16>(),partial.as<float>(),1,1,sp,nullptr));}
                output_transform<<<(N/128+3)/4,128>>>(partial.as<float>(),v.as<uint16_t>(),bias.as<float>(),out.as<float>(),N,1,sp,1,0);HIP(hipGetLastError());
            }
            HIP(hipEventRecord(stop));HIP(hipEventSynchronize(stop));float ms;HIP(hipEventElapsedTime(&ms,start,stop));if(rep>=0)times[fused].push_back(ms/clones);
        }
        w.trellis=p.as<uint32_t>();
        for(int fused=0;fused<2;++fused) {
            printf("sweep_ms route=%s splits=%d",fused?"fused2":"baseline3",sp);for(float t:times[fused])printf(" %.9g",t*clones);puts("");
            std::sort(times[fused].begin(),times[fused].end());auto& t=times[fused];
            printf("BENCH file=%s route=%s I=%d N=%d K=%u splits=%d sweeps=11 clones=%d rotation_bytes=%zu min_us=%.9g median_us=%.9g p90_us=%.9g compressed_GBs=%.9g\n",path,fused?"fused2":"baseline3",I,N,f.bits,sp,clones,bytes*clones,t[0]*1000,t[5]*1000,t[9]*1000,bytes/(t[5]*1e6));
        }
        HIP(hipEventDestroy(start));HIP(hipEventDestroy(stop));
    }
}
int main(int argc,char** argv) {
    setvbuf(stdout,nullptr,_IOLBF,0);
    if(argc<2){fprintf(stderr,"Usage: fused_input_bench fixture... [--bench fixture...]\n");return 2;}
    try {
        hipDeviceProp_t p{};HIP(hipGetDeviceProperties(&p,0));
        printf("device=%s arch=%s warp=%d WGP=%d nominal_GPU_kHz=%d nominal_mem_kHz=%d bus_bits=%d geometry=N/64_by_splits block=128 dynamic_LDS=ceil(H/splits)*256 max_LDS=16384\n",p.name,p.gcnArchName,p.warpSize,p.multiProcessorCount,p.clockRate,p.memoryClockRate,p.memoryBusWidth);
        decode_checks();bool bench=false;
        for(int k=1;k<argc;++k){if(!strcmp(argv[k],"--bench")){bench=true;continue;}fixture(argv[k],bench);}
        puts("ALL_PASS direct_backend_optional_fusion");return 0;
    } catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 2;}
}
