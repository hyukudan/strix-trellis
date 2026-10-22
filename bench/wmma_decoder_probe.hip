// Minimal decoder physical qualification. Independent project fixture lifecycle.
#include "../src/backend.hpp"
#define input_transform prefill_test_input_transform
#define output_transform prefill_test_output_transform
#include "../src/transforms.hpp"
#undef input_transform
#undef output_transform
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
namespace strix_trellis {
hipError_t launch_wmma_reuse_variant(const DeviceWeights&,const _Float16*,float*,int,int,hipStream_t,int);
}
static void hip_check(hipError_t s) {
    if (s != hipSuccess) throw std::runtime_error(hipGetErrorString(s));
}
struct Device {
    inline static size_t live=0,peak=0;
    void* p = nullptr;
    size_t bytes;
    explicit Device(size_t n):bytes(n) {
        if(n>=(1ull<<30)-live) throw std::runtime_error("Explicit GPU allocation budget");
        if (n) hip_check(hipMalloc(&p, n));
        live+=n;peak=std::max(peak,live);
    }
    ~Device() { if (p) (void)hipFree(p);live-=bytes; }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
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
        if (!i || !n || i%128 || n%128 || i>8192 || n>131072 || !rows || rows>512 ||
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
static void compare(const std::string& label, const std::vector<float>& actual,
                    const std::vector<float>& ref) {
    double maxabs=0, maxref=0, err2=0, ref2=0;
    size_t worst=0;
    for(size_t k=0;k<ref.size();++k) {
        if(!std::isfinite(actual[k]) || !std::isfinite(ref[k])) throw std::runtime_error(label+": nonfinite");
        double e=std::abs(double(actual[k])-ref[k]);
        if(e>maxabs) { maxabs=e; worst=k; }
        maxref=std::max(maxref,std::abs(double(ref[k]))); err2+=e*e; ref2+=double(ref[k])*ref[k];
    }
    const double nrmse=std::sqrt(err2/std::max(ref2,1e-30));
    std::cout<<label<<" maxabs="<<maxabs<<" maxref="<<maxref<<" nrmse="<<nrmse<<" worst="<<worst<<'\n';
    if(maxabs>5e-4*std::max(1.0,maxref) || nrmse>3e-5) throw std::runtime_error(label+": tolerance exceeded");
}
namespace {
constexpr size_t cold_bytes=256ull<<20;
constexpr int rotations=11;
constexpr size_t guard=256;
struct Stream {
    hipStream_t s{};
    Stream() { hip_check(hipSetDevice(0)); hip_check(hipStreamCreateWithFlags(&s,hipStreamNonBlocking)); }
    ~Stream() { (void)hipStreamSynchronize(s); (void)hipStreamDestroy(s); }
};
struct Guarded {
    Device allocation;
    size_t bytes;
    unsigned char* p;
    explicit Guarded(size_t n):allocation(n+2*guard),bytes(n),p(static_cast<unsigned char*>(allocation.p)+guard) {
        hip_check(hipMemset(allocation.p,0xa5,n+2*guard));
        hip_check(hipDeviceSynchronize());
    }
    void check() {
        unsigned char front[guard],back[guard];
        hip_check(hipMemcpy(front,allocation.p,guard,hipMemcpyDeviceToHost));
        hip_check(hipMemcpy(back,p+bytes,guard,hipMemcpyDeviceToHost));
        for(size_t k=0;k<guard;++k) if(front[k]!=0xa5 || back[k]!=0xa5) throw std::runtime_error("Canary changed");
    }
};
void copy(void* dst,const std::vector<unsigned char>& src,hipStream_t s,size_t bytes=0) {
    hip_check(hipMemcpyAsync(dst,src.data(),bytes?bytes:src.size(),hipMemcpyHostToDevice,s));
}
}
#include "../src/wmma_decoder.hpp"
#include "../src/packed_decoder.hpp"
#include "../src/gemm.hpp"
#include <array>
#include <chrono>
#include <filesystem>
namespace experiment {
namespace wd = strix_trellis::decoder;
void require(bool ok,const std::string& message) { if(!ok) throw std::runtime_error(message); }
template<class T> std::vector<T> read_file(const std::string& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    require(bool(f),"Cannot open "+path);
    auto bytes=f.tellg();require(bytes>=0 && bytes<512*1024*1024 && size_t(bytes)%sizeof(T)==0,"Invalid oracle size");
    std::vector<T> v(size_t(bytes)/sizeof(T));f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()),bytes);require(bool(f),"Truncated oracle");return v;
}
void exact(const void* device,const void* ref,size_t bytes,hipStream_t stream,const std::string& label) {
    std::vector<unsigned char> got(bytes);
    hip_check(hipMemcpyAsync(got.data(),device,bytes,hipMemcpyDeviceToHost,stream));
    hip_check(hipStreamSynchronize(stream));
    if(std::memcmp(got.data(),ref,bytes)) {
        std::ofstream("build/failure-actual.bin",std::ios::binary).write(reinterpret_cast<const char*>(got.data()),bytes);
        std::ofstream("build/failure-expected.bin",std::ios::binary).write(static_cast<const char*>(ref),bytes);
        std::ofstream("build/failure-label.txt")<<label<<'\n';
        if(label=="Exhaustive hashes and both half components") {
            std::ofstream actual_file("build/exhaustive-actual.bin",std::ios::binary),expected_file("build/exhaustive-expected.bin",std::ios::binary);
            actual_file.write(reinterpret_cast<const char*>(got.data()),bytes);
            expected_file.write(static_cast<const char*>(ref),bytes);
        }
        if(label.find("checkpoint")!=std::string::npos) {
            std::ofstream actual_file("build/checkpoint-actual.bin",std::ios::binary),expected_file("build/checkpoint-expected.bin",std::ios::binary);
            actual_file.write(reinterpret_cast<const char*>(got.data()),bytes);
            expected_file.write(static_cast<const char*>(ref),bytes);
        }
        size_t k=0;while(k<bytes && got[k]==static_cast<const unsigned char*>(ref)[k]) ++k;
        throw std::runtime_error(label+" byte="+std::to_string(k));
    }
}
void finite(const std::vector<float>& data) {
    for(float x:data) require(std::isfinite(x),"Nonfinite partial");
}
struct Weights {
    Guarded packed,u,v,bias;
    strix_trellis::DeviceWeights w;
    Weights(const Fixture& f,hipStream_t s,size_t clones=1):packed(clones*f.parts.at("trellis").size()),u(2*f.i),v(2*f.n),bias(f.bias?4*f.n:0),
        w{reinterpret_cast<uint32_t*>(packed.p),reinterpret_cast<uint16_t*>(u.p),reinterpret_cast<uint16_t*>(v.p),f.bias?reinterpret_cast<float*>(bias.p):nullptr,int(f.i),int(f.n),int(f.bits),int(f.cb)} {
        for(size_t c=0;c<clones;++c) copy(packed.p+c*f.parts.at("trellis").size(),f.parts.at("trellis"),s);
        copy(u.p,f.parts.at("suh"),s);copy(v.p,f.parts.at("svh"),s);
        if(f.bias) copy(bias.p,f.parts.at("bias"),s);
        hip_check(hipStreamSynchronize(s));
    }
    void check() {packed.check();u.check();v.check();bias.check();}
};
std::vector<float> download(const Guarded& p,hipStream_t s) {
    std::vector<float> v(p.bytes/4);
    hip_check(hipMemcpyAsync(v.data(),p.p,p.bytes,hipMemcpyDeviceToHost,s));
    hip_check(hipStreamSynchronize(s));return v;
}
void input(const Fixture& f,const Weights& w,const Guarded& x,Guarded& a,int rows,hipStream_t s) {
    strix_trellis::prefill_test_input_transform<<<(size_t(rows)*(f.i/128)+3)/4,128,0,s>>>(reinterpret_cast<const float*>(x.p),w.w.suh,reinterpret_cast<_Float16*>(a.p),f.i,rows);
    hip_check(hipGetLastError());
}
void output(const Fixture& f,const Weights& w,const Guarded& p,Guarded& y,int rows,int splits,float alpha,float beta,hipStream_t s) {
    strix_trellis::prefill_test_output_transform<<<(size_t(rows)*(f.n/128)+3)/4,128,0,s>>>(reinterpret_cast<const float*>(p.p),w.w.svh,w.w.bias,reinterpret_cast<float*>(y.p),f.n,rows,splits,alpha,beta);
    hip_check(hipGetLastError());
}

__global__ void native_probe(float* out,int c,int selected_sub,int selected_m) {
    using H16=_Float16 __attribute__((ext_vector_type(16)));
    using F8=float __attribute__((ext_vector_type(8)));
    int test=blockIdx.x,lane=threadIdx.x&31,wave=threadIdx.x/32,col=lane%16;
    F8 acc[4]={};
    for(int sub=0;sub<c;++sub) for(int m=0;m<4;++m) {
        H16 a={},b={};
        if(sub==selected_sub && m==selected_m) for(int k=0;k<16;++k) {
            int ar=col*16+k,bc=k*16+col;
            a[k]=test<256 ? _Float16(ar==test?2:0) : test<512 ? _Float16(col==k) : _Float16((ar*7%13)-6);
            b[k]=test<256 ? _Float16(k==col) : test<512 ? _Float16(bc==test-256?-3:0) : _Float16((bc*11%17)-8);
        }
        acc[m]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b,acc[m]);
    }
    for(int m=0;m<4;++m) for(int j=0;j<8;++j)
        out[(((size_t(test)*4+wave)*4+m)*16+2*j+lane/16)*16+col]=acc[m][j];
}
void native_gate(const std::string& dir,hipStream_t s) {
    Guarded d(513*4*4*256*4);
    double maxabs=0,maxratio=0;
    // The reference is an analytic expression for impulses. Mixed-sign integer
    // reference/sumabs are only 256 dots of length16, prepared once below by
    // closed-form tables saved by --prepare (no CPU matmul during the turn).
    auto oracle=read_file<double>(dir+"/native.bin");require(oracle.size()==512,"Bad native oracle");
    for(int c:{1}) for(int sub=0;sub<c;++sub) for(int selected=0;selected<4;++selected) {
        native_probe<<<513,128,0,s>>>(reinterpret_cast<float*>(d.p),c,sub,selected);hip_check(hipGetLastError());
        auto got=download(d,s);
        for(int test=0;test<513;++test) for(int wave=0;wave<4;++wave) for(int m=0;m<4;++m) for(int r=0;r<16;++r) for(int n=0;n<16;++n) {
            double expected=0,sumabs=0;
            if(m==selected) {
                if(test<256) expected=(r*16+n==test)?2:0;
                else if(test<512) expected=(r*16+n==test-256)?-3:0;
                else {expected=oracle[r*16+n];sumabs=oracle[256+r*16+n];}
            }
            double actual=got[(((size_t(test)*4+wave)*4+m)*16+r)*16+n];
            double error=std::abs(actual-expected);require(std::isfinite(actual),"Native nonfinite");
            if(test<512 || m!=selected) require(error==0,"Native impulse/map mismatch");
            else require(error<=16*std::ldexp(1.0,-24)*sumabs,"Native mixed envelope");
            maxabs=std::max(maxabs,error);if(sumabs) maxratio=std::max(maxratio,error/sumabs);
        }
        d.check();
    }
    std::cout<<"GATE native_512_impulses=PASS sub_m16_combinations=4 all_waves=4 maxabs="<<maxabs<<" max_error_sumabs="<<maxratio<<"\n";
}

using Handle=std::unique_ptr<const wd::Token>;
constexpr std::array<int,5> splits_list={1,3,4,16,32};
constexpr std::array<int,16> rows_list={1,8,15,16,17,63,64,65,127,128,129,255,256,257,511,512};
size_t cases=0,partial_values=0;
auto begin_time=std::chrono::steady_clock::now();
void budget() {require(std::chrono::duration<double>(std::chrono::steady_clock::now()-begin_time).count()<145,"145 second campaign budget exhausted");}
wd::Request request(const strix_trellis::DeviceWeights& w,const Guarded& a,const Guarded& p,int rows,int splits,int layout,int mode,hipStream_t s,size_t packed) {
    return {w,{w.trellis,packed},{a.p,a.bytes},{p.p,p.bytes},rows,splits,w.input,w.output,rows,static_cast<wd::Layout>(layout),static_cast<wd::Mode>(mode),s,false};
}
Handle prepare(const wd::Request& r) {Handle t;hip_check(wd::prepare(r,t));require(bool(t),"Unready token");return t;}
__global__ void hash_half_gate(uint32_t* out) {
    unsigned state=blockIdx.x*blockDim.x+threadIdx.x;
    if(state>=65536) return;
    constexpr uint32_t high[]={0,0x12340000u,0xa5a50000u,0xffff0000u,0x80000000u};
    const uint32_t a=__builtin_bit_cast(uint32_t,strix_trellis::optimized::mul1_half2(state,65535-state));
    const uint32_t b=__builtin_bit_cast(uint32_t,strix_trellis::packed_decode::mul1_half2(state|high[blockIdx.y],(65535-state)|high[4-blockIdx.y]));
    const size_t index=(size_t(blockIdx.y)*65536+state)*8;
    out[index]=a&65535u;out[index+1]=a>>16;
    out[index+2]=b&65535u;out[index+3]=b>>16;
    out[index+4]=strix_trellis::packed_decode::hash16_isa<0x83dcd12du>(state|high[blockIdx.y]);
    out[index+5]=__builtin_bit_cast(uint16_t,_Float16(strix_trellis::codebook_value(state,2)));
    out[index+6]=a;out[index+7]=b;
}
template<int B,int D> __global__ void basis_reader(const uint32_t* rings,uint16_t* out) {
    int lane=threadIdx.x&31;const uint32_t* ptr=rings+size_t(blockIdx.x)*B*8;
    using Reader=std::conditional_t<D==1,strix_trellis::optimized::AlignedLane<B,true>,strix_trellis::packed_decode::SaddrLane<B,true>>;
    Reader tile(ptr);
    #define STATE(J) out[size_t(blockIdx.x)*256+8*lane+J]=tile.template state<J>()
    STATE(0);STATE(1);STATE(2);STATE(3);STATE(4);STATE(5);STATE(6);STATE(7);
    #undef STATE
}
void decoder_gate(const std::string& dir,hipStream_t s) {
    auto lut=read_file<uint16_t>(dir+"/mul1.bin");require(lut.size()==65536,"Bad half LUT");
    std::vector<uint32_t> ref(65536*5*8);
    for(unsigned h=0;h<5;++h) for(unsigned j=0;j<65536;++j) {
        const size_t index=(size_t(h)*65536+j)*8;
        ref[index]=ref[index+2]=ref[index+5]=lut[j];
        ref[index+1]=ref[index+3]=lut[65535-j];ref[index+4]=j*0x83dcd12du;
        ref[index+6]=ref[index+7]=uint32_t(lut[j])|(uint32_t(lut[65535-j])<<16);
    }
    Guarded hash(ref.size()*4);hash_half_gate<<<dim3(256,5),256,0,s>>>(reinterpret_cast<uint32_t*>(hash.p));hip_check(hipGetLastError());
    std::vector<uint32_t> actual(ref.size());
    hip_check(hipMemcpyAsync(actual.data(),hash.p,hash.bytes,hipMemcpyDeviceToHost,s));hip_check(hipStreamSynchronize(s));
    std::ofstream("build/exhaustive-actual.bin",std::ios::binary).write(reinterpret_cast<const char*>(actual.data()),hash.bytes);
    std::ofstream("build/exhaustive-expected.bin",std::ios::binary).write(reinterpret_cast<const char*>(ref.data()),hash.bytes);
    exact(hash.p,ref.data(),hash.bytes,s,"Exhaustive hashes and both half components");hash.check();
    for(int bits:{5,6}) {
        auto rings=read_file<unsigned char>(dir+"/basis"+std::to_string(bits)+".bin");
        auto states=read_file<uint16_t>(dir+"/states"+std::to_string(bits)+".bin");
        Guarded r(rings.size()),d(states.size()*2);copy(r.p,rings,s);
        for(int mode:{1,2}) {
            #define RUN(B,D) basis_reader<B,D><<<B*256+2,32,0,s>>>(reinterpret_cast<uint32_t*>(r.p),reinterpret_cast<uint16_t*>(d.p))
            if(bits==5) {if(mode==1) {RUN(5,1);}else {RUN(5,2);}}
            else {if(mode==1) {RUN(6,1);}else {RUN(6,2);}}
            #undef RUN
            hip_check(hipGetLastError());exact(d.p,states.data(),d.bytes,s,"Every physical basis state");
        }
        exact(r.p,rings.data(),r.bytes,s,"Immutable basis input");r.check();d.check();
    }
    std::cout<<"GATE exhaustive_hash_half=65536 both_components=PASS high_containers=PASS K5_K6_all_basis=PASS\n";
}
void guards(const wd::Request& good) {
    size_t count=0;
    auto reject=[&](wd::Request r) {auto token=prepare(good);require(wd::prepare(r,token)!=hipSuccess && !token,"Invalid request accepted or token retained");(void)hipGetLastError();++count;};
    for(int value:{-1,3,17}) {auto r=good;r.mode=static_cast<wd::Mode>(value);reject(r);}
    for(int value:{-1,3,64}) {auto r=good;r.layout=static_cast<wd::Layout>(value);reject(r);}
    for(int value:{0,513,-1}) {auto r=good;r.rows=value;reject(r);}
    for(int value:{0,33,-1}) {auto r=good;r.splits=value;reject(r);}
    for(int value:{0,127,8320}) {auto r=good;r.weights.input=value;reject(r);}
    for(int value:{0,127,32896}) {auto r=good;r.weights.output=value;reject(r);}
    for(int value:{0,1,3}) {auto r=good;r.weights.codebook=value;reject(r);}
    for(int value:{0,4,7}) {auto r=good;r.weights.bits=value;reject(r);}
    {auto r=good;r.a_stride++;reject(r);r=good;r.partial_stride++;reject(r);r=good;r.total_rows++;reject(r);}
    for(int i=0;i<3;++i) {
        auto r=good;wd::Range* v[]={&r.packed,&r.a,&r.partial};v[i]->capacity=1;reject(r);
        r=good;wd::Range* z[]={&r.packed,&r.a,&r.partial};z[i]->pointer=static_cast<const unsigned char*>(z[i]->pointer)+1;if(!i) r.weights.trellis=static_cast<const uint32_t*>(z[i]->pointer);reject(r);
        r=good;wd::Range* y[]={&r.packed,&r.a,&r.partial};y[i]->capacity=SIZE_MAX;reject(r);
        r=good;wd::Range* x[]={&r.packed,&r.a,&r.partial};x[i]->pointer=nullptr;if(!i)r.weights.trellis=nullptr;reject(r);
    }
    {auto r=good;r.partial=r.a;reject(r);r=good;r.a=r.packed;reject(r);r=good;r.partial=r.packed;reject(r);}
    {auto r=good;r.stream=nullptr;reject(r);r=good;r.stream=hipStreamLegacy;reject(r);r=good;r.stream=hipStreamPerThread;reject(r);}
    {auto r=good;r.diagnostic=true;reject(r);}
    {auto r=good;r.a.pointer=reinterpret_cast<const void*>(UINTPTR_MAX-1);reject(r);}
    std::cout<<"GATE invalid_host_guards="<<count<<" PASS output_not_launched=structural\n";
}
void staging(const Fixture& f,const Weights& w,const Guarded& a,hipStream_t s) {
    Guarded p(4ull*512*f.n),da(64*16*2),dp(2048),dl(2048);
    size_t dumps=0;
    for(int layout=0;layout<3;++layout) for(int mode=0;mode<3;++mode) {
        int m=layout?64:16;
        auto t=prepare(request(w.w,a,p,512,1,layout,mode,s,f.parts.at("trellis").size()));
        for(int k=0;k<int(f.i/16);++k) for(int n=0;n<int(f.n/16);n+=4) for(int fr:{0,511}) {
            hip_check(wd::dump(*t,reinterpret_cast<uint16_t*>(da.p),reinterpret_cast<uint16_t*>(dp.p),reinterpret_cast<uint16_t*>(dl.p),fr,n,k));
            std::vector<uint16_t> ar(m*16),physical(1024),logical(1024);
            for(int r=0;r<m;++r) for(int c=0;c<16;++c) if(fr+r<512)
                std::memcpy(&ar[r*16+c],f.parts.at("a").data()+(size_t(fr+r)*f.i+k*16+c)*2,2);
            for(int wave=0;wave<4;++wave) for(int r=0;r<16;++r) for(int c=0;c<16;++c) if(n+wave<int(f.n/16)) {
                uint16_t v;std::memcpy(&v,f.parts.at("q").data()+(size_t(k*16+r)*f.n+(n+wave)*16+c)*2,2);
                logical[wave*256+r*16+c]=v;physical[wave*256+r*16+(c^(layout?(r&7)*2:0))]=v;
            }
            exact(da.p,ar.data(),ar.size()*2,s,"A full staging");exact(dp.p,physical.data(),2048,s,"Physical Q all bytes");exact(dl.p,logical.data(),2048,s,"Logical Q all bytes");++dumps;
        }
    }
    da.check();dp.check();dl.check();p.check();std::cout<<"GATE staging_all_tiles_A_Q="<<dumps<<" PASS\n";
}
void validate(const std::string& path,hipStream_t s) {
    Fixture f(path);Weights w(f,s);Guarded x(4ull*512*f.i),a(2ull*512*f.i);
    copy(x.p,f.parts.at("x"),s);input(f,w,x,a,512,s);exact(a.p,f.parts.at("a").data(),a.bytes,s,"A exact full transform");
    staging(f,w,a,s);
    auto oracle=read_file<double>(path+".partial");auto impulse=read_file<unsigned char>(path+".impulse");
    const size_t matrix=512ull*f.n;
    for(int kind=0;kind<2;++kind) {
        if(kind) copy(a.p,impulse,s);
        for(size_t rowcase=0;rowcase<rows_list.size()+5;++rowcase) {
            budget();int rows=rowcase<16?rows_list[rowcase]:17;
            int split_index=rowcase<16?int(rowcase%5):int(rowcase-16),splits=splits_list[split_index];
            Guarded p(size_t(splits)*rows*f.n*4),y(size_t(rows)*f.n*4);
            if(!kind && !rowcase) guards(request(w.w,a,p,rows,splits,0,0,s,f.parts.at("trellis").size()));
            int preceding=0;for(int z=0;z<split_index;++z) preceding+=splits_list[z];
            size_t base=size_t(kind*56+preceding)*matrix*2;
            for(int layout=0;layout<3;++layout) {
                std::vector<float> control,control_y;
                for(int mode=0;mode<3;++mode) {
                    auto r=request(w.w,a,p,rows,splits,layout,mode,s,f.parts.at("trellis").size());auto token=prepare(r);
                    hip_check(token->submit());auto got=download(p,s);
                    if(mode==0) control=got;else require(!std::memcmp(control.data(),got.data(),p.bytes),"Full partial byte parity");
                    // Every partial compared with the independent CPU dot and sumabs.
                    double e2=0,r2=0,ma=0,mr=0;
                    for(int split=0;split<splits;++split) for(size_t j=0;j<size_t(rows)*f.n;++j) {
                        double ref=oracle[base+size_t(split)*matrix*2+j],sumabs=oracle[base+size_t(split)*matrix*2+matrix+j];
                        double v=got[size_t(split)*rows*f.n+j],e=std::abs(v-ref);
                        require(std::isfinite(v),"Nonfinite partial");
                        e2+=e*e;r2+=ref*ref;ma=std::max(ma,e);mr=std::max(mr,std::abs(ref));
                        if(kind) require(e<=16*std::ldexp(1.,-24)*sumabs,"Impulse partial envelope");
                        if((f.i/16)*(split+1)/splits==(f.i/16)*split/splits) require(v==0 && !std::signbit(v),"Vacant split not +0");
                    }
                    require(std::sqrt(e2/std::max(r2,1e-30))<=3e-5 && ma<=5e-4*std::max(1.,mr),"Independent full partial tolerance");
                    hip_check(token->submit());exact(p.p,got.data(),p.bytes,s,"Deterministic partial replay");
                    if(!kind) {
                        auto ref=f.floats("ref"),old=f.floats("old");ref.resize(size_t(rows)*f.n);old.resize(ref.size());
                        for(auto ab:std::array<std::pair<float,float>,3>{{{1,0},{-.75f,.25f},{0,-.5f}}}) {
                            if(ab.second==0) hip_check(hipMemsetAsync(y.p,0xff,y.bytes,s));
                            else hip_check(hipMemcpyAsync(y.p,old.data(),y.bytes,hipMemcpyHostToDevice,s));
                            output(f,w,p,y,rows,splits,ab.first,ab.second,s);auto actual=download(y,s),expected=ref;
                            for(size_t j=0;j<ref.size();++j) {volatile float av=ab.first*ref[j],bv=ab.second*old[j];expected[j]=av+bv;}
                            compare("SMALL_FULL",actual,expected);
                            if(mode==0) {if(ab.first==1)control_y=actual;}
                            else if(ab.first==1) require(!std::memcmp(control_y.data(),actual.data(),y.bytes),"Y byte parity");
                        }
                    }
                    ++cases;partial_values+=got.size();
                }
            }
            p.check();y.check();
        }
    }
    // A offset 2 is a positive contract case, including native WMMA.
    Guarded offset(a.bytes+2),p(3ull*17*f.n*4);hip_check(hipMemcpyAsync(offset.p+2,a.p,a.bytes,hipMemcpyDeviceToDevice,s));
    for(int mode=0;mode<3;++mode) {
        auto r=request(w.w,a,p,17,3,0,mode,s,f.parts.at("trellis").size());auto t=prepare(r);hip_check(t->submit());auto ref=download(p,s);
        r.a={offset.p+2,a.bytes};t=prepare(r);hip_check(t->submit());exact(p.p,ref.data(),p.bytes,s,"A offset two valid");
    }
    exact(x.p,f.parts.at("x").data(),x.bytes,s,"Immutable X");exact(w.packed.p,f.parts.at("trellis").data(),w.packed.bytes,s,"Immutable packed");
    exact(w.u.p,f.parts.at("suh").data(),w.u.bytes,s,"Immutable suh");exact(w.v.p,f.parts.at("svh").data(),w.v.bytes,s,"Immutable svh");
    if(f.bias)exact(w.bias.p,f.parts.at("bias").data(),w.bias.bytes,s,"Immutable bias");
    x.check();a.check();w.check();offset.check();p.check();
    std::cout<<"GATE small_full path="<<path<<" cases="<<cases<<" partial_values="<<partial_values<<" PASS\n";
}


bool operator_ranges(const std::vector<std::pair<wd::Range,size_t>>& ranges,int device) {
    for(size_t i=0;i<ranges.size();++i) {
        const auto& [r,alignment]=ranges[i];uintptr_t ptr=reinterpret_cast<uintptr_t>(r.pointer);
        if(!ptr || ptr%alignment || r.capacity>UINTPTR_MAX-ptr)return false;
        for(size_t j=0;j<i;++j) {
            uintptr_t other=reinterpret_cast<uintptr_t>(ranges[j].first.pointer);
            if(ptr<other ? r.capacity>other-ptr : ranges[j].first.capacity>ptr-other)return false;
        }
        hipPointerAttribute_t attr{};
        if(hipPointerGetAttributes(&attr,r.pointer)!=hipSuccess){(void)hipGetLastError();return false;}
        if(attr.type!=hipMemoryTypeDevice || attr.isManaged || attr.device!=device)return false;
        void* base=nullptr;size_t capacity=0;
        if(hipMemGetAddressRange(&base,&capacity,const_cast<void*>(r.pointer))!=hipSuccess){(void)hipGetLastError();return false;}
        uintptr_t origin=reinterpret_cast<uintptr_t>(base);
        if(ptr<origin || ptr-origin>capacity || r.capacity>capacity-(ptr-origin))return false;
    }
    return true;
}
void operator_preflight(const Fixture& f,const Weights& w,const Guarded& x,const Guarded& a,const Guarded& p,const Guarded& y,int rows,int splits) {
    require(x.bytes==size_t(rows)*f.i*4 && a.bytes==size_t(rows)*f.i*2 && p.bytes==size_t(splits)*rows*f.n*4 && y.bytes==size_t(rows)*f.n*4,"Dense operator strides");
    std::vector<std::pair<wd::Range,size_t>> ranges={{{x.p,x.bytes},4},{{a.p,a.bytes},2},{{p.p,p.bytes},4},{{y.p,y.bytes},4},{{w.packed.p,w.packed.bytes},4},{{w.u.p,w.u.bytes},2},{{w.v.p,w.v.bytes},2}};
    if(f.bias)ranges.push_back({{w.bias.p,w.bias.bytes},4});
    int device=-1;hip_check(hipGetDevice(&device));require(operator_ranges(ranges,device),"Operator preflight");
    auto invalid=ranges;invalid[3]=invalid[0];require(!operator_ranges(invalid,device),"X/Y alias accepted");
    invalid=ranges;invalid[0].first.capacity=SIZE_MAX;require(!operator_ranges(invalid,device),"Operator overflow accepted");
    invalid=ranges;invalid[0].first.pointer=x.p+1;require(!operator_ranges(invalid,device),"Operator alignment accepted");
    std::cout<<"GATE operator_ranges_noalias_alignment_capacity=PASS\n";
}
int policy_s(int shape,int rows) {
    constexpr int table[6][3]={{1,1,1},{4,2,1},{8,4,2},{4,2,1},{16,8,4},{16,8,4}};
    return table[shape][rows==64?0:rows==128?1:2];
}
void performance(const std::string& path,int shape,hipStream_t s) {
    Fixture f(path);const size_t packed=f.parts.at("trellis").size(),clones=(cold_bytes+packed-1)/packed;
    for(const char* key:{"q","core","scaled","inh"}) f.parts.erase(key);
    Weights w(f,s,clones);
    for(int rows:{64,128,256}) {
        budget();int splits=policy_s(shape,rows);require(rows<=int(f.rows),"Missing R256 reference");
        Guarded x(size_t(rows)*f.i*4),a(size_t(rows)*f.i*2),p(size_t(splits)*rows*f.n*4),y(size_t(rows)*f.n*4);
        size_t bound=w.packed.bytes+w.u.bytes+w.v.bytes+w.bias.bytes+x.bytes+a.bytes+p.bytes+y.bytes+16*guard;
        require(bound<(1ull<<30),"GPU byte budget");
        copy(x.p,f.parts.at("x"),s,x.bytes);input(f,w,x,a,rows,s);exact(a.p,f.parts.at("a").data(),a.bytes,s,"Performance exact A");
        operator_preflight(f,w,x,a,p,y,rows,splits);
        auto ref=f.floats("ref");ref.resize(size_t(rows)*f.n);
        std::vector<std::vector<Handle>> tokens(3);
        for(int mode=0;mode<3;++mode) for(size_t c=0;c<clones;++c) {
            auto cw=w.w;cw.trellis=reinterpret_cast<const uint32_t*>(w.packed.p+c*packed);
            tokens[mode].push_back(prepare(request(cw,a,p,rows,splits,2,mode,s,packed)));
        }
        // Independent immutable V11 control, same policy layout and split.
        hip_check(strix_trellis::launch_wmma_core(w.w,reinterpret_cast<_Float16*>(a.p),reinterpret_cast<float*>(p.p),rows,splits,s));
        auto control=download(p,s);output(f,w,p,y,rows,splits,1,0,s);auto control_y=download(y,s);
        compare("IMMUTABLE_V11_FULL",control_y,ref);
        auto partial_ref=read_file<double>("build/oracles/perf-"+std::to_string(shape)+"-"+std::to_string(rows)+".bin");
        require(partial_ref.size()==control.size()*2,"Full large partial oracle dimensions");
        double e2=0,r2=0,ma=0,mr=0;
        const size_t plane=size_t(rows)*f.n;
        for(int split=0;split<splits;++split) for(size_t j=0;j<plane;++j) {
            const double expected=partial_ref[size_t(split)*2*plane+j];
            const double value=control[size_t(split)*plane+j],e=std::abs(value-expected);
            require(std::isfinite(value),"Nonfinite large partial");
            e2+=e*e;r2+=expected*expected;ma=std::max(ma,e);mr=std::max(mr,std::abs(expected));
        }
        require(std::sqrt(e2/std::max(r2,1e-30))<=3e-5 && ma<=5e-4*std::max(1.,mr),"Independent full large partial tolerance");
        std::cout<<"GATE large_full_CPU_partials shape="<<shape<<" rows="<<rows<<" splits="<<splits<<" values="<<control.size()<<" PASS\n";
        auto submit=[&](int mode,size_t c,bool full) {
            if(full)input(f,w,x,a,rows,s);
            hip_check(tokens[mode][c]->submit());
            if(full)output(f,w,p,y,rows,splits,1,0,s);
        };
        auto immutable_inputs=[&]() {
            exact(x.p,f.parts.at("x").data(),x.bytes,s,"Immutable performance X");
            for(size_t c=0;c<clones;++c) exact(w.packed.p+c*packed,f.parts.at("trellis").data(),packed,s,"Immutable disjoint packed clone");
            exact(w.u.p,f.parts.at("suh").data(),w.u.bytes,s,"Immutable performance suh");
            exact(w.v.p,f.parts.at("svh").data(),w.v.bytes,s,"Immutable performance svh");
            if(f.bias)exact(w.bias.p,f.parts.at("bias").data(),w.bias.bytes,s,"Immutable performance bias");
            w.check();x.check();a.check();p.check();y.check();
        };
        auto check=[&](int mode) {
            for(size_t c=0;c<clones;++c) {
                submit(mode,c,true);exact(p.p,control.data(),p.bytes,s,"All clone full partial parity V11");
                exact(y.p,control_y.data(),y.bytes,s,"All clone full output parity V11");
                compare("ALL_CLONE_FULL_REF",download(y,s),ref);
            }
            exact(a.p,f.parts.at("a").data(),a.bytes,s,"All clone exact A");
        };
        immutable_inputs();for(int mode=0;mode<3;++mode)check(mode);
        for(bool full:{false,true}) {
            hipEvent_t start{},stop{};hip_check(hipEventCreate(&start));hip_check(hipEventCreate(&stop));
            for(int candidate:{1,2}) {
                std::array<std::vector<double>,2> times;
                // Exactly one warm rotation for each side of each pair.
                for(int mode:{0,candidate}) for(size_t c=0;c<clones;++c)submit(mode,c,full);
                hip_check(hipStreamSynchronize(s));
                for(int rep=0;rep<11;++rep) {
                    budget();
                    for(int side=0;side<2;++side) {
                        int index=(rep&1)?1-side:side,mode=index?candidate:0;
                        hip_check(hipEventRecord(start,s));
                        for(size_t c=0;c<clones;++c)submit(mode,(c+rep)%clones,full);
                        hip_check(hipEventRecord(stop,s));hip_check(hipEventSynchronize(stop));
                        float ms=0;hip_check(hipEventElapsedTime(&ms,start,stop));times[index].push_back(ms*1000./clones);
                        // Last timed clone checked each sweep; full arrays, outside events.
                        exact(p.p,control.data(),p.bytes,s,"Last timed full partial V11");
                        if(full) {exact(y.p,control_y.data(),y.bytes,s,"Last timed full Y V11");compare("LAST_TIMED_REF",download(y,s),ref);}
                    }
                }
                check(0);check(candidate);immutable_inputs();
                for(int side=0;side<2;++side) {
                    auto sorted=times[side];std::sort(sorted.begin(),sorted.end());
                    std::cout<<"RESULT shape="<<shape<<" rows="<<rows<<" splits="<<splits<<" layout=64/18/XOR mode="<<(full?"operator":"core")
                        <<" pair="<<candidate<<" decoder="<<(side?candidate:0)<<" clones="<<clones<<" cloned_bytes="<<clones*packed<<" gpu_bound="<<bound
                        <<" min_us="<<sorted[0]<<" median_us="<<sorted[5]<<" p90_us="<<sorted[9]<<" samples=";
                    for(double value:times[side])std::cout<<value<<",";
                    std::cout<<" parity=PASS CPU_ref=PASS canaries=PASS\n";
                }
            }
            hip_check(hipEventDestroy(start));hip_check(hipEventDestroy(stop));
        }
    }
}
void inactive(const std::string& path,hipStream_t s) {
    Fixture f(path);Weights w(f,s);Guarded a(17ull*f.i*2);
    hip_check(hipMemcpyAsync(a.p,f.parts.at("a").data(),a.bytes,hipMemcpyHostToDevice,s));
    for(int n:{16,32,48}) {
        auto ref=read_file<double>(path+".diag"+std::to_string(n));Guarded p(3ull*17*n*4),da(2048),dp(2048),dl(2048);
        auto weights=w.w;weights.output=n;
        for(int layout=0;layout<3;++layout) {
            std::vector<float> control;
            for(int mode=0;mode<3;++mode) {
                auto r=request(weights,a,p,17,3,layout,mode,s,size_t(f.i)*n*f.bits/8);r.diagnostic=true;
                auto t=prepare(r);hip_check(t->submit());auto got=download(p,s);
                if(!mode)control=got;else require(!std::memcmp(control.data(),got.data(),p.bytes),"Inactive wave partial parity");
                std::vector<float> cpu(ref.begin(),ref.end());compare("INACTIVE_FULL_CPU",got,cpu);
                hip_check(wd::dump(*t,reinterpret_cast<uint16_t*>(da.p),reinterpret_cast<uint16_t*>(dp.p),reinterpret_cast<uint16_t*>(dl.p),16,0,0));
                auto physical=read_file<uint16_t>(path+".diagq"+std::to_string(n));
                std::vector<uint16_t> expected(1024);
                for(int wave=0;wave<n/16;++wave) for(int row=0;row<16;++row)for(int col=0;col<16;++col)
                    expected[wave*256+row*16+(col^(layout?(row&7)*2:0))]=physical[row*n+wave*16+col];
                exact(dp.p,expected.data(),2048,s,"Inactive complete physical Q");
            }
        }
        p.check();da.check();dp.check();dl.check();
    }
    a.check();w.check();std::cout<<"GATE inactive_N16_32_48 all_layouts_modes=PASS\n";
}
} // namespace experiment
int main(int argc,char** argv) {
    try {
        using namespace experiment;
        std::cout<<std::unitbuf<<std::setprecision(10);
        require(argc==9,"Usage: probe ORACLE_DIR SIX_REAL_FIXTURES CAMPAIGN");
        Stream stream;hipDeviceProp_t props{};hip_check(hipGetDeviceProperties(&props,0));
        require(std::string(props.gcnArchName).find("gfx1151")==0 && props.warpSize==32,"gfx1151 wave32 required");
        std::cout<<"HARDWARE name="<<props.name<<" arch="<<props.gcnArchName<<" wave="<<props.warpSize<<" multiprocessors="<<props.multiProcessorCount<<" nominal_gpu_kHz="<<props.clockRate<<" nominal_mem_kHz="<<props.memoryClockRate<<" totalGlobalMem="<<props.totalGlobalMem<<" campaign="<<argv[8]<<"\n";
        std::string dir=argv[1];begin_time=std::chrono::steady_clock::now();
        decoder_gate(dir,stream.s);native_gate(dir,stream.s);
        for(int bits:{5,6})for(int inputs:{128,256,384}) {
            std::string file=dir+"/core-k"+std::to_string(bits)+"-i"+std::to_string(inputs)+"-n"+std::to_string(inputs==256?128:256)+".stgpu";
            validate(file,stream.s);if(inputs==128)inactive(file,stream.s);
        }
        std::cout<<"GATE correctness_complete=PASS cases="<<cases<<" full_partial_values="<<partial_values<<" seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-begin_time).count()<<"\n";
        for(int shape=0;shape<6;++shape)performance(argv[2+shape],shape,stream.s);
        hip_check(hipStreamSynchronize(stream.s));budget();
        std::cout<<"PASS campaign_complete=1 shapes=6 rows=64,128,256 modes=3 sweeps=11 criterion=2percent_both_campaigns explicit_gpu_peak="<<Device::peak<<" seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-begin_time).count()<<"\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
}
