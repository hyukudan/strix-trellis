// Independent whole-device operator split experiment; no C API submission or validation.
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
    void* p = nullptr;
    explicit Device(size_t n) { if (n) hip_check(hipMalloc(&p, n)); }
    ~Device() { if (p) (void)hipFree(p); }
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
int default_tile(const Fixture& f,int m) {
    return ((m>=256 && f.n>=2304) || (m>=64 && f.n>=32768 && f.bits==5 && f.cb==2))?-65:16;
}
int policy_splits(const Fixture& f,int m,int tile) {
    size_t groups=size_t((m+tile-1)/tile)*((f.n+63)/64);
    int s=1; while(groups*s<160 && s<32) s*=2;
    return std::min(s,int(f.i/16));
}
void run(const std::string& path,hipStream_t stream) {
    Fixture f(path);
    const size_t packed=f.parts.at("trellis").size();
    const size_t clones=(cold_bytes+packed-1)/packed;
    std::cout<<"FIXTURE path="<<path<<" I="<<f.i<<" N="<<f.n<<" bits="<<f.bits<<" codebook="<<f.cb<<" fixture_rows="<<f.rows<<" clones="<<clones<<" cloned_bytes="<<clones*packed<<"\n";
    for(const char* key:{"q","core","scaled","inh","old"}) f.parts.erase(key);
    Device weights(clones*packed),u(2*f.i),v(2*f.n),bias(f.bias?4*f.n:0);
    for(size_t c=0;c<clones;++c) copy(static_cast<char*>(weights.p)+c*packed,f.parts.at("trellis"),stream);
    copy(u.p,f.parts.at("suh"),stream); copy(v.p,f.parts.at("svh"),stream);
    if(f.bias) copy(bias.p,f.parts.at("bias"),stream);
    hip_check(hipStreamSynchronize(stream));
    for(int m:{32,64,128,129}) {
        if(m>int(f.rows)) throw std::runtime_error("Required M reference unavailable");
        Guarded x(size_t(m)*f.i*4),y(size_t(m)*f.n*4);
        copy(x.p,f.parts.at("x"),stream,x.bytes);
        auto ref=f.floats("ref"); ref.resize(size_t(m)*f.n);
        std::vector<float> actual(ref.size());
        // Baseline and sweep share exact kernels. Report their identities explicitly.
        const int dt=default_tile(f,m), ds=policy_splits(f,m,dt==16?16:64);
        std::cout<<"BASELINE rows="<<m<<" backend_tile="<<dt<<" backend_splits="<<ds
                 <<" forced_m16_policy_splits="<<policy_splits(f,m,16)<<"\n";
        for(int tile:{16,-65}) for(int splits:{1,2,4,8,16}) {
            const size_t abytes=size_t(m)*f.i*2;
            const size_t pbytes=size_t(splits)*m*f.n*4;
            Guarded a(abytes),partial(pbytes);
            const size_t bound=clones*packed+2ull*(f.i+f.n)+(f.bias?4ull*f.n:0)+x.bytes+y.bytes+abytes+pbytes+8*guard;
            if(bound>=(1ull<<30)) throw std::runtime_error("Allocation bound exceeds 1 GiB");
            auto submit=[&](size_t c) {
                strix_trellis::DeviceWeights w{reinterpret_cast<const uint32_t*>(static_cast<char*>(weights.p)+c*packed),
                    static_cast<const uint16_t*>(u.p),static_cast<const uint16_t*>(v.p),static_cast<const float*>(bias.p),int(f.i),int(f.n),int(f.bits),int(f.cb)};
                strix_trellis::prefill_test_input_transform<<<(size_t(m)*(f.i/128)+3)/4,128,0,stream>>>(reinterpret_cast<float*>(x.p),w.suh,reinterpret_cast<_Float16*>(a.p),f.i,m);
                hip_check(hipGetLastError());
                hip_check(strix_trellis::launch_wmma_reuse_variant(w,reinterpret_cast<_Float16*>(a.p),reinterpret_cast<float*>(partial.p),m,splits,stream,tile));
                strix_trellis::prefill_test_output_transform<<<(size_t(m)*(f.n/128)+3)/4,128,0,stream>>>(reinterpret_cast<float*>(partial.p),w.svh,w.bias,reinterpret_cast<float*>(y.p),f.n,m,splits,1.0f,0.0f);
                hip_check(hipGetLastError());
            };
            std::string id=" rows="+std::to_string(m)+" tile="+std::to_string(tile)+" splits="+std::to_string(splits);
            auto check=[&](const char* phase,bool all) {
                for(size_t c=0;c<(all?clones:1);++c) {
                    if(all) submit(c);
                    hip_check(hipMemcpyAsync(actual.data(),y.p,y.bytes,hipMemcpyDeviceToHost,stream));
                    hip_check(hipStreamSynchronize(stream));
                    compare(std::string("CHECK phase=")+phase+id+" clone="+(all?std::to_string(c):"final"),actual,ref);
                }
                x.check();y.check();a.check();partial.check();
            };
            // Exact transformed operand gate before timing; full prefix, signed zeros included.
            submit(0);
            std::vector<unsigned char> operand(abytes);
            hip_check(hipMemcpyAsync(operand.data(),a.p,abytes,hipMemcpyDeviceToHost,stream));
            hip_check(hipStreamSynchronize(stream));
            if(std::memcmp(operand.data(),f.parts.at("a").data(),abytes))
                throw std::runtime_error("Input FP16 operand bytes differ from CPU"+id);
            std::cout<<"INPUT_GATE"<<id<<" bytes="<<abytes<<" exact=PASS\n";
            check("before",true);
            // One discarded full rotation. Every sample consumes >=256 MiB distinct compressed addresses.
            for(size_t c=0;c<clones;++c) submit(c);
            hip_check(hipStreamSynchronize(stream));
            hipEvent_t start,stop; hip_check(hipEventCreate(&start)); hip_check(hipEventCreate(&stop));
            std::vector<double> times;
            for(int rep=0;rep<rotations;++rep) {
                hip_check(hipEventRecord(start,stream));
                // Rotate the first clone as well as traversing every clone.
                for(size_t c=0;c<clones;++c) submit((c+rep)%clones);
                hip_check(hipEventRecord(stop,stream)); hip_check(hipEventSynchronize(stop));
                float ms; hip_check(hipEventElapsedTime(&ms,start,stop)); times.push_back(ms*1000.0/clones);
            }
            hip_check(hipEventDestroy(start));hip_check(hipEventDestroy(stop));
            check("final_timed",false);check("after",true);
            std::sort(times.begin(),times.end());
            std::cout<<"RESULT"<<id<<" backend_default="<<(tile==dt && splits==ds)
                     <<" m16_policy="<<(tile==16 && splits==policy_splits(f,m,16))
                     <<" partial_shape="<<splits<<","<<m<<","<<f.n<<" workspace_bytes="<<abytes+pbytes
                     <<" gpu_allocation_bound="<<bound<<" rotations="<<rotations
                     <<" event_min_us="<<times.front()<<" event_median_us="<<times[5]<<" event_p90_us="<<times[9]
                     <<" useful_TOPS="<<2.0*m*f.i*f.n/times[5]/1e6
                     <<" checksum="<<std::accumulate(actual.begin(),actual.end(),0.0)<<" canaries=PASS\n";
        }
    }
}
}
int main(int argc,char** argv) {
    try {
        if(argc<2) throw std::runtime_error("Usage: prefill_split_bench FIXTURE...");
        std::cout<<std::unitbuf<<std::setprecision(10);
        Stream stream;
        hipDeviceProp_t p{}; hip_check(hipGetDeviceProperties(&p,0));
        if(std::string(p.gcnArchName).find("gfx1151")!=0 || p.warpSize!=32) throw std::runtime_error("Requires gfx1151 wave32");
        std::cout<<"HARDWARE name="<<p.name<<" arch="<<p.gcnArchName<<" wave="<<p.warpSize<<" multiprocessors="<<p.multiProcessorCount
                 <<" nominal_clock_kHz="<<p.clockRate<<" nominal_memory_clock_kHz="<<p.memoryClockRate<<" totalGlobalMem="<<p.totalGlobalMem
                 <<" whole_device_operator=1 C_API=0 event_rotation_amortized=1\n";
        for(int k=1;k<argc;++k) run(argv[k],stream.s);
        hip_check(hipStreamSynchronize(stream.s));
        std::cout<<"PASS prefill_splits\n"; return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL "<<e.what()<<"\n"; return 1; }
}
