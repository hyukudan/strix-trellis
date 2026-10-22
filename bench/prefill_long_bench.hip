// Long-prefill experiment. Fixture, Guarded and staged raw path adapted from
// our bench/prefill_split_bench.hip; API lifecycle adapted from tests/gpu_runtime.hip.
#include "strix_trellis.h"
#include "../src/launch_policy.hpp"
#include <chrono>
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
static void api_check(strix_status s) {
    if(s!=STRIX_SUCCESS) throw std::runtime_error(std::string("C API: ")+strix_status_string(s));
}
struct Device {
    void* p = nullptr;
    explicit Device(size_t n) { if (n) hip_check(hipMalloc(&p, n)); }
    ~Device() { if (p) (void)hipFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
struct Runtime {
    hipStream_t stream = nullptr;
    strix_stream s{};
    strix_context context = nullptr;
    Runtime() {
        int device = 0;
        hip_check(hipSetDevice(device));
        hipDeviceProp_t prop{};
        hip_check(hipGetDeviceProperties(&prop, device));
        std::cout << "device=" << prop.name << " arch=" << prop.gcnArchName
                  << " wave=" << prop.warpSize << '\n';
        hip_check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        s.struct_size = sizeof(s); s.abi_version = STRIX_TRELLIS_ABI_VERSION;
        s.hip_stream = reinterpret_cast<void*>(stream);
        strix_context_desc d{};
        d.struct_size = sizeof(d); d.abi_version = STRIX_TRELLIS_ABI_VERSION;
        d.device_ordinal = device;
        api_check(strix_context_create(&d, &context));
    }
    ~Runtime() {
        if (stream) (void)hipStreamSynchronize(stream);
        if (context) (void)strix_context_destroy(context);
        if (stream) (void)hipStreamDestroy(stream);
    }
    void close() {
        hip_check(hipStreamSynchronize(stream));
        api_check(strix_context_destroy(context)); context=nullptr;
        hip_check(hipStreamDestroy(stream)); stream=nullptr;
    }
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
        if (!i || !n || i%128 || n%128 || i>8192 || n>131072 || !rows || rows>1024 ||
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
struct Weight {
    strix_weights w=nullptr;
    Runtime& runtime;
    Weight(Runtime& rt, const Fixture& f) : runtime(rt) {
        std::vector<unsigned char> blob;
        strix_exl3_metadata_v1 m{};
        m.struct_size=sizeof(m); m.transport_version=1; m.rows=f.n; m.columns=f.i;
        m.bits_x2=2*f.bits; m.codebook=static_cast<strix_codebook>(f.cb); m.marker=f.marker;
        auto append=[&](const char* key, uint64_t& offset, uint64_t& bytes) {
            const auto& p=f.parts.at(key);
            if (p.empty()) { offset=bytes=0; return; }
            while (blob.size()%4) blob.push_back(0);
            offset=blob.size(); bytes=p.size(); blob.insert(blob.end(),p.begin(),p.end());
        };
        append("trellis",m.trellis_offset,m.trellis_bytes); append("suh",m.suh_offset,m.suh_bytes);
        append("svh",m.svh_offset,m.svh_bytes); append("bias",m.bias_offset,m.bias_bytes);
        strix_exl3_transport_desc d{};
        d.struct_size=sizeof(d); d.abi_version=1; d.format=STRIX_TRANSPORT_EXL3_CANONICAL;
        d.transport_version=1; d.data=blob.data(); d.data_bytes=blob.size();
        d.metadata=&m; d.metadata_bytes=sizeof(m); d.rows=f.n; d.columns=f.i;
        api_check(strix_weights_prepare(rt.context,&d,&rt.s,&w));
        // All host transport storage expires here: prepare must have completed uploads.
    }
    ~Weight() {
        // Drain on exception paths too, before freeing prepared weights.
        if(w) { (void)hipStreamSynchronize(runtime.stream); (void)strix_weights_destroy(w); }
    }
    void close() {
        hip_check(hipStreamSynchronize(runtime.stream));
        api_check(strix_weights_destroy(w)); w=nullptr;
    }
    Weight(const Weight&)=delete;
    Weight& operator=(const Weight&)=delete;
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
static strix_tensor_desc tensor(void* p, uint32_t rows, uint32_t width, bool vector) {
    strix_tensor_desc d{}; d.struct_size=sizeof(d); d.abi_version=1; d.dtype=STRIX_DTYPE_F32;
    d.data=p; d.buffer_bytes=size_t(rows)*width*4; d.rank=vector?1:2;
    d.shape[0]=vector?width:rows; d.shape[1]=vector?0:width;
    d.strides[0]=vector?1:width; d.strides[1]=vector?0:1; return d;
}
static strix_matmul_desc desc(strix_weights w, void* x, void* y, uint32_t rows,
                              const Fixture& f, bool vector, float alpha, float beta) {
    strix_matmul_desc d; strix_matmul_desc_init(&d); d.weights=w;
    d.input=tensor(x,rows,f.i,vector); d.output=tensor(y,rows,f.n,vector);
    d.alpha=alpha; d.beta=beta; return d;
}
namespace {
constexpr size_t cold_bytes=256ull<<20;
constexpr int rotations=11;
constexpr size_t guard=256;
struct Guarded {
    Device allocation;
    size_t bytes;
    unsigned char* p;
    explicit Guarded(size_t n):allocation(n+2*guard),bytes(n),p(static_cast<unsigned char*>(allocation.p)+guard) {
        hip_check(hipMemset(allocation.p,0xa5,n+2*guard));
        // Explicit dependency after default-stream memset: required by nonblocking stream.
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
    (void)f;
    return m<256?-65:16;
}
int policy_splits(const Fixture& f,int m,int tile) {
    size_t groups=size_t((m+tile-1)/tile)*((f.n+63)/64);
    int s=1; while(groups*s<160 && s<32) s*=2;
    return std::min(s,int(f.i/16));
}
void run(const std::string& path,hipStream_t stream) {
    Fixture f(path);
    if(f.rows!=512 || f.cb!=2 || !((f.i==5120 && f.n==1280 && f.bits==6) ||
       (f.i==4096 && f.n==1024 && f.bits==5))) throw std::runtime_error("Unexpected long-prefill geometry");
    const size_t packed=f.parts.at("trellis").size();
    const size_t clones=(cold_bytes+packed-1)/packed;
    std::cout<<"FIXTURE path="<<path<<" I="<<f.i<<" N="<<f.n<<" bits="<<f.bits<<" codebook="<<f.cb<<" fixture_rows="<<f.rows<<" clones="<<clones<<" cloned_bytes="<<clones*packed<<"\n";
    for(const char* key:{"q","core","scaled","inh","old"}) f.parts.erase(key);
    Device weights(clones*packed),u(2*f.i),v(2*f.n),bias(f.bias?4*f.n:0);
    for(size_t c=0;c<clones;++c) copy(static_cast<char*>(weights.p)+c*packed,f.parts.at("trellis"),stream);
    copy(u.p,f.parts.at("suh"),stream); copy(v.p,f.parts.at("svh"),stream);
    if(f.bias) copy(bias.p,f.parts.at("bias"),stream);
    hip_check(hipStreamSynchronize(stream));
    for(int m:{255,256,257,512}) {
        if(m>int(f.rows)) throw std::runtime_error("Required M reference unavailable");
        const size_t y_stride=size_t(m)*f.n*4;
        Guarded x(size_t(m)*f.i*4),y(clones*y_stride);
        copy(x.p,f.parts.at("x"),stream,x.bytes);
        auto ref=f.floats("ref"); ref.resize(size_t(m)*f.n);
        std::vector<float> actual(ref.size());
        // Baseline and sweep share exact kernels. Report their identities explicitly.
        const int dt=default_tile(f,m);
        int ds=policy_splits(f,m,16);
        if(dt==-65) { ds=1; while(size_t((m+63)/64)*((f.n+63)/64)*ds<512 && ds<16) ds*=2; }
        std::cout<<"BASELINE rows="<<m<<" backend_tile="<<dt<<" backend_splits="<<ds
                 <<" forced_m16_policy_splits="<<policy_splits(f,m,16)<<"\n";
        for(int tile:{16,-65}) for(int splits:{1,2,4,8,16}) {
            // Boundary points are correctness gates for both old and new policies.
            const size_t groups=size_t((m+63)/64)*((f.n+63)/64);
            int candidate=1; const size_t target=m>64 && m%64?512:256;
            while(groups*candidate<target && candidate<16) candidate*=2;
            if((m==255 || m==257) && splits!=(tile==16?policy_splits(f,m,16):candidate)) continue;
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
                strix_trellis::prefill_test_output_transform<<<(size_t(m)*(f.n/128)+3)/4,128,0,stream>>>(reinterpret_cast<float*>(partial.p),w.svh,w.bias,reinterpret_cast<float*>(y.p+c*y_stride),f.n,m,splits,1.0f,0.0f);
                hip_check(hipGetLastError());
            };
            std::string id=" rows="+std::to_string(m)+" tile="+std::to_string(tile)+" splits="+std::to_string(splits);
            auto check=[&](const char* phase,bool all,bool rerun=true) {
                for(size_t c=0;c<(all?clones:1);++c) {
                    if(rerun) submit(c);
                    hip_check(hipMemcpyAsync(actual.data(),y.p+c*y_stride,y_stride,hipMemcpyDeviceToHost,stream));
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
            check("final_timed",true,false);check("after",true);
            std::vector<double> warm;
            for(int rep=0;rep<rotations;++rep) {
                hip_check(hipEventCreate(&start));hip_check(hipEventCreate(&stop));
                hip_check(hipEventRecord(start,stream));
                for(int k=0;k<16;++k) submit(0);
                hip_check(hipEventRecord(stop,stream));hip_check(hipEventSynchronize(stop));
                float ms;hip_check(hipEventElapsedTime(&ms,start,stop));warm.push_back(ms*1000.0/16);
                hip_check(hipEventDestroy(start));hip_check(hipEventDestroy(stop));
            }
            check("warm_final_timed",false,false);
            std::sort(warm.begin(),warm.end());
            std::sort(times.begin(),times.end());
            std::cout<<"RESULT"<<id<<" mode=cold timed_outputs_all_clones=1"
                     <<" output_per_clone_bytes="<<y_stride<<" resident_outputs_bytes="<<y.bytes
                     <<" original_v7_default="<<(tile==dt && splits==ds)
                     <<" candidate_policy="<<(tile==-65 && splits==candidate)
                     <<" m16_policy="<<(tile==16 && splits==policy_splits(f,m,16))
                     <<" partial_shape="<<splits<<","<<m<<","<<f.n<<" workspace_bytes="<<abytes+pbytes
                     <<" gpu_allocation_bound="<<bound<<" rotations="<<rotations
                     <<" event_min_us="<<times.front()<<" event_median_us="<<times[5]<<" event_p90_us="<<times[9]
                     <<" warm_min_us="<<warm.front()<<" warm_median_us="<<warm[5]<<" warm_p90_us="<<warm[9]
                     <<" useful_TOPS="<<2.0*m*f.i*f.n/times[5]/1e6
                     <<" checksum="<<std::accumulate(actual.begin(),actual.end(),0.0)<<" canaries=PASS\n";
        }
    }
}
}
static void run_api(Runtime& r,const std::string& path) {
    Fixture f(path);
    for(const char* key:{"q","core","scaled","inh","a","old"}) f.parts.erase(key);
    const size_t packed=f.parts.at("trellis").size();
    const size_t clones=(cold_bytes+packed-1)/packed;
    std::vector<std::unique_ptr<Weight>> weights;
    for(size_t c=0;c<clones;++c) weights.emplace_back(new Weight(r,f));
    for(int m:{255,256,257,512}) {
        const size_t y_stride=size_t(m)*f.n*4;
        Guarded x(size_t(m)*f.i*4), y(clones*y_stride);
        copy(x.p,f.parts.at("x"),r.stream,x.bytes);
        auto d=desc(weights[0]->w,x.p,y.p,m,f,false,1,0);
        uint64_t bytes=0;
        api_check(strix_matmul_workspace_bytes(r.context,&d,&bytes));
        strix_trellis::DeviceWeights shape{nullptr,nullptr,nullptr,nullptr,int(f.i),int(f.n),int(f.bits),int(f.cb)};
        int splits=1; const size_t groups=size_t((m+63)/64)*((f.n+63)/64);
        const size_t target=m%64?512:256;
        while(groups*splits<target && splits<16) splits*=2;
        const size_t expected=((size_t(m)*f.i*2+255)&~size_t(255))+size_t(splits)*m*f.n*4;
        if(strix_trellis::wmma_tile_rows(shape,m)!=64 || bytes!=expected)
            throw std::runtime_error("Candidate query/tile mismatch");
        Guarded workspace(bytes); d.workspace=workspace.p;d.workspace_bytes=bytes;
        const size_t bound=clones*(packed+2ull*(f.i+f.n)+4*255)+x.bytes+y.bytes+bytes+6*guard;
        if(bound>=(1ull<<30)) throw std::runtime_error("API exceeds 1 GiB");
        // Exercise driver pointer validation with this actual shape and allocation.
        auto bad=d;bad.input.data=f.parts.at("x").data();
        if(strix_gemm(r.context,&bad,&r.s)!=STRIX_ERROR_INVALID_ARGUMENT)
            throw std::runtime_error("Host pointer was not rejected");
        bad=d;bad.workspace_bytes=bytes-1;
        if(strix_gemm(r.context,&bad,&r.s)!=STRIX_ERROR_INSUFFICIENT_BUFFER)
            throw std::runtime_error("Short workspace was not rejected");
        auto ref=f.floats("ref");ref.resize(size_t(m)*f.n);
        std::vector<float> actual(ref.size());
        auto submit=[&](size_t c) {
            d.weights=weights[c]->w;d.output.data=y.p+c*y_stride;
            api_check(strix_gemm(r.context,&d,&r.s));
        };
        const std::string id=" rows="+std::to_string(m)+" splits="+std::to_string(splits);
        auto check=[&](const char* phase,bool all,bool rerun=true) {
            for(size_t c=0;c<(all?clones:1);++c) {
                if(rerun) submit(c);
                hip_check(hipMemcpyAsync(actual.data(),y.p+c*y_stride,y_stride,hipMemcpyDeviceToHost,r.stream));
                hip_check(hipStreamSynchronize(r.stream));
                compare(std::string("API_CHECK phase=")+phase+id+" clone="+(all?std::to_string(c):"final"),actual,ref);
            }
            x.check();y.check();workspace.check();
        };
        check("before",true);
        for(size_t c=0;c<clones;++c) submit(c);
        hip_check(hipStreamSynchronize(r.stream));
        // Event time includes real pointer validation submission effects; wall
        // time additionally synchronizes each operator, as in our V7 API bench.
        for(bool wall:{false,true}) {
            hipEvent_t start,stop;hip_check(hipEventCreate(&start));hip_check(hipEventCreate(&stop));
            std::vector<double> times;
            for(int rep=0;rep<rotations;++rep) {
                auto begin=std::chrono::steady_clock::now();
                hip_check(hipEventRecord(start,r.stream));
                for(size_t c=0;c<clones;++c) {
                    submit((c+rep)%clones);
                    if(wall) hip_check(hipStreamSynchronize(r.stream));
                }
                hip_check(hipEventRecord(stop,r.stream));hip_check(hipEventSynchronize(stop));
                float ms;hip_check(hipEventElapsedTime(&ms,start,stop));
                double us=wall?std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count():ms*1000.0;
                times.push_back(us/clones);
            }
            hip_check(hipEventDestroy(start));hip_check(hipEventDestroy(stop));
            check(wall?"wall_final_timed":"event_final_timed",true,false);
            std::sort(times.begin(),times.end());
            std::cout<<"API_RESULT"<<id<<" mode=cold timing="<<(wall?"wall_sync_each":"event")
                     <<" timed_outputs_all_clones=1 output_per_clone_bytes="<<y_stride<<" resident_outputs_bytes="<<y.bytes
                     <<" tile=-65 pointer_validation=PASS query=PASS workspace_bytes="<<bytes
                     <<" clones="<<clones<<" cloned_bytes="<<clones*packed<<" gpu_allocation_bound="<<bound
                     <<" rotations="<<rotations<<" min_us="<<times[0]<<" median_us="<<times[5]<<" p90_us="<<times[9]
                     <<" useful_TOPS="<<2.0*m*f.i*f.n/times[5]/1e6<<"\n";
        }
        check("after",true);
    }
}
int main(int argc,char** argv) {
    try {
        if(argc<2) throw std::runtime_error("Usage: prefill_long_bench FIXTURE...");
        std::cout<<std::unitbuf<<std::setprecision(10);
        Runtime runtime;
        auto& stream=runtime;
        hipDeviceProp_t p{}; hip_check(hipGetDeviceProperties(&p,0));
        if(std::string(p.gcnArchName).find("gfx1151")!=0 || p.warpSize!=32) throw std::runtime_error("Requires gfx1151 wave32");
        std::cout<<"HARDWARE name="<<p.name<<" arch="<<p.gcnArchName<<" wave="<<p.warpSize<<" multiprocessors="<<p.multiProcessorCount
                 <<" nominal_clock_kHz="<<p.clockRate<<" nominal_memory_clock_kHz="<<p.memoryClockRate<<" totalGlobalMem="<<p.totalGlobalMem
                 <<" whole_device_operator=1 C_API=separate event_rotation_amortized=1\n";
        for(int k=1;k<argc;++k) { run(argv[k],stream.stream); run_api(runtime,argv[k]); }
        hip_check(hipStreamSynchronize(stream.stream));
        runtime.close();
        std::cout<<"PASS prefill_long\n"; return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL "<<e.what()<<"\n"; return 1; }
}
