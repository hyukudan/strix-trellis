// Real HIP/C API correctness harness. No backend substitutes or success stubs.
#include "strix_trellis.h"
// Give header-defined stage kernels test-only symbols so linking a backend that
// also includes transforms.hpp cannot introduce duplicate global definitions.
#define input_transform gpu_test_input_transform
#define output_transform gpu_test_output_transform
#include "../src/transforms.hpp"
#undef input_transform
#undef output_transform
#include "../src/trellis_device.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

static void hip_check(hipError_t s) {
    if (s != hipSuccess) throw std::runtime_error(hipGetErrorString(s));
}
static void api_check(strix_status s) {
    if (s != STRIX_SUCCESS) throw std::runtime_error(std::string("C API: ") + strix_status_string(s));
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
static void upload(void* dst, const void* src, size_t bytes, Runtime& r) {
    if(bytes) hip_check(hipMemcpyAsync(dst,src,bytes,hipMemcpyHostToDevice,r.stream));
}
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
// Driver-backed validation must reject these calls without changing output.
static void negative_storage(Runtime& r, const Fixture& f, strix_weights w) {
    Device x(size_t(f.i)*4), y(size_t(f.n)*4);
    auto d=desc(w,x.p,y.p,1,f,false,1,0);
    uint64_t bytes=0; api_check(strix_matmul_workspace_bytes(r.context,&d,&bytes));
    Device workspace(bytes+256); d.workspace=workspace.p; d.workspace_bytes=bytes;
    std::vector<unsigned char> sentinel(size_t(f.n)*4,0x5a), actual(sentinel.size());
    upload(y.p,sentinel.data(),sentinel.size(),r);
    auto reject=[&](const char* label, const strix_matmul_desc& bad, strix_status expected) {
        const auto status=strix_gemv(r.context,&bad,&r.s);
        if(status!=expected && !(std::string(label)=="allocation_subrange" && status==STRIX_ERROR_INVALID_ARGUMENT))
            throw std::runtime_error(std::string(label)+": unexpected status "+strix_status_string(status));
        hip_check(hipGetLastError());
        hip_check(hipMemcpyAsync(actual.data(),y.p,actual.size(),hipMemcpyDeviceToHost,r.stream));
        hip_check(hipStreamSynchronize(r.stream));
        if(actual!=sentinel) throw std::runtime_error(std::string(label)+": output modified");
        std::cout<<"negative_storage="<<label<<" PASS status="<<strix_status_string(status)<<'\n';
    };
    auto bad=d; bad.input.data=const_cast<unsigned char*>(f.parts.at("x").data());
    reject("host_input",bad,STRIX_ERROR_INVALID_ARGUMENT);
    Device managed(0); hip_check(hipMallocManaged(&managed.p,size_t(f.i)*4));
    bad=d; bad.input.data=managed.p; reject("managed_input",bad,STRIX_ERROR_INVALID_ARGUMENT);
    // The supplied pointer remains inside x, but its logical tensor exceeds x.
    // Adjacent HIP allocations may make it overlap y/workspace first; either
    // invalid argument or insufficient buffer is a valid rejection precedence.
    bad=d; bad.input.data=static_cast<char*>(x.p)+4;
    reject("allocation_subrange",bad,STRIX_ERROR_INSUFFICIENT_BUFFER);
    bad=d; bad.workspace_bytes=bytes-1; reject("workspace_short",bad,STRIX_ERROR_INSUFFICIENT_BUFFER);
    bad=d; bad.workspace=static_cast<char*>(workspace.p)+4;
    reject("workspace_misaligned",bad,STRIX_ERROR_INVALID_ARGUMENT);
    bad=d; bad.workspace=y.p; reject("workspace_alias",bad,STRIX_ERROR_INVALID_ARGUMENT);
}
static std::vector<float> expected(const Fixture& f, uint32_t rows, float alpha, float beta) {
    auto ref=f.floats("ref"), old=f.floats("old"); ref.resize(size_t(rows)*f.n);
    for(size_t k=0;k<ref.size();++k) {
        // Volatile prevents host FMA contraction across the specified RN32 stages.
        volatile float a=alpha*ref[k]; volatile float b=beta*old[k]; ref[k]=a+b;
    }
    return ref;
}
static std::vector<float> individual(Runtime& r, const Fixture& f, strix_weights w,
    uint32_t rows, bool vector, bool gemv, float alpha, float beta, bool poison) {
    Device x(size_t(rows)*f.i*4), y(size_t(rows)*f.n*4);
    auto d=desc(w,x.p,y.p,rows,f,vector,alpha,beta);
    uint64_t bytes=0; api_check(strix_matmul_workspace_bytes(r.context,&d,&bytes));
    if(!rows && bytes) throw std::runtime_error("Empty matmul requires nonzero workspace");
    Device workspace(bytes); d.workspace=workspace.p; d.workspace_bytes=bytes;
    if(reinterpret_cast<uintptr_t>(workspace.p)%STRIX_TRELLIS_WORKSPACE_ALIGNMENT)
        throw std::runtime_error("HIP workspace allocation lacks required alignment");
    upload(x.p,f.parts.at("x").data(),d.input.buffer_bytes,r);
    if(beta!=0) upload(y.p,f.parts.at("old").data(),d.output.buffer_bytes,r);
    else if(poison && rows) hip_check(hipMemsetAsync(y.p,0xff,d.output.buffer_bytes,r.stream));
    // beta=0 normal mode deliberately leaves output uninitialized.
    api_check(gemv?strix_gemv(r.context,&d,&r.s):strix_gemm(r.context,&d,&r.s));
    std::vector<float> result(size_t(rows)*f.n);
    if(rows) hip_check(hipMemcpyAsync(result.data(),y.p,d.output.buffer_bytes,hipMemcpyDeviceToHost,r.stream));
    hip_check(hipStreamSynchronize(r.stream));
    std::string label=std::string(gemv?"gemv":"gemm")+" rows="+std::to_string(rows)+
        " rank="+(vector?"1":"2")+" alpha="+std::to_string(alpha)+" beta="+std::to_string(beta)+
        (poison?" poison":"");
    compare(label,result,expected(f,rows,alpha,beta)); return result;
}

// Test-only checkpoint probe: use the existing transform helper, never a core kernel.
__global__ void input_checkpoints(const float* x, const uint16_t* u, uint16_t* scaled,
                                  float* inh, uint16_t* a, int width, int rows) {
    const int group=blockIdx.x*4+threadIdx.x/32;
    if(group>=rows*(width/128)) return;
    const int lane=threadIdx.x&31;
    const size_t start=size_t(group)*128;
    float v[4];
    for(int j=0;j<4;++j) {
        size_t k=start+lane+32*j;
        float product=strix_trellis::materialize_fp32(x[k]*__half2float(__ushort_as_half(u[k%width])));
        __half h=__float2half_rn(product);
        scaled[k]=__half_as_ushort(h); v[j]=__half2float(h);
    }
    strix_trellis::hadamard128(v);
    for(int j=0;j<4;++j) {
        size_t k=start+lane+32*j; inh[k]=v[j]; a[k]=__half_as_ushort(__float2half_rn(v[j]));
    }
}
static void exact_device(Runtime& r, void* device, const std::vector<unsigned char>& ref,
                         const char* label) {
    std::vector<unsigned char> result(ref.size());
    hip_check(hipMemcpyAsync(result.data(),device,result.size(),hipMemcpyDeviceToHost,r.stream));
    hip_check(hipStreamSynchronize(r.stream));
    size_t mismatches=0, first=0;
    for(size_t k=0;k<ref.size();++k) if(result[k]!=ref[k]) { if(!mismatches) first=k; ++mismatches; }
    std::cout<<label<<" mismatched_bytes="<<mismatches<<" first_byte="<<first<<'\n';
    if(mismatches) throw std::runtime_error(std::string(label)+": bit mismatch");
}
// Probe the production decode primitive against independent Python logical Q[I,N].
// This is a weight checkpoint test, not a replacement matrix multiplication.
template<int Bits> __global__ void decoded_weights(const uint32_t* packed, uint16_t* q,
                                                  int output, int codebook) {
    const int tile=blockIdx.x;
    strix_trellis::WaveTile<Bits> data(packed+size_t(tile)*Bits*8);
    const int lane=threadIdx.x&31;
    for(int p=lane;p<256;p+=32) {
        const int row=p/16, col=p%16;
        const int position=((col&7)<<5)|((row&6)<<2)|(row&1)|((row&8)>>2)|((col&8)>>1);
        const float value=strix_trellis::codebook_value(data.state(position),codebook);
        const size_t index=size_t((tile/(output/16))*16+row)*output+(tile%(output/16))*16+col;
        q[index]=__half_as_ushort(__float2half_rn(value));
    }
}
template<int Bits> static void decode_launch(Runtime& r, const Fixture& f, void* packed, void* q) {
    hipLaunchKernelGGL((decoded_weights<Bits>),dim3((f.i/16)*(f.n/16)),dim3(32),0,r.stream,
        static_cast<const uint32_t*>(packed),static_cast<uint16_t*>(q),int(f.n),int(f.cb));
}
static void stages(Runtime& r, const Fixture& f) {
    Device packed(f.parts.at("trellis").size()), q(f.parts.at("q").size());
    upload(packed.p,f.parts.at("trellis").data(),f.parts.at("trellis").size(),r);
    switch(f.bits) {
        case 1: decode_launch<1>(r,f,packed.p,q.p); break;
        case 2: decode_launch<2>(r,f,packed.p,q.p); break;
        case 3: decode_launch<3>(r,f,packed.p,q.p); break;
        case 4: decode_launch<4>(r,f,packed.p,q.p); break;
        case 5: decode_launch<5>(r,f,packed.p,q.p); break;
        case 6: decode_launch<6>(r,f,packed.p,q.p); break;
        case 7: decode_launch<7>(r,f,packed.p,q.p); break;
        case 8: decode_launch<8>(r,f,packed.p,q.p); break;
    }
    hip_check(hipGetLastError()); exact_device(r,q.p,f.parts.at("q"),"decoded_Q_logical_I_N");
    Device x(f.parts.at("x").size()), u(2*f.i), scaled(f.parts.at("scaled").size()),
        inh(f.parts.at("inh").size()), a(f.parts.at("a").size());
    upload(x.p,f.parts.at("x").data(),f.parts.at("x").size(),r);
    upload(u.p,f.parts.at("suh").data(),2*f.i,r);
    int blocks=(f.rows*(f.i/128)+3)/4;
    hipLaunchKernelGGL(input_checkpoints,dim3(blocks),dim3(128),0,r.stream,
        static_cast<const float*>(x.p),static_cast<const uint16_t*>(u.p),
        static_cast<uint16_t*>(scaled.p),static_cast<float*>(inh.p),static_cast<uint16_t*>(a.p),int(f.i),int(f.rows));
    hip_check(hipGetLastError());
    exact_device(r,scaled.p,f.parts.at("scaled"),"scaled_fp16");
    exact_device(r,inh.p,f.parts.at("inh"),"input_hadamard_fp32");
    exact_device(r,a.p,f.parts.at("a"),"core_input_fp16");
    hipLaunchKernelGGL(strix_trellis::gpu_test_input_transform,dim3(blocks),dim3(128),0,r.stream,
        static_cast<const float*>(x.p),static_cast<const uint16_t*>(u.p),static_cast<_Float16*>(a.p),int(f.i),int(f.rows));
    hip_check(hipGetLastError()); exact_device(r,a.p,f.parts.at("a"),"production_input_transform");
    // Feed the known CPU core checkpoint directly to isolate output H/scale/bias.
    Device core(f.parts.at("core").size()), v(2*f.n), bias(f.parts.at("bias").size()), y(f.parts.at("ref").size());
    upload(core.p,f.parts.at("core").data(),f.parts.at("core").size(),r);
    upload(v.p,f.parts.at("svh").data(),2*f.n,r);
    upload(bias.p,f.parts.at("bias").data(),f.parts.at("bias").size(),r);
    hipLaunchKernelGGL(strix_trellis::gpu_test_output_transform,dim3((f.rows*(f.n/128)+3)/4),dim3(128),0,r.stream,
        static_cast<const float*>(core.p),static_cast<const uint16_t*>(v.p),static_cast<const float*>(bias.p),
        static_cast<float*>(y.p),int(f.n),int(f.rows),1,1.0f,0.0f);
    hip_check(hipGetLastError());
    std::vector<float> result(size_t(f.rows)*f.n);
    hip_check(hipMemcpyAsync(result.data(),y.p,result.size()*4,hipMemcpyDeviceToHost,r.stream));
    hip_check(hipStreamSynchronize(r.stream)); compare("output_from_known_core",result,f.floats("ref"));
}

static void grouped(Runtime& r, const Fixture& f, strix_weights w) {
    // A distinct expert with zero svh has exactly bias-only Linear(X). This catches
    // ignoring a group's weights handle, independently of any dot reduction order.
    Fixture other=f;
    std::fill(other.parts.at("svh").begin(),other.parts.at("svh").end(),0);
    std::vector<float> bias(f.n,0);
    if(f.bias) std::memcpy(bias.data(),f.parts.at("bias").data(),4*f.n);
    for(uint32_t row=0;row<f.rows;++row)
        std::memcpy(other.parts.at("ref").data()+size_t(row)*f.n*4,bias.data(),4*f.n);
    Weight second(r,other);
    const uint32_t counts[]={1,2,4,8,16};
    size_t total=0; for(auto rows:counts) if(rows<=f.rows) total+=rows;
    Device x(total*f.i*4), y(total*f.n*4);
    std::vector<strix_matmul_desc> groups;
    std::vector<std::vector<float>> individual_results;
    size_t offset=0;
    for(auto rows:counts) if(rows<=f.rows) {
        float alpha=offset? -0.75f:1.0f, beta=offset?0.25f:0.0f;
        const bool alternate=groups.size()%2==1;
        auto handle=alternate?second.w:w;
        auto* xp=static_cast<float*>(x.p)+offset*f.i;
        auto* yp=static_cast<float*>(y.p)+offset*f.n;
        upload(xp,f.parts.at("x").data(),size_t(rows)*f.i*4,r);
        if(beta!=0) upload(yp,f.parts.at("old").data(),size_t(rows)*f.n*4,r);
        groups.push_back(desc(handle,xp,yp,rows,f,rows==1,alpha,beta));
        individual_results.push_back(individual(r,alternate?other:f,handle,rows,rows==1,rows<=8,alpha,beta,false));
        offset+=rows;
    }
    // Empty groups with NULL buffers must coexist with nonempty slices.
    groups.push_back(desc(w,nullptr,nullptr,0,f,false,1,0));
    strix_grouped_moe_desc d{}; d.struct_size=sizeof(d); d.abi_version=1;
    d.groups=groups.data(); d.group_count=groups.size();
    uint64_t bytes=0; api_check(strix_grouped_moe_workspace_bytes(r.context,&d,&bytes));
    Device workspace(bytes); d.workspace=workspace.p; d.workspace_bytes=bytes;
    api_check(strix_grouped_moe(r.context,&d,&r.s));
    std::vector<float> actual(total*f.n);
    hip_check(hipMemcpyAsync(actual.data(),y.p,actual.size()*4,hipMemcpyDeviceToHost,r.stream));
    hip_check(hipStreamSynchronize(r.stream));
    offset=0;
    for(size_t k=0;k<individual_results.size();++k) {
        const size_t count=individual_results[k].size();
        std::vector<float> slice(actual.begin()+offset,actual.begin()+offset+count);
        compare("grouped_vs_individual group="+std::to_string(k),slice,individual_results[k]);
        if(std::memcmp(slice.data(),individual_results[k].data(),count*4))
            throw std::runtime_error("Grouped/individual same-route results differ in bits");
        offset+=count;
    }
    second.close();
}

int main(int argc, char** argv) {
    try {
        std::string route="all"; std::vector<std::string> paths;
        for(int k=1;k<argc;++k) {
            std::string arg=argv[k];
            if(arg=="--route" && k+1<argc) route=argv[++k]; else paths.push_back(arg);
        }
        if(paths.empty() || (route!="all" && route!="gemv" && route!="gemm" && route!="grouped" && route!="stages"))
            throw std::runtime_error("Usage: gpu_runtime [--route all|gemv|gemm|grouped|stages] fixture.stgpu [...]");
        Runtime r;
        bool checked_storage=false;
        for(const auto& path:paths) {
            Fixture f(path);
            std::cout<<"fixture="<<path<<" I="<<f.i<<" N="<<f.n<<" bits="<<f.bits
                     <<" codebook="<<f.cb<<" seed="<<f.seed<<'\n';
            Weight w(r,f);
            if(route=="all" && !checked_storage) { negative_storage(r,f,w.w); checked_storage=true; }
            if(route=="all" || route=="stages") stages(r,f);
            for(uint32_t rows: {0u,1u,2u,3u,4u,5u,6u,7u,8u,16u,17u,32u,64u,65u,128u,129u,255u,256u,257u,511u,512u}) if(rows<=f.rows) {
                for(int vector=0;vector<=(rows==1?1:0);++vector) {
                    if(route=="all" || route=="gemv") if(rows<=8)
                        individual(r,f,w.w,rows,vector!=0,true,1,0,false);
                    if(!vector && (route=="all" || route=="gemm"))
                        individual(r,f,w.w,rows,false,false,1,0,false);
                }
                if(rows && (route=="all" || route=="gemv" || route=="gemm")) {
                    bool gemv=route=="gemv" || (route=="all" && rows<=8);
                    if(!gemv || rows<=8) {
                        individual(r,f,w.w,rows,false,gemv,-0.75f,0.25f,false);
                        individual(r,f,w.w,rows,false,gemv,0,-0.5f,false);
                        individual(r,f,w.w,rows,false,gemv,1,-0.0f,true);
                    }
                }
            }
            if(route=="all" || route=="grouped") grouped(r,f,w.w);
            hip_check(hipStreamSynchronize(r.stream));
            w.close();
        }
        r.close();
        std::cout<<"PASS route="<<route<<" fixtures="<<paths.size()<<" (no performance claim)\n"; return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
