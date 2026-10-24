// Synthetic, deterministic complete-operator A/B. No external implementation.
#include "strix_trellis.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef DENSE_R1_BASELINE
#include "../src/dense_r1.hpp"
#include "../src/dense_r1_policy.hpp"
#include "../src/tp_splitk.hpp"
#include "../src/gemv_lds.hpp"
#define input_transform dense_reference_input_transform
#define output_transform dense_reference_output_transform
#include "../src/transforms.hpp"
#undef input_transform
#undef output_transform
#endif

namespace {
void need(bool ok, const char* text) { if(!ok) throw std::runtime_error(text); }
void hip_ok(hipError_t s) { if(s != hipSuccess) throw std::runtime_error(hipGetErrorString(s)); }
void api_ok(strix_status s) { if(s != STRIX_SUCCESS) throw std::runtime_error(strix_status_string(s)); }
struct Buffer {
    void* allocation=nullptr;
    unsigned char* data=nullptr;
    size_t bytes;
    explicit Buffer(size_t n):bytes(n) {
        hip_ok(hipMalloc(&allocation,n+512)); data=static_cast<unsigned char*>(allocation)+256;
        hip_ok(hipMemset(allocation,0xa5,n+512));
        // Preparation uses the default stream, while kernels use a nonblocking
        // stream. Complete poison writes before exposing these allocations.
        hip_ok(hipDeviceSynchronize());
    }
    ~Buffer() { if(allocation) (void)hipFree(allocation); }
    void upload(const void* p,size_t n) {
        need(n<=bytes,"upload extent"); hip_ok(hipMemcpy(data,p,n,hipMemcpyHostToDevice));
        hip_ok(hipDeviceSynchronize());
    }
    std::vector<unsigned char> download() const {
        std::vector<unsigned char> v(bytes); hip_ok(hipMemcpy(v.data(),data,bytes,hipMemcpyDeviceToHost)); return v;
    }
    void canary() const {
        std::array<unsigned char,512> v{};
        hip_ok(hipMemcpy(v.data(),allocation,256,hipMemcpyDeviceToHost));
        hip_ok(hipMemcpy(v.data()+256,data+bytes,256,hipMemcpyDeviceToHost));
        for(auto c:v) need(c==0xa5,"allocation canary");
    }
};
struct Runtime {
    hipStream_t stream=nullptr;
    strix_context context=nullptr;
    strix_stream descriptor{};
    Runtime() {
        hip_ok(hipSetDevice(0)); hipDeviceProp_t p{}; hip_ok(hipGetDeviceProperties(&p,0));
        std::printf("DENSE_DEVICE name=%s arch=%s wave=%d reported_CUs=%d memory=%zu clock_kHz=%d\n",
            p.name,p.gcnArchName,p.warpSize,p.multiProcessorCount,p.totalGlobalMem,p.clockRate);
        hip_ok(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
        descriptor.struct_size=sizeof(descriptor); descriptor.abi_version=1; descriptor.hip_stream=stream;
        strix_context_desc c{}; c.struct_size=sizeof(c); c.abi_version=1; api_ok(strix_context_create(&c,&context));
    }
    ~Runtime(){ (void)hipStreamSynchronize(stream); (void)strix_context_destroy(context); (void)hipStreamDestroy(stream); }
};
uint32_t random32(uint32_t& s) { s^=s<<13; s^=s>>17; s^=s<<5; return s; }
struct Data {
    int input,output,bits;
    std::vector<uint32_t> codes;
    std::vector<uint16_t> suh,svh;
    std::vector<float> bias,x,old;
    Data(int i,int n,int b):input(i),output(n),bits(b),codes(size_t(i)*n*b/32),suh(i),svh(n),bias(n),x(size_t(i)*8),old(size_t(n)*8) {
        uint32_t s=0x67dca521u+uint32_t(i*17+n*3+b);
        for(auto& c:codes)c=random32(s);
        // Both signs, non-unit scales, no overflow. Stable prefix across R.
        for(auto& c:suh) {
            const auto magnitude=random32(s)&0x3ffu,sign=random32(s)&1u;
            c=uint16_t(0x2c00u+magnitude+(sign<<15));
        }
        for(auto& c:svh) {
            const auto magnitude=random32(s)&0x3ffu,sign=random32(s)&1u;
            c=uint16_t(0x3000u+magnitude+(sign<<15));
        }
        for(auto& c:bias)c=float(int(random32(s)%2049)-1024)*0x1p-16f;
        for(auto& c:x)c=float(int(random32(s)%4097)-2048)*0x1p-10f;
        for(auto& c:old)c=float(int(random32(s)%2049)-1024)*0x1p-12f;
    }
};
struct Weight {
    strix_weights handle=nullptr;
    Weight(Runtime& rt,const Data& d):Weight(rt.context,rt.descriptor,d) {}
    Weight(strix_context context,const strix_stream& stream,const Data& d) {
        // Transport v1 is also accepted by the frozen v12 archive.
        strix_exl3_metadata_v1 m{};m.struct_size=sizeof(m);m.transport_version=1;
        m.bits_x2=d.bits*2;m.rows=d.output;m.columns=d.input;m.codebook=STRIX_CODEBOOK_MUL1;m.marker=STRIX_EXL3_MARKER_MUL1;
        m.trellis_bytes=d.codes.size()*4;m.suh_bytes=d.suh.size()*2;m.svh_bytes=d.svh.size()*2;m.bias_bytes=d.bias.size()*4;
        m.suh_offset=m.trellis_bytes;m.svh_offset=m.suh_offset+m.suh_bytes;m.bias_offset=m.svh_offset+m.svh_bytes;
        std::vector<unsigned char> blob(m.bias_offset+m.bias_bytes);
        std::memcpy(blob.data(),d.codes.data(),m.trellis_bytes);
        std::memcpy(blob.data()+m.suh_offset,d.suh.data(),m.suh_bytes);
        std::memcpy(blob.data()+m.svh_offset,d.svh.data(),m.svh_bytes);
        std::memcpy(blob.data()+m.bias_offset,d.bias.data(),m.bias_bytes);
        strix_exl3_transport_desc p{};p.struct_size=sizeof(p);p.abi_version=1;p.format=STRIX_TRANSPORT_EXL3_CANONICAL;
        p.transport_version=1;p.data=blob.data();p.data_bytes=blob.size();p.metadata=&m;p.metadata_bytes=sizeof(m);p.rows=d.output;p.columns=d.input;
        api_ok(strix_weights_prepare(context,&p,&stream,&handle));
    }
    ~Weight(){ if(handle)(void)strix_weights_destroy(handle); }
};
strix_tensor_desc tensor(void* p,size_t bytes,int r,int w,int stride) {
    strix_tensor_desc t{};t.struct_size=sizeof(t);t.abi_version=1;t.data=p;t.buffer_bytes=bytes;
    t.dtype=STRIX_DTYPE_F32;t.rank=2;t.shape[0]=r;t.shape[1]=w;t.strides[0]=stride;t.strides[1]=1;return t;
}
std::vector<unsigned char> strided(const std::vector<float>& data,int r,int w,int stride) {
    std::vector<unsigned char> v(size_t(r)*stride*4,0xa5);
    for(int j=0;j<r;++j)std::memcpy(v.data()+size_t(j)*stride*4,data.data()+size_t(j)*w,size_t(w)*4);
    return v;
}
std::vector<unsigned char> compact(const Buffer& buffer,int r,int w,int stride) {
    const auto raw=buffer.download();std::vector<unsigned char> v(size_t(r)*w*4);
    for(int j=0;j<r;++j){
        std::memcpy(v.data()+size_t(j)*w*4,raw.data()+size_t(j)*stride*4,size_t(w)*4);
        for(int k=w*4;k<stride*4;++k)need(raw[size_t(j)*stride*4+k]==0xa5,"row padding");
    }
    return v;
}
void equal(const std::vector<unsigned char>& a,const std::vector<unsigned char>& b,const char* stage) {
    if(a!=b) {
        size_t k=0;while(k<std::min(a.size(),b.size()) && a[k]==b[k])++k;
        std::printf("DENSE_MISMATCH stage=%s byte=%zu sizes=%zu/%zu\n",stage,k,a.size(),b.size());
        if(k+4<=a.size() && k+4<=b.size()) {
            const size_t offset=k&~size_t(3);float av,bv;
            std::memcpy(&av,a.data()+offset,4);std::memcpy(&bv,b.data()+offset,4);
            std::printf("DENSE_MISMATCH_VALUE float=%zu actual=%a expected=%a\n",offset/4,double(av),double(bv));
        }
        throw std::runtime_error(stage);
    }
}
uint64_t digest_bytes(const void* p,size_t n) {
    const auto* b=static_cast<const unsigned char*>(p);uint64_t h=1469598103934665603ull;
    for(size_t i=0;i<n;++i){h^=b[i];h*=1099511628211ull;}return h;
}
uint64_t digest(const std::vector<unsigned char>& b) { return digest_bytes(b.data(),b.size()); }
struct Graph {
    hipGraph_t graph=nullptr;hipGraphExec_t exec=nullptr;
    Graph(Runtime& rt,const std::function<void()>& call,int expected) {
        hip_ok(hipStreamBeginCapture(rt.stream,hipStreamCaptureModeThreadLocal));call();
        hip_ok(hipStreamEndCapture(rt.stream,&graph));
        size_t count=0;hip_ok(hipGraphGetNodes(graph,nullptr,&count));
        need(expected<0 || count==size_t(expected),"graph node count");
        std::vector<hipGraphNode_t> nodes(count);hip_ok(hipGraphGetNodes(graph,nodes.data(),&count));
        for(auto n:nodes){hipGraphNodeType t{};hip_ok(hipGraphNodeGetType(n,&t));need(t==hipGraphNodeTypeKernel,"non-kernel graph node");}
        hip_ok(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
    }
    ~Graph(){if(exec)(void)hipGraphExecDestroy(exec);if(graph)(void)hipGraphDestroy(graph);}
    void call(Runtime& rt){hip_ok(hipGraphLaunch(exec,rt.stream));}
};
double measure(Runtime& rt,const std::string& label,const std::function<void()>& call,int repeats,int sweeps) {
    hipEvent_t b{},e{};hip_ok(hipEventCreate(&b));hip_ok(hipEventCreate(&e));std::vector<double> v;
    for(int s=-1;s<sweeps;++s){
        hip_ok(hipEventRecord(b,rt.stream));for(int k=0;k<repeats;++k)call();hip_ok(hipEventRecord(e,rt.stream));
        hip_ok(hipEventSynchronize(e));float ms=0;hip_ok(hipEventElapsedTime(&ms,b,e));
        if(s>=0){v.push_back(ms*1000./repeats);std::printf("DENSE_SAMPLE %s sweep=%d repeats=%d us=%.6f\n",label.c_str(),s,repeats,v.back());}
    }
    hip_ok(hipEventDestroy(b));hip_ok(hipEventDestroy(e));std::sort(v.begin(),v.end());
    std::printf("DENSE_RESULT %s samples=%d median_us=%.6f pmax_us=%.6f\n",label.c_str(),sweeps,v[v.size()/2],v.back());
    std::fflush(stdout);return v[v.size()/2];
}
struct Site {const char* name;int input,output;};
const Site sites[]={{"q_a",5120,1280},{"kv",5120,512},{"q_b",1280,16384},{"o_b",4096,5120},
    {"shared_gate",5120,1152},{"shared_down",1152,5120},{"head",5120,64640}};

#ifndef DENSE_R1_BASELINE
struct RawWeight {
    Buffer codes,suh,svh,bias;
    strix_trellis::DeviceWeights w{};
    explicit RawWeight(const Data& d):codes(d.codes.size()*4),suh(d.suh.size()*2),svh(d.svh.size()*2),bias(d.bias.size()*4){
        codes.upload(d.codes.data(),codes.bytes);suh.upload(d.suh.data(),suh.bytes);svh.upload(d.svh.data(),svh.bytes);bias.upload(d.bias.data(),bias.bytes);
        w={reinterpret_cast<const uint32_t*>(codes.data),reinterpret_cast<const uint16_t*>(suh.data),reinterpret_cast<const uint16_t*>(svh.data),
            reinterpret_cast<const float*>(bias.data),d.input,d.output,d.bits,2};
    }
};
void epilogue_gate(Runtime& rt,const Data& data,const strix_trellis::DeviceWeights& weights,
    const Buffer& reference_p,Buffer& ref_y,Buffer& x,Buffer& y,int rows,int xs,int ys,int identity,
    const char* site,int repeats,int sweeps) {
    using namespace strix_trellis;
    const int splits=dense_r1_splits(weights.input,weights.output,identity!=0);
    const size_t pb=size_t(splits)*rows*weights.output*4;
    Buffer p(pb),counter(size_t(weights.output/128)*4);
    hip_ok(hipMemset(counter.data,0,counter.bytes));
    Weight api_weight(rt,data);
    const float scalars[][2]={{1,0},{-0.75f,0.25f},{0,1},{0,0}};
    for(int k=0;k<5;++k) {
        const auto scalar=scalars[k<4?k:0];
        auto w=weights;if(k==4)w.bias=nullptr;
        auto initial=strided(data.old,rows,w.output,ys);
        if(scalar[1]==0) for(int r=0;r<rows;++r)std::memset(initial.data()+size_t(r)*ys*4,0xff,size_t(w.output)*4);
        y.upload(initial.data(),initial.size());
        ref_y.upload(data.old.data(),ref_y.bytes);
        dense_reference_output_transform<<<(size_t(rows)*(w.output/128)+3)/4,128,0,rt.stream>>>(
            reinterpret_cast<const float*>(reference_p.data),w.svh,w.bias,reinterpret_cast<float*>(ref_y.data),w.output,rows,splits,scalar[0],scalar[1]);
        hip_ok(hipGetLastError());hip_ok(hipStreamSynchronize(rt.stream));const auto expected=ref_y.download();
        DenseR1Launch a{};a.projection={w,reinterpret_cast<const float*>(x.data),reinterpret_cast<float*>(y.data),uint64_t(xs),uint64_t(ys),rows,scalar[0],scalar[1],rows};
        a.partial=reinterpret_cast<float*>(p.data);a.counters=reinterpret_cast<uint32_t*>(counter.data);a.splits=splits;
        a.columns_per_block=w.output<=512?32:64;a.prefetch=2;
        hip_ok(launch_dense_r1_trusted(a,rt.stream));hip_ok(hipStreamSynchronize(rt.stream));
        equal(compact(y,rows,w.output,ys),expected,"epilogue raw exact");
        if(k==4)continue;
        strix_matmul_desc d{};strix_matmul_desc_init(&d);d.weights=api_weight.handle;
        d.input=tensor(x.data,x.bytes,rows,w.input,xs);d.output=tensor(y.data,y.bytes,rows,w.output,ys);d.alpha=scalar[0];d.beta=scalar[1];
        uint64_t bytes=0;api_ok(strix_tp_dense_r1_workspace_bytes(rt.context,&d,0,identity,&bytes));
        Buffer workspace(bytes);d.workspace=workspace.data;d.workspace_bytes=bytes;
        strix_tp_dense_plan plan=nullptr;
        api_ok(strix_tp_dense_plan_create_r1(rt.context,&d,&rt.descriptor,0,identity,&plan));
        y.upload(initial.data(),initial.size());
        {
            Graph graph(rt,[&]{api_ok(strix_tp_dense_plan_submit(plan));},1);
            d.input.data=d.output.data=nullptr;d.alpha=17; // The plan froze its values.
            graph.call(rt);hip_ok(hipStreamSynchronize(rt.stream));
            equal(compact(y,rows,w.output,ys),expected,"epilogue C API graph exact");
            const auto contents=workspace.download();
            equal(std::vector<unsigned char>(contents.begin(),contents.begin()+pb),reference_p.download(),"C API complete P exact");
            if(k==0 && repeats>0) {
                const auto schedule=dense_r1_schedule(w.input,w.output,w.bits,rows,splits);
                const std::string label="backend=dense_r1_api site="+std::string(site)+" bits="+std::to_string(w.bits)+
                    " rows="+std::to_string(rows)+" splits="+std::to_string(splits)+
                    " implementation="+(schedule.use_splitk04?"splitk04":"native");
                measure(rt,label,[&]{graph.call(rt);},repeats,sweeps);
                equal(compact(y,rows,w.output,ys),expected,"timed C API graph exact");
            }
        }
        api_ok(strix_tp_dense_plan_destroy(plan));workspace.canary();
    }
    p.canary();counter.canary();
    std::printf("DENSE_EPILOGUE input=%d output=%d bits=%d rows=%d splits=%d scalars=4 absent_bias=PASS NaN_old_beta0=PASS frozen_plan_graph=PASS\n",
        weights.input,weights.output,weights.bits,rows,splits);
}
void concurrent_api_gate(Runtime& rt,const Data& data,Buffer& x,int rows,int xs,int identity,
    const std::vector<unsigned char>& expected,const Buffer& reference_p) {
    Weight weight(rt,data);
    Buffer y0(expected.size()),y1(expected.size());
    strix_matmul_desc d{};strix_matmul_desc_init(&d);d.weights=weight.handle;
    d.input=tensor(x.data,x.bytes,rows,data.input,xs);
    d.output=tensor(y0.data,y0.bytes,rows,data.output,data.output);
    uint64_t bytes=0;api_ok(strix_tp_dense_r1_workspace_bytes(rt.context,&d,0,identity,&bytes));
    Buffer ws0(bytes),ws1(bytes);
    hipStream_t other=nullptr;hip_ok(hipStreamCreateWithFlags(&other,hipStreamNonBlocking));
    auto stream=rt.descriptor;stream.hip_stream=other;
    strix_tp_dense_plan p0=nullptr,p1=nullptr;
    d.workspace=ws0.data;d.workspace_bytes=bytes;
    api_ok(strix_tp_dense_plan_create_r1(rt.context,&d,&rt.descriptor,0,identity,&p0));
    d.workspace=ws1.data;d.output=tensor(y1.data,y1.bytes,rows,data.output,data.output);
    api_ok(strix_tp_dense_plan_create_r1(rt.context,&d,&stream,0,identity,&p1));
    for(int k=0;k<64;++k){api_ok(strix_tp_dense_plan_submit(p0));api_ok(strix_tp_dense_plan_submit(p1));}
    hip_ok(hipStreamSynchronize(rt.stream));hip_ok(hipStreamSynchronize(other));
    equal(y0.download(),expected,"concurrent API stream 0 Y");equal(y1.download(),expected,"concurrent API stream 1 Y");
    const auto ep=reference_p.download(),a=ws0.download(),b=ws1.download();
    equal(std::vector<unsigned char>(a.begin(),a.begin()+ep.size()),ep,"concurrent API stream 0 P");
    equal(std::vector<unsigned char>(b.begin(),b.begin()+ep.size()),ep,"concurrent API stream 1 P");
    api_ok(strix_tp_dense_plan_destroy(p0));api_ok(strix_tp_dense_plan_destroy(p1));hip_ok(hipStreamDestroy(other));
    y0.canary();y1.canary();ws0.canary();ws1.canary();
    std::printf("DENSE_CONCURRENT input=%d output=%d bits=%d rows=%d identity=%d shared_readonly_weights_X=PASS disjoint_workspaces=PASS P=PASS Y=PASS repeats=64\n",
        data.input,data.output,data.bits,rows,identity);
}
void candidates(Runtime& rt,const Site& site,const Data& data,int rows,const std::string& mode,int repeats,int sweeps,const std::string& v12_dump) {
    using namespace strix_trellis;
    RawWeight weight(data);const auto w=weight.w;
    Buffer input(size_t(rows)*data.input*4),rotated(size_t(rows)*data.input*2),ref_y(size_t(rows)*data.output*4);
    input.upload(data.x.data(),input.bytes);
    dense_reference_input_transform<<<(size_t(rows)*(w.input/128)+3)/4,128,0,rt.stream>>>(reinterpret_cast<const float*>(input.data),w.suh,reinterpret_cast<_Float16*>(rotated.data),w.input,rows);
    hip_ok(hipGetLastError());hip_ok(hipStreamSynchronize(rt.stream));const auto reference_a=rotated.download();
    const int xs=data.input+7,ys=data.output+11;
    Buffer x(size_t(rows)*xs*4),y(size_t(rows)*ys*4);
    const auto x_initial=strided(data.x,rows,data.input,xs),y_initial=strided(data.old,rows,data.output,ys);
    x.upload(x_initial.data(),x_initial.size());
    for(int identity=0;identity<2;++identity) {
        const int splits=dense_r1_splits(w.input,w.output,identity!=0);
        if(identity && splits==4)continue;
        Buffer ref_p(size_t(splits)*rows*data.output*4);
        hip_ok(launch_gemv_lds_variant(w,reinterpret_cast<const _Float16*>(rotated.data),reinterpret_cast<float*>(ref_p.data),rows,rows,splits,rt.stream,0));
        dense_reference_output_transform<<<(size_t(rows)*(w.output/128)+3)/4,128,0,rt.stream>>>(reinterpret_cast<const float*>(ref_p.data),w.svh,w.bias,
            reinterpret_cast<float*>(ref_y.data),w.output,rows,splits,1,0);
        hip_ok(hipGetLastError());hip_ok(hipStreamSynchronize(rt.stream));
        const auto reference_p=ref_p.download(),reference_y=ref_y.download();
        if(!v12_dump.empty() && splits==dense_r1_splits(w.input,w.output,true)) {
            const auto path=v12_dump+"/"+site.name+"-k"+std::to_string(data.bits)+"-r"+std::to_string(rows)+".f32";
            std::ifstream file(path,std::ios::binary|std::ios::ate);
            need(bool(file) && size_t(file.tellg())==reference_y.size(),"v12 dump missing or wrong extent");
            std::vector<unsigned char> expected(reference_y.size());file.seekg(0);
            file.read(reinterpret_cast<char*>(expected.data()),expected.size());need(bool(file),"v12 dump read failed");
            equal(reference_y,expected,"LEGACY full output bytes == frozen v12");
            std::printf("DENSE_V12_EXACT site=%s bits=%d rows=%d splits=%d bytes=%zu PASS\n",site.name,data.bits,rows,splits,expected.size());
        }
        std::printf("DENSE_REFERENCE site=%s bits=%d rows=%d splits=%d A=%016llx P=%016llx Y=%016llx\n",
            site.name,data.bits,rows,splits,(unsigned long long)digest(reference_a),(unsigned long long)digest(reference_p),(unsigned long long)digest(reference_y));
        // R>1 must reproduce independent one-row calls at every row position.
        if(rows>1) {
            Buffer p1(size_t(splits)*w.output*4),y1(size_t(w.output)*4);
            for(int r=0;r<rows;++r) {
                hip_ok(launch_gemv_lds_variant(w,reinterpret_cast<const _Float16*>(rotated.data)+size_t(r)*w.input,
                    reinterpret_cast<float*>(p1.data),1,1,splits,rt.stream,0));
                dense_reference_output_transform<<<(w.output/128+3)/4,128,0,rt.stream>>>(reinterpret_cast<const float*>(p1.data),w.svh,w.bias,
                    reinterpret_cast<float*>(y1.data),w.output,1,splits,1,0);
                hip_ok(hipGetLastError());hip_ok(hipStreamSynchronize(rt.stream));
                std::vector<unsigned char> p_row(size_t(splits)*w.output*4);
                for(int s=0;s<splits;++s)std::memcpy(p_row.data()+size_t(s)*w.output*4,
                    reference_p.data()+(size_t(s)*rows+r)*w.output*4,size_t(w.output)*4);
                equal(p1.download(),p_row,"batched P == independent R1");
                equal(y1.download(),std::vector<unsigned char>(reference_y.begin()+size_t(r)*w.output*4,reference_y.begin()+size_t(r+1)*w.output*4),"batched == independent R1");
            }
        }
        std::vector<int> tiles{rows};if(rows>4)tiles.push_back(4);
        for(int tile:tiles) for(int columns:{8,16,32,64,128}) for(int outputs:{8,16}) for(int prefetch:{1,2,4}) {
            if(columns==8 && outputs==16)continue;
            if(mode=="final" || mode=="api-qualify") {
                const auto schedule=dense_r1_schedule(w.input,w.output,w.bits,rows,splits);
                if(tile!=rows || columns!=schedule.columns || prefetch!=schedule.prefetch || outputs!=schedule.outputs_per_wave)continue;
            }
            const size_t pb=size_t(splits)*rows*w.output*4;
            Buffer p(pb),counter(size_t((rows+tile-1)/tile)*(w.output/128)*4),a_dump(size_t(rows)*w.input*2);
            hip_ok(hipMemset(counter.data,0,counter.bytes));y.upload(y_initial.data(),y_initial.size());
            DenseR1Launch a{};a.projection={w,reinterpret_cast<const float*>(x.data),reinterpret_cast<float*>(y.data),uint64_t(xs),uint64_t(ys),rows,1,0,tile};
            a.partial=reinterpret_cast<float*>(p.data);a.counters=reinterpret_cast<uint32_t*>(counter.data);
            a.splits=splits;a.columns_per_block=columns;a.prefetch=prefetch;
            a.outputs_per_wave=outputs;
            bool supported=false;hip_ok(query_dense_r1_resources(a,65536,&supported));need(supported,"function resource admission");
            a.activation=reinterpret_cast<uint16_t*>(a_dump.data);hip_ok(launch_dense_r1_trusted(a,rt.stream));hip_ok(hipStreamSynchronize(rt.stream));
            equal(a_dump.download(),reference_a,"A exact");equal(p.download(),reference_p,"P exact");equal(compact(y,rows,w.output,ys),reference_y,"Y exact");
            a.activation=nullptr;
            for(unsigned rotation:{1u,17u,43u}) {
                a.schedule_rotation=rotation;hip_ok(launch_dense_r1_trusted(a,rt.stream));hip_ok(hipStreamSynchronize(rt.stream));
                equal(p.download(),reference_p,"rotated P exact");equal(compact(y,rows,w.output,ys),reference_y,"rotated Y exact");
            }
            a.schedule_rotation=0;
            const std::string label="backend=dense_r1 site="+std::string(site.name)+" bits="+std::to_string(data.bits)+" rows="+std::to_string(rows)+
                " tile="+std::to_string(tile)+" splits="+std::to_string(splits)+" columns="+std::to_string(columns)+" prefetch="+std::to_string(prefetch)+" outputs_per_wave="+std::to_string(outputs);
            Graph graph(rt,[&]{hip_ok(launch_dense_r1_trusted(a,rt.stream));},1);
            if(mode!="qualify" && mode!="api-qualify")measure(rt,label,[&]{graph.call(rt);},repeats,sweeps);
            else for(int k=0;k<64;++k)graph.call(rt);
            hip_ok(hipStreamSynchronize(rt.stream));
            equal(p.download(),reference_p,"post graph P exact");equal(compact(y,rows,w.output,ys),reference_y,"post graph Y exact");
            for(auto byte:counter.download())need(byte==0,"counter did not reset");
            p.canary();counter.canary();a_dump.canary();
            std::printf("DENSE_EXACT %s A=PASS P=PASS Y=PASS graph=1 rotations=3 counters=zero\n",label.c_str());
        }
        if(mode=="qualify" || mode=="api-qualify" || mode=="final")
            epilogue_gate(rt,data,w,ref_p,ref_y,x,y,rows,xs,ys,identity,site.name,mode=="final"?repeats:0,sweeps);
        if(mode=="api-qualify" && splits==dense_r1_splits(w.input,w.output,true) && (rows==1 || rows==6))
            concurrent_api_gate(rt,data,x,rows,xs,identity,reference_y,ref_p);
        equal(x.download(),x_initial,"post identity input unchanged");
        equal(rotated.download(),reference_a,"post identity A unchanged");
        if(splits==4 && tp_dense_geometry(w.input,w.output) && mode!="qualify" && mode!="api-qualify") {
            y.upload(y_initial.data(),y_initial.size());
            const TpDenseArguments args{w,reinterpret_cast<const float*>(x.data),reinterpret_cast<float*>(y.data),uint64_t(xs),uint64_t(ys),rows,1,0,rows};
            Graph graph(rt,[&]{hip_ok(launch_tp_dense_trusted(args,rt.stream));},1);graph.call(rt);hip_ok(hipStreamSynchronize(rt.stream));
            equal(compact(y,rows,w.output,ys),reference_y,"tp03 S4 exact");
            measure(rt,"backend=tp03 site="+std::string(site.name)+" bits="+std::to_string(data.bits)+" rows="+std::to_string(rows)+" splits=4",
                [&]{graph.call(rt);},repeats,sweeps);
        }
        // Existing splitk04 is a separate baseline with its original selector.
        if(tp_split_geometry(w.input,w.output) && splits==tp_split_count(w.input,w.output) && mode!="qualify" && mode!="api-qualify") {
            Buffer p(size_t(splits)*rows*w.output*4),c(size_t(w.output/128)*4);hip_ok(hipMemset(c.data,0,c.bytes));
            TpSplitLaunch a{};a.member[0]={w,reinterpret_cast<const float*>(x.data),reinterpret_cast<float*>(y.data),uint64_t(xs),uint64_t(ys),rows,1,0,rows};
            a.partial[0]=reinterpret_cast<float*>(p.data);a.counters=reinterpret_cast<uint32_t*>(c.data);
            Graph graph(rt,[&]{hip_ok(launch_tp_split_trusted(a,rt.stream));},1);graph.call(rt);hip_ok(hipStreamSynchronize(rt.stream));
            equal(p.download(),reference_p,"splitk04 P exact");equal(compact(y,rows,w.output,ys),reference_y,"splitk04 Y exact");
            const auto label="backend=splitk04 site="+std::string(site.name)+" bits="+std::to_string(data.bits)+" rows="+std::to_string(rows)+" splits="+std::to_string(splits);
            measure(rt,label,[&]{graph.call(rt);},repeats,sweeps);
        }
    }
    equal(x.download(),x_initial,"input unchanged");x.canary();y.canary();input.canary();rotated.canary();
}
#endif
void api_baseline(Runtime& rt,const Site& site,const Data& data,int rows,int repeats,int sweeps,const std::string& dump) {
    Weight weights(rt,data);Buffer x(size_t(rows)*data.input*4),y(size_t(rows)*data.output*4);x.upload(data.x.data(),x.bytes);
    strix_matmul_desc d{};strix_matmul_desc_init(&d);d.weights=weights.handle;
    d.input=tensor(x.data,x.bytes,rows,data.input,data.input);d.output=tensor(y.data,y.bytes,rows,data.output,data.output);
    uint64_t n=0;api_ok(strix_matmul_workspace_bytes(rt.context,&d,&n));Buffer ws(n);d.workspace=ws.data;d.workspace_bytes=n;
    auto call=[&]{api_ok(strix_gemv(rt.context,&d,&rt.descriptor));};call();hip_ok(hipStreamSynchronize(rt.stream));
    const auto expected=y.download();
    for(size_t k=0;k<expected.size();k+=4){float f;std::memcpy(&f,expected.data()+k,4);need(std::isfinite(f),"baseline nonfinite");}
    #ifdef DENSE_R1_BASELINE
    const char* backend="v12";
    #else
    const char* backend="v16_api";
    #endif
    const auto label="backend="+std::string(backend)+" site="+site.name+" bits="+std::to_string(data.bits)+" rows="+std::to_string(rows);
    Graph graph(rt,call,-1);measure(rt,label,[&]{graph.call(rt);},repeats,sweeps);
    equal(y.download(),expected,"baseline stable");
    std::printf("DENSE_DIGEST %s Y=%016llx\n",label.c_str(),(unsigned long long)digest(expected));
    if(!dump.empty()) {
        std::filesystem::create_directories(dump);
        std::ofstream f(dump+"/"+site.name+"-k"+std::to_string(data.bits)+"-r"+std::to_string(rows)+".f32",std::ios::binary);
        f.write(reinterpret_cast<const char*>(expected.data()),expected.size());need(bool(f),"dump output");
    }
    x.canary();y.canary();ws.canary();
}
}
int main(int argc,char** argv) {
    try {
        const std::string mode=argc>1?argv[1]:"tune",filter=argc>2?argv[2]:"all";
        const int repeats=argc>3?std::stoi(argv[3]):30,sweeps=argc>4?std::stoi(argv[4]):7;
        const std::string dump=argc>5?argv[5]:"";
        need(repeats>0 && repeats<=1000 && sweeps>0 && sweeps<=31,"timing arguments");
        Runtime rt;
        for(const auto& site:sites)if(filter=="all" || filter==site.name)for(int bits:{4,5}) {
            Data data(site.input,site.output,bits);
            std::printf("DENSE_FIXTURE site=%s bits=%d codes=%016llx suh=%016llx svh=%016llx X=%016llx\n",site.name,bits,
                (unsigned long long)digest_bytes(data.codes.data(),data.codes.size()*4),
                (unsigned long long)digest_bytes(data.suh.data(),data.suh.size()*2),
                (unsigned long long)digest_bytes(data.svh.data(),data.svh.size()*2),
                (unsigned long long)digest_bytes(data.x.data(),data.x.size()*4));
            for(int rows=1;rows<=8;++rows) {
                if(mode!="qualify" && mode!="api-qualify" && rows!=1 && rows!=2 && rows!=4 && rows!=6 && rows!=8)continue;
                #ifdef DENSE_R1_BASELINE
                api_baseline(rt,site,data,rows,repeats,sweeps,dump);
                #else
                if(mode=="baseline")api_baseline(rt,site,data,rows,repeats,sweeps,dump);
                else candidates(rt,site,data,rows,mode,repeats,sweeps,dump);
                #endif
            }
        }
        std::puts("PASS dense_r1_probe");return 0;
    } catch(const std::exception& e){std::fprintf(stderr,"FAIL dense_r1_probe: %s\n",e.what());return 1;}
}
