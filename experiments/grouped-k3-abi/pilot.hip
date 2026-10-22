// Native container/ownership/capture probe only. No GEMV/H128/decode.
#include <hip/hip_runtime.h>
#include "abi.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <vector>
namespace pilot {
using namespace grouped_k3_review;
static void require(bool b,const char* text) { if(!b) throw std::runtime_error(text); }
static void hip(hipError_t e) { if(e!=hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
static void cleanup(hipError_t e) noexcept {
    if(e!=hipSuccess) { std::cerr<<"STOP cleanup HIP error="<<int(e)<<"\n";std::terminate(); }
}
__device__ __forceinline__ void record(Bits& d,const NativeGroup& s) {
    // Convert pointer VALUES only. None of the eight captured pointers is dereferenced.
    d.pointers[0]=reinterpret_cast<uintptr_t>(s.packed);
    d.pointers[1]=reinterpret_cast<uintptr_t>(s.x);
    d.pointers[2]=reinterpret_cast<uintptr_t>(s.suh);
    d.pointers[3]=reinterpret_cast<uintptr_t>(s.svh);
    d.pointers[4]=reinterpret_cast<uintptr_t>(s.bias);
    d.pointers[5]=reinterpret_cast<uintptr_t>(s.y);
    d.pointers[6]=reinterpret_cast<uintptr_t>(s.a);
    d.pointers[7]=reinterpret_cast<uintptr_t>(s.partial);
    d.alpha=__float_as_uint(s.alpha);d.beta=__float_as_uint(s.beta);
}
template<unsigned Stage> __device__ __forceinline__ void trace(const NativeBatch& b,Trace* t) {
    const unsigned g=Stage==1?blockIdx.z:blockIdx.y;
    const unsigned index=(blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x;
    // The separate trusted grid and Trace allocation bound every store; Batch.count/N
    // never controls a store or pointer dereference. Every CTA owns a distinct Slot.
    Slot* slots=Stage==0?t->input:Stage==1?t->core:t->output;
    if(threadIdx.x==0) {
        Slot& s=slots[index];record(s.group,b.groups[g]);
        s.metadata[0]=b.input;s.metadata[1]=b.output;
        s.metadata[2]=b.splits;s.metadata[3]=b.count;
        s.x=blockIdx.x;s.y=blockIdx.y;s.z=blockIdx.z;
        unsigned predecessor=0;
        if constexpr(Stage==1) predecessor=t->input[g*((gridDim.x==36)?10:5)].tag;
        if constexpr(Stage==2) {
            // Predecessor indexing uses trusted launch geometry, NOT Batch fields.
            const unsigned coreX=gridDim.x==5?36:80;
            const unsigned coreY=gridDim.x==5?16:4;
            predecessor=t->core[g*coreX*coreY].tag;
        }
        s.predecessor=predecessor;
        s.tag=(0x71300000u+Stage*256+g)^predecessor;
    }
    if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x<32) {
        record(t->headers[Stage].groups[threadIdx.x],b.groups[threadIdx.x]);
        if(threadIdx.x==0) {
            t->headers[Stage].metadata[0]=b.input;t->headers[Stage].metadata[1]=b.output;
            t->headers[Stage].metadata[2]=b.splits;t->headers[Stage].metadata[3]=b.count;
        }
    }
}
extern "C" __global__ __launch_bounds__(128) void trace_inputs(NativeBatch b,Trace* t) { trace<0>(b,t); }
extern "C" __global__ __launch_bounds__(128) void trace_core(NativeBatch b,Trace* t) { trace<1>(b,t); }
extern "C" __global__ __launch_bounds__(128) void trace_outputs(NativeBatch b,Trace* t) { trace<2>(b,t); }

class Owner {
    struct Allocation { void* ptr; size_t bytes; bool trace; };
    std::vector<Allocation> allocations_;
    hipStream_t streams_[2]{};
    const uint64_t generation_;
    int device_;
    size_t bytes_=0;
    Owner(uint64_t generation,int device):generation_(generation),device_(device) {
        for(auto& s:streams_) {
            hip(hipStreamCreateWithFlags(&s,hipStreamNonBlocking));
            unsigned flags=0;hip(hipStreamGetFlags(s,&flags));
            require(flags==hipStreamNonBlocking,"prepared stream flags");
        }
    }
public:
    static std::shared_ptr<Owner> create(uint64_t generation,int device) {
        return std::shared_ptr<Owner>(new Owner(generation,device));
    }
    ~Owner() {
        // No raw-free operation is exposed. Shared plans and graphs pin this owner.
        // Synchronization and destruction take place outside Submit/capture.
        for(auto s:streams_) if(s)cleanup(hipStreamSynchronize(s));
        for(auto a:allocations_)cleanup(hipFree(a.ptr));
        for(auto s:streams_)if(s)cleanup(hipStreamDestroy(s));
        std::cout<<"owner_destroy generation="<<generation_<<" owned_peak_bytes="<<bytes_<<"\n";
    }
    uint64_t generation() const { return generation_; }
    hipStream_t stream(unsigned i) const { require(i<2,"stream index");return streams_[i]; }
    Environment environment(unsigned i) const {
        require(i<2 && streams_[i],"stream ownership");
        return {generation_,generation_,device_,device_,true};
    }
    Region allocate(size_t payload,bool traceBuffer=false) {
        void* p=nullptr;hip(hipMalloc(&p,payload+512));
        allocations_.push_back({p,payload+512,traceBuffer});bytes_+=payload+512;
        require(bytes_<128u*1024*1024,"explicit GPU budget");
        require(reinterpret_cast<uintptr_t>(p)%256==0,"allocation alignment");
        hipPointerAttribute_t attrs{};hip(hipPointerGetAttributes(&attrs,p));
        require(attrs.device==device_ && attrs.type==hipMemoryTypeDevice && attrs.devicePointer==p,"allocation device");
        hip(hipMemsetAsync(p,0xa5,payload+512,streams_[0]));
        return {reinterpret_cast<uintptr_t>(p),payload+512,256,payload,generation_,device_,true};
    }
    bool authentic(const Region& r) const {
        if(!r.allocation)return true; // optional null bias is checked by the guard
        for(auto a:allocations_) if(reinterpret_cast<uintptr_t>(a.ptr)==r.allocation)
            return !a.trace && r.allocation_bytes==a.bytes && r.owner==generation_ &&
                r.device==device_ && r.live && r.offset>=256 &&
                r.offset<=a.bytes-256 && r.capacity<=a.bytes-256-r.offset;
        return false;
    }
    bool authentic_trace(const Trace* p) const {
        for(auto a:allocations_)if(a.trace && a.bytes==sizeof(Trace)+512 &&
            reinterpret_cast<uintptr_t>(p)==reinterpret_cast<uintptr_t>(a.ptr)+256)return true;
        return false;
    }
    void check_buffers() const {
        for(auto a:allocations_) {
            std::vector<unsigned char> bytes(a.bytes);
            hip(hipMemcpy(bytes.data(),a.ptr,a.bytes,hipMemcpyDeviceToHost));
            if(a.trace) {
                require(std::all_of(bytes.begin(),bytes.begin()+256,[](auto c){return c==0xa5;}),"Trace left guard");
                require(std::all_of(bytes.end()-256,bytes.end(),[](auto c){return c==0xa5;}),"Trace right guard");
            } else require(std::all_of(bytes.begin(),bytes.end(),[](auto c){return c==0xa5;}),"payload or canary changed");
        }
    }
};
struct Fixture {
    std::array<Descriptor,32> descriptors;
    Trace* trace;
};
static Fixture fixture(const std::shared_ptr<Owner>& owner,unsigned geometry,unsigned variant,const Fixture* shared=nullptr) {
    Fixture f{};
    int i=geometry?2304:5120,n=geometry?5120:2304,s=geometry?4:16;
    const uint64_t sizes[8]={uint64_t(i)*n*3/8,uint64_t(i)*4,uint64_t(i)*2,
        uint64_t(n)*2,uint64_t(n)*4,uint64_t(n)*4,uint64_t(i)*2,uint64_t(s)*n*4};
    Region reads[5];for(unsigned k=0;k<5;++k)
        reads[k]=shared?shared->descriptors[0].regions[k]:owner->allocate(sizes[k]);
    // One allocation per field; SUM slices include 256-byte guards on each side.
    Region writes[3];size_t strides[3];
    for(unsigned k=0;k<3;++k) {
        strides[k]=(sizes[k+5]+512+255)&~size_t(255);
        writes[k]=owner->allocate(32*strides[k]);
    }
    for(unsigned g=0;g<32;++g) {
        auto& d=f.descriptors[g];d.input=i;d.output=n;d.splits=s;
        d.alpha=float(1+g+variant*64)/64;d.beta=g==0?-0.0f:-float(g+variant*32)/128;
        for(unsigned k=0;k<5;++k)d.regions[k]=reads[k];
        for(unsigned k=0;k<3;++k) {
            d.regions[k+5]=writes[k];d.regions[k+5].offset+=g*strides[k]+256;
            d.regions[k+5].capacity=sizes[k+5];
        }
    }
    auto r=owner->allocate(sizeof(Trace),true);
    f.trace=reinterpret_cast<Trace*>(r.allocation+r.offset);return f;
}
class Plan {
public:
    const NativeBatch batch;
    const std::shared_ptr<Owner> lease;
    const uint64_t generation;
    const unsigned streamIndex;
    const Trace* const traceCertificate;
    Plan(NativeBatch b,std::shared_ptr<Owner> o,unsigned stream,Trace* trace):
        batch(b),lease(std::move(o)),generation(lease->generation()),streamIndex(stream),traceCertificate(trace) {}
};
static std::optional<Plan> prepare(const std::shared_ptr<Owner>& o,const Fixture& f,
                                  unsigned count,unsigned stream,uint64_t generation) {
    if(!o || generation!=o->generation() || stream>=2 || count>32 || !o->authentic_trace(f.trace))return std::nullopt;
    for(unsigned g=0;g<count;++g)for(const auto& r:f.descriptors[g].regions)
        if(!o->authentic(r))return std::nullopt;
    grouped_k3_review::Plan p;
    if(!grouped_k3_review::prepare(f.descriptors.data(),count,o->environment(stream),p))return std::nullopt;
    return std::optional<Plan>(std::in_place,bridge(p.batch()),o,stream,f.trace);
}
struct Grids { dim3 input,core,output; };
static Grids grids(const NativeBatch& b) {
    require(b.count>=1 && b.count<=32,"trusted count");
    require((b.input==5120 && b.output==2304 && b.splits==16)||
        (b.input==2304 && b.output==5120 && b.splits==4),"trusted geometry");
    return {dim3((b.input+511)/512,b.count),dim3(b.output/64,b.splits,b.count),
        dim3((b.output+511)/512,b.count)};
}
// Only three kernel enqueues and immediate error retrieval. No storage/resource
// queries, allocations, explicit copies, synchronization or fallback here.
static hipError_t submit(const NativeBatch& b,Trace* t,const Grids& g,hipStream_t stream) {
    hipLaunchKernelGGL(trace_inputs,g.input,dim3(128),0,stream,b,t);
    auto e=hipGetLastError();if(e!=hipSuccess)return e;
    hipLaunchKernelGGL(trace_core,g.core,dim3(128),0,stream,b,t);
    e=hipGetLastError();if(e!=hipSuccess)return e;
    hipLaunchKernelGGL(trace_outputs,g.output,dim3(128),0,stream,b,t);
    return hipGetLastError();
}
static std::unique_ptr<Trace> expected(const NativeBatch& b) {
    auto t=std::make_unique<Trace>();std::memset(t.get(),0xa5,sizeof(Trace));
    auto g=grids(b);dim3 dims[3]={g.input,g.core,g.output};
    for(unsigned stage=0;stage<3;++stage) {
        auto& h=t->headers[stage];for(unsigned j=0;j<32;++j)h.groups[j]=bits(b.groups[j]);
        h.metadata[0]=b.input;h.metadata[1]=b.output;h.metadata[2]=b.splits;h.metadata[3]=b.count;
        auto d=dims[stage];Slot* slots=stage==0?t->input:stage==1?t->core:t->output;
        for(unsigned z=0;z<d.z;++z)for(unsigned y=0;y<d.y;++y)for(unsigned x=0;x<d.x;++x) {
            unsigned group=stage==1?z:y,index=(z*d.y+y)*d.x+x;
            auto& s=slots[index];s.group=bits(b.groups[group]);
            std::copy(h.metadata,h.metadata+4,s.metadata);s.x=x;s.y=y;s.z=z;
            s.predecessor=stage==0?0:stage==1?t->input[group*g.input.x].tag:
                t->core[group*g.core.x*g.core.y].tag;
            s.tag=tag(stage,group)^s.predecessor;
        }
    }
    return t;
}
static void verify(Trace* device,const Trace& wanted) {
    auto actual=std::make_unique<Trace>();hip(hipMemcpy(actual.get(),device,sizeof(Trace),hipMemcpyDeviceToHost));
    if(std::memcmp(actual.get(),&wanted,sizeof(Trace))) {
        auto a=reinterpret_cast<const unsigned char*>(actual.get());
        auto w=reinterpret_cast<const unsigned char*>(&wanted);
        for(size_t k=0;k<sizeof(Trace);++k)if(a[k]!=w[k]) {
            std::cerr<<"first_trace_difference byte="<<k<<" expected="<<unsigned(w[k])<<" actual="<<unsigned(a[k])<<"\n";break;
        }
        throw std::runtime_error("trace comparison failed; STOP");
    }
}
class Graph {
    const std::shared_ptr<Owner> lease_;
    hipGraph_t graph_{};hipGraphExec_t executable_{};
    unsigned stream_;
public:
    Graph(const Plan& p,const Plan& replacement):lease_(p.lease),stream_(p.streamIndex) {
        // Reuse the EXACT host storage after ending the captured object's lifetime.
        alignas(NativeBatch) unsigned char storage[sizeof(NativeBatch)];
        NativeBatch* temporary=new(storage) NativeBatch(p.batch);
        const auto g=grids(p.batch);auto stream=lease_->stream(stream_);
        hip(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal));
        hip(submit(*temporary,const_cast<Trace*>(p.traceCertificate),g,stream));
        hip(hipStreamEndCapture(stream,&graph_));
        temporary->~NativeBatch();
        auto second=new(storage) NativeBatch(replacement.batch);
        // Volatile writes and an observed checksum prevent dead-store elimination.
        auto bytes=reinterpret_cast<volatile unsigned char*>(second);
        const auto source=reinterpret_cast<const unsigned char*>(&replacement.batch);
        uint64_t sum=0;
        for(size_t j=0;j<sizeof(NativeBatch);++j)bytes[j]=source[j];
        std::atomic_signal_fence(std::memory_order_seq_cst);
        for(size_t j=0;j<sizeof(NativeBatch);++j)sum=sum*131+bytes[j];
        uint64_t expectedSum=0;for(size_t j=0;j<sizeof(NativeBatch);++j)expectedSum=expectedSum*131+source[j];
        require(sum==expectedSum && std::memcmp(second,&p.batch,sizeof(NativeBatch))!=0,"storage mutation");
        std::cout<<"capture_storage_reused checksum="<<sum<<" bytes="<<sizeof(NativeBatch)<<"\n";
        second->~NativeBatch(); // both temporary objects are gone BEFORE instantiate
        size_t count=0;hip(hipGraphGetNodes(graph_,nullptr,&count));require(count==3,"three captured nodes");
        hipGraphNode_t nodes[3];hip(hipGraphGetNodes(graph_,nodes,&count));
        void* functions[3]={reinterpret_cast<void*>(trace_inputs),reinterpret_cast<void*>(trace_core),reinterpret_cast<void*>(trace_outputs)};
        hipGraphNode_t stageNodes[3]{};
        dim3 dims[3]={g.input,g.core,g.output};
        for(auto node:nodes) {
            hipGraphNodeType type;hip(hipGraphNodeGetType(node,&type));require(type==hipGraphNodeTypeKernel,"kernel node type");
            hipKernelNodeParams params{};hip(hipGraphKernelNodeGetParams(node,&params));
            int stage=-1;for(int j=0;j<3;++j)if(params.func==functions[j])stage=j;
            require(stage>=0 && !stageNodes[stage],"distinct trace function identity");stageNodes[stage]=node;
            auto d=dims[stage];require(params.gridDim.x==d.x && params.gridDim.y==d.y && params.gridDim.z==d.z &&
                params.blockDim.x==128 && params.blockDim.y==1 && params.blockDim.z==1 && params.sharedMemBytes==0,"node geometry");
            require(params.kernelParams && std::memcmp(params.kernelParams[0],&p.batch,sizeof(NativeBatch))==0,"node Batch all bytes copied");
            Trace* traceArg=nullptr;std::memcpy(&traceArg,params.kernelParams[1],sizeof(traceArg));
            require(traceArg==p.traceCertificate,"separate trace argument");
        }
        size_t edges=0;hip(hipGraphGetEdges(graph_,nullptr,nullptr,&edges));require(edges==2,"two edges");
        hipGraphNode_t from[2],to[2];hip(hipGraphGetEdges(graph_,from,to,&edges));
        bool first=false,last=false;for(unsigned j=0;j<2;++j) {
            first|=from[j]==stageNodes[0] && to[j]==stageNodes[1];
            last|=from[j]==stageNodes[1] && to[j]==stageNodes[2];
        }
        require(first && last,"input to core to output chain");
        hip(hipGraphInstantiate(&executable_,graph_,nullptr,nullptr,0));
        std::cout<<"graph kernels=3 edges=2 functions/grids/Batch2320/TraceArg=PASS\n";
    }
    ~Graph() {
        cleanup(hipStreamSynchronize(lease_->stream(stream_)));
        if(executable_)cleanup(hipGraphExecDestroy(executable_));if(graph_)cleanup(hipGraphDestroy(graph_));
    }
    void enqueue() { hip(hipGraphLaunch(executable_,lease_->stream(stream_))); }
};
static void resource_gate() {
    for(auto f:{reinterpret_cast<const void*>(trace_inputs),reinterpret_cast<const void*>(trace_core),reinterpret_cast<const void*>(trace_outputs)}) {
        hipFuncAttributes a{};hip(hipFuncGetAttributes(&a,f));
        std::cout<<"resource maxThreads="<<a.maxThreadsPerBlock<<" local="<<a.localSizeBytes<<" static="<<a.sharedSizeBytes<<" regs="<<a.numRegs<<"\n";
        require(a.maxThreadsPerBlock>=128 && a.localSizeBytes==0 && a.sharedSizeBytes==0,"function resource gate; STOP");
    }
}
static void negatives(const std::shared_ptr<Owner>& owner,const Fixture& f,unsigned stream) {
    auto empty=std::make_unique<Trace>();std::memset(empty.get(),0xa5,sizeof(Trace));
    for(unsigned fault=0;fault<5;++fault) {
        auto bad=f;unsigned count=32;uint64_t generation=owner->generation();
        if(fault==0)bad.descriptors[31].regions[P].capacity--;
        if(fault==1)bad.descriptors[31].regions[A]=bad.descriptors[0].regions[A];
        if(fault==2)count=33;
        if(fault==3)bad.descriptors[31].alpha=INFINITY;
        if(fault==4)++generation;
        auto s=owner->stream(stream);hip(hipStreamBeginCapture(s,hipStreamCaptureModeThreadLocal));
        auto p=prepare(owner,bad,count,stream,generation);
        // No enqueue is reachable unless the entire real-address guard succeeds.
        if(p)hip(submit(p->batch,f.trace,grids(p->batch),s));
        hipGraph_t graph{};hip(hipStreamEndCapture(s,&graph));
        size_t nodes=0;hip(hipGraphGetNodes(graph,nullptr,&nodes));hip(hipGraphDestroy(graph));
        require(!p && nodes==0,"negative must fail before any enqueue");
        verify(f.trace,*empty);owner->check_buffers();
        std::cout<<"negative fault="<<fault<<" nodes=0 fullTrace/allBuffers=unchanged PASS\n";
    }
}
static void run() {
    auto begin=std::chrono::steady_clock::now();
    int device=0;hip(hipSetDevice(device));hipDeviceProp_t props{};hip(hipGetDeviceProperties(&props,device));
    int runtime=0,driver=0;hip(hipRuntimeGetVersion(&runtime));hip(hipDriverGetVersion(&driver));
    std::cout<<"hardware="<<props.name<<" arch="<<props.gcnArchName<<" wave="<<props.warpSize
        <<" MPs="<<props.multiProcessorCount<<" runtime="<<runtime<<" driver="<<driver<<"\n";
    require(std::string(props.gcnArchName).find("gfx1151")==0 && props.warpSize==32 && props.maxThreadsPerBlock>=128 &&
        props.maxThreadsDim[0]>=128 && props.maxGridSize[0]>=80 && props.maxGridSize[1]>=32 && props.maxGridSize[2]>=32,"device gate");
    resource_gate(); // Exactly once, inside the acquired lease and before all use.
    uint64_t generation=1;
    for(unsigned geometry=0;geometry<2;++geometry)for(unsigned count:{1,2,4,8,16,31,32}) {
        auto owner=Owner::create(generation++,device);std::weak_ptr<Owner> weak=owner;
        Fixture a=fixture(owner,geometry,0),b=fixture(owner,geometry,1,&a);
        Fixture replacement=fixture(owner,1-geometry,2);
        hip(hipStreamSynchronize(owner->stream(0))); // finish owned initialization before capture
        auto pa=prepare(owner,a,count,0,owner->generation());
        auto pb=prepare(owner,b,count,1,owner->generation());
        auto pr=prepare(owner,replacement,count==32?31:32,0,owner->generation());
        require(pa && pb && pr,"prepare valid plans");
        if(count==32)negatives(owner,a,0);
        auto ea=expected(pa->batch),eb=expected(pb->batch),er=expected(pr->batch);
        // Direct SECOND Batch produces its own complete, different descriptor trace.
        hip(submit(pr->batch,replacement.trace,grids(pr->batch),owner->stream(0)));
        hip(hipStreamSynchronize(owner->stream(0)));verify(replacement.trace,*er);
        require(std::memcmp(ea.get(),er.get(),sizeof(Trace))!=0,"second descriptor trace differs");
        Graph ga(*pa,*pr),gb(*pb,*pr);
        auto streamA=owner->stream(0),streamB=owner->stream(1);
        owner.reset();require(!weak.expired(),"external owner reset must remain pinned");
        for(unsigned replay=0;replay<2;++replay) {
            hip(hipMemsetAsync(a.trace,0xa5,sizeof(Trace),streamA));
            hip(hipMemsetAsync(b.trace,0xa5,sizeof(Trace),streamB));
            // Both independent requests enqueue BEFORE either stream synchronization.
            ga.enqueue();gb.enqueue();
            hip(hipStreamSynchronize(streamA));hip(hipStreamSynchronize(streamB));
            verify(a.trace,*ea);verify(b.trace,*eb);pa->lease->check_buffers();
            std::cout<<"geometry="<<geometry<<" G="<<count<<" replay="<<replay<<" twoStreams=PASS all32fields/allMetadata/allCTAs/allCanaries=PASS\n";
        }
    }
    std::cout<<"PASS native container trace only wall_seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()<<"\n";
}
} // namespace pilot
int main() {
    std::cout.setf(std::ios::unitbuf);
    try { pilot::run();return 0; }
    catch(const std::exception& e) { std::cerr<<"STOP first probe failure: "<<e.what()<<"\n";return 1; }
}
