// SOURCE/BUILD ONLY. No controller in this experiment requests a GPU turn.
#include "kernels.hip"
#include "owner.hpp"
#include <chrono>
#include <cmath>
#include <iostream>

namespace grouped_k3_math {
struct Snapshot { std::vector<unsigned char> a,p,y; };
static std::vector<Snapshot> snapshot(const NativeBatch& b) {
    std::vector<Snapshot> result(b.count);
    for(int g=0;g<b.count;++g) {
        auto& s=result[g];const auto& v=b.groups[g];
        s.a.resize(size_t(b.input)*2);s.p.resize(size_t(b.splits)*b.output*4);s.y.resize(size_t(b.output)*4);
        hip(hipMemcpy(s.a.data(),v.a,s.a.size(),hipMemcpyDeviceToHost));
        hip(hipMemcpy(s.p.data(),v.partial,s.p.size(),hipMemcpyDeviceToHost));
        hip(hipMemcpy(s.y.data(),v.y,s.y.size(),hipMemcpyDeviceToHost));
    }
    return result;
}
static void exact(const std::vector<Snapshot>& a,const std::vector<Snapshot>& b,bool yonly=false) {
    need(a.size()==b.size(),"snapshot group count");
    for(size_t g=0;g<a.size();++g)need(a[g].y==b[g].y && (yonly || (a[g].a==b[g].a && a[g].p==b[g].p)),"original/grouped A/P/Y bytes differ; STOP");
}
static void metrics(const unsigned char* actual,const unsigned char* reference,size_t count,unsigned g,int split,const char* stage,bool enforce=true) {
    double maxabs=0,maxref=0,ee=0,rr=0;size_t worst=0;
    for(size_t j=0;j<count;++j) {
        float a;double r;std::memcpy(&a,actual+4*j,4);std::memcpy(&r,reference+8*j,8);
        need(std::isfinite(a) && std::isfinite(r),"nonfinite comparison; STOP");
        double e=double(a)-r;if(std::abs(e)>maxabs){maxabs=std::abs(e);worst=j;}
        maxref=std::max(maxref,std::abs(r));ee+=e*e;rr+=r*r;
    }
    double nrms=std::sqrt(ee/std::max(rr,1e-30));
    std::printf("CPU %s group=%u split=%d maxabs=%.12g nrms=%.12g maxref=%.12g worst=%zu enforce=%d\n",stage,g,split,maxabs,nrms,maxref,worst,enforce);
    if(enforce)need(maxabs<=5e-4*std::max(1.,maxref) && nrms<=3e-5,"frozen CPU envelope exceeded; STOP");
}
static void cpu_check(const LabRAIIOwner& owner,const NativeBatch& b,unsigned geometry,const std::vector<Snapshot>& result) {
    for(int g=0;g<b.count;++g) {
        const auto& c=owner.cpu(geometry,g);const auto& s=result[g];
        need(s.a==c.get("a"),"RN32/RN16/H128 A not exact; STOP");
        for(int split=0;split<b.splits;++split)
            metrics(s.p.data()+size_t(split)*b.output*4,c.get("p64").data()+size_t(split)*b.output*8,b.output,g,split,"P64");
        std::vector<double> y(b.output);
        for(int j=0;j<b.output;++j){float v;std::memcpy(&v,c.get("y").data()+4*j,4);y[j]=v;}
        metrics(s.y.data(),reinterpret_cast<const unsigned char*>(y.data()),b.output,g,-1,"Yordered");
        // Full original-basis FP64 reports transform/quantization deviation
        // separately; it is never described as native fdot2 scalar identity.
        metrics(s.y.data(),c.get("y64").data(),b.output,g,-1,"YoriginalFP64",false);
        double ratio=0;
        for(size_t j=0;j<s.p.size()/4;++j) {
            float actual;double ref,abs;std::memcpy(&actual,s.p.data()+4*j,4);
            std::memcpy(&ref,c.get("p64").data()+8*j,8);std::memcpy(&abs,c.get("sumabs").data()+8*j,8);
            ratio=std::max(ratio,std::abs(double(actual)-ref)/std::max(abs,1e-300));
        }
        std::printf("P_error_over_sumabs group=%d max=%.12g empirical_only=1\n",g,ratio);
    }
}
class Graph {
    const std::shared_ptr<LabRAIIOwner> lease_;
    const NativeBatch captured_;
    const hipStream_t stream_;
    hipGraph_t graph_{};
    hipGraphExec_t executable_{};
    void release() noexcept {
        AttrRegion observation(attribute_trace,AttrPhase::cleanup,lease_->creation());
        cleanup(hipStreamSynchronize(stream_));
        if(executable_)cleanup(hipGraphExecDestroy(executable_));
        if(graph_)cleanup(hipGraphDestroy(graph_));
        executable_=nullptr;graph_=nullptr;
    }
public:
    Graph(const PreparedPlan& p,const PreparedPlan& replacement):lease_(p.lease_),captured_(p.batch_),stream_(p.stream_) {
        AttrRegion observation(attribute_trace,AttrPhase::capture,lease_->creation());
        struct ConstructionGuard {Graph* graph;bool armed=true;~ConstructionGuard(){if(armed)graph->release();}} guard{this};
        alignas(NativeBatch) unsigned char storage[sizeof(NativeBatch)];
        auto a=new(storage) NativeBatch(p.batch_);
        hip(hipStreamBeginCapture(stream_,hipStreamCaptureModeThreadLocal));
        auto e=launch_three(*a,p.grids_,stream_);
        auto end=hipStreamEndCapture(stream_,&graph_);
        a->~NativeBatch();
        if(e!=hipSuccess || end!=hipSuccess) {
            if(graph_)cleanup(hipGraphDestroy(graph_));graph_=nullptr;
            hip(e!=hipSuccess?e:end);
        }
        // Every active group's pointers change to valid live other-geometry
        // allocations, metadata changes, and scalars change by group rotation.
        NativeBatch b=replacement.batch_;
        for(int g=0;g<b.count;++g)b.groups[g]=replacement.batch_.groups[(g+1)%b.count];
        auto second=new(storage) NativeBatch(b);
        auto observed=reinterpret_cast<volatile unsigned char*>(second);
        auto source=reinterpret_cast<const unsigned char*>(&b);
        uint64_t checksum=0,wanted=0;
        for(size_t j=0;j<sizeof(b);++j)observed[j]=source[j];
        std::atomic_signal_fence(std::memory_order_seq_cst);
        for(size_t j=0;j<sizeof(b);++j){checksum=checksum*131+observed[j];wanted=wanted*131+source[j];}
        need(checksum==wanted && std::memcmp(second,&captured_,sizeof(b)),"observed replacement storage");
        second->~NativeBatch(); // BOTH lifetimes end before instantiate.
        size_t count=0;hip(hipGraphGetNodes(graph_,nullptr,&count));need(count==3,"three actual kernel nodes");
        hipGraphNode_t nodes[3],ordered[3]{};hip(hipGraphGetNodes(graph_,nodes,&count));
        const void* funcs[]={reinterpret_cast<void*>(strix_trellis::grouped_input_transform),
            reinterpret_cast<void*>(strix_trellis::grouped_baseline_core),reinterpret_cast<void*>(strix_trellis::grouped_output_transform)};
        const dim3 dims[]={p.grids_.input,p.grids_.core,p.grids_.output};
        for(auto node:nodes) {
            hipGraphNodeType type;hip(hipGraphNodeGetType(node,&type));need(type==hipGraphNodeTypeKernel,"kernel node only");
            hipKernelNodeParams params{};hip(hipGraphKernelNodeGetParams(node,&params));int stage=-1;
            for(int k=0;k<3;++k)if(params.func==funcs[k])stage=k;
            need(stage>=0 && !ordered[stage],"math function exactly once");ordered[stage]=node;
            need(params.gridDim.x==dims[stage].x && params.gridDim.y==dims[stage].y && params.gridDim.z==dims[stage].z &&
                params.blockDim.x==128 && params.blockDim.y==1 && params.blockDim.z==1 && !params.sharedMemBytes,"captured geometry");
            need(params.kernelParams && !std::memcmp(params.kernelParams[0],&captured_,sizeof(captured_)),"all2320 argument bits incl inactive zero");
        }
        size_t edges=0;hip(hipGraphGetEdges(graph_,nullptr,nullptr,&edges));need(edges==2,"two edges");
        hipGraphNode_t from[2],to[2];hip(hipGraphGetEdges(graph_,from,to,&edges));bool first=false,last=false;
        for(int j=0;j<2;++j){first|=from[j]==ordered[0] && to[j]==ordered[1];last|=from[j]==ordered[1] && to[j]==ordered[2];}
        need(first && last,"input/core/output chain");
        hip(hipGraphInstantiate(&executable_,graph_,nullptr,nullptr,0));
        // Real mathematical second Batch, not a Trace tag. Allocations stay live.
        hip(launch_three(b,grids(b),replacement.stream_));hip(hipStreamSynchronize(replacement.stream_));
        auto second_result=snapshot(b);
        need(!second_result.empty() && second_result[0].a.size()!=size_t(captured_.input)*2 &&
            second_result[0].y.size()!=size_t(captured_.output)*4,"second Batch mathematical A/P/Y differs");
        for(int g=0;g<b.count;++g) {
            const auto& c=lease_->cpu(b.input==5120?0:1,(g+1)%b.count);
            need(second_result[g].a==c.get("a"),"second Batch CPU A");
            for(int split=0;split<b.splits;++split)metrics(second_result[g].p.data()+size_t(split)*b.output*4,
                c.get("p64").data()+size_t(split)*b.output*8,b.output,g,split,"secondBatchP");
            std::vector<double> y(b.output);for(int j=0;j<b.output;++j){float v;std::memcpy(&v,c.get("y").data()+j*4,4);y[j]=v;}
            metrics(second_result[g].y.data(),reinterpret_cast<unsigned char*>(y.data()),b.output,g,-1,"secondBatchY");
        }
        std::printf("capture_copy checksum=%llu Batch2320 functions3 edges2 secondBatch_fullCPU=PASS\n",static_cast<unsigned long long>(checksum));
        guard.armed=false;
    }
    ~Graph() {
        release();
    }
    void reset(unsigned geometry,unsigned request) const {lease_->reset(geometry,request,captured_.count);}
    void enqueue() const {AttrRegion observation(attribute_trace,AttrPhase::replay,lease_->creation());hip(hipGraphLaunch(executable_,stream_));}
    void wait() const {hip(hipStreamSynchronize(stream_));}
    std::vector<Snapshot> verify(unsigned geometry) const {
        auto s=snapshot(captured_);cpu_check(*lease_,captured_,geometry,s);lease_->immutable_and_canaries();return s;
    }
};

static void run(const std::string& data) {
    auto start=std::chrono::steady_clock::now();
    // One actual context/resource-probe set for the complete campaign. Context,
    // weights and all writer generations are retained by the graph leases.
    auto owner=LabRAIIOwner::create(data,1);
    for(unsigned geometry=0;geometry<2;++geometry) {
        auto plan=owner->prepare(geometry,0,32);need(bool(plan),"negative control plan");
        auto before=snapshot(plan->batch());owner->negatives(geometry);exact(before,snapshot(plan->batch()));
        owner->immutable_and_canaries();
    }
    for(unsigned geometry=0;geometry<2;++geometry)for(unsigned count:{1,2,4,8,16,31,32}) {
        auto a=owner->prepare(geometry,0,count),b=owner->prepare(geometry,1,count);
        need(a && b,"all groups prepared before enqueue");
        need(!owner->prepare(geometry,0,0) && !owner->prepare(geometry,0,33) && !owner->prepare(geometry,0,count,2),"count/stale preparation negatives");
        owner->reset(geometry,0,count);owner->original(*a,true);
        auto full=snapshot(a->batch());
        owner->reset(geometry,0,count);owner->original(*a,false);hip(hipStreamSynchronize(owner->stream(0)));
        auto original=snapshot(a->batch());exact(full,original,true);cpu_check(*owner,a->batch(),geometry,original);
        owner->reset(geometry,0,count);owner->reset(geometry,1,count);
        hip(submit(*a));hip(submit(*b)); // BOTH enqueue before either wait.
        hip(hipStreamSynchronize(owner->stream(0)));hip(hipStreamSynchronize(owner->stream(1)));
        exact(original,snapshot(a->batch()));exact(original,snapshot(b->batch()));owner->immutable_and_canaries();
        owner->reset(geometry,0,count);owner->reset(geometry,1,count);
        auto ra=owner->prepare(1-geometry,0,count==32?31:32),rb=owner->prepare(1-geometry,1,count==32?31:32);
        owner->reset(1-geometry,0,count==32?31:32);owner->reset(1-geometry,1,count==32?31:32);
        auto ga=std::make_unique<Graph>(*a,*ra),gb=std::make_unique<Graph>(*b,*rb);
        std::weak_ptr<LabRAIIOwner> weak=owner;
        a.reset();b.reset();ra.reset();rb.reset();owner.reset(); // ONLY graph leases retain everything.
        need(!weak.expired(),"graph-only lease lost actual owner");
        for(unsigned replay=0;replay<2;++replay) {
            ga->reset(geometry,0);gb->reset(geometry,1);
            ga->enqueue();gb->enqueue();ga->wait();gb->wait();
            exact(original,ga->verify(geometry));exact(original,gb->verify(geometry));
        }
        // Retain the same real owner before destroying this pair of graph leases.
        owner=weak.lock();need(bool(owner),"live graph lease");
        {AttrRegion observation(attribute_trace,AttrPhase::reuse,owner->creation());ga.reset();gb.reset();}
        need(attribute_trace.complete(),"late attribute query or overflow; STOP");
        std::printf("PASS config geometry=%u G=%u all_A_P_Y direct/replay/twostream/LAST\n",geometry,count);
        need(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<145.,"145s budget exhausted; STOP; ROOT split decision required");
    }
    owner.reset();need(attribute_trace.complete(),"attribute final gate; STOP");std::cout<<"PASS native mathematical QA only; no performance claim\n";
}
}
static void dump_attributes() {
    using namespace grouped_k3_math;
    const char* names[]={"idle","creation","math","prepare","submit","capture","replay","reuse","cleanup"};
    const char* bindings[]={"V13-g4-K5-Direct","V13-g4-K6-Direct","V13-grouped-K5","V13-grouped-K6",
        "math-input","math-core","math-output"};
    size_t known=0,admitted=0;
    for(size_t j=0;j<attribute_trace.creations;++j) {
        known+=attribute_trace.contexts[j].handle!=nullptr;
        admitted+=attribute_trace.contexts[j].admitted;
    }
    const void* math[]={reinterpret_cast<const void*>(strix_trellis::grouped_input_transform),
        reinterpret_cast<const void*>(strix_trellis::grouped_baseline_core),reinterpret_cast<const void*>(strix_trellis::grouped_output_transform)};
    std::printf("attribute_scope=observed_link_boundary_user_and_static_library_calls_not_internal_driver_queries logical_creations=%zu contexts_with_handle=%zu contexts_admitted=%zu calls=%zu stored=%zu overflow=%d failed=%d first_error=%d trace_bytes=%zu\n",
        attribute_trace.creations,known,admitted,attribute_trace.calls,attribute_trace.stored,attribute_trace.overflow,attribute_trace.failed,attribute_trace.first_error,sizeof(attribute_trace));
    for(size_t j=0;j<attribute_trace.stored;++j) {
        const auto& e=attribute_trace.events[j];
        const auto begin=e.creation?attribute_trace.contexts[e.creation-1].begin:0;
        const auto ordinal=j-begin;
        const char* binding="UNEXPECTED";
        if(e.phase==AttrPhase::creation && ordinal<4)binding=bindings[ordinal];
        if(e.phase==AttrPhase::math)for(unsigned k=0;k<3;++k)if(e.function==math[k])binding=bindings[4+k];
        std::printf("attribute_event=%zu creation=%llu context=%p function=%p phase=%s result=%d valid=%d source_order_binding=%s",
            j,static_cast<unsigned long long>(e.creation),e.context,e.function,names[static_cast<unsigned>(e.phase)],e.result,e.valid,
            binding);
        if(e.valid)std::printf(" maxthreads=%d local=%zu static=%zu const=%zu regs=%d binary=%d ptx=%d",
            e.attributes.maxThreadsPerBlock,e.attributes.localSizeBytes,e.attributes.sharedSizeBytes,
            e.attributes.constSizeBytes,e.attributes.numRegs,e.attributes.binaryVersion,e.attributes.ptxVersion);
        std::printf("\n");
    }
    for(unsigned j=3;j<9;++j)std::printf("attribute_zero_region phase=%s regions=%zu delta=%zu\n",names[j],attribute_trace.regions[j],attribute_trace.deltas[j]);
}
int main(int argc,char** argv) {
    try {grouped_k3_math::need(argc==2,"usage: math_native_qa DATA (future ROOT lease only)");grouped_k3_math::run(argv[1]);dump_attributes();return 0;}
    catch(const std::exception& e){dump_attributes();std::cerr<<"STOP "<<e.what()<<"\n";return 1;}
}
