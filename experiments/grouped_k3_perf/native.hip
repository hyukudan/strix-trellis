// Build only until ROOT supplies actual native QA PASS and reviews this harness.
#include "../grouped_k3_math/kernels.hip"
#include "pool.hpp"
#include "heap_guard.hpp"
#include <chrono>
#include <iostream>
namespace strix_trellis {
template<int Bits,bool Mul1,int Rows,bool Fast> __global__ void gemv_baseline_kernel(const uint32_t*,const _Float16*,float*,int,int,int,int,int,int);
}
namespace grouped_k3_perf {
using Clock=std::chrono::steady_clock;
static Clock::time_point began;
static void budget(){require(std::chrono::duration<double>(Clock::now()-began).count()<270,"270s inner budget; STOP without retry");}
struct Snapshot {std::vector<unsigned char> a,p,y;};
static Snapshot snapshot(const NativeBatch& b,unsigned j) {
    Snapshot r;const auto& g=b.groups[j];r.a.resize(b.input*2);r.p.resize(b.splits*b.output*4);r.y.resize(b.output*4);
    hip(hipMemcpy(r.a.data(),g.a,r.a.size(),hipMemcpyDeviceToHost));hip(hipMemcpy(r.p.data(),g.partial,r.p.size(),hipMemcpyDeviceToHost));hip(hipMemcpy(r.y.data(),g.y,r.y.size(),hipMemcpyDeviceToHost));return r;
}
static void metric(const unsigned char* actual,const unsigned char* ref,size_t n,unsigned id,int split,const char* stage,bool enforce=true) {
    double maximum=0,maxref=0,ee=0,rr=0;size_t worst=0;
    for(size_t j=0;j<n;++j){float a;double r;std::memcpy(&a,actual+4*j,4);std::memcpy(&r,ref+8*j,8);
        require(std::isfinite(a) && std::isfinite(r),"nonfinite full numeric check");double e=double(a)-r;
        if(std::abs(e)>maximum){maximum=std::abs(e);worst=j;}maxref=std::max(maxref,std::abs(r));ee+=e*e;rr+=r*r;}
    double nrms=std::sqrt(ee/std::max(rr,1e-30));
    std::printf("CHECK id=%u split=%d stage=%s maxabs=%.12g nrms=%.12g maxref=%.12g worst=%zu enforce=%d\n",id,split,stage,maximum,nrms,maxref,worst,enforce);
    if(enforce)require(maximum<=5e-4*std::max(1.,maxref) && nrms<=3e-5,"frozen numeric tolerance; STOP including baseline");
}
static void verify(const std::vector<std::shared_ptr<const Prepared>>& old,const std::vector<std::shared_ptr<const Prepared>>& now) {
    require(clean_submit(),"forbidden submit side effect; STOP");
    Last last;
    for(size_t c=0;c<old.size();++c)for(int j=0;j<old[c]->batch.count;++j){unsigned id=old[c]->first+j;const auto& b=old[c]->batch;
        auto a=snapshot(b,j),n=snapshot(now[c]->batch,j);require(a.a==n.a && a.p==n.p && a.y==n.y,"all A/P/Y original launch_linear vs grouped bytes");
        const auto& cpu=old[c]->lease->cpu[id];require(a.a==cpu.get("a"),"full CPU A exact");
        for(int s=0;s<b.splits;++s)metric(a.p.data()+s*b.output*4,cpu.get("p64").data()+s*b.output*8,b.output,id,s,"P64");
        std::vector<double> y(b.output);for(int k=0;k<b.output;++k){float f;std::memcpy(&f,cpu.get("y").data()+k*4,4);y[k]=f;}
        metric(a.y.data(),reinterpret_cast<const unsigned char*>(y.data()),b.output,id,-1,"Yordered");
        metric(a.y.data(),cpu.get("y64").data(),b.output,id,-1,"YoriginalFP64",false);
        double ratio=0;for(size_t k=0;k<a.p.size()/4;++k){float v;double r,sum;std::memcpy(&v,a.p.data()+4*k,4);std::memcpy(&r,cpu.get("p64").data()+8*k,8);std::memcpy(&sum,cpu.get("sumabs").data()+8*k,8);ratio=std::max(ratio,std::abs(v-r)/std::max(sum,1e-300));}
        std::printf("SUMABS id=%u ratio=%.12g empirical_only=1\n",id,ratio);last.mark(0,id);last.mark(1,id);
    }
    last.complete();old.front()->lease->immutable_and_guards();require(attribute_trace.complete(),"attribute counters changed");budget();
    std::printf("PASS_VERIFY input=%d G=%d coverage=64 ownership=PASS guards=PASS immutable=PASS RSS=PASS ledger=PASS A_P_Y=PASS\n",old.front()->batch.input,old.front()->batch.count);
}
template<class T> static void arg(const hipKernelNodeParams& p,unsigned k,const T& expected){require(p.kernelParams && !std::memcmp(p.kernelParams[k],&expected,sizeof(T)),"captured original scalar/pointer argument");}
class Captured {
    std::shared_ptr<const Prepared> plan;
    hipGraph_t graph{};hipGraphExec_t executable{};
    void inspect() {
        const auto& b=plan->batch;size_t count=0;hip(hipGraphGetNodes(graph,nullptr,&count));require(count==size_t(plan->grouped?3:3*b.count),"graph node count");
        std::vector<hipGraphNode_t> nodes(count),from(count-1),to(count-1);hip(hipGraphGetNodes(graph,nodes.data(),&count));
        size_t edges=0;hip(hipGraphGetEdges(graph,nullptr,nullptr,&edges));require(edges==count-1,"graph sequential edges");hip(hipGraphGetEdges(graph,from.data(),to.data(),&edges));
        hipGraphNode_t current=nullptr;for(auto node:nodes)if(std::find(to.begin(),to.end(),node)==to.end()){require(!current,"unique graph root");current=node;}
        const void* grouped_funcs[]={reinterpret_cast<const void*>(strix_trellis::grouped_input_transform),reinterpret_cast<const void*>(strix_trellis::grouped_baseline_core),reinterpret_cast<const void*>(strix_trellis::grouped_output_transform)};
        const void* old_funcs[]={reinterpret_cast<const void*>(strix_trellis::input_transform),reinterpret_cast<const void*>(strix_trellis::gemv_baseline_kernel<3,true,1,false>),reinterpret_cast<const void*>(strix_trellis::output_transform)};
        for(size_t index=0;index<count;++index){require(current,"connected graph chain");hipGraphNodeType type;hip(hipGraphNodeGetType(current,&type));require(type==hipGraphNodeTypeKernel,"only kernel nodes");hipKernelNodeParams p{};hip(hipGraphKernelNodeGetParams(current,&p));unsigned stage=index%3;
            require(p.func==(plan->grouped?grouped_funcs[stage]:old_funcs[stage]) && p.blockDim.x==128 && p.blockDim.y==1 && p.blockDim.z==1 && !p.sharedMemBytes,"actual graph function/block/resources");
            NativeBatch shape=b;if(!plan->grouped)shape.count=1;auto grid=grids(shape);dim3 dims[]={grid.input,grid.core,grid.output};
            require(p.gridDim.x==dims[stage].x && p.gridDim.y==dims[stage].y && p.gridDim.z==dims[stage].z,"actual graph grid");
            if(plan->grouped){require(p.kernelParams && !std::memcmp(p.kernelParams[0],&b,sizeof(b)),"captured all Batch2320 bytes");}
            else {const auto& g=b.groups[index/3];int one=1,cb=2;
                if(stage==0){arg(p,0,g.x);arg(p,1,g.suh);arg(p,2,g.a);arg(p,3,b.input);arg(p,4,one);}
                if(stage==1){arg(p,0,g.packed);arg(p,1,g.a);arg(p,2,g.partial);arg(p,3,b.input);arg(p,4,b.output);arg(p,5,one);arg(p,6,one);arg(p,7,b.splits);arg(p,8,cb);}
                if(stage==2){arg(p,0,g.partial);arg(p,1,g.svh);arg(p,2,g.bias);arg(p,3,g.y);arg(p,4,b.output);arg(p,5,one);arg(p,6,b.splits);arg(p,7,g.alpha);arg(p,8,g.beta);}}
            std::printf("GRAPH clone_start=%u variant=%d index=%zu func=%p grid=%u,%u,%u block=128 shared=0\n",plan->first,plan->grouped,index,p.func,p.gridDim.x,p.gridDim.y,p.gridDim.z);
            auto next=std::find(from.begin(),from.end(),current);current=next==from.end()?nullptr:to[next-from.begin()];
        }
        require(!current,"graph ends after exact chain");
    }
public:
    explicit Captured(std::shared_ptr<const Prepared> p):plan(std::move(p)) {
        struct Guard {Captured* self;bool active=true;~Guard(){if(active){if(self->executable)cleanup(hipGraphExecDestroy(self->executable));if(self->graph)cleanup(hipGraphDestroy(self->graph));}}} guard{this};
        {AttrRegion observation(attribute_trace,AttrPhase::capture,plan->lease->creation);
        hip(hipStreamBeginCapture(plan->lease->stream,hipStreamCaptureModeThreadLocal));auto e=submit(*plan);auto end=hipStreamEndCapture(plan->lease->stream,&graph);hip(e);hip(end);}
        inspect();hip(hipGraphInstantiate(&executable,graph,nullptr,nullptr,0));guard.active=false;
    }
    ~Captured(){cleanup(hipStreamSynchronize(plan->lease->stream));if(executable)cleanup(hipGraphExecDestroy(executable));if(graph)cleanup(hipGraphDestroy(graph));}
    void enqueue() const {hipError_t result;{SubmitScope prohibition;AttrRegion observation(attribute_trace,AttrPhase::replay,plan->lease->creation);result=hipGraphLaunch(executable,plan->lease->stream);}hip(result);}
};
struct Events {hipEvent_t a{},b{};Events(){hip(hipEventCreate(&a));hip(hipEventCreate(&b));}~Events(){cleanup(hipEventDestroy(a));cleanup(hipEventDestroy(b));}};
static void run(const std::string& data) {
    began=Clock::now();
    for(int input:{5120,2304}) {
        auto pool=Pool::create(data,input);
        for(unsigned g:selections) {
            std::vector<std::shared_ptr<const Prepared>> plans[2];
            for(unsigned v=0;v<2;++v)for(unsigned c=0;c<clones(g);++c)plans[v].push_back(pool->prepare(g,c,v));
            // Poison beta0 test outside all event intervals; every writer remains live.
            pool->poison();for(auto& variant:plans)for(const auto& p:variant)hip(submit(*p));hip(hipStreamSynchronize(pool->stream));verify(plans[0],plans[1]);
            for(unsigned mode=0;mode<2;++mode) {
                std::vector<std::unique_ptr<Captured>> graphs[2];
                if(mode)for(unsigned v=0;v<2;++v)for(auto p:plans[v])graphs[v].push_back(std::make_unique<Captured>(p));
                // Validate graph execution before timing, with NaN Y under beta0.
                pool->poison();for(unsigned v=0;v<2;++v)for(unsigned c=0;c<clones(g);++c){if(mode)graphs[v][c]->enqueue();else hip(submit(*plans[v][c]));}
                hip(hipStreamSynchronize(pool->stream));verify(plans[0],plans[1]);Events events;
                for(unsigned sweep=0;sweep<11;++sweep) {
                    for(unsigned turn=0;turn<2;++turn){unsigned v=(sweep+turn)%2;
                        for(unsigned c=0;c<clones(g);++c){budget();hip(hipEventRecord(events.a,pool->stream));auto t=Clock::now();
                            if(mode)graphs[v][c]->enqueue();else hip(submit(*plans[v][c]));
                            hip(hipEventRecord(events.b,pool->stream));hip(hipStreamSynchronize(pool->stream));auto end=Clock::now();float ms=0;hip(hipEventElapsedTime(&ms,events.a,events.b));
                            std::printf("SAMPLE {\"input\":%d,\"G\":%u,\"mode\":\"%s\",\"sweep\":%u,\"variant\":\"%s\",\"clone\":%u,\"event_ms\":%.9g,\"wall_ms\":%.9g}\n",input,g,mode?"graph":"direct",sweep,v?"new":"V13",c,ms,std::chrono::duration<double,std::milli>(end-t).count());}}
                    // LAST of EACH sweep checks all 128 instances before any overwrite.
                    hip(hipStreamSynchronize(pool->stream));verify(plans[0],plans[1]);
                }
                verify(plans[0],plans[1]);
            }
        }
        pool.reset();require(attribute_trace.complete(),"cleanup attribute delta");
    }
    std::printf("PASS physical performance campaign all fixed cells; contexts=%zu calls=%zu\n",attribute_trace.creations,attribute_trace.calls);
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
    // No --help/prepare execution is supported for a HIP-linked executable.
    try {grouped_k3_perf::require(argc==3 && std::string(argv[1])=="--inside-root-lease","ROOT payload only");grouped_k3_perf::run(argv[2]);dump_attributes();return 0;}
    catch(const std::exception& e){dump_attributes();std::cerr<<"STOP "<<e.what()<<"\n";return 1;}
}
