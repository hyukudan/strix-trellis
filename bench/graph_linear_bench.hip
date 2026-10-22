// Own complete-operator graph experiment; reuse only local fixture helpers.
#define main strix_graph_unused_runtime_main
#include "../tests/gpu_runtime.hip"
#undef main
#include <chrono>
#include <iomanip>
#include <memory>
#include <numeric>

namespace {
using Clock=std::chrono::steady_clock;
constexpr size_t cold_bytes=256ull<<20;
constexpr int rotations=11;
double elapsed(Clock::time_point t) { return std::chrono::duration<double,std::micro>(Clock::now()-t).count(); }
void graph_check(hipError_t s,const char* stage) {
    if(s!=hipSuccess) throw std::runtime_error(std::string("HIP_GRAPH stage=")+stage+" code="+std::to_string(int(s))+" name="+hipGetErrorName(s)+" error="+hipGetErrorString(s));
}
struct Graph {
    hipGraph_t graph{}; hipGraphExec_t exec{};
    double capture_us=0,instantiate_us=0; size_t nodes=0;
    ~Graph() { if(exec) (void)hipGraphExecDestroy(exec); if(graph) (void)hipGraphDestroy(graph); }
    template<class Submit> void capture(Runtime& r,size_t operators,Submit submit) {
        auto t=Clock::now();
        graph_check(hipStreamBeginCapture(r.stream,hipStreamCaptureModeThreadLocal),"begin_capture");
        try { submit(); } catch(...) { (void)hipStreamEndCapture(r.stream,&graph); throw; }
        graph_check(hipStreamEndCapture(r.stream,&graph),"end_capture"); capture_us=elapsed(t);
        graph_check(hipGraphGetNodes(graph,nullptr,&nodes),"get_nodes");
        std::vector<hipGraphNode_t> list(nodes);
        graph_check(hipGraphGetNodes(graph,list.data(),&nodes),"get_nodes_list");
        if(nodes!=3*operators && nodes!=2*operators && nodes!=operators)
            throw std::runtime_error("Expected one Direct, two fused or three staged nodes per complete operator");
        for(auto n:list) { hipGraphNodeType type; graph_check(hipGraphNodeGetType(n,&type),"node_type");
            if(type!=hipGraphNodeTypeKernel) throw std::runtime_error("Captured non-kernel node"); }
        t=Clock::now(); graph_check(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0),"instantiate"); instantiate_us=elapsed(t);
    }
    void launch(Runtime& r) { graph_check(hipGraphLaunch(exec,r.stream),"launch"); }
};
struct Stats { double minimum,median,p90; };
Stats stats(std::vector<double> v) { std::sort(v.begin(),v.end()); return {v.front(),v[5],v[9]}; }
void print(const char* name,Stats s) { std::cout<<" "<<name<<"_min_us="<<s.minimum<<" "<<name<<"_median_us="<<s.median<<" "<<name<<"_p90_us="<<s.p90; }
template<class Submit> Stats measure(Runtime& r,size_t launches,size_t operators,bool events,Submit submit) {
    hipEvent_t a{},b{}; hip_check(hipEventCreate(&a)); hip_check(hipEventCreate(&b));
    for(size_t c=0;c<launches;++c) submit(c);
    hip_check(hipStreamSynchronize(r.stream));
    std::vector<double> times;
    for(int rep=0;rep<rotations;++rep) {
        auto t=Clock::now(); if(events) hip_check(hipEventRecord(a,r.stream));
        for(size_t c=0;c<launches;++c) { submit(c); if(!events) hip_check(hipStreamSynchronize(r.stream)); }
        if(events) { hip_check(hipEventRecord(b,r.stream)); hip_check(hipEventSynchronize(b));
            float ms=0; hip_check(hipEventElapsedTime(&ms,a,b)); times.push_back(ms*1000.0/operators);
        } else times.push_back(elapsed(t)/operators);
    }
    hip_check(hipEventDestroy(a)); hip_check(hipEventDestroy(b)); return stats(times);
}
void run(Runtime& r,const std::string& path) {
    Fixture f(path); size_t packed=f.parts.at("trellis").size(),clones=(cold_bytes+packed-1)/packed;
    std::cout<<"FIXTURE path="<<path<<" I="<<f.i<<" N="<<f.n<<" bits="<<f.bits<<" bias="<<f.bias<<"\n";
    for(const char* k:{"q","core","scaled","inh","a","old"}) f.parts.erase(k);
    std::vector<std::unique_ptr<Weight>> weights;
    for(size_t c=0;c<clones;++c) weights.emplace_back(new Weight(r,f));
    for(uint32_t rows:{1u,8u}) {
        Device x(size_t(rows)*f.i*4),y(clones*size_t(rows)*f.n*4);
        auto d=desc(weights[0]->w,x.p,y.p,rows,f,false,1,0);
        uint64_t bytes=0; api_check(strix_matmul_workspace_bytes(r.context,&d,&bytes));
        Device workspace(bytes); d.workspace=workspace.p; d.workspace_bytes=bytes;
        size_t bound=clones*(packed+2ull*(f.i+f.n)+(f.bias?4ull*f.n:0)+4*255)+d.input.buffer_bytes+clones*d.output.buffer_bytes+bytes;
        if(bound>=(1ull<<30)) throw std::runtime_error("GPU allocation bound exceeds 1 GiB");
        upload(x.p,f.parts.at("x").data(),d.input.buffer_bytes,r); hip_check(hipStreamSynchronize(r.stream));
        auto direct=[&](size_t c) { d.weights=weights[c]->w; d.output.data=static_cast<char*>(y.p)+c*d.output.buffer_bytes;
            api_check(strix_gemv(r.context,&d,&r.s)); };
        // Warm runtime/kernel module loading only. Prepared weights remain compressed.
        direct(0); hip_check(hipStreamSynchronize(r.stream));
        std::vector<std::unique_ptr<Graph>> graphs; double cap=0,inst=0;
        for(size_t c=0;c<clones;++c) { auto g=std::make_unique<Graph>(); g->capture(r,1,[&]{direct(c);});
            cap+=g->capture_us; inst+=g->instantiate_us; graphs.push_back(std::move(g)); }
        Graph rotation; rotation.capture(r,clones,[&]{for(size_t c=0;c<clones;++c) direct(c);});
        std::cout<<"SETUP rows="<<rows<<" clones="<<clones<<" single_nodes="<<graphs.front()->nodes<<" rotation_nodes="<<rotation.nodes
                 <<" individual_capture_total_us="<<cap<<" individual_instantiate_total_us="<<inst
                 <<" rotation_capture_us="<<rotation.capture_us<<" rotation_instantiate_us="<<rotation.instantiate_us<<" update=unused\n";
        auto ref=f.floats("ref"); ref.resize(size_t(rows)*f.n);
        size_t comparisons=0; double checksum=0;
        auto check=[&](const std::string& phase) { std::vector<float> actual(ref.size());
            for(size_t c=0;c<clones;++c) {
                hip_check(hipMemcpyAsync(actual.data(),static_cast<char*>(y.p)+c*d.output.buffer_bytes,d.output.buffer_bytes,hipMemcpyDeviceToHost,r.stream));
                hip_check(hipStreamSynchronize(r.stream));
                compare("CHECK mode="+phase+" rows="+std::to_string(rows)+" clone="+std::to_string(c),actual,ref);
                ++comparisons; checksum+=std::accumulate(actual.begin(),actual.end(),0.0);
            }
        };
        for(int mode=0;mode<3;++mode) {
            const char* name=mode==0?"direct":mode==1?"graph_individual":"graph_rotation";
            auto submit=[&](size_t c) { if(mode==0) direct(c); else if(mode==1) graphs[c]->launch(r); else rotation.launch(r); };
            size_t launches=mode==2?1:clones;
            for(size_t c=0;c<launches;++c) submit(c);
            hip_check(hipStreamSynchronize(r.stream)); check(std::string(name)+"_before");
            auto event=measure(r,launches,clones,true,submit); check(std::string(name)+"_last_event");
            auto wall=measure(r,launches,clones,false,submit); check(std::string(name)+"_last_wall");
            for(size_t c=0;c<launches;++c) submit(c);
            hip_check(hipStreamSynchronize(r.stream)); check(std::string(name)+"_after");
            std::cout<<"RESULT mode="<<name<<" rows="<<rows<<" clones="<<clones<<" cloned_bytes="<<clones*packed
                     <<" workspace_bytes="<<bytes<<" gpu_allocation_bound="<<bound<<" rotations="<<rotations
                     <<" operators_per_launch="<<(mode==2?clones:1)<<" event_rotation_amortized=1 wall_sync_per_launch=1";
            print("event",event); print("wall",wall);
            std::cout<<" compressed_GBs="<<packed/event.median/1000<<" useful_TOPS="<<2.0*rows*f.i*f.n/event.median/1e6<<"\n";
        }
        std::cout<<"CORRECTNESS rows="<<rows<<" full_comparisons="<<comparisons<<" checksum="<<checksum<<"\n";
    }
}
}
int main(int argc,char** argv) {
    try {
        if(argc<2) throw std::runtime_error("Usage: graph_linear_bench FIXTURE...");
        std::cout<<std::setprecision(10)<<std::unitbuf; Runtime r;
        hipDeviceProp_t p{}; hip_check(hipGetDeviceProperties(&p,0));
        std::cout<<"HARDWARE multiprocessors="<<p.multiProcessorCount<<" clockRate_kHz="<<p.clockRate<<" memoryClockRate_kHz="<<p.memoryClockRate<<"\n";
        for(int k=1;k<argc;++k) run(r,argv[k]);
        r.close(); std::cout<<"PASS complete_operator_graphs\n"; return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL "<<e.what()<<"\n"; return 1; }
}
