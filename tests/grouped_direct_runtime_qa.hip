// Native integration gates. This executable may run only inside ROOT's GPU turn.
// The generated helper header contains host-only prefixes of our existing tests.
#include "grouped_direct_runtime_qa_helpers.hpp"
#include "../src/backend.hpp"
#include "../src/fused_output_policy.hpp"
#include "qa_private_layout.hpp"
hipError_t qa_same_group_control(const strix_trellis::DeviceWeights&,const _Float16*,float*,hipStream_t,float,float);
#include <atomic>
#include <memory>
#include <sys/resource.h>

static std::atomic<unsigned> attribute_calls{0};
static const void* attribute_functions[4]={};
extern "C" hipError_t __real_hipFuncGetAttributes(hipFuncAttributes*,const void*);
extern "C" hipError_t __wrap_hipFuncGetAttributes(hipFuncAttributes* a,const void* f) {
    const unsigned index=attribute_calls.fetch_add(1);
    if(index<4)attribute_functions[index]=f;
    const auto status=__real_hipFuncGetAttributes(a,f);
    if(status==hipSuccess)std::cout<<"REAL_ATTRIBUTES index="<<index<<" function="<<f
        <<" max_threads="<<a->maxThreadsPerBlock<<" shared_bytes="<<a->sharedSizeBytes
        <<" local_bytes="<<a->localSizeBytes<<'\n';
    else std::cout<<"REAL_ATTRIBUTES index="<<index<<" error="<<int(status)<<'\n';
    return status;
}
namespace {
void require(bool ok,const std::string& why) { if(!ok) throw std::runtime_error(why); }
void no_queries() { require(attribute_calls.load()==4,"Unexpected function attribute query count"); }
void host_budget() {
    rusage usage{};require(getrusage(RUSAGE_SELF,&usage)==0,"Host RSS query failed");
    require(uint64_t(usage.ru_maxrss)*1024 < (2ull<<30),"Host peak RSS exceeds 2 GiB");
    std::cout<<"HOST_PEAK_RSS bytes="<<uint64_t(usage.ru_maxrss)*1024<<'\n';
}
std::vector<unsigned char> download(Runtime& r,const void* p,size_t n) {
    std::vector<unsigned char> v(n);
    hip_check(hipMemcpyAsync(v.data(),p,n,hipMemcpyDeviceToHost,r.stream));
    hip_check(hipStreamSynchronize(r.stream));return v;
}
struct Data {
    uint32_t i,n;std::vector<unsigned char> x,a,old,reference;
    std::vector<unsigned char> cpu,baseline_s4;
    explicit Data(const Fixture& f):i(f.i),n(f.n) {
        require(f.rows>=8,"Eight fixture rows required");
        x=f.parts.at("x");a=f.parts.at("a");old=f.parts.at("old");reference=f.parts.at("ref");
        x.resize(8ull*i*4);a.resize(8ull*i*2);old.resize(8ull*n*4);reference.resize(8ull*n*4);
    }
    Data(const HeadData& f,const fs::path& refs):i(f.i),n(f.n) {
        x.resize(f.x.size()*4);std::memcpy(x.data(),f.x.data(),x.size());
        a=HeadData::read(refs/"r8-a.f16",8ull*i*2);
        reference=HeadData::read(refs/"r8-y.f32",8ull*n*4);
        old.resize(reference.size());
        for(size_t k=0;k<old.size()/4;++k){float v=(int(k%41)-20)*0.015625f;std::memcpy(old.data()+k*4,&v,4);}
    }
    std::vector<float> expected(unsigned rows,float alpha,float beta) const {
        std::vector<float> v(size_t(rows)*n),o(v.size());
        std::memcpy(v.data(),reference.data(),v.size()*4);std::memcpy(o.data(),old.data(),o.size()*4);
        for(size_t k=0;k<v.size();++k) {
            volatile float av=alpha*v[k];
            if(beta!=0) { volatile float bv=beta*o[k];v[k]=av+bv; } else v[k]=av;
        }
        return v;
    }
};
struct Case {
    Runtime& r;const Data& f;unsigned rows;Guarded x,y,workspace;
    strix_matmul_desc d{};size_t aligned;bool direct;
    static size_t need(Runtime& r,const Data& f,strix_weights w,unsigned rows,void* x,void* y) {
        strix_matmul_desc d;strix_matmul_desc_init(&d);d.weights=w;
        d.input=tensor(x,rows,f.i,false);d.output=tensor(y,rows,f.n,false);
        uint64_t n=0;api_check(strix_matmul_workspace_bytes(r.context,&d,&n));no_queries();return n;
    }
    Case(Runtime& rt,const Data& data,strix_weights w,unsigned count,bool is_direct):
        r(rt),f(data),rows(count),x(size_t(rows)*f.i*4,r),y(size_t(rows)*f.n*4,r),
        workspace(need(r,f,w,rows,x.p,y.p),r),aligned((size_t(rows)*f.i*2+255)&~size_t(255)),direct(is_direct) {
        strix_matmul_desc_init(&d);d.weights=w;
        d.input=tensor(x.p,rows,f.i,false);d.output=tensor(y.p,rows,f.n,false);
        d.workspace=workspace.p;d.workspace_bytes=workspace.bytes;
        uint64_t bytes=0;api_check(strix_matmul_workspace_bytes(r.context,&d,&bytes));
        require(bytes==workspace.bytes,"Workspace query changed");
        if(rows==8) require(bytes==aligned+4ull*rows*f.n*4,"S4 workspace changed");
        upload(x.p,f.x.data(),x.bytes,r);reset();hip_check(hipStreamSynchronize(r.stream));no_queries();
    }
    void reset(float alpha=1,float beta=0) {
        d.alpha=alpha;d.beta=beta;
        hip_check(hipMemsetAsync(workspace.p,0xff,workspace.bytes,r.stream));
        if(beta==0)hip_check(hipMemsetAsync(y.p,0xff,y.bytes,r.stream));
        else upload(y.p,f.old.data(),y.bytes,r);
    }
    void submit(bool gemm=false,const strix_stream* stream=nullptr) {
        api_check(gemm?strix_gemm(r.context,&d,stream?stream:&r.s):strix_gemv(r.context,&d,stream?stream:&r.s));no_queries();
    }
    std::vector<unsigned char> check(const std::string& label,const std::vector<float>* override_ref=nullptr) {
        auto out=download(r,y.p,y.bytes);std::vector<float> actual(out.size()/4);
        std::memcpy(actual.data(),out.data(),out.size());auto expected=override_ref?*override_ref:f.expected(rows,d.alpha,d.beta);
        compare(label,actual,expected);
        if(f.n==129280)for(unsigned row=0;row<rows;++row)compare(label+" row="+std::to_string(row),
            std::vector<float>(actual.begin()+row*f.n,actual.begin()+(row+1)*f.n),
            std::vector<float>(expected.begin()+row*f.n,expected.begin()+(row+1)*f.n));
        if(!override_ref && !f.cpu.empty()) {
            Data independent=f;independent.reference=f.cpu;independent.cpu.clear();
            auto cpuref=independent.expected(rows,d.alpha,d.beta);compare(label+" full_CPU",actual,cpuref);
            for(unsigned row=0;row<rows;++row)compare(label+" CPU_row="+std::to_string(row),
                std::vector<float>(actual.begin()+row*f.n,actual.begin()+(row+1)*f.n),
                std::vector<float>(cpuref.begin()+row*f.n,cpuref.begin()+(row+1)*f.n));
        }
        if(!override_ref && direct && (f.i==1280 || f.n==129280)) {
            Device control(y.bytes);
            if(d.beta==0)hip_check(hipMemsetAsync(control.p,255,y.bytes,r.stream));
            else upload(control.p,f.old.data(),y.bytes,r);
            hip_check(qa_same_group_control(d.weights->device,
                reinterpret_cast<const _Float16*>(workspace.p),static_cast<float*>(control.p),r.stream,d.alpha,d.beta));
            require(download(r,control.p,y.bytes)==out,label+": rebuilt same-G prototype bytes differ");
        }
        if(direct && f.n==5120 && !f.baseline_s4.empty() && d.alpha==1 && d.beta==0 && !override_ref)
            require(out==f.baseline_s4,label+": unchanged G4 full V12 bytes differ");
        auto ws=download(r,workspace.p,workspace.bytes);
        require(std::equal(ws.begin(),ws.begin()+size_t(rows)*f.i*2,f.a.begin()),label+": full A mismatch");
        if(direct) require(std::all_of(ws.begin()+aligned,ws.end(),[](unsigned char v){return v==255;}),label+": Direct touched global P");
        else {
            for(size_t k=aligned;k<ws.size();k+=4) {
                float v;std::memcpy(&v,ws.data()+k,4);
                require(std::isfinite(v),label+": fallback left poisoned/nonfinite P");
            }
        }
        require(download(r,x.p,x.bytes)==std::vector<unsigned char>(f.x.begin(),f.x.begin()+x.bytes),label+": X changed");
        x.check(r);y.check(r);workspace.check(r);no_queries();
        std::cout<<"CHECK "<<label<<" rows="<<rows<<" all_values="<<actual.size()<<" A_all_bytes=1 P="
            <<(direct?"entirely_ff":"all_finite_written")<<" X_unchanged=1 canaries=1\n";return out;
    }
};
struct Graph {
    hipGraph_t graph=nullptr;hipGraphExec_t exec=nullptr;
    ~Graph(){if(exec)(void)hipGraphExecDestroy(exec);if(graph)(void)hipGraphDestroy(graph);}
    template<class F>void capture(Runtime& r,size_t expected,F submit) {
        hip_check(hipStreamBeginCapture(r.stream,hipStreamCaptureModeThreadLocal));
        try{submit();}catch(...){(void)hipStreamEndCapture(r.stream,&graph);throw;}
        hip_check(hipStreamEndCapture(r.stream,&graph));no_queries();
        size_t nodes=0;hip_check(hipGraphGetNodes(graph,nullptr,&nodes));require(nodes==expected,"Wrong captured node count");
        std::vector<hipGraphNode_t> list(nodes);if(nodes)hip_check(hipGraphGetNodes(graph,list.data(),&nodes));
        for(auto node:list){hipGraphNodeType type;hip_check(hipGraphNodeGetType(node,&type));require(type==hipGraphNodeTypeKernel,"Non-kernel capture node");}
        if(nodes)hip_check(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        std::cout<<"CAPTURE kernel_nodes="<<nodes<<" attributes=4\n";
    }
    void replay(Runtime& r){hip_check(hipGraphLaunch(exec,r.stream));no_queries();}
};
struct SecondStream {
    hipStream_t stream=nullptr;strix_stream s{};
    SecondStream(){hip_check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));s.struct_size=sizeof(s);s.abi_version=1;s.hip_stream=reinterpret_cast<void*>(stream);}
    ~SecondStream(){if(stream){(void)hipStreamSynchronize(stream);(void)hipStreamDestroy(stream);}}
};
void groups(Runtime& r,Case& a,Case& b) {
    auto ga=a.d,gb=b.d;ga.workspace=gb.workspace=nullptr;ga.workspace_bytes=gb.workspace_bytes=0;
    strix_matmul_desc descriptors[3]={ga,gb,ga};
    strix_grouped_moe_desc g{};g.struct_size=sizeof(g);g.abi_version=1;g.groups=descriptors;g.group_count=2;
    uint64_t need=0;api_check(strix_grouped_moe_workspace_bytes(r.context,&g,&need));
    require(need==a.workspace.bytes && need==b.workspace.bytes,"Grouped maximum workspace changed");
    g.workspace=a.workspace.p;g.workspace_bytes=need;
    // The last group is disjoint, so failures test final validation, not aliasing.
    Guarded lastx(a.x.bytes,r),lasty(a.y.bytes,r);Device short_output(4);upload(lastx.p,a.f.x.data(),lastx.bytes,r);
    descriptors[2].input=tensor(lastx.p,a.rows,a.f.i,false);descriptors[2].output=tensor(lasty.p,a.rows,a.f.n,false);
    for(int storage=0;storage<2;++storage) {
        a.reset();b.reset();hip_check(hipMemsetAsync(lasty.p,0xff,lasty.bytes,r.stream));
        hip_check(hipMemsetAsync(short_output.p,0xff,4,r.stream));
        hip_check(hipStreamSynchronize(r.stream));
        g.group_count=3;auto saved=descriptors[2];
        if(storage)descriptors[2].output.data=short_output.p;else descriptors[2].output.reserved[0]=1;
        Graph rejection;rejection.capture(r,0,[&]{
            auto status=strix_grouped_moe(r.context,&g,&r.s);
            require(status==(storage?STRIX_ERROR_INSUFFICIENT_BUFFER:STRIX_ERROR_INVALID_ARGUMENT),"Wrong final-group rejection");
        });
        descriptors[2]=saved;
        for(Case* c:{&a,&b}) {
            auto y=download(r,c->y.p,c->y.bytes),ws=download(r,c->workspace.p,c->workspace.bytes);
            require(std::all_of(y.begin(),y.end(),[](unsigned char v){return v==255;}),"Rejected group changed earlier Y");
            require(std::all_of(ws.begin(),ws.end(),[](unsigned char v){return v==255;}),"Rejected group changed workspace");
            require(download(r,c->x.p,c->x.bytes)==c->f.x,"Rejected group changed earlier X");
            c->x.check(r);c->y.check(r);c->workspace.check(r);
        }
        require(download(r,lastx.p,lastx.bytes)==a.f.x,"Rejected group changed last X");
        auto untouched=download(r,lasty.p,lasty.bytes);
        require(std::all_of(untouched.begin(),untouched.end(),[](unsigned char v){return v==255;}),"Rejected group changed last Y");
        auto short_bytes=download(r,short_output.p,4);
        require(std::all_of(short_bytes.begin(),short_bytes.end(),[](unsigned char v){return v==255;}),"Rejected storage changed short output");
        lastx.check(r);lasty.check(r);no_queries();
    }
    g.group_count=2;a.reset();b.reset();api_check(strix_grouped_moe(r.context,&g,&r.s));
    a.check("valid_group_first");
    // Both outputs share the same maximum workspace in stream order.
    b.d.workspace=a.workspace.p;b.d.workspace_bytes=need;
    // Check b's output and its own X/canaries; inspect the actual shared A/P via a.
    auto out=download(r,b.y.p,b.y.bytes);std::vector<float> actual(out.size()/4);std::memcpy(actual.data(),out.data(),out.size());
    compare("valid_group_second",actual,b.f.expected(b.rows,1,0));

    require(download(r,b.x.p,b.x.bytes)==b.f.x,"Grouped X changed");b.x.check(r);b.y.check(r);b.workspace.check(r);
    b.d.workspace=b.workspace.p;no_queries();
}
void suite(Runtime& r,const Data& f,strix_weights w,bool dense) {
    auto immutable_weights=download(r,w->allocation,w->allocation_bytes);
    {
        Case a(r,f,w,8,true),b(r,f,w,8,true);SecondStream second;
        a.submit();auto baseline=a.check("public_direct");
        Graph graph;graph.capture(r,2,[&]{a.submit();});
        for(int rep=0;rep<2;++rep){a.reset();graph.replay(r);require(a.check("direct_replay")==baseline,"Replay bytes changed");}
        // Initialize on the primary stream, then publish all uploads before both enqueues.
        a.reset();b.reset();hip_check(hipStreamSynchronize(r.stream));
        a.submit();b.submit(false,&second.s);
        // Both submissions precede either synchronization.
        hip_check(hipStreamSynchronize(second.stream));hip_check(hipStreamSynchronize(r.stream));
        require(a.check("concurrent_first")==baseline,"Concurrent first bytes changed");
        require(b.check("concurrent_second")==baseline,"Concurrent second bytes changed");
        a.reset(1,.5f);a.submit();a.d.alpha=-.75f;a.d.beta=.5f;a.submit();
        auto ref=f.expected(8,1,.5f),linear=f.expected(8,1,0);
        for(size_t k=0;k<ref.size();++k){volatile float av=-.75f*linear[k];volatile float bv=.5f*ref[k];ref[k]=av+bv;}
        a.check("ordered_workspace_reuse",&ref);
        // Re-run with beta zero to check complete A/P/X/canaries after ordered calls.
        a.reset();a.submit();require(a.check("ordered_final")==baseline,"Ordered bytes changed");
        groups(r,a,b);
    }
    {
        Case ep(r,f,w,8,true);
        const float scales[][2]={{0,0},{-0.0f,-0.0f},{1,-0.0f},{-.75f,.5f},{1,1},{-0.0f,.5f}};
        for(const auto& s:scales){ep.reset(s[0],s[1]);ep.submit();ep.check("all_route_epilogue");}
    }
    if(dense) {
        Case c(r,f,w,8,false);c.submit(true);c.check("prefer_wmma_fallback");
        Graph graph;graph.capture(r,3,[&]{c.submit(true);});
        for(int rep=0;rep<2;++rep){c.reset();graph.replay(r);c.check("wmma_replay");}
        Case ep(r,f,w,8,true);
        const float scales[][2]={{0,0},{-0.0f,-0.0f},{1,-0.0f},{-.75f,.5f},{1,1},{-0.0f,.5f}};
        for(const auto& s:scales){ep.reset(s[0],s[1]);ep.submit();ep.check("epilogue alpha="+std::to_string(s[0])+" beta="+std::to_string(s[1]));}
    } else std::cout<<"NOT_RUN head_prefer_wmma: CPU source guard only; excluded from ROOT turn budget, high-dimensional WMMA numeric risk\n";
    require(download(r,w->allocation,w->allocation_bytes)==immutable_weights,"Prepared weight bytes changed");

}
void mixed_groups(Runtime& r,const std::vector<Case*>& cases) {
    std::vector<std::vector<unsigned char>> immutable_weights;
    for(Case* c:cases)immutable_weights.push_back(download(r,c->d.weights->allocation,c->d.weights->allocation_bytes));
    size_t maximum=0;std::vector<strix_matmul_desc> descriptors;
    std::vector<std::vector<unsigned char>> baselines;
    for(Case* c:cases){c->reset();c->submit();baselines.push_back(c->check("mixed_individual_baseline"));
        maximum=std::max(maximum,c->workspace.bytes);auto d=c->d;d.workspace=nullptr;d.workspace_bytes=0;descriptors.push_back(d);}
    Guarded workspace(maximum,r);strix_grouped_moe_desc g{};g.struct_size=sizeof(g);g.abi_version=1;
    g.groups=descriptors.data();g.group_count=descriptors.size();g.workspace=workspace.p;g.workspace_bytes=maximum;
    uint64_t need=0;api_check(strix_grouped_moe_workspace_bytes(r.context,&g,&need));require(need==maximum,"Mixed group maximum changed");
    auto reset=[&]{for(Case* c:cases)c->reset();hip_check(hipMemsetAsync(workspace.p,255,maximum,r.stream));hip_check(hipStreamSynchronize(r.stream));};
    for(int storage=0;storage<2;++storage){reset();auto saved=descriptors.back();Device short_output(4);
        hip_check(hipMemsetAsync(short_output.p,255,4,r.stream));hip_check(hipStreamSynchronize(r.stream));
        if(storage)descriptors.back().output.data=short_output.p;else descriptors.back().output.reserved[0]=1;
        Graph rejected;rejected.capture(r,0,[&]{require(strix_grouped_moe(r.context,&g,&r.s)==
            (storage?STRIX_ERROR_INSUFFICIENT_BUFFER:STRIX_ERROR_INVALID_ARGUMENT),"Mixed last-group rejection changed");});
        descriptors.back()=saved;
        for(Case* c:cases){for(auto region:{std::make_pair(c->y.p,c->y.bytes),std::make_pair(c->workspace.p,c->workspace.bytes)}){
            auto v=download(r,region.first,region.second);require(std::all_of(v.begin(),v.end(),[](unsigned char b){return b==255;}),"Rejected mixed group changed Y/workspace");}
            require(download(r,c->x.p,c->x.bytes)==std::vector<unsigned char>(c->f.x.begin(),c->f.x.begin()+c->x.bytes),"Rejected mixed group changed X");
            c->x.check(r);c->y.check(r);c->workspace.check(r);}
        auto v=download(r,workspace.p,maximum);require(std::all_of(v.begin(),v.end(),[](unsigned char b){return b==255;}),"Rejected mixed shared workspace changed");
        v=download(r,short_output.p,4);require(std::all_of(v.begin(),v.end(),[](unsigned char b){return b==255;}),"Rejected mixed storage changed");workspace.check(r);
    }
    reset();Graph graph;graph.capture(r,9,[&]{api_check(strix_grouped_moe(r.context,&g,&r.s));});
    for(int rep=0;rep<2;++rep){reset();graph.replay(r);
        for(size_t k=0;k<cases.size();++k){Case* c=cases[k];require(download(r,c->y.p,c->y.bytes)==baselines[k],"Mixed grouped output differs from own producer");require(download(r,c->x.p,c->x.bytes)==std::vector<unsigned char>(c->f.x.begin(),c->f.x.begin()+c->x.bytes),"Mixed valid group changed X");c->x.check(r);c->y.check(r);c->workspace.check(r);}
        Case* last=cases.back();auto ws=download(r,workspace.p,maximum);
        require(std::equal(ws.begin(),ws.begin()+last->rows*last->f.i*2,last->f.a.begin()),"Mixed final A mismatch");
        const size_t end=last->workspace.bytes;
        for(size_t k=last->aligned;k<end;k+=4){float v;std::memcpy(&v,ws.data()+k,4);require(std::isfinite(v),"Mixed fallback P not written");}
        require(std::all_of(ws.begin()+end,ws.end(),[](unsigned char v){return v==255;}),"Mixed group changed unused workspace tail");
        workspace.check(r);no_queries();
    }
    for(size_t k=0;k<cases.size();++k){auto w=cases[k]->d.weights;
        require(download(r,w->allocation,w->allocation_bytes)==immutable_weights[k],"Mixed prepared weights changed");}
    std::cout<<"CHECK mixed_groups G1_G2_G4_R7 max_workspace="<<maximum<<" nodes=9 full_producer_identity=1 invalid_last_descriptor_and_storage=1\n";
}
void private_false(Runtime& r,strix_trellis::DeviceWeights w,const Data& data) {
    w.fused_output_capable=false;
    size_t bytes=strix_trellis::linear_workspace_bytes(w,8);
    Guarded x(8ull*w.input*4,r),y(8ull*w.output*4,r),ws(bytes,r);
    upload(x.p,data.x.data(),x.bytes,r);
    auto reset=[&]{hip_check(hipMemsetAsync(y.p,255,y.bytes,r.stream));hip_check(hipMemsetAsync(ws.p,255,ws.bytes,r.stream));hip_check(hipStreamSynchronize(r.stream));};
    auto submit=[&]{hip_check(strix_trellis::launch_linear(w,reinterpret_cast<float*>(x.p),reinterpret_cast<float*>(y.p),8,ws.p,r.stream,1,0,false));no_queries();};
    auto check=[&]{auto out=download(r,y.p,y.bytes);
        require(out==data.baseline_s4,"False capability full Y differs from standalone frozen V12 S4");
        auto workspace=download(r,ws.p,ws.bytes);require(std::equal(data.a.begin(),data.a.end(),workspace.begin()),"False fallback A mismatch");
        for(size_t k=(data.a.size()+255)&~size_t(255);k<workspace.size();k+=4){float v;std::memcpy(&v,workspace.data()+k,4);require(std::isfinite(v),"False fallback P not written");}
        require(download(r,x.p,x.bytes)==data.x,"False fallback X changed");x.check(r);y.check(r);ws.check(r);return out;};
    reset();submit();auto baseline=check();Graph graph;graph.capture(r,3,submit);
    for(int rep=0;rep<2;++rep){reset();graph.replay(r);require(check()==baseline,"False fallback replay bytes changed");}
}

}
int main(int argc,char** argv) {
    try {
        require(argc==4 || argc==5,"Usage: grouped_direct_runtime_qa SCRATCH_ROOT CUDA_REFS V12_OUTPUTS [STRICT_CPU_HEAD_F32] (ROOT GPU turn only)");
        std::cout<<std::unitbuf<<std::setprecision(12);
        fs::path root=fs::canonical(argv[1]),refs=fs::canonical(argv[2]),oldrefs=fs::canonical(argv[3]);
        require(root==fs::path(std::getenv("HOME"))/"gpu-site/scratch-trellis","Unexpected scratch root");
        require(refs==root/"runtime-v13-qa/reference","Unexpected reference directory");
        Runtime r;require(attribute_calls.load()==4,"Context must query four actual Direct functions exactly once");
        for(unsigned i=0;i<4;++i){require(attribute_functions[i],"Null function query");for(unsigned j=0;j<i;++j)require(attribute_functions[i]!=attribute_functions[j],"Distinct Direct function queries required");}
        // Pure null-argument error: no device access and no resource query.
        require(strix_trellis::query_fused_output_resources(0,24576,nullptr)==hipErrorInvalidValue,"Null capability result accepted");no_queries();
        const char* names[]={"layers.0.attn.wq_b.stgpu","layers.0.attn.wo_b.stgpu","layers.0.ffn.shared_experts.w2.stgpu"};
        for(unsigned index=0;index<3;++index) {
            Fixture f((root/"gpu-dense-prefill-tests"/names[index]).string());
            require(f.bits==5 && f.cb==2 && !f.bias,"Unexpected dense fixture");Data data(f);data.baseline_s4=HeadData::read(oldrefs/(std::to_string(index)+"-y.f32"),8ull*f.n*4);require(HeadData::read(oldrefs/(std::to_string(index)+"-a.f16"),8ull*f.i*2)==data.a,"V12/CPU full A mismatch");
            qa_weight_reserve=f.parts.at("trellis").size()+2ull*(f.i+f.n)+4ull*f.n+1024;
            std::cout<<"GEOMETRY I="<<f.i<<" N="<<f.n<<" bits="<<f.bits<<" bias=0\n";
            host_budget();{Weight weight(r,f);no_queries();suite(r,data,weight.w,true);private_false(r,weight.w->device,data);
                if(index==0){Case transition(r,data,weight.w,7,false);transition.submit();transition.check("R7_fallback_transition");}
                weight.close();}
            if(index==0) {
                // Patch only our host copy; deterministic bias follows the same FP32 epilogue.
                f.bias=1;std::vector<float> bias(f.n);for(size_t c=0;c<bias.size();++c)bias[c]=(int(c%29)-14)*0.0009765625f;
                auto& raw=f.parts["bias"];raw.resize(bias.size()*4);std::memcpy(raw.data(),bias.data(),raw.size());
                auto ref=f.floats("ref");for(size_t k=0;k<ref.size();++k)ref[k]+=bias[k%f.n];
                std::memcpy(f.parts["ref"].data(),ref.data(),ref.size()*4);Data biased(f);
                Weight weight(r,f);no_queries();Case ep(r,biased,weight.w,8,true);
                auto immutable=download(r,weight.w->allocation,weight.w->allocation_bytes);
                const float scales[][2]={{0,0},{-0.0f,-0.0f},{1,-0.0f},{-.75f,.5f},{1,1},{-0.0f,.5f}};
                for(const auto& s:scales){ep.reset(s[0],s[1]);ep.submit();ep.check("bias_epilogue alpha="+std::to_string(s[0])+" beta="+std::to_string(s[1]));}
                require(download(r,weight.w->allocation,weight.w->allocation_bytes)==immutable,"Bias weights changed");
            }
        }
        host_budget();{HeadData f(root/"gpu-head-eaq-ref-v2");require(f.actual_model_values,"Real head required");
            qa_weight_reserve=f.blob_bytes+1024;
            Data data(f,refs);require(HeadData::read(oldrefs/"head-a.f16",8ull*f.i*2)==data.a,"V12/CUDA full A mismatch");data.baseline_s4=HeadData::read(oldrefs/"head-y.f32",8ull*f.n*4);
            if(argc==5)data.cpu=HeadData::read(argv[4],8ull*f.n*4);
            else std::cout<<"PENDING full_CPU_head_oracle: CUDA full gate only\n";
            {PreparedHead weight(r,f);host_budget();no_queries();suite(r,data,weight.w,false);
             private_false(r,weight.w->device,data);
             {Fixture q((root/"gpu-dense-prefill-tests/layers.0.attn.wq_b.stgpu").string());
              Fixture wo((root/"gpu-dense-prefill-tests/layers.0.attn.wo_b.stgpu").string());
              Data qd(q),wd(wo);qa_weight_reserve=f.blob_bytes+q.parts.at("trellis").size()+wo.parts.at("trellis").size()+2ull*(q.i+q.n+wo.i+wo.n)+4096;
              Weight qw(r,q),ww(r,wo);Case g1(r,qd,qw.w,8,true),g2(r,data,weight.w,8,true),g4(r,wd,ww.w,8,true),fallback(r,qd,qw.w,7,false);
              mixed_groups(r,{&g1,&g2,&g4,&fallback});qw.close();ww.close();}
             qa_weight_reserve=f.blob_bytes+1024;weight.close();}
            std::vector<float> bias(f.n);for(size_t c=0;c<bias.size();++c)bias[c]=(int(c%29)-14)*0.0009765625f;
            for(auto* raw:{&data.reference,&data.cpu}) if(!raw->empty())for(size_t k=0;k<raw->size()/4;++k){float v;std::memcpy(&v,raw->data()+k*4,4);v+=bias[k%f.n];std::memcpy(raw->data()+k*4,&v,4);}
            PreparedHead biased(r,f,&bias);Case ep(r,data,biased.w,8,true);
            auto immutable=download(r,biased.w->allocation,biased.w->allocation_bytes);
            const float scales[][2]={{0,0},{-0.0f,-0.0f},{1,-0.0f},{-.75f,.5f},{1,1},{-0.0f,.5f}};
            for(const auto& s:scales){ep.reset(s[0],s[1]);ep.submit();ep.check("head_bias_epilogue");}
            require(download(r,biased.w->allocation,biased.w->allocation_bytes)==immutable,"Head bias weights changed");
            biased.close();host_budget();}
        no_queries();r.close();std::cout<<"PASS native_grouped_Direct_integration physical_execution_only attributes=4 peak_explicit_gpu_bound="<<qa_device_peak<<'\n';return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
