// Prepared bounded native QA. Execution is exclusively ROOT's leased gate.
#include "runtime_v15_case.hpp"
#include "../src/gemv_fused_both.hpp"
#include "../src/fused_both_policy.hpp"
#include "qa_private_layout.hpp"
#include <atomic>
#include <memory>
hipError_t qa_literal(const strix_trellis::DeviceWeights&,const float*,float*,hipStream_t,float,float);
static std::atomic<unsigned> attributes{0};static const void* functions[5]={};static hipFuncAttributes limits[5]={};
extern "C" hipError_t __real_hipFuncGetAttributes(hipFuncAttributes*,const void*);
extern "C" hipError_t __wrap_hipFuncGetAttributes(hipFuncAttributes* a,const void* f){unsigned i=attributes.fetch_add(1);auto e=__real_hipFuncGetAttributes(a,f);if(i<5){functions[i]=f;if(e==hipSuccess)limits[i]=*a;}std::cout<<"ATTR i="<<i<<" function="<<f<<" status="<<int(e);if(e==hipSuccess)std::cout<<" threads="<<a->maxThreadsPerBlock<<" static="<<a->sharedSizeBytes<<" local="<<a->localSizeBytes<<" dynamic="<<a->maxDynamicSharedSizeBytes;std::cout<<'\n';return e;}
static void no_queries(){require(attributes==5,"Attributes queried after creation or wrong initial count");}
struct Graph {
    std::vector<void*> funcs;hipGraph_t graph=nullptr;hipGraphExec_t exec=nullptr;
    ~Graph(){if(exec)(void)hipGraphExecDestroy(exec);if(graph)(void)hipGraphDestroy(graph);}
    template<class F>void capture(Runtime& r,size_t expected,F submit){hip_check(hipStreamBeginCapture(r.stream,hipStreamCaptureModeThreadLocal));submit();hip_check(hipStreamEndCapture(r.stream,&graph));size_t n=0;hip_check(hipGraphGetNodes(graph,nullptr,&n));require(n==expected,"Wrong graph node count expected="+std::to_string(expected)+" actual="+std::to_string(n));std::vector<hipGraphNode_t> nodes(n);if(n)hip_check(hipGraphGetNodes(graph,nodes.data(),&n));for(auto node:nodes){hipGraphNodeType t;hip_check(hipGraphNodeGetType(node,&t));require(t==hipGraphNodeTypeKernel,"Non-kernel node");}std::vector<hipGraphNode_t> ordered;
        while(ordered.size()<n){bool progress=false;for(auto node:nodes){if(std::find(ordered.begin(),ordered.end(),node)!=ordered.end())continue;size_t count=0;hip_check(hipGraphNodeGetDependencies(node,nullptr,&count));std::vector<hipGraphNode_t> deps(count);if(count)hip_check(hipGraphNodeGetDependencies(node,deps.data(),&count));bool ready=true;for(auto dep:deps)if(std::find(ordered.begin(),ordered.end(),dep)==ordered.end())ready=false;if(ready){ordered.push_back(node);hipKernelNodeParams params{};hip_check(hipGraphKernelNodeGetParams(node,&params));funcs.push_back(params.func);std::cout<<"NODE ordinal="<<funcs.size()-1<<" function="<<params.func<<'\n';progress=true;}}require(progress,"Graph dependency cycle");}
        if(n)hip_check(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));no_queries();}
    void replay(hipStream_t stream){hip_check(hipGraphLaunch(exec,stream));no_queries();}
};
static std::vector<double> oracle(const fs::path& p){auto raw=readbytes(p);require(raw.size()==8*5120*8 || raw.size()==8*32768*8,"Oracle shape");std::vector<double> v(raw.size()/8);std::memcpy(v.data(),raw.data(),raw.size());return v;}
static void full_partial_oracle(Case& c,const fs::path& inputs,int id,bool h128){
    auto raw=readbytes(inputs/(std::to_string(id)+(h128?".ph-f64":".p-f64")));
    require(raw.size()==size_t(4)*8*c.f.n*8,"Full partial oracle size");
    auto got=get(c.r,c.ws.p+c.aligned,c.ws.bytes-c.aligned);
    require(got.size()==size_t(4)*c.rows*c.f.n*4,"Full partial workspace size");
    double e2=0,r2=0,ma=0,mr=0;
    for(int g=0;g<4;++g)for(int r=0;r<c.rows;++r)for(unsigned col=0;col<c.f.n;++col){
        float v;double ref;size_t k=(size_t(g)*c.rows+r)*c.f.n+col,j=(size_t(g)*8+r)*c.f.n+col;
        std::memcpy(&v,got.data()+k*4,4);std::memcpy(&ref,raw.data()+j*8,8);
        require(std::isfinite(v) && std::isfinite(ref),"Full partial nonfinite");
        double e=std::abs(double(v)-ref);e2+=e*e;r2+=ref*ref;ma=std::max(ma,e);mr=std::max(mr,std::abs(ref));
    }
    require(ma<=5e-4*std::max(1.,mr) && std::sqrt(e2/std::max(r2,1e-30))<=3e-5,"Full partial FP64 tolerance");
    std::cout<<"P_FP64 id="<<id<<" rows="<<c.rows<<" H128="<<h128<<" values="<<got.size()/4<<" maxabs="<<ma<<" NRMS="<<std::sqrt(e2/std::max(r2,1e-30))<<'\n';
}
static void exact_producer(Case& c){Guarded reference(c.y.bytes,c.r);if(c.d.beta==0)hip_check(hipMemsetAsync(reference.p,255,reference.bytes,c.r.stream));else upload(reference.p,c.f.parts.at("old").data(),reference.bytes,c.r);hip_check(qa_literal(c.d.weights->device,reinterpret_cast<const float*>(c.x.p),reinterpret_cast<float*>(reference.p),c.r.stream,c.d.alpha,c.d.beta));require(get(c.r,reference.p,reference.bytes)==get(c.r,c.y.p,c.y.bytes),"Literal qualified Mode Direct Y bytes");reference.check(c.r);no_queries();}
static void groups(Runtime& r,Case& a,Case& b,const std::vector<double>& ref,bool negatives){
    strix_matmul_desc ds[]={a.d,b.d};for(auto& d:ds){d.workspace=nullptr;d.workspace_bytes=0;}
    strix_grouped_moe_desc g{};g.struct_size=sizeof(g);g.abi_version=1;g.groups=ds;g.group_count=2;g.workspace=a.ws.p;g.workspace_bytes=a.ws.bytes;
    uint64_t need=0;api_check(strix_grouped_moe_workspace_bytes(r.context,&g,&need));require(need==a.ws.bytes,"Grouped workspace contract");
    if(negatives){
        // Tail is derived exclusively from a real allocation during ROOT's lease.
        // Host prequery must succeed before testing driver storage rejection.
        std::vector<std::unique_ptr<Guarded>> storage;void* short_y=nullptr;
        for(int attempt=0;attempt<4 && !short_y;++attempt){storage.emplace_back(new Guarded(4096+4096*attempt,r));hipDeviceptr_t base{};size_t extent=0;hip_check(hipMemGetAddressRange(&base,&extent,storage.back()->allocation.p));uintptr_t start=reinterpret_cast<uintptr_t>(base);require(extent>=4 && start<=UINTPTR_MAX-extent,"Extent overflow");uintptr_t p=(start+extent-4)&~uintptr_t(3);auto saved=ds[1];ds[1].output.data=reinterpret_cast<void*>(p);uint64_t n=0;auto status=strix_grouped_moe_workspace_bytes(r.context,&g,&n);ds[1]=saved;std::cout<<"EXTENT attempt="<<attempt<<" remaining="<<start+extent-p<<" query="<<int(status)<<'\n';if(status==STRIX_SUCCESS && n==g.workspace_bytes && start+extent-p<b.y.bytes){hipPointerAttribute_t attr{};hipDeviceptr_t actual_base{};size_t actual_size=0;auto pointer=reinterpret_cast<void*>(p);auto pe=hipPointerGetAttributes(&attr,pointer);auto re=hipMemGetAddressRange(&actual_base,&actual_size,pointer);if(pe==hipSuccess && re==hipSuccess && attr.type==hipMemoryTypeDevice && !attr.isManaged && attr.device==r.context->device && actual_base==base && actual_size==extent)short_y=pointer;}}
        require(short_y,"Cannot isolate actual extent negative");
        auto weights=get(r,a.d.weights->allocation,a.d.weights->allocation_bytes);
        for(int mode=0;mode<3;++mode){a.reset(1,0);b.reset(1,0);auto saved=ds[1];if(mode==0)ds[1].output.reserved[0]=1;if(mode==1)ds[1].output.buffer_bytes=4;if(mode==2)ds[1].output.data=short_y;auto expected=mode==0?STRIX_ERROR_INVALID_ARGUMENT:STRIX_ERROR_INSUFFICIENT_BUFFER;Graph rejected;rejected.capture(r,0,[&]{auto got=strix_grouped_moe(r.context,&g,&r.s);require(got==expected,"Isolated last-group status expected="+std::to_string(int(expected))+" actual="+std::to_string(int(got)));});ds[1]=saved;require(poison(get(r,a.y.p,a.y.bytes)) && poison(get(r,b.y.p,b.y.bytes)) && poison(get(r,a.ws.p,a.ws.bytes)) && poison(get(r,b.ws.p,b.ws.bytes)),"Rejected capture wrote buffers");require(get(r,a.d.weights->allocation,a.d.weights->allocation_bytes)==weights,"Rejected capture changed weights");a.intact();b.intact();for(auto& s:storage){auto bytes=get(r,s->allocation.p,s->bytes+512);require(std::all_of(bytes.begin(),bytes.end(),[](unsigned char v){return v==0xa5;}),"Rejected capture changed extent storage");}std::cout<<"PASS negative="<<mode<<" status="<<int(expected)<<" nodes=0\n";}
    }
    a.reset(1,0);b.reset(1,0);Graph graph;graph.capture(r,a.rows==8?3:2,[&]{api_check(strix_grouped_moe(r.context,&g,&r.s));});graph.replay(r.stream);a.oracle(ref);b.oracle(ref);a.intact();b.intact();if(a.rows==1)require(poison(get(r,a.ws.p,a.ws.bytes)),"Direct grouped workspace written");no_queries();
}
#include "q1_complete.hpp"
int main(int argc,char** argv){try{
    require(argc==3,"Usage: runtime_v15_qa INPUT_DIRECTORY V14B_SEPARATE_OUTPUT_DIRECTORY (ROOT lease only)");fs::path inputs=argv[1],baseline=argv[2];Runtime r;no_queries();require(r.context->fused_both_capable && r.context->fused_output_capable,"Native active baseline must support all five actual images");
    const int required[]={1024,1024,256,512};for(int i=0;i<5;++i){require(functions[i],"Missing function");for(int j=0;j<i;++j)require(functions[i]!=functions[j],"Duplicate function address");if(i<4)require(limits[i].maxThreadsPerBlock>=required[i] && !limits[i].localSizeBytes,"Old actual function limits");}
    require(functions[4]==strix_trellis::fused_both::direct_function(),"Wrong actual new function");hipDeviceProp_t props{};hip_check(hipGetDeviceProperties(&props,0));require(strix_trellis::fused_both_device_limits(props.maxThreadsPerBlock,props.maxThreadsDim[0],props.sharedMemPerBlock) && strix_trellis::fused_both_image_limits(limits[4].maxThreadsPerBlock,limits[4].sharedSizeBytes,limits[4].localSizeBytes,limits[4].maxDynamicSharedSizeBytes,props.sharedMemPerBlock),"Actual new function/device thresholds");
    for(int id=0;id<2;++id)for(int bias=0;bias<2;++bias){Fixture f((inputs/(std::to_string(id)+".stgpu")).string());setbias(f,bias);auto ref=oracle(inputs/(std::to_string(id)+".f64"));qa_weight_reserve=f.parts.at("trellis").size()+2ull*(f.i+f.n)+4ull*f.n+1024;Weight w(r,f);const bool observed_both=w.w->device.fused_both_capable;no_queries();require(w.w->device.fused_both_capable,"Prepared flag lost");auto immutable=get(r,w.w->allocation,w.w->allocation_bytes);
        for(int ep=0;ep<8;++ep){Case c(r,f,w.w,1);c.reset(epilogues[ep][0],epilogues[ep][1]);Graph graph;graph.capture(r,1,[&]{c.submit();});graph.replay(r.stream);c.oracle(ref);exact_producer(c);require(poison(get(r,c.ws.p,c.ws.bytes)),"Direct wrote A/P workspace");c.intact();
            // Disable only the new field. Existing V13 fused-input R1 is TWO nodes.
            w.w->device.fused_both_capable=false;for(int rows:{1,8})for(bool gemm:{false,true}){if(rows==8 && gemm)continue;Case fb(r,f,w.w,rows);fb.reset(epilogues[ep][0],epilogues[ep][1]);Graph old;old.capture(r,gemm?3:2,[&]{fb.submit(gemm);});old.replay(r.stream);fb.oracle(ref);if(rows==1)full_partial_oracle(fb,inputs,id,!gemm);require(get(r,fb.y.p,fb.y.bytes)==readbytes(baseline/(tag(id,bias,rows,ep,gemm)+"-y")),"Separate V14b Y differs");auto bytes=get(r,fb.ws.p,fb.ws.bytes);require(bytes==readbytes(baseline/(tag(id,bias,rows,ep,gemm)+"-ws")),"Separate V14b whole workspace differs");if(rows==1){finite32(bytes.data()+fb.aligned,bytes.size()-fb.aligned);if(!gemm)require(poison(std::vector<unsigned char>(bytes.begin(),bytes.begin()+fb.aligned)),"V13 fused-input A must stay untouched");else for(size_t k=0;k<size_t(f.i)*2;k+=2){uint16_t v;std::memcpy(&v,bytes.data()+k,2);require((v&0x7c00)!=0x7c00,"Nonfinite staged A");}}fb.intact();}w.w->device.fused_both_capable=observed_both;
        }
        {Case a(r,f,w.w,1),b(r,f,w.w,1);groups(r,a,b,ref,true);Case r8(r,f,w.w,8);groups(r,r8,b,ref,false);}
        {Case a(r,f,w.w,1),b(r,f,w.w,1);a.reset(-0.75f,-0.5f);b.reset(1.25f,0.25f);hip_check(hipStreamSynchronize(r.stream));hipStream_t other{};hip_check(hipStreamCreateWithFlags(&other,hipStreamNonBlocking));strix_stream s=r.s;s.hip_stream=other;Graph graph;graph.capture(r,1,[&]{b.submit();});a.submit();graph.replay(other);no_queries();hip_check(hipStreamSynchronize(r.stream));hip_check(hipStreamSynchronize(other));a.oracle(ref);b.oracle(ref);a.intact();b.intact();hip_check(hipStreamDestroy(other));}
        {Case c(r,f,w.w,1);c.reset(1,-0.0f);c.submit();auto previous_raw=get(r,c.y.p,c.y.bytes);std::vector<float> previous(previous_raw.size()/4);std::memcpy(previous.data(),previous_raw.data(),previous_raw.size());Guarded reference(c.y.bytes,r);upload(reference.p,previous_raw.data(),previous_raw.size(),r);c.d.alpha=-0.75f;c.d.beta=-0.5f;c.submit();hip_check(qa_literal(w.w->device,reinterpret_cast<float*>(c.x.p),reinterpret_cast<float*>(reference.p),r.stream,c.d.alpha,c.d.beta));require(get(r,reference.p,reference.bytes)==get(r,c.y.p,c.y.bytes),"Ordered reuse producer bytes");c.oracle(ref,&previous);c.intact();require(poison(get(r,c.ws.p,c.ws.bytes)),"Ordered reuse wrote workspace");}
        require(get(r,w.w->allocation,w.w->allocation_bytes)==immutable,"Prepared weights/scales/bias changed");w.close();no_queries();host_budget();
    }q1(r,inputs,baseline);r.close();no_queries();std::cout<<"PASS prepared V16 native contract\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
