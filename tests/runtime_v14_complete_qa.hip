// Prepared integration QA; execution belongs exclusively to ROOT's leased turn.
#include "v14_helpers.hpp"
#include "../src/backend.hpp"
#include "../src/gemv_lds.hpp"
#include "../src/gemm.hpp"
#include "../src/gemv_k2_multirow.hpp"
#include "../src/wmma_packed.hpp"
#include "../src/v14_core_policy.hpp"
#include "qa_private_layout.hpp"
#include "v14_native_probe.hpp"
#include <atomic>
#include <memory>
#include "runtime_v14_extent.hpp"
namespace strix_trellis::v13_control {
hipError_t launch_gemv_lds_core(const DeviceWeights&,const _Float16*,float*,int,int,int,hipStream_t);
hipError_t launch_wmma_core(const DeviceWeights&,const _Float16*,float*,int,int,hipStream_t);
}
static std::atomic<unsigned> attributes{0};
static const void* queried[4]={};
extern "C" hipError_t __real_hipFuncGetAttributes(hipFuncAttributes*,const void*);
extern "C" hipError_t __wrap_hipFuncGetAttributes(hipFuncAttributes* a,const void* f) {
    unsigned i=attributes.fetch_add(1);if(i<4)queried[i]=f;
    return __real_hipFuncGetAttributes(a,f);
}
static void require(bool v,const std::string& message) {if(!v)throw std::runtime_error(message);}
static void no_queries(){require(attributes==4,"Additional function attribute query");}
template<class T>static std::vector<T> read(const fs::path& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    require(bool(f),"Missing reference: "+path.string());auto size=f.tellg();
    require(size>=0 && size<512ll*1024*1024 && size%sizeof(T)==0,"Invalid reference size");
    std::vector<T> out(size_t(size)/sizeof(T));f.seekg(0);
    f.read(reinterpret_cast<char*>(out.data()),size);require(bool(f),"Truncated reference");return out;
}
static std::vector<unsigned char> download(Runtime& r,const void* p,size_t bytes) {
    std::vector<unsigned char> out(bytes);
    hip_check(hipMemcpyAsync(out.data(),p,bytes,hipMemcpyDeviceToHost,r.stream));
    hip_check(hipStreamSynchronize(r.stream));return out;
}
struct Graph {
    hipGraph_t graph=nullptr;hipGraphExec_t exec=nullptr;
    ~Graph(){if(exec)(void)hipGraphExecDestroy(exec);if(graph)(void)hipGraphDestroy(graph);}
    template<class F>void capture(Runtime& r,size_t expected,F submit) {
        hip_check(hipStreamBeginCapture(r.stream,hipStreamCaptureModeThreadLocal));
        submit();hip_check(hipStreamEndCapture(r.stream,&graph));
        size_t count=0;hip_check(hipGraphGetNodes(graph,nullptr,&count));
        require(count==expected,"Wrong graph node count");
        std::vector<hipGraphNode_t> nodes(count);if(count)hip_check(hipGraphGetNodes(graph,nodes.data(),&count));
        for(auto node:nodes){hipGraphNodeType type;hip_check(hipGraphNodeGetType(node,&type));require(type==hipGraphNodeTypeKernel,"Non-kernel graph node");}
        if(count)hip_check(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));no_queries();
    }
    void replay(Runtime& r){hip_check(hipGraphLaunch(exec,r.stream));no_queries();}
};
static std::vector<float> expected(const Fixture& f,int rows,float alpha,float beta) {
    auto ref=f.floats("ref"),old=f.floats("old");ref.resize(size_t(rows)*f.n);
    for(size_t j=0;j<ref.size();++j){volatile float av=alpha*ref[j];if(beta!=0){volatile float bv=beta*old[j];ref[j]=av+bv;}else ref[j]=av;}
    return ref;
}
static void native_envelope(Runtime& r,const fs::path& root) {
    auto oracle=read<double>(root/"wmma-packed-probe/build/oracles/native.bin");
    require(oracle.size()==512,"Native oracle shape");Guarded d(513ull*4*4*256*4,r);
    for(int selected=0;selected<4;++selected){
        native_probe<<<513,128,0,r.stream>>>(reinterpret_cast<float*>(d.p),1,0,selected);hip_check(hipGetLastError());
        auto raw=download(r,d.p,d.bytes);std::vector<float> got(raw.size()/4);std::memcpy(got.data(),raw.data(),raw.size());
        for(int test=0;test<513;++test)for(int wave=0;wave<4;++wave)for(int m=0;m<4;++m)for(int row=0;row<16;++row)for(int col=0;col<16;++col){
            double ref=0,sumabs=0;
            if(m==selected){if(test<256)ref=row*16+col==test?2:0;else if(test<512)ref=row*16+col==test-256?-3:0;else{ref=oracle[row*16+col];sumabs=oracle[256+row*16+col];}}
            double value=got[(((size_t(test)*4+wave)*4+m)*16+row)*16+col],error=std::abs(value-ref);
            require(std::isfinite(value),"Native nonfinite");
            require(test<512 || m!=selected ? error==0 : error<=16*std::ldexp(1.,-24)*sumabs,"Original WMMA impulse/mixed envelope");
        }
        d.check(r);
    }
    no_queries();
}
struct Case {
    Runtime& r;Fixture& f;strix_weights weights;int rows,splits;bool wmma;
    Guarded x,y,workspace,control_p,control_y;size_t aligned;
    strix_matmul_desc d{};
    static size_t need(Runtime& r,Fixture& f,strix_weights w,int rows,void* x,void* y) {
        auto d=desc(w,x,y,unsigned(rows),f,false,1,0);uint64_t bytes=0;
        api_check(strix_matmul_workspace_bytes(r.context,&d,&bytes));return bytes;
    }
    Case(Runtime& rt,Fixture& fixture,strix_weights w,int m,int s):r(rt),f(fixture),weights(w),rows(m),splits(s),wmma(m>8),
        x(size_t(m)*f.i*4,r),y(size_t(m)*f.n*4,r),workspace(need(r,f,w,m,x.p,y.p),r),
        control_p(size_t(s)*m*f.n*4,r),control_y(y.bytes,r),aligned((size_t(m)*f.i*2+255)&~size_t(255)) {
        require(f.rows>=unsigned(m),"Missing fixture rows");
        require(workspace.bytes==aligned+control_p.bytes,"Actual workspace split policy differs");
        d=desc(w,x.p,y.p,unsigned(m),f,false,1,0);d.workspace=workspace.p;d.workspace_bytes=workspace.bytes;
        upload(x.p,f.parts.at("x").data(),x.bytes,r);reset();no_queries();
    }
    void reset(float alpha=1,float beta=0) {
        d.alpha=alpha;d.beta=beta;
        hip_check(hipMemsetAsync(workspace.p,255,workspace.bytes,r.stream));
        hip_check(hipMemsetAsync(control_p.p,255,control_p.bytes,r.stream));
        if(beta==0){hip_check(hipMemsetAsync(y.p,255,y.bytes,r.stream));hip_check(hipMemsetAsync(control_y.p,255,y.bytes,r.stream));}
        else {upload(y.p,f.parts.at("old").data(),y.bytes,r);upload(control_y.p,f.parts.at("old").data(),y.bytes,r);}
    }
    void submit(const strix_stream* stream=nullptr) {
        api_check(wmma?strix_gemm(r.context,&d,stream?stream:&r.s):strix_gemv(r.context,&d,stream?stream:&r.s));no_queries();
    }
    std::vector<unsigned char> check(const std::string& label,bool independent=true) {
        auto a=reinterpret_cast<const _Float16*>(workspace.p);auto p=reinterpret_cast<float*>(control_p.p);
        auto& w=weights->device;
        hip_check(wmma?strix_trellis::v13_control::launch_wmma_core(w,a,p,rows,splits,r.stream):
            strix_trellis::v13_control::launch_gemv_lds_core(w,a,p,rows,rows,splits,r.stream));
        auto actual_p=download(r,workspace.p+aligned,control_p.bytes);
        require(actual_p==download(r,control_p.p,control_p.bytes),label+": full original V13 P bytes differ");
        strix_trellis::gpu_test_output_transform<<<(size_t(rows)*(f.n/128)+3)/4,128,0,r.stream>>>(
            p,w.svh,w.bias,reinterpret_cast<float*>(control_y.p),f.n,rows,splits,d.alpha,d.beta);
        hip_check(hipGetLastError());auto out=download(r,y.p,y.bytes);
        require(out==download(r,control_y.p,y.bytes),label+": full original V13 Y bytes differ");
        if(independent){std::vector<float> floats(out.size()/4);std::memcpy(floats.data(),out.data(),out.size());compare(label,floats,expected(f,rows,d.alpha,d.beta));}
        auto ws=download(r,workspace.p,workspace.bytes);
        require(std::equal(ws.begin(),ws.begin()+size_t(rows)*f.i*2,f.parts.at("a").begin()),label+": original FP32 input transform A differs");
        require(download(r,x.p,x.bytes)==std::vector<unsigned char>(f.parts.at("x").begin(),f.parts.at("x").begin()+x.bytes),"X mutated");
        x.check(r);y.check(r);workspace.check(r);control_p.check(r);control_y.check(r);no_queries();return out;
    }
    void partial_reference(const std::vector<double>& refs,bool k2) {
        auto raw=download(r,workspace.p+aligned,control_p.bytes);std::vector<float> got(raw.size()/4);std::memcpy(got.data(),raw.data(),raw.size());
        const size_t plane=size_t(rows)*f.n;double e2=0,r2=0,ma=0,mr=0;
        require(refs.size()==(k2?size_t(4)*8*f.n*2:got.size()*2),"FP64 dimensions");
        for(int s=0;s<splits;++s)for(int row=0;row<rows;++row)for(unsigned col=0;col<f.n;++col){
            size_t j=size_t(row)*f.n+col,index=size_t(s)*plane+j;
            size_t ri=k2?(size_t(s)*8+row)*f.n+col:size_t(s)*2*plane+j;
            size_t ai=k2?size_t(4)*8*f.n+ri:ri+plane;
            double ref=refs[ri],sumabs=refs[ai],value=got[index],error=std::abs(value-ref);
            require(std::isfinite(value) && std::isfinite(ref) && std::isfinite(sumabs),"Nonfinite partial/oracle");
            if(k2)require(error<=2e-6*std::max(1.,sumabs),"K2 full S4/8/N FP64 partial envelope");
            e2+=error*error;r2+=ref*ref;ma=std::max(ma,error);mr=std::max(mr,std::abs(ref));
        }
        if(!k2)require(std::sqrt(e2/std::max(r2,1e-30))<=3e-5 && ma<=5e-4*std::max(1.,mr),"WMMA full FP64 partial tolerance");
    }
};
static void groups(Runtime& r,Case& a,Case& b) {
    auto first=a.d,second=b.d;first.workspace=second.workspace=nullptr;first.workspace_bytes=second.workspace_bytes=0;
    strix_matmul_desc descriptors[]={first,second};strix_grouped_moe_desc g{};
    g.struct_size=sizeof(g);g.abi_version=1;g.groups=descriptors;g.group_count=2;
    g.workspace=a.workspace.p;g.workspace_bytes=a.workspace.bytes;
    // Prepare outside capture. A full-declared hypothetical range must pass the
    // real host query, while the real HIP allocation must be provably too short.
    std::vector<std::unique_ptr<Guarded>> candidates;
    unsigned char* short_pointer=nullptr;size_t remaining=0;
    for(unsigned attempt=0;attempt<4 && !short_pointer;++attempt){
        candidates.emplace_back(new Guarded(b.y.bytes+attempt*4096,r));
        auto& candidate=*candidates.back();hipDeviceptr_t base{};size_t extent=0;
        hip_check(hipMemGetAddressRange(&base,&extent,candidate.allocation.p));
        uintptr_t pointer=0;size_t available=0;
        if(!v14_extent::tail(reinterpret_cast<uintptr_t>(candidate.allocation.p),candidate.bytes+2*guard_bytes,
            reinterpret_cast<uintptr_t>(base),extent,guard_bytes,b.y.bytes,pointer,available))continue;
        bool separate=true;
        for(Case* c:{&a,&b}){
            for(Guarded* buffer:{&c->x,&c->y,&c->workspace,&c->control_p,&c->control_y})
                separate &= !v14_extent::overlap(pointer,b.y.bytes,reinterpret_cast<uintptr_t>(buffer->allocation.p),buffer->bytes+2*guard_bytes);
            separate &= !v14_extent::overlap(pointer,b.y.bytes,reinterpret_cast<uintptr_t>(c->weights->allocation),c->weights->allocation_bytes);
        }
        if(!separate)continue;
        auto saved=descriptors[1];descriptors[1].output.data=reinterpret_cast<void*>(pointer);
        uint64_t needed=0;auto status=strix_grouped_moe_workspace_bytes(r.context,&g,&needed);descriptors[1]=saved;
        std::cout<<"extent_candidate rows="<<b.rows<<" I="<<b.f.i<<" N="<<b.f.n<<" attempt="<<attempt
                 <<" base="<<base<<" extent="<<extent<<" output="<<reinterpret_cast<void*>(pointer)
                 <<" remaining="<<available<<" required="<<b.y.bytes<<" query="<<int(status)<<'\n';
        if(status!=STRIX_SUCCESS)continue;
        require(needed==g.workspace_bytes,"Extent prequery workspace mismatch");
        short_pointer=reinterpret_cast<unsigned char*>(pointer);remaining=available;
    }
    require(short_pointer && remaining<b.y.bytes,"Cannot isolate bounded actual-extent negative");
    no_queries();
    auto ax=download(r,a.x.p,a.x.bytes),bx=download(r,b.x.p,b.x.bytes);
    auto weights=download(r,a.weights->allocation,a.weights->allocation_bytes);
    const char* names[]={"descriptor_invalid","declared_buffer_insufficient","actual_extent_insufficient"};
    for(int mode=0;mode<3;++mode){a.reset();b.reset();hip_check(hipStreamSynchronize(r.stream));
        auto saved=descriptors[1];
        if(mode==0)descriptors[1].output.reserved[0]=1;
        if(mode==1)descriptors[1].output.buffer_bytes=4;
        if(mode==2)descriptors[1].output.data=short_pointer;
        const auto expected=mode==0?STRIX_ERROR_INVALID_ARGUMENT:STRIX_ERROR_INSUFFICIENT_BUFFER;
        Graph invalid;invalid.capture(r,0,[&]{auto status=strix_grouped_moe(r.context,&g,&r.s);
            require(status==expected,std::string("Last-invalid ")+names[mode]+" expected="+std::to_string(int(expected))+" actual="+std::to_string(int(status)));});
        descriptors[1]=saved;
        for(Case* c:{&a,&b})for(auto range:{std::make_pair(c->y.p,c->y.bytes),std::make_pair(c->workspace.p,c->workspace.bytes)}){
            auto raw=download(r,range.first,range.second);require(std::all_of(raw.begin(),raw.end(),[](unsigned char v){return v==255;}),"Rejected group wrote output/workspace");}
        require(download(r,a.x.p,a.x.bytes)==ax && download(r,b.x.p,b.x.bytes)==bx,"Rejected group mutated input");
        require(download(r,a.weights->allocation,a.weights->allocation_bytes)==weights,"Rejected group mutated weights");
        for(auto& candidate:candidates){
            auto raw=download(r,candidate->allocation.p,candidate->bytes+2*guard_bytes);
            require(std::all_of(raw.begin(),raw.end(),[](unsigned char v){return v==0xa5;}),"Rejected group mutated short allocation/guards");
        }
        for(Case* c:{&a,&b}){c->x.check(r);c->y.check(r);c->workspace.check(r);c->control_p.check(r);c->control_y.check(r);}
        std::cout<<"PASS last_group "<<names[mode]<<" capture_nodes=0 outputs_workspace_inputs_weights_guards_unchanged=1\n";
    }
    a.reset();b.reset();Graph graph;graph.capture(r,6,[&]{api_check(strix_grouped_moe(r.context,&g,&r.s));});
    graph.replay(r);a.check("group first");
    // Equal X/weights/rows: shared workspace is also the complete second result.
    hip_check(hipMemcpyAsync(b.workspace.p,a.workspace.p,a.workspace.bytes,hipMemcpyDeviceToDevice,r.stream));b.check("group second");
}
static void k2_fallback_views(Runtime& r,Fixture& f,strix_weights weight) {
    auto w=weight->device;Guarded a(8ull*f.i*2+4,r),p(4ull*8*f.n*4,r),old(p.bytes,r);
    upload(a.p+2,f.parts.at("a").data(),8ull*f.i*2,r);
    hip_check(hipMemsetAsync(p.p,255,p.bytes,r.stream));hip_check(hipMemsetAsync(old.p,255,old.bytes,r.stream));
    auto x=reinterpret_cast<const _Float16*>(a.p+2);
    hip_check(strix_trellis::launch_gemv_lds_core(w,x,reinterpret_cast<float*>(p.p),2,8,4,r.stream));
    hip_check(strix_trellis::v13_control::launch_gemv_lds_core(w,x,reinterpret_cast<float*>(old.p),2,8,4,r.stream));
    require(download(r,p.p,p.bytes)==download(r,old.p,old.bytes),"A2 fallback full padded P differs");
    upload(a.p,f.parts.at("a").data(),8ull*f.i*2,r);
    hip_check(hipMemsetAsync(p.p,255,p.bytes,r.stream));hip_check(hipMemsetAsync(old.p,255,old.bytes,r.stream));
    x=reinterpret_cast<const _Float16*>(a.p)+3*f.i;
    auto out=reinterpret_cast<float*>(p.p)+3*f.n,control=reinterpret_cast<float*>(old.p)+3*f.n;
    Graph invalid;invalid.capture(r,0,[&]{require(strix_trellis::launch_gemv_k2_multirow(w,x,out,2,8,4,r.stream)==hipErrorInvalidValue,"Larger-operation chunk accepted");});
    hip_check(strix_trellis::launch_gemv_lds_core(w,x,out,2,8,4,r.stream));
    hip_check(strix_trellis::v13_control::launch_gemv_lds_core(w,x,control,2,8,4,r.stream));
    auto actual=download(r,p.p,p.bytes);require(actual==download(r,old.p,old.bytes),"Offset chunk fallback bytes differ");
    for(int split=0;split<4;++split)for(int row=0;row<8;++row)if(row!=3 && row!=4){
        auto begin=actual.begin()+(size_t(split)*8+row)*f.n*4;
        require(std::all_of(begin,begin+f.n*4,[](unsigned char v){return v==255;}),"Partial chunk wrote other rows");
    }
    a.check(r);p.check(r);old.check(r);no_queries();
}
static void set_bias(Fixture& f,bool enabled) {
    auto previous=f.floats("bias"),ref=f.floats("ref");
    std::vector<float> next(enabled?f.n:0);for(size_t col=0;col<next.size();++col)next[col]=(int(col%29)-14)*.0009765625f;
    for(size_t j=0;j<ref.size();++j){if(!previous.empty())ref[j]-=previous[j%f.n];if(enabled)ref[j]+=next[j%f.n];}
    f.bias=enabled;f.parts["bias"].resize(next.size()*4);
    if(!next.empty())std::memcpy(f.parts["bias"].data(),next.data(),next.size()*4);
    std::memcpy(f.parts["ref"].data(),ref.data(),ref.size()*4);
}
static void suite(Runtime& r,Fixture& f,strix_weights w,int rows,int splits,const std::vector<double>& refs,bool k2) {
    std::cout<<"suite I="<<f.i<<" N="<<f.n<<" rows="<<rows<<" splits="<<splits<<" bias="<<f.bias<<'\n';
    auto immutable=download(r,w->allocation,w->allocation_bytes);
    Case a(r,f,w,rows,splits),b(r,f,w,rows,splits);
    Graph rejected;rejected.capture(r,0,[&]{
        auto& dev=w->device;
        hipError_t status=rows>8?strix_trellis::launch_wmma_core(dev,nullptr,reinterpret_cast<float*>(a.workspace.p+a.aligned),rows,splits,r.stream):
            strix_trellis::launch_gemv_lds_core(dev,nullptr,reinterpret_cast<float*>(a.workspace.p+a.aligned),rows,rows,splits,r.stream);
        require(status==hipErrorInvalidValue,"Null A error did not propagate without enqueue");
    });
    a.submit();auto baseline=a.check("API first");if(!refs.empty())a.partial_reference(refs,k2);
    Graph graph;graph.capture(r,3,[&]{a.submit();});
    for(int replay=0;replay<2;++replay){a.reset();graph.replay(r);require(a.check("graph replay")==baseline,"Replay byte mismatch");}
    hipStream_t second=nullptr;hip_check(hipStreamCreateWithFlags(&second,hipStreamNonBlocking));
    strix_stream stream=r.s;stream.hip_stream=second;
    a.reset();b.reset();hip_check(hipStreamSynchronize(r.stream));a.submit();b.submit(&stream);
    hip_check(hipStreamSynchronize(second));hip_check(hipStreamSynchronize(r.stream));
    require(a.check("two streams first")==baseline && b.check("two streams second")==baseline,"Concurrent bytes differ");
    hip_check(hipStreamDestroy(second));groups(r,a,b);
    const float scales[][2]={{0,0},{-0.0f,-0.0f},{1,-0.0f},{.75f,0},{1,.5f},{-.5f,-.25f},{-0.0f,.5f}};
    for(auto& scale:scales){a.reset(scale[0],scale[1]);a.submit();a.check("alpha beta");}
    require(download(r,w->allocation,w->allocation_bytes)==immutable,"Prepared weights mutated");no_queries();
}
int main(int argc,char** argv) {
    try {
        require(argc==2,"Usage: runtime_v14_qa SCRATCH_ROOT (ROOT leased GPU turn only)");
        std::cout<<std::unitbuf<<std::setprecision(12);fs::path root=fs::canonical(argv[1]);
        require(root==fs::path(std::getenv("HOME"))/"gpu-site/scratch-trellis","Unexpected scratch root");
        Runtime r;no_queries();for(int i=0;i<4;++i){require(queried[i],"Null probe");for(int j=0;j<i;++j)require(queried[i]!=queried[j],"Repeated function probe");}
        native_envelope(r,root);
        for(const char* name:{"wo_b","q_b"}) {
            Fixture f((root/"k2-core/data"/(std::string(name)+".stgpu")).string());
            auto bytes=read<unsigned char>(root/"k2-rows-confirm"/"data/derived"/(std::string(name)+".fp64"));
            require(bytes.size()==24+size_t(4)*8*f.n*16 && !std::memcmp(bytes.data(),"K2RFP641",8),"K2 independent multirow oracle");
            require(le(bytes.data()+8,4)==f.i && le(bytes.data()+12,4)==f.n && le(bytes.data()+16,4)==4 && le(bytes.data()+20,4)==8,"K2 oracle shape");
            std::vector<double> refs((bytes.size()-24)/8);std::memcpy(refs.data(),bytes.data()+24,bytes.size()-24);
            Weight w(r,f);k2_fallback_views(r,f,w.w);for(int rows:{2,4,8})suite(r,f,w.w,rows,4,refs,true);
            for(int rows:{3,5,6,7})suite(r,f,w.w,rows,4,{},true);
            // Misaligned A is rejected by the pure selector, never by executing a new kernel.
            Case a(r,f,w.w,2,4);Graph invalid;invalid.capture(r,0,[&]{require(strix_trellis::launch_gemv_k2_multirow(w.w->device,reinterpret_cast<_Float16*>(a.workspace.p+2),reinterpret_cast<float*>(a.workspace.p+a.aligned),2,2,4,r.stream)==hipErrorInvalidValue,"A2 accepted by new trusted guard");});
            w.close();set_bias(f,!f.bias);Weight epilogue(r,f);
            for(int rows:{2,4,8})suite(r,f,epilogue.w,rows,4,{},true);
        }
        const char* names[]={"layers.0.attn.wq_b","layers.0.attn.wo_b","layers.0.ffn.shared_experts.w1","layers.0.ffn.shared_experts.w2","layers.0.attn.wq_a","layers.0.attn.wo_a.slice.0"};
        const int splits[6][3]={{1,1,1},{4,2,1},{8,4,2},{4,2,1},{16,8,4},{16,8,4}};
        for(int shape=0;shape<6;++shape){Fixture f((root/"wmma-packed-probe/fixtures"/(std::string(names[shape])+".stgpu")).string());
            Weight w(r,f);for(int slot=0;slot<3;++slot){int rows=64<<slot;if(shape==4 && rows!=256)continue;
                auto refs=read<double>(root/"wmma-packed-probe/build/oracles"/("perf-"+std::to_string(shape)+"-"+std::to_string(rows)+".bin"));
                suite(r,f,w.w,rows,splits[shape][slot],refs,false);
            }
            w.close();
            // Bias transport and its independent fixture output are patched only in host QA storage.
            set_bias(f,!f.bias);
            Weight biased(r,f);
            for(int slot=0;slot<3;++slot){int rows=64<<slot;if(shape==4 && rows!=256)continue;
                suite(r,f,biased.w,rows,splits[shape][slot],{},false);
            }
        }
        no_queries();r.close();std::cout<<"PASS native_runtime_v14 attributes=4 K2_all_partials=1 WMMA_tuples=16\n";return 0;
    } catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
