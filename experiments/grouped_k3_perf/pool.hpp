// Private one-geometry pool. Never instantiate the two-geometry QA owner.
#pragma once
#include "../grouped_k3_math/owner.hpp"
#include "contract.hpp"
#include "submit_hooks.hpp"
namespace grouped_k3_perf {
using namespace grouped_k3_math;
class Pool;
class Prepared {
    friend class Pool;
    Prepared(std::shared_ptr<Pool> p,NativeBatch b,unsigned first,bool grouped):lease(std::move(p)),batch(b),first(first),grouped(grouped) {}
public:
    const std::shared_ptr<Pool> lease;
    const NativeBatch batch;
    const unsigned first;
    const bool grouped;
};
class Pool:public std::enable_shared_from_this<Pool> {
    struct Allocation {void* base;size_t bytes;bool immutable;std::vector<unsigned char> initial;};
    std::vector<Allocation> buffers;
    std::array<strix_weights,instances> weights{};
    std::array<std::array<Descriptor,instances>,2> descriptors{};
    std::vector<Region> certificates;
    strix_context context{};
    int device=-1;
    size_t bytes=0;
    Region alloc(size_t size,const std::vector<unsigned char>* initial=nullptr) {
        require(size && bytes+size+512<(1ull<<30),"explicit allocation limit");
        void* p=nullptr;hip(hipMalloc(&p,size+512));buffers.push_back({p,size+512,initial!=nullptr,{}});bytes+=size+512;
        hip(hipMemsetAsync(p,0xa5,size+512,stream));
        if(initial){buffers.back().initial=*initial;hip(hipMemcpyAsync(static_cast<char*>(p)+256,initial->data(),size,hipMemcpyHostToDevice,stream));}
        certify(p,size+512);
        Region r{reinterpret_cast<uintptr_t>(p),size+512,256,size,1,device,true};certificates.push_back(r);return r;
    }
    void certify(const void* p,size_t size) {
        hipPointerAttribute_t a{};hip(hipPointerGetAttributes(&a,p));
        require(a.device==device && a.type==hipMemoryTypeDevice,"actual pointer device");
        void* base=nullptr;size_t extent=0;hip(hipMemGetAddressRange(&base,&extent,const_cast<void*>(p)));
        auto offset=reinterpret_cast<uintptr_t>(p)-reinterpret_cast<uintptr_t>(base);
        require(offset<=extent && size<=extent-offset,"actual pointer extent");
    }
    bool authentic(const Region& r) const {
        return authentic_certificate(r,certificates.data(),certificates.size());
    }
    void weight(unsigned j,const CpuCase& c) {
        strix_exl3_metadata_v1 m{};m.struct_size=sizeof(m);m.transport_version=1;
        m.rows=c.output;m.columns=c.input;m.bits_x2=6;m.codebook=STRIX_CODEBOOK_MUL1;m.marker=STRIX_EXL3_MARKER_MUL1;
        std::vector<unsigned char> blob;
        auto append=[&](const char* name,uint64_t& at,uint64_t& size){const auto& t=c.get(name);if(t.empty())return;
            blob.resize((blob.size()+3)&~size_t(3));at=blob.size();size=t.size();blob.insert(blob.end(),t.begin(),t.end());};
        append("packed",m.trellis_offset,m.trellis_bytes);append("suh",m.suh_offset,m.suh_bytes);append("svh",m.svh_offset,m.svh_bytes);
        strix_exl3_transport_desc t{};t.struct_size=sizeof(t);t.abi_version=1;t.format=STRIX_TRANSPORT_EXL3_CANONICAL;
        t.transport_version=1;t.data=blob.data();t.data_bytes=blob.size();t.metadata=&m;t.metadata_bytes=sizeof(m);t.rows=m.rows;t.columns=m.columns;
        strix_stream s{};s.struct_size=sizeof(s);s.abi_version=1;s.hip_stream=stream;
        require(bytes+blob.size()+1024<(1ull<<30),"weight budget");api(strix_weights_prepare(context,&t,&s,&weights[j]));
        const auto w=weights[j];require(w->owner==context,"literal private9 weight owner");
        bytes+=w->allocation_bytes;certify(w->allocation,w->allocation_bytes);
        const auto& d=w->device;require(d.input==c.input && d.output==c.output && d.bits==3 && d.codebook==2 && !d.bias,"original K3 CB2 no bias");
        const void* ptrs[]={d.trellis,d.suh,d.svh};const char* tags[]={"packed","suh","svh"};
        for(unsigned f=0;f<3;++f) {
            auto base=reinterpret_cast<uintptr_t>(w->allocation),at=reinterpret_cast<uintptr_t>(ptrs[f]);auto size=c.get(tags[f]).size();
            require(at>=base && at-base<=w->allocation_bytes && size<=w->allocation_bytes-(at-base),"weight logical slice");
            Region r{base,w->allocation_bytes,at-base,size,1,device,true};certificates.push_back(r);
            for(auto& v:descriptors)v[j].regions[f==0?Packed:f==1?Suh:Svh]=r;
        }
        // strix_weights_prepare consumes host transport synchronously in V13.
    }
    Pool()=default;
public:
    hipStream_t stream{};
    uint64_t creation=0;
    std::array<CpuCase,instances> cpu;
    ~Pool() {
        AttrRegion observation(attribute_trace,AttrPhase::cleanup,creation);
        if(stream)cleanup(hipStreamSynchronize(stream));
        for(auto w:weights)if(w && strix_weights_destroy(w)!=STRIX_SUCCESS)std::terminate();
        for(const auto& a:buffers)cleanup(hipFree(a.base));
        if(stream)cleanup(hipStreamDestroy(stream));
        if(context && strix_context_destroy(context)!=STRIX_SUCCESS)std::terminate();
    }
    static std::shared_ptr<Pool> create(const std::string& path,int input) {
        require(input==5120 || input==2304,"fixed geometry");auto p=std::shared_ptr<Pool>(new Pool);
        p->certificates.reserve(1024);p->buffers.reserve(320);
        hip(hipGetDevice(&p->device));strix_context_desc d{};d.struct_size=sizeof(d);d.abi_version=1;d.device_ordinal=p->device;
        p->creation=attribute_trace.reserve();require(p->creation!=0,"actual context counter");
        auto result=strix_context_create(&d,&p->context);if(p->context)attribute_trace.associate(p->creation,p->context);api(result);
        attribute_trace.phase=AttrPhase::math;hip(hipStreamCreate(&p->stream));
        hipDeviceProp_t prop{};hip(hipGetDeviceProperties(&prop,p->device));
        require(std::string(prop.gcnArchName).find("gfx1151")==0 && prop.warpSize==32 && prop.maxThreadsPerBlock>=128 &&
            prop.maxGridSize[0]>=80 && prop.maxGridSize[1]>=32 && prop.maxGridSize[2]>=32,"native device admission");
        std::printf("hardware=%s arch=%s MPs=%d wave=%d\n",prop.name,prop.gcnArchName,prop.multiProcessorCount,prop.warpSize);
        const void* funcs[]={reinterpret_cast<const void*>(strix_trellis::grouped_input_transform),reinterpret_cast<const void*>(strix_trellis::grouped_baseline_core),reinterpret_cast<const void*>(strix_trellis::grouped_output_transform)};
        for(auto f:funcs){hipFuncAttributes a{};hip(hipFuncGetAttributes(&a,f));require(a.maxThreadsPerBlock>=128 && !a.localSizeBytes && !a.sharedSizeBytes,"native resources");}
        require(attribute_trace.admit(p->creation,funcs),"four old plus three math per actual context");
        AttrRegion region(attribute_trace,AttrPhase::prepare,p->creation);
        for(unsigned j=0;j<instances;++j) {
            char name[80];std::snprintf(name,sizeof(name),"/i%d-g%02u.gk3",input,j);auto& c=p->cpu[j];c=CpuCase::read(path+name);
            require(c.input==input && c.group==int(j) && c.scalar(0)==1 && c.scalar(1)==0 && c.get("bias").empty() && c.get("packed").size()==packed_bytes,"perf fixture contract");
            p->weight(j,c);auto x=p->alloc(c.get("x").size(),&c.get("x"));
            Layout l{};require(layout(c.input,c.output,c.splits,1,l) && l.stride==(input==5120?157696:86528),"literal SUM workspace stride");
            for(unsigned v=0;v<2;++v) {
                auto& desc=p->descriptors[v][j];desc.input=c.input;desc.output=c.output;desc.splits=c.splits;desc.alpha=1;desc.beta=0;
                desc.regions[X]=x;desc.regions[Y]=p->alloc(c.get("old").size());
                auto workspace=p->alloc(l.stride);auto a=workspace,partial=workspace;
                a.capacity=c.input*2;partial.offset+=l.a;partial.capacity=l.p;
                p->certificates.push_back(a);p->certificates.push_back(partial);desc.regions[A]=a;desc.regions[P]=partial;
                require(strix_trellis::linear_workspace_bytes(weights_device(p,j),1)==l.stride,"actual V13 workspace contract");
            }
        }
        disjoint(p->descriptors[0],p->descriptors[1]);hip(hipStreamSynchronize(p->stream));host_budget();
        std::printf("ledger explicit_gpu_bytes=%zu packed_resident=%llu host_limit=2147483648 contexts_observed=%zu\n",p->bytes,static_cast<unsigned long long>(instances*packed_bytes),attribute_trace.creations);
        return p;
    }
    static const strix_trellis::DeviceWeights& weights_device(const std::shared_ptr<Pool>& p,unsigned j){return p->weights.at(j)->device;}
    std::shared_ptr<const Prepared> prepare(unsigned g,unsigned clone,bool grouped) {
        AttrRegion observation(attribute_trace,AttrPhase::prepare,creation);
        auto first=start(g,clone);std::array<Descriptor,32> frozen{};
        for(unsigned j=0;j<g;++j){frozen[j]=descriptors[grouped][first+j];for(const auto& r:frozen[j].regions)require(authentic(r),"unregistered cert");}
        hipDevice_t sd=-1;unsigned flags=0;hip(hipStreamGetDevice(stream,&sd));hip(hipStreamGetFlags(stream,&flags));require(sd==device,"owned stream device");
        grouped_k3_review::Plan checked;require(grouped_k3_review::prepare(frozen.data(),g,{1,1,device,device,true},checked),"all descriptors before submit");
        return std::shared_ptr<const Prepared>(new Prepared(shared_from_this(),bridge(checked.batch()),first,grouped));
    }
    int old_launch(const Prepared& p,unsigned j) const {
        const auto& g=p.batch.groups[j];return int(strix_trellis::launch_linear(weights[p.first+j]->device,g.x,g.y,1,g.a,stream,g.alpha,g.beta,false));
    }
    void poison() const {for(const auto& variant:descriptors)for(const auto& d:variant)hip(hipMemsetAsync(reinterpret_cast<void*>(d.regions[Y].allocation+d.regions[Y].offset),0xff,d.output*4,stream));hip(hipStreamSynchronize(stream));}
    void immutable_and_guards() const {
        host_budget();
        for(const auto& a:buffers){std::vector<unsigned char> raw(a.bytes);hip(hipMemcpy(raw.data(),a.base,a.bytes,hipMemcpyDeviceToHost));
            require(std::all_of(raw.begin(),raw.begin()+256,[](auto b){return b==0xa5;}) && std::all_of(raw.end()-256,raw.end(),[](auto b){return b==0xa5;}),"outer guard failure (no internal A/P canary)");
            if(a.immutable)require(!std::memcmp(raw.data()+256,a.initial.data(),a.initial.size()),"immutable X");}
        for(unsigned j=0;j<instances;++j){const auto& d=weights[j]->device;const void* ptrs[]={d.trellis,d.suh,d.svh};const char* tags[]={"packed","suh","svh"};
            for(unsigned k=0;k<3;++k){const auto& expected=cpu[j].get(tags[k]);std::vector<unsigned char> raw(expected.size());hip(hipMemcpy(raw.data(),ptrs[k],raw.size(),hipMemcpyDeviceToHost));require(raw==expected,"immutable original weights");}}
    }
};
inline int old_callback(void* state,unsigned j){const auto& p=*static_cast<const Prepared*>(state);return p.lease->old_launch(p,j);}
inline int new_callback(void* state){const auto& p=*static_cast<const Prepared*>(state);return int(launch_three(p.batch,grids(p.batch),p.lease->stream));}
inline hipError_t submit(const Prepared& p) {
    SubmitScope prohibition;
    AttrRegion observation(attribute_trace,AttrPhase::submit,p.lease->creation);
    return hipError_t(dispatch({const_cast<Prepared*>(&p),old_callback,new_callback},p.grouped,p.batch.count));
}
}
