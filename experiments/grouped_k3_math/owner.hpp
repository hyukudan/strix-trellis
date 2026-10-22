// Private lab owner. Only native_qa.hip may construct it inside a future lease.
#pragma once
#include "abi.hpp"
#include "private_views.hpp"
#include "first_error.hpp"
#include <hip/hip_runtime.h>
#include "attribute_hook.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>

namespace strix_trellis {
__global__ void grouped_input_transform(grouped_k3_math::NativeBatch);
__global__ void grouped_baseline_core(grouped_k3_math::NativeBatch);
__global__ void grouped_output_transform(grouped_k3_math::NativeBatch);
__global__ void input_transform(const float*,const uint16_t*,_Float16*,int,int);
__global__ void output_transform(const float*,const uint16_t*,const float*,float*,int,int,int,float,float);
hipError_t baseline_core(const DeviceWeights&,const _Float16*,float*,int,int,int,hipStream_t);
}
namespace grouped_k3_math {
using namespace grouped_k3_review;
static_assert(sizeof(attribute_trace)<=128*1024,"fixed observation log budget");
inline void need(bool b,const char* text) { if(!b)throw std::runtime_error(text); }
inline void hip(hipError_t e) { if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e)); }
inline void api(strix_status e) { need(e==STRIX_SUCCESS,"actual V13 public API failed"); }
inline void cleanup(hipError_t e) { if(e!=hipSuccess)std::terminate(); }
inline void host_budget() {
    rusage usage{};need(getrusage(RUSAGE_SELF,&usage)==0,"host RSS query");
    need(uint64_t(usage.ru_maxrss)*1024<(2ull<<30),"host RSS >=2GiB; STOP");
}
struct CpuCase {
    int input=0,output=0,splits=0,group=0;
    std::map<std::string,std::vector<unsigned char>> tensors;
    static CpuCase read(const std::string& path) {
        std::ifstream f(path,std::ios::binary);need(bool(f),"CPU case missing");
        char magic[8];uint32_t h[6];f.read(magic,8);f.read(reinterpret_cast<char*>(h),sizeof(h));
        need(f && !std::memcmp(magic,"GK3CPU01",8) && h[4]==3 && h[5]==12,"CPU case header");
        CpuCase c;c.input=h[0];c.output=h[1];c.splits=h[2];c.group=h[3];
        for(unsigned k=0;k<h[5];++k) {
            char tag[9]{};uint64_t size=0;f.read(tag,8);f.read(reinterpret_cast<char*>(&size),8);
            need(f && size<32u*1024*1024,"CPU tensor bound");
            auto& v=c.tensors[tag];need(v.empty(),"duplicate tensor");v.resize(size);
            f.read(reinterpret_cast<char*>(v.data()),size);need(bool(f),"CPU tensor truncated");
        }
        need(f.peek()==EOF,"CPU case trailing bytes");return c;
    }
    const std::vector<unsigned char>& get(const char* name) const { return tensors.at(name); }
    float scalar(unsigned k) const { float v;std::memcpy(&v,get("scalars").data()+4*k,4);return v; }
};
struct Grids { dim3 input,core,output; };
inline Grids grids(const NativeBatch& b) {
    return {dim3((b.input+511)/512,b.count),dim3(b.output/64,b.splits,b.count),dim3((b.output+511)/512,b.count)};
}
class LabRAIIOwner;
class Graph;
class PreparedPlan {
    const NativeBatch batch_;
    const Grids grids_;
    const std::shared_ptr<LabRAIIOwner> lease_;
    const hipStream_t stream_;
    const uint64_t generation_;
    PreparedPlan(NativeBatch b,std::shared_ptr<LabRAIIOwner> owner,hipStream_t stream,uint64_t generation):
        batch_(b),grids_(grids(b)),lease_(std::move(owner)),stream_(stream),generation_(generation) {}
    friend class LabRAIIOwner;
    friend class Graph;
    friend hipError_t submit(const PreparedPlan&);
public:
    const NativeBatch& batch() const {return batch_;}
};
// The raw operation is private to this experiment; prepared submit never borrows
// mutable host descriptors or does allocations, pointer queries, copies or sync.
inline hipError_t launch_three(const NativeBatch& b,const Grids& g,hipStream_t stream) {
    return enqueue_three([&](unsigned stage) {
        if(stage==0) {hipLaunchKernelGGL(strix_trellis::grouped_input_transform,g.input,dim3(128),0,stream,b);}
        else if(stage==1) {hipLaunchKernelGGL(strix_trellis::grouped_baseline_core,g.core,dim3(128),0,stream,b);}
        else {hipLaunchKernelGGL(strix_trellis::grouped_output_transform,g.output,dim3(128),0,stream,b);}
        return hipGetLastError();
    },hipSuccess);
}
hipError_t submit(const PreparedPlan& p);

class LabRAIIOwner : public std::enable_shared_from_this<LabRAIIOwner> {
    struct Allocation { void* base;size_t bytes;Region region;bool writable;std::vector<unsigned char> initial; };
    struct Weight { strix_weights handle=nullptr; std::array<Region,4> regions; };
    struct Request { std::array<Descriptor,32> descriptors;unsigned geometry,stream; };
    int device_=-1;
    uint64_t generation_;
    uint64_t creation_=0;
    strix_context context_=nullptr;
    hipStream_t streams_[2]{};
    std::vector<Allocation> allocations_;
    std::vector<Weight> weights_;
    std::array<std::array<CpuCase,32>,2> cpu_;
    std::array<Request,4> requests_;
    size_t gpu_bytes_=0;
    LabRAIIOwner(uint64_t generation):generation_(generation) {}
    void account(size_t bytes) {
        need(bytes<(1ull<<30) && gpu_bytes_<(1ull<<30)-bytes,"explicit GPU <1GiB");gpu_bytes_+=bytes;
    }
    Region allocation(size_t bytes,const std::vector<unsigned char>* initial,bool writable) {
        need(bytes && bytes<32u*1024*1024,"bounded allocation");
        need(gpu_bytes_<(1ull<<30)-(bytes+512),"allocation would exceed explicit GPU budget");
        void* p=nullptr;hip(hipMalloc(&p,bytes+512));
        Allocation a{p,bytes+512,{reinterpret_cast<uintptr_t>(p),bytes+512,256,bytes,generation_,device_,true},writable,{}};
        allocations_.push_back(std::move(a));account(bytes+512);
        certify(p,bytes+512);
        hip(hipMemsetAsync(p,0xa5,bytes+512,streams_[0]));
        if(initial) {
            need(initial->size()==bytes,"initial extent");allocations_.back().initial=*initial;
            hip(hipMemcpyAsync(static_cast<char*>(p)+256,initial->data(),bytes,hipMemcpyHostToDevice,streams_[0]));
            // CpuCase vectors persist for the owner's lifetime.
        }
        return allocations_.back().region;
    }
    void certify(const void* pointer,size_t bytes) const {
        hipPointerAttribute_t attrs{};hip(hipPointerGetAttributes(&attrs,pointer));
        need(attrs.device==device_ && attrs.type==hipMemoryTypeDevice,"device pointer");
        void* base=nullptr;size_t extent=0;hip(hipMemGetAddressRange(&base,&extent,const_cast<void*>(pointer)));
        auto offset=reinterpret_cast<uintptr_t>(pointer)-reinterpret_cast<uintptr_t>(base);
        need(offset<=extent && bytes<=extent-offset,"actual allocation extent");
    }
    Weight weight(const CpuCase& c) {
        strix_exl3_metadata_v1 m{};m.struct_size=sizeof(m);m.transport_version=1;
        m.rows=c.output;m.columns=c.input;m.bits_x2=6;m.codebook=STRIX_CODEBOOK_MUL1;m.marker=STRIX_EXL3_MARKER_MUL1;
        std::vector<unsigned char> blob;
        auto append=[&](const char* name,uint64_t& at,uint64_t& bytes) {
            const auto& t=c.get(name);if(t.empty())return;
            blob.resize((blob.size()+3)&~size_t(3));at=blob.size();bytes=t.size();blob.insert(blob.end(),t.begin(),t.end());
        };
        append("packed",m.trellis_offset,m.trellis_bytes);append("suh",m.suh_offset,m.suh_bytes);
        append("svh",m.svh_offset,m.svh_bytes);append("bias",m.bias_offset,m.bias_bytes);
        strix_exl3_transport_desc t{};t.struct_size=sizeof(t);t.abi_version=1;t.format=STRIX_TRANSPORT_EXL3_CANONICAL;
        t.transport_version=1;t.data=blob.data();t.data_bytes=blob.size();t.metadata=&m;t.metadata_bytes=sizeof(m);t.rows=m.rows;t.columns=m.columns;
        strix_stream stream{};stream.struct_size=sizeof(stream);stream.abi_version=1;stream.hip_stream=streams_[0];
        need(gpu_bytes_<(1ull<<30)-(blob.size()+1024),"weights would exceed explicit GPU budget");
        weights_.emplace_back();auto& owned=weights_.back();
        api(strix_weights_prepare(context_,&t,&stream,&owned.handle));
        // Literal api.hip views, extracted and pinned; no guessed opaque layout.
        const auto& w=owned;
        need(w.handle->owner==context_,"actual weights owner");account(w.handle->allocation_bytes);
        certify(w.handle->allocation,w.handle->allocation_bytes);
        const auto& d=w.handle->device;
        need(d.input==c.input && d.output==c.output && d.bits==3 && d.codebook==2,"actual weights metadata");
        const void* ptrs[4]={d.trellis,d.suh,d.svh,d.bias};
        const size_t sizes[4]={c.get("packed").size(),c.get("suh").size(),c.get("svh").size(),c.get("bias").size()};
        for(unsigned f=0;f<4;++f) {
            if(!sizes[f]) {need(!ptrs[f],"null bias");continue;}
            auto base=reinterpret_cast<uintptr_t>(w.handle->allocation),at=reinterpret_cast<uintptr_t>(ptrs[f]);
            need(at>=base && at-base<=w.handle->allocation_bytes && sizes[f]<=w.handle->allocation_bytes-(at-base),"private weights slice");
            owned.regions[f]={base,w.handle->allocation_bytes,at-base,sizes[f],generation_,device_,true};
        }
        return owned;
    }
    bool authentic(const Region& r) const {
        if(!r.allocation)return true; // guard checks every absent-bias field
        auto same=[&](const Region& a) {return a.allocation==r.allocation && a.allocation_bytes==r.allocation_bytes &&
            a.offset==r.offset && a.capacity==r.capacity && a.owner==r.owner && a.device==r.device && a.live==r.live;};
        for(const auto& a:allocations_)if(same(a.region))return true;
        for(const auto& w:weights_)for(const auto& a:w.regions)if(same(a))return true;
        return false;
    }
    std::unique_ptr<const PreparedPlan> checked(const std::array<Descriptor,32>& descriptors,unsigned count,unsigned request) {
        if(!count || count>32 || request>=2)return {};
        for(unsigned g=0;g<count;++g)for(const auto& r:descriptors[g].regions)if(!authentic(r))return {};
        grouped_k3_review::Plan validated;
        if(!grouped_k3_review::prepare(descriptors.data(),count,{generation_,generation_,device_,device_,true},validated))return {};
        auto b=bridge(validated.batch());
        NativeGroup zero{};for(unsigned g=count;g<32;++g)need(!std::memcmp(&b.groups[g],&zero,sizeof(zero)),"inactive zero");
        return std::unique_ptr<const PreparedPlan>(new PreparedPlan(b,shared_from_this(),streams_[request],generation_));
    }
    void resources() const {
        hipDeviceProp_t p{};hip(hipGetDeviceProperties(&p,device_));
        need(std::string(p.gcnArchName).find("gfx1151")==0 && p.warpSize==32 && p.maxThreadsPerBlock>=128 &&
            p.maxThreadsDim[0]>=128 && p.maxGridSize[0]>=80 && p.maxGridSize[1]>=32 && p.maxGridSize[2]>=32,"device admission");
        std::printf("hardware=%s arch=%s wave=%d MPs=%d\n",p.name,p.gcnArchName,p.warpSize,p.multiProcessorCount);
        const void* functions[]={reinterpret_cast<const void*>(strix_trellis::grouped_input_transform),
            reinterpret_cast<const void*>(strix_trellis::grouped_baseline_core),reinterpret_cast<const void*>(strix_trellis::grouped_output_transform)};
        for(auto f:functions) {
            hipFuncAttributes a{};hip(hipFuncGetAttributes(&a,f)); // Exactly once/function/context.
            std::printf("math_resource maxthreads=%d local=%zu static=%zu regs=%d\n",a.maxThreadsPerBlock,a.localSizeBytes,a.sharedSizeBytes,a.numRegs);
            need(a.maxThreadsPerBlock>=128 && a.localSizeBytes==0 && a.sharedSizeBytes==0,"math resource admission");
        }
    }
public:
    LabRAIIOwner(const LabRAIIOwner&)=delete;
    LabRAIIOwner& operator=(const LabRAIIOwner&)=delete;
    ~LabRAIIOwner() {
        AttrRegion observation(attribute_trace,AttrPhase::cleanup,creation_);
        for(auto s:streams_)if(s)cleanup(hipStreamSynchronize(s));
        for(auto& w:weights_)if(w.handle && strix_weights_destroy(w.handle)!=STRIX_SUCCESS)std::terminate();
        for(auto& a:allocations_)cleanup(hipFree(a.base));
        for(auto s:streams_)if(s)cleanup(hipStreamDestroy(s));
        if(context_ && strix_context_destroy(context_)!=STRIX_SUCCESS)std::terminate();
    }
    static std::shared_ptr<LabRAIIOwner> create(const std::string& data,uint64_t generation) {
        need(generation!=0,"nonzero generation");
        auto o=std::shared_ptr<LabRAIIOwner>(new LabRAIIOwner(generation));
        o->weights_.reserve(64);o->allocations_.reserve(448);
        hip(hipGetDevice(&o->device_));strix_context_desc d{};d.struct_size=sizeof(d);d.abi_version=1;d.device_ordinal=o->device_;
        o->creation_=attribute_trace.reserve();
        need(o->creation_!=0,"attribute context capacity; STOP");
        auto context_result=strix_context_create(&d,&o->context_);
        if(o->context_)attribute_trace.associate(o->creation_,o->context_);
        api(context_result);
        attribute_trace.phase=AttrPhase::math;
        for(auto& s:o->streams_)hip(hipStreamCreate(&s));
        o->resources();
        const void* math[]={reinterpret_cast<const void*>(strix_trellis::grouped_input_transform),
            reinterpret_cast<const void*>(strix_trellis::grouped_baseline_core),reinterpret_cast<const void*>(strix_trellis::grouped_output_transform)};
        need(attribute_trace.admit(o->creation_,math),"actual attribute queries must be four V13 plus three math per context; STOP");
        AttrRegion observation(attribute_trace,AttrPhase::prepare,o->creation_);
        for(unsigned geometry=0;geometry<2;++geometry)for(unsigned g=0;g<32;++g) {
            int i=geometry?2304:5120;char name[64];std::snprintf(name,sizeof(name),"/i%d-g%02u.gk3",i,g);
            auto& c=o->cpu_[geometry][g];c=CpuCase::read(data+name);
            auto w=o->weight(c);
            auto x=o->allocation(c.get("x").size(),&c.get("x"),false);
            for(unsigned request=0;request<2;++request) {
                auto& r=o->requests_[geometry*2+request];r.geometry=geometry;r.stream=request;
                auto& v=r.descriptors[g];v.input=c.input;v.output=c.output;v.splits=c.splits;
                v.alpha=c.scalar(0);v.beta=c.scalar(1);
                v.regions[Packed]=w.regions[0];v.regions[X]=x;v.regions[Suh]=w.regions[1];
                v.regions[Svh]=w.regions[2];v.regions[Bias]=w.regions[3];
                v.regions[Y]=o->allocation(c.get("old").size(),&c.get("old"),true);
                v.regions[A]=o->allocation(size_t(c.input)*2,nullptr,true);
                v.regions[P]=o->allocation(size_t(c.splits)*c.output*4,nullptr,true);
            }
        }
        hip(hipStreamSynchronize(o->streams_[0]));
        host_budget();
        std::printf("explicit_gpu_bytes=%zu limit=1073741824 host_limit=2147483648\n",o->gpu_bytes_);
        return o;
    }
    std::unique_ptr<const PreparedPlan> prepare(unsigned geometry,unsigned request,unsigned count,uint64_t generation=0) {
        AttrRegion observation(attribute_trace,AttrPhase::prepare,creation_);
        if(geometry>=2 || request>=2 || !count || count>32 || (generation && generation!=generation_))return {};
        const auto descriptors=requests_[geometry*2+request].descriptors; // freeze once
        auto stream=streams_[request];hipDevice_t device=-1;unsigned flags=0;
        hip(hipStreamGetDevice(stream,&device));hip(hipStreamGetFlags(stream,&flags));
        if(device!=device_)return {};
        return checked(descriptors,count,request);
    }
    hipStream_t stream(unsigned index) const {need(index<2,"stream index");return streams_[index];}
    uint64_t generation() const {return generation_;}
    uint64_t creation() const {return creation_;}
    void negatives(unsigned geometry) {
        AttrRegion observation(attribute_trace,AttrPhase::prepare,creation_);
        for(unsigned fault=0;fault<23;++fault) {
            auto d=requests_[geometry*2].descriptors;auto& last=d[31];
            if(fault<8)last.regions[fault].capacity--;
            else if(fault<11)last.regions[Y+fault-8]=d[0].regions[Y+fault-8];
            else if(fault==11)last.regions[Y]=d[0].regions[A];
            else if(fault==12)last.regions[P]=d[0].regions[X];
            else if(fault==13)last.regions[A]=d[0].regions[Packed];
            else if(fault==14)last.input++;
            else if(fault==15)last.alpha=std::numeric_limits<float>::infinity();
            else if(fault==16)last.splits++;
            else if(fault==17)last.bits=2;
            else if(fault==18)last.rows=2;
            else if(fault==19)last.regions[P].owner++;
            else if(fault==20)last.regions[P].device++;
            else if(fault==21)last.regions[P].offset=UINT64_MAX;
            else {last.regions[Bias].allocation=0;} // malformed absent bias
            auto plan=checked(d,32,0); // all preparation, including auth, BEFORE capture
            hipGraph_t graph{};hip(hipStreamBeginCapture(streams_[0],hipStreamCaptureModeThreadLocal));
            if(plan)hip(submit(*plan));
            hip(hipStreamEndCapture(streams_[0],&graph));
            size_t nodes=0;hip(hipGraphGetNodes(graph,nullptr,&nodes));hip(hipGraphDestroy(graph));
            need(!plan && nodes==0,"last descriptor fault enqueued work; STOP");
            std::printf("negative_geometry=%u fault=%u native_nodes=0 preparation_guard=PASS\n",geometry,fault);
        }
        // Actual public dtype0 rejection, separate from private eligibility.
        strix_matmul_desc desc{};strix_matmul_desc_init(&desc);desc.weights=weights_[geometry*32].handle;
        const auto& c=cpu_[geometry][0];const auto& r=requests_[geometry*2].descriptors[0];
        desc.input.data=reinterpret_cast<void*>(r.regions[X].allocation+r.regions[X].offset);
        desc.input.buffer_bytes=size_t(c.input)*4;desc.input.rank=1;desc.input.shape[0]=c.input;desc.input.strides[0]=1;
        desc.output.data=reinterpret_cast<void*>(r.regions[Y].allocation+r.regions[Y].offset);
        desc.output.buffer_bytes=size_t(c.output)*4;desc.output.rank=1;desc.output.shape[0]=c.output;desc.output.strides[0]=1;
        desc.input.dtype=STRIX_DTYPE_INVALID;uint64_t bytes=0;
        need(strix_matmul_workspace_bytes(context_,&desc,&bytes)==STRIX_ERROR_UNSUPPORTED_FORMAT,"public dtype0 must reject");
    }
    const CpuCase& cpu(unsigned geometry,unsigned g) const {return cpu_.at(geometry).at(g);}
    void reset(unsigned geometry,unsigned request,unsigned count) const {
        for(unsigned g=0;g<count;++g) {
            const auto& r=requests_[geometry*2+request].descriptors[g].regions[Y];const auto& old=cpu_[geometry][g].get("old");
            hip(hipMemcpyAsync(reinterpret_cast<void*>(r.allocation+r.offset),old.data(),old.size(),hipMemcpyHostToDevice,streams_[request]));
        }
        hip(hipStreamSynchronize(streams_[request]));
    }
    void original(const PreparedPlan& p,bool full) const {
        for(int g=0;g<p.batch_.count;++g) {
            const auto& v=p.batch_.groups[g];
            const strix_trellis::DeviceWeights* w=nullptr;
            for(const auto& owner:weights_)if(owner.handle->device.trellis==v.packed)w=&owner.handle->device;
            need(w,"actual original weights");
            if(full) {
                // Full V13 workspace has contiguous A/P. Allocate outside submit/timing.
                const size_t bytes=strix_trellis::linear_workspace_bytes(*w,1);void* workspace=nullptr;
                need(bytes<=157696 && gpu_bytes_<(1ull<<30)-bytes,"transient original workspace GPU budget");
                hip(hipMalloc(&workspace,bytes));
                try { hip(strix_trellis::launch_linear(*w,v.x,v.y,1,workspace,p.stream_,v.alpha,v.beta,false));hip(hipStreamSynchronize(p.stream_)); }
                catch(...) {cleanup(hipFree(workspace));throw;}
                cleanup(hipFree(workspace));
            } else {
                hipLaunchKernelGGL(strix_trellis::input_transform,dim3((w->input+511)/512),dim3(128),0,p.stream_,v.x,v.suh,v.a,w->input,1);
                hip(hipGetLastError());hip(strix_trellis::baseline_core(*w,v.a,v.partial,1,1,p.batch_.splits,p.stream_));
                hipLaunchKernelGGL(strix_trellis::output_transform,dim3((w->output+511)/512),dim3(128),0,p.stream_,v.partial,v.svh,v.bias,v.y,w->output,1,p.batch_.splits,v.alpha,v.beta);
                hip(hipGetLastError());
            }
        }
    }
    void immutable_and_canaries() const {
        host_budget();
        for(const auto& a:allocations_) {
            std::vector<unsigned char> raw(a.bytes);hip(hipMemcpy(raw.data(),a.base,a.bytes,hipMemcpyDeviceToHost));
            need(std::all_of(raw.begin(),raw.begin()+256,[](auto b){return b==0xa5;}) &&
                std::all_of(raw.end()-256,raw.end(),[](auto b){return b==0xa5;}),"canary corruption");
            if(!a.writable)need(!std::memcmp(raw.data()+256,a.initial.data(),a.initial.size()),"immutable X changed");
        }
        for(unsigned geometry=0;geometry<2;++geometry)for(unsigned g=0;g<32;++g) {
            const auto& w=weights_[geometry*32+g].handle->device;const auto& c=cpu_[geometry][g];
            const void* ptrs[]={w.trellis,w.suh,w.svh,w.bias};const char* tags[]={"packed","suh","svh","bias"};
            for(unsigned k=0;k<4;++k) {
                const auto& expected=c.get(tags[k]);if(expected.empty())continue;
                std::vector<unsigned char> raw(expected.size());hip(hipMemcpy(raw.data(),ptrs[k],raw.size(),hipMemcpyDeviceToHost));
                need(raw==expected,"immutable weight changed");
            }
        }
    }
};
inline hipError_t submit(const PreparedPlan& p) {
    AttrRegion observation(attribute_trace,AttrPhase::submit,p.lease_->creation());
    if(p.generation_!=p.lease_->generation())return hipErrorInvalidValue;
    return launch_three(p.batch_,p.grids_,p.stream_);
}
} // namespace grouped_k3_math
