// Private owning factory. No ownership claims for public C API raw pointers.
#pragma once
#include "kernels.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
namespace tp_decode {
inline void require(bool ok,const char* what){if(!ok)throw std::runtime_error(what);}
inline void hip(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
inline uint64_t live_bytes=0,peak_bytes=0;
class Allocation {
    void* pointer_=nullptr;size_t bytes_=0;
public:
    explicit Allocation(size_t bytes):bytes_(bytes){
        require(bytes&&live_bytes+bytes<(1ull<<30),"STOP private resident allocation budget");
        hip(hipMalloc(&pointer_,bytes));live_bytes+=bytes;peak_bytes=std::max(peak_bytes,live_bytes);
    }
    Allocation(const Allocation&)=delete;Allocation& operator=(const Allocation&)=delete;
    ~Allocation(){if(pointer_)hip(hipFree(pointer_));live_bytes-=bytes_;}
    void* data()const{return pointer_;}size_t bytes()const{return bytes_;}
    Region region()const{return {uintptr_t(pointer_),bytes_};}
    void upload(const void* source,size_t n,size_t offset=0){require(offset<=bytes_&&n<=bytes_-offset,"upload extent");hip(hipMemcpy(static_cast<char*>(pointer_)+offset,source,n,hipMemcpyHostToDevice));}
    template<class T> std::vector<T> download(size_t count,size_t offset=0)const {
        require(offset<=bytes_&&count<=((bytes_-offset)/sizeof(T)),"download extent");std::vector<T> v(count);
        hip(hipMemcpy(v.data(),static_cast<char*>(pointer_)+offset,count*sizeof(T),hipMemcpyDeviceToHost));return v;
    }
    void poison(){hip(hipMemset(pointer_,0xa5,bytes_));}
};
class Stream {
    hipStream_t stream_=nullptr;
public:
    Stream(){hip(hipStreamCreate(&stream_));}~Stream(){if(stream_)hip(hipStreamDestroy(stream_));}
    Stream(const Stream&)=delete;Stream& operator=(const Stream&)=delete;
    hipStream_t get()const{return stream_;}void wait()const{hip(hipStreamSynchronize(stream_));}
};
struct Weight {
    std::shared_ptr<Allocation> allocation;strix_trellis::DeviceWeights value{};
    Weight(int i,int n,const std::vector<uint32_t>& packed,const std::vector<uint16_t>& u,
           const std::vector<uint16_t>& v,const std::vector<float>& bias){
        require(primary(i,n)&&packed.size()*4==13107200&&u.size()==size_t(i)&&v.size()==size_t(n),"weight geometry");
        size_t offset=packed.size()*4;allocation=std::make_shared<Allocation>(offset+2*i+2*n+4*bias.size()+256);
        allocation->poison();allocation->upload(packed.data(),offset);
        auto* b=static_cast<char*>(allocation->data());value.trellis=reinterpret_cast<uint32_t*>(b);
        value.suh=reinterpret_cast<uint16_t*>(b+offset);allocation->upload(u.data(),2*i,offset);offset+=2*i;
        value.svh=reinterpret_cast<uint16_t*>(b+offset);allocation->upload(v.data(),2*n,offset);offset+=2*n;
        if(!bias.empty()){require(bias.size()==size_t(n),"bias shape");value.bias=reinterpret_cast<float*>(b+offset);allocation->upload(bias.data(),4*n,offset);}
        value.input=i;value.output=n;value.bits=5;value.codebook=2;
        // Match actual V16 creation-time capabilities for the real whole operator control.
        value.fused_output_capable=true;value.fused_both_capable=true;
    }
};
static_assert(sizeof(strix_trellis::DeviceWeights)==56,"TEN");
static_assert(offsetof(strix_trellis::DeviceWeights,fused_output_capable)==48,"TEN flag1");
static_assert(offsetof(strix_trellis::DeviceWeights,fused_both_capable)==49,"TEN flag2");
struct Resources {
    struct Entry{const void* fn;hipFuncAttributes attributes;};
    std::vector<Entry> entries;hipDeviceProp_t device{};
    Resources(){int ordinal;hip(hipGetDevice(&ordinal));hip(hipGetDeviceProperties(&device,ordinal));
        require(device.warpSize==32&&std::strncmp(device.gcnArchName,"gfx1151",7)==0,"STOP required gfx1151 wave32 context");
        std::cout<<"HARDWARE name="<<device.name<<" arch="<<device.gcnArchName<<" global_bytes="<<device.totalGlobalMem
                 <<" HIP_multiprocessors="<<device.multiProcessorCount<<" warp="<<device.warpSize
                 <<" LDS_per_block="<<device.sharedMemPerBlock<<" max_threads_block="<<device.maxThreadsPerBlock<<'\n';
        size_t count;auto table=functions(count);require(count==30,"function inventory");
        for(size_t j=0;j<count;++j){hipFuncAttributes attr{};hip(hipFuncGetAttributes(&attr,table[j].pointer));entries.push_back({table[j].pointer,attr});
            std::cout<<"ATTRIBUTE "<<table[j].name<<" threads="<<attr.maxThreadsPerBlock<<" static="<<attr.sharedSizeBytes<<" dynamic_cap="<<attr.maxDynamicSharedSizeBytes<<" local="<<attr.localSizeBytes<<" regs="<<attr.numRegs<<'\n';
            require(attr.maxThreadsPerBlock>=table[j].threads&&attr.localSizeBytes==0,"STOP native function resources");}
    }
    void admit(Variant v,int rows,int i)const{
        size_t count;auto table=functions(count);
        for(size_t j=0;j<count;++j)if(table[j].variant==v&&table[j].rows==rows){
            auto attr=entries[j].attributes;
            require(attr.sharedSizeBytes+shared_bytes(table[j],i)<=size_t(device.sharedMemPerBlock),"STOP actual LDS capacity");
            require(shared_bytes(table[j],i)<=size_t(attr.maxDynamicSharedSizeBytes),"STOP actual function dynamic LDS capacity");
            int limit=(v==Variant::A||v==Variant::B)?(rows==6?64:256):96;
            require(attr.numRegs<=limit,"STOP native register resource");}
    }
};
class Prepared {
    std::shared_ptr<Weight> weights_;
    std::shared_ptr<Resources> resources_;
public:
    std::shared_ptr<Allocation> x,y,workspace;
    std::shared_ptr<Stream> stream;
    tp_desc descriptor{};
    Prepared(std::shared_ptr<Weight> weights,std::shared_ptr<Resources> resources,int rows,uint64_t xs,uint64_t ys,
             std::shared_ptr<Stream> s=std::make_shared<Stream>()):weights_(std::move(weights)),resources_(std::move(resources)),stream(std::move(s)){
        int i=weights_->value.input,n=weights_->value.output;require(rows>=1&&rows<=8&&xs>=uint64_t(i)&&ys>=uint64_t(n),"prepare shape");
        x=std::make_shared<Allocation>(4*((rows-1)*xs+i)+256);y=std::make_shared<Allocation>(4*((rows-1)*ys+n)+256);
        size_t a=(2*rows*i+255)&~size_t(255);workspace=std::make_shared<Allocation>(a+16*rows*n+256);
        descriptor.struct_size=sizeof(tp_desc);descriptor.abi_version=1;descriptor.numerics=TP_LOGICAL_K16_V1;descriptor.logical_splits=4;
        auto& m=descriptor.linear;m.struct_size=sizeof(m);m.abi_version=1;m.alpha=1;
        auto tensor=[&](Allocation& allocation,int width,uint64_t stride){strix_tensor_desc t{};t.struct_size=sizeof(t);t.abi_version=1;t.dtype=STRIX_DTYPE_F32;t.rank=2;
            t.data=allocation.data();t.buffer_bytes=allocation.bytes()-256;t.shape[0]=rows;t.shape[1]=width;t.strides[0]=stride;t.strides[1]=1;return t;};
        m.input=tensor(*x,i,xs);m.output=tensor(*y,n,ys);m.workspace=workspace->data();m.workspace_bytes=workspace->bytes()-256;
        // Resource calls happen only here, outside Submit/capture.
        for(auto v:{Variant::A,Variant::B,Variant::C,Variant::D,Variant::DiagnosticD})if(eligible(v,i,n,rows))resources_->admit(v,rows,i);
    }
    Prepared(const Prepared&)=delete;Prepared& operator=(const Prepared&)=delete;
    ~Prepared(){stream->wait();} // All owners remain alive until LAST completion, including exception paths.
    const Weight& weight()const{return *weights_;}
    Error check(const tp_desc& d,Variant v,Layout& l)const{
        Region owners[4]={x->region(),y->region(),workspace->region(),weights_->allocation->region()};
        auto e=validate(d,weights_->value.input,weights_->value.output,v,owners,l);if(e!=Error::Ok)return e;
        if(v==Variant::Old&&(l.xs!=uint64_t(l.input)||l.ys!=uint64_t(l.output)))return Error::Stride;
        return Error::Ok;
    }
    Error submit(const tp_desc& d,Variant v)const {
        Layout l;auto err=check(d,v,l);if(err!=Error::Ok)return err; // Before ANY enqueue.
        Arguments a{weights_->value,static_cast<const float*>(d.linear.input.data),static_cast<float*>(d.linear.output.data),
            static_cast<_Float16*>(d.linear.workspace),reinterpret_cast<float*>(static_cast<char*>(d.linear.workspace)+l.a_bytes),
            l.xs,l.ys,l.rows,d.linear.alpha,d.linear.beta};
        SubmitScope guard;
        return enqueue(a,v,stream->get())==hipSuccess?Error::Ok:Error::Resource;
    }
};
struct Graph {
    // Retention is explicit: graph owns the Prepared owner, never borrowed pointers.
    std::shared_ptr<Prepared> keep;hipGraph_t graph=nullptr;hipGraphExec_t executable=nullptr;
    Graph(std::shared_ptr<Prepared> p,Variant v):keep(std::move(p)){
        hip(hipStreamBeginCapture(keep->stream->get(),hipStreamCaptureModeGlobal));
        require(keep->submit(keep->descriptor,v)==Error::Ok,"graph submit");
        hip(hipStreamEndCapture(keep->stream->get(),&graph));
        hip(hipGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    }
    Graph(const Graph&)=delete;Graph& operator=(const Graph&)=delete;
    ~Graph(){keep->stream->wait();if(executable)hip(hipGraphExecDestroy(executable));if(graph)hip(hipGraphDestroy(graph));}
    void replay(){hip(hipGraphLaunch(executable,keep->stream->get()));}
};
} // namespace tp_decode
