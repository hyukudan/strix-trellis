// Complete C API head benchmark with independently checked H128 blocks.
#define main strix_head_unused_runtime_main
#include "../tests/gpu_runtime.hip"
#undef main
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <numeric>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "head_fixture.hpp"

namespace {
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
constexpr size_t guard_bytes=256;
using strix_trellis::bench::HeadData;
struct Mapping {
    int fd=-1;void* data=MAP_FAILED;size_t bytes;
    Mapping(const fs::path& path,size_t expected):bytes(expected) {
        fd=open(path.c_str(),O_RDONLY|O_CLOEXEC);
        if(fd<0) throw std::runtime_error("Cannot open head blob");
        struct stat info{};
        if(fstat(fd,&info) || info.st_size!=off_t(bytes)) {close(fd);fd=-1;throw std::runtime_error("Head blob size mismatch");}
        data=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);
        if(data==MAP_FAILED) {close(fd);fd=-1;throw std::runtime_error("Head blob mmap failed");}
    }
    ~Mapping(){if(data!=MAP_FAILED)munmap(data,bytes);if(fd>=0)close(fd);}
    Mapping(const Mapping&)=delete;
};
struct PreparedHead {
    Runtime& r;strix_weights w=nullptr;
    PreparedHead(Runtime& runtime,const HeadData& f):r(runtime) {
        Mapping blob(f.directory/"head.blob",f.blob_bytes);
        strix_exl3_metadata_v1 m{};m.struct_size=sizeof(m);m.transport_version=1;
        m.rows=f.n;m.columns=f.i;m.bits_x2=2*f.bits;m.codebook=static_cast<strix_codebook>(2);m.marker=f.marker;
        m.trellis_bytes=f.packed_bytes;m.suh_offset=f.suh_offset;m.suh_bytes=2*f.i;
        m.svh_offset=f.svh_offset;m.svh_bytes=2*f.n;
        strix_exl3_transport_desc d{};d.struct_size=sizeof(d);d.abi_version=1;
        d.format=STRIX_TRANSPORT_EXL3_CANONICAL;d.transport_version=1;d.data=blob.data;d.data_bytes=blob.bytes;
        d.metadata=&m;d.metadata_bytes=sizeof(m);d.rows=f.n;d.columns=f.i;
        api_check(strix_weights_prepare(r.context,&d,&r.s,&w));
        // The source mapping expires here; all prepared device storage must be owned.
    }
    ~PreparedHead(){if(w){(void)hipStreamSynchronize(r.stream);(void)strix_weights_destroy(w);}}
    void close(){hip_check(hipStreamSynchronize(r.stream));api_check(strix_weights_destroy(w));w=nullptr;}
};
struct Guarded {
    Device allocation;unsigned char* p;size_t bytes;
    Guarded(size_t n,Runtime& r):allocation(n+2*guard_bytes),p(static_cast<unsigned char*>(allocation.p)+guard_bytes),bytes(n) {
        hip_check(hipMemsetAsync(allocation.p,0xa5,n+2*guard_bytes,r.stream));
    }
    void check(Runtime& r) {
        unsigned char front[guard_bytes],back[guard_bytes];
        hip_check(hipMemcpyAsync(front,allocation.p,guard_bytes,hipMemcpyDeviceToHost,r.stream));
        hip_check(hipMemcpyAsync(back,p+bytes,guard_bytes,hipMemcpyDeviceToHost,r.stream));
        hip_check(hipStreamSynchronize(r.stream));
        for(size_t k=0;k<guard_bytes;++k) if(front[k]!=0xa5 || back[k]!=0xa5) throw std::runtime_error("Head canary changed");
    }
};
struct Stats{double minimum,median,p90;};
template<class Submit> Stats measure(Runtime& r,bool events,Submit submit) {
    hipEvent_t start{},stop{};hip_check(hipEventCreate(&start));hip_check(hipEventCreate(&stop));
    submit();submit();hip_check(hipStreamSynchronize(r.stream));
    std::vector<double> times;times.reserve(11);
    for(int rep=0;rep<11;++rep) {
        if(events) {
            hip_check(hipEventRecord(start,r.stream));submit();hip_check(hipEventRecord(stop,r.stream));
            hip_check(hipEventSynchronize(stop));float ms=0;hip_check(hipEventElapsedTime(&ms,start,stop));times.push_back(ms*1000.0);
        } else {
            auto begin=Clock::now();submit();hip_check(hipStreamSynchronize(r.stream));
            times.push_back(std::chrono::duration<double,std::micro>(Clock::now()-begin).count());
        }
    }
    hip_check(hipEventDestroy(start));hip_check(hipEventDestroy(stop));std::sort(times.begin(),times.end());
    return {times.front(),times[5],times[9]};
}
void print(const char* name,const Stats& s) {
    std::cout<<' '<<name<<"_min_us="<<s.minimum<<' '<<name<<"_median_us="<<s.median<<' '<<name<<"_p90_us="<<s.p90;
}
void check_output(Runtime& r,const HeadData& f,const Guarded& y,int rows,const char* phase) {
    std::vector<float> actual(size_t(rows)*f.n);
    hip_check(hipMemcpyAsync(actual.data(),y.p,actual.size()*4,hipMemcpyDeviceToHost,r.stream));hip_check(hipStreamSynchronize(r.stream));
    for(float v:actual) if(!std::isfinite(v)) throw std::runtime_error("Nonfinite full-head output");
    double worst_abs=0,worst_nrms=0;
    for(size_t b=0;b<f.blocks.size();++b) for(int row=0;row<rows;++row) {
        double maxabs=0,maxref=0,err2=0,ref2=0;
        for(int c=0;c<128;++c) {
            double ref=f.expected[(b*f.rows+row)*128+c],value=actual[size_t(row)*f.n+f.blocks[b]+c],e=value-ref;
            maxabs=std::max(maxabs,std::abs(e));maxref=std::max(maxref,std::abs(ref));err2+=e*e;ref2+=ref*ref;
        }
        double nrms=std::sqrt(err2/std::max(ref2,1e-30));
        if(maxabs>5e-4*std::max(1.0,maxref) || nrms>3e-5) throw std::runtime_error("Sampled head numerical gate failed");
        worst_abs=std::max(worst_abs,maxabs);worst_nrms=std::max(worst_nrms,nrms);
    }
    std::cout<<"HEAD_CHECK phase="<<phase<<" rows="<<rows<<" sampled_values="<<rows*f.blocks.size()*128
             <<" total_values="<<actual.size()<<" maxabs="<<worst_abs<<" max_row_block_nrmse="<<worst_nrms
             <<" full_output_finite=1 checksum="<<std::accumulate(actual.begin(),actual.end(),0.0)<<'\n';
}
void run_rows(Runtime& r,const HeadData& f,strix_weights w,int rows) {
    Guarded x(size_t(rows)*f.i*4,r),y(size_t(rows)*f.n*4,r);
    upload(x.p,f.x.data(),x.bytes,r);hip_check(hipMemsetAsync(y.p,0xff,y.bytes,r.stream));
    strix_matmul_desc d;strix_matmul_desc_init(&d);d.weights=w;
    d.input=tensor(x.p,rows,f.i,false);d.output=tensor(y.p,rows,f.n,false);
    uint64_t need=0;api_check(strix_matmul_workspace_bytes(r.context,&d,&need));
    uint64_t bound=f.blob_bytes+x.bytes+y.bytes+need+6*guard_bytes;
    if(bound>=1ull<<30) throw std::runtime_error("Head device allocation bound exceeded");
    Guarded workspace(need,r);d.workspace=workspace.p;d.workspace_bytes=need;
    auto submit=[&](){api_check(strix_gemv(r.context,&d,&r.s));};
    submit();check_output(r,f,y,rows,"before");x.check(r);y.check(r);workspace.check(r);
    auto event=measure(r,true,submit);check_output(r,f,y,rows,"last_event");
    auto wall=measure(r,false,submit);check_output(r,f,y,rows,"last_wall");
    x.check(r);y.check(r);workspace.check(r);
    std::cout<<"HEAD_RESULT source_weights="<<f.source_kind()<<" reference_coverage=3_of_1010_H128_blocks api=gemv rows="<<rows
             <<" clones=1 cloned_bytes="<<f.packed_bytes<<" workspace_bytes="<<need<<" device_allocation_bound="<<bound<<" observations=11";
    print("event",event);print("wall",wall);
    std::cout<<" compressed_GBs="<<f.packed_bytes/event.median/1000<<" useful_TOPS="<<2.0*rows*f.i*f.n/event.median/1e6<<" canaries=PASS\n";
}
__global__ void head_read(const uint32_t* values,size_t count,unsigned long long* sums) {
    unsigned long long sum=0;
    for(size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;k<count;k+=size_t(gridDim.x)*blockDim.x) sum+=values[k];
    __shared__ unsigned long long local[256];local[threadIdx.x]=sum;__syncthreads();
    for(int step=128;step;step/=2){if(threadIdx.x<step)local[threadIdx.x]+=local[threadIdx.x+step];__syncthreads();}
    if(!threadIdx.x)sums[blockIdx.x]=local[0];
}
void read_control(Runtime& r,const HeadData& f) {
    Device packed(f.packed_bytes),sums(256*8);
    unsigned long long reference=0;
    {
        Mapping blob(f.directory/"head.blob",f.blob_bytes);
        auto* bytes=static_cast<const unsigned char*>(blob.data);
        for(size_t k=0;k<f.packed_bytes;k+=4){uint32_t value;std::memcpy(&value,bytes+k,4);reference+=value;}
        upload(packed.p,blob.data,f.packed_bytes,r);hip_check(hipStreamSynchronize(r.stream));
    }
    auto submit=[&](){head_read<<<256,256,0,r.stream>>>(static_cast<const uint32_t*>(packed.p),f.packed_bytes/4,static_cast<unsigned long long*>(sums.p));hip_check(hipGetLastError());};
    auto check=[&](){std::vector<unsigned long long> host(256);hip_check(hipMemcpyAsync(host.data(),sums.p,host.size()*8,hipMemcpyDeviceToHost,r.stream));hip_check(hipStreamSynchronize(r.stream));if(std::accumulate(host.begin(),host.end(),0ull)!=reference)throw std::runtime_error("Read checksum mismatch");};
    submit();check();auto event=measure(r,true,submit);check();auto wall=measure(r,false,submit);check();
    std::cout<<"HEAD_READ bytes="<<f.packed_bytes<<" checksum="<<reference;print("event",event);print("wall",wall);
    std::cout<<" compressed_GBs="<<f.packed_bytes/event.median/1000<<'\n';
}
} // namespace
int main(int argc,char** argv) {
    try {
        if(argc!=2) throw std::runtime_error("Usage: head_linear_bench fixture-directory");
        std::cout<<std::unitbuf<<std::setprecision(10);HeadData f(argv[1]);Runtime r;
        std::cout<<"HEAD_FIXTURE source_weights="<<f.source_kind()<<" actual_model_values="<<int(f.actual_model_values)<<" I="<<f.i<<" N="<<f.n<<" bits="<<f.bits<<" sampled_H128_blocks=3 total_H128_blocks=1010\n";
        PreparedHead weights(r,f);
        for(int rows=1;rows<=8;++rows)run_rows(r,f,weights.w,rows);
        weights.close();read_control(r,f);r.close();
        std::cout<<"PASS sampled_head_operator (sampled numerical coverage)\n";return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
