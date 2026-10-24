// Complete C API benchmark. Reuse only this project's fixture/lifecycle helpers.
#define main strix_correctness_harness_main
#include "../tests/gpu_runtime.hip"
#undef main
#include <chrono>
#include <iomanip>
#include <memory>
#include <numeric>

namespace {
constexpr size_t cold_bytes = 256ull << 20;
constexpr int rotations = 11;
using Clock = std::chrono::steady_clock;
struct Events {
    hipEvent_t start{}, stop{};
    Events() { hip_check(hipEventCreate(&start)); hip_check(hipEventCreate(&stop)); }
    ~Events() { (void)hipEventDestroy(start); (void)hipEventDestroy(stop); }
};
struct Stats { double minimum, median, p90; };
Stats stats(std::vector<double> v) {
    std::sort(v.begin(),v.end()); return {v.front(),v[v.size()/2],v[9]};
}
template<class Submit> Stats measure(Runtime& r, size_t clones, bool events, Submit submit) {
    Events e;
    // Discard one complete rotation, including touching every independent address.
    for(size_t c=0;c<clones;++c) submit(c);
    hip_check(hipStreamSynchronize(r.stream));
    std::vector<double> times;
    for(int rep=0;rep<rotations;++rep) {
        auto start=Clock::now();
        if(events) hip_check(hipEventRecord(e.start,r.stream));
        for(size_t c=0;c<clones;++c) {
            submit(c);
            if(!events) hip_check(hipStreamSynchronize(r.stream));
        }
        if(events) {
            hip_check(hipEventRecord(e.stop,r.stream));
            hip_check(hipEventSynchronize(e.stop));
            float ms=0; hip_check(hipEventElapsedTime(&ms,e.start,e.stop));
            times.push_back(double(ms)*1000/clones);
        } else times.push_back(std::chrono::duration<double,std::micro>(Clock::now()-start).count()/clones);
    }
    return stats(times);
}
void print_stats(const char* prefix,const Stats& s) {
    std::cout<<" "<<prefix<<"_min_us="<<s.minimum<<" "<<prefix<<"_median_us="<<s.median
             <<" "<<prefix<<"_p90_us="<<s.p90;
}
// Exact sum consumes every compressed word, without decoding or expanding Q.
__global__ void contiguous_read(const uint32_t* data,size_t count,unsigned long long* sums) {
    unsigned long long sum=0;
    for(size_t k=size_t(blockIdx.x)*blockDim.x+threadIdx.x;k<count;k+=size_t(gridDim.x)*blockDim.x)
        sum+=data[k];
    __shared__ unsigned long long local[256];
    local[threadIdx.x]=sum; __syncthreads();
    for(int stride=128;stride;stride/=2) {
        if(threadIdx.x<stride) local[threadIdx.x]+=local[threadIdx.x+stride];
        __syncthreads();
    }
    if(!threadIdx.x) sums[blockIdx.x]=local[0];
}
void read_baseline(Runtime& r,const Fixture& f,size_t clones) {
    const auto& packed=f.parts.at("trellis");
    Device data(clones*packed.size()), sums(256*sizeof(unsigned long long));
    for(size_t c=0;c<clones;++c) upload(static_cast<char*>(data.p)+c*packed.size(),packed.data(),packed.size(),r);
    hip_check(hipStreamSynchronize(r.stream));
    unsigned long long reference=0;
    for(size_t k=0;k<packed.size();k+=4) reference+=le(packed.data()+k,4);
    auto submit=[&](size_t c) {
        hipLaunchKernelGGL(contiguous_read,dim3(256),dim3(256),0,r.stream,
            reinterpret_cast<const uint32_t*>(static_cast<char*>(data.p)+c*packed.size()),
            packed.size()/4,static_cast<unsigned long long*>(sums.p));
        hip_check(hipGetLastError());
    };
    auto check=[&]() {
        std::vector<unsigned long long> host(256);
        for(size_t c=0;c<clones;++c) {
            submit(c);
            hip_check(hipMemcpyAsync(host.data(),sums.p,host.size()*sizeof(host[0]),hipMemcpyDeviceToHost,r.stream));
            hip_check(hipStreamSynchronize(r.stream));
            if(std::accumulate(host.begin(),host.end(),0ull)!=reference) throw std::runtime_error("Read checksum mismatch");
        }
    };
    check(); auto event=measure(r,clones,true,submit); auto wall=measure(r,clones,false,submit); check();
    std::cout<<"READ clones="<<clones<<" cloned_bytes="<<clones*packed.size()<<" checksum="<<reference;
    print_stats("event",event); print_stats("wall",wall);
    std::cout<<" compressed_GBs="<<packed.size()/event.median/1000<<"\n";
}
void run(Runtime& r,const std::string& path) {
    Fixture f(path);
    const size_t packed=f.parts.at("trellis").size();
    const size_t clones=(cold_bytes+packed-1)/packed;
    std::cout<<"FIXTURE path="<<path<<" I="<<f.i<<" N="<<f.n<<" bits="<<f.bits
             <<" codebook="<<f.cb<<" marker="<<f.marker<<" fixture_rows="<<f.rows<<" bias="<<f.bias<<"\n";
    // The parser's Q/checkpoints stay on host; no Q upload, expansion or CPU MM.
    for(const char* key:{"q","core","scaled","inh","a","old"}) f.parts.erase(key);
    {
        std::vector<std::unique_ptr<Weight>> weights;
        for(size_t c=0;c<clones;++c) weights.emplace_back(new Weight(r,f));
        for(uint32_t rows:{1u,2u,3u,4u,5u,6u,7u,8u,32u,64u,128u,129u,255u,256u,257u,511u,512u}) {
            if(rows>f.rows) continue;
            Device x(size_t(rows)*f.i*4),y(size_t(rows)*f.n*4);
            auto d=desc(weights[0]->w,x.p,y.p,rows,f,false,1,0);
            uint64_t bytes=0; api_check(strix_matmul_workspace_bytes(r.context,&d,&bytes));
            Device workspace(bytes); d.workspace=workspace.p; d.workspace_bytes=bytes;
            const size_t payload=packed+2ull*(f.i+f.n)+(f.bias?4ull*f.n:0);
            // API private slices each have at most 255 padding bytes.
            const size_t gpu_bound=clones*(payload+4*255)+d.input.buffer_bytes+d.output.buffer_bytes+bytes;
            if(gpu_bound>=(1ull<<30)) throw std::runtime_error("GPU allocation bound exceeds 1 GiB");
            upload(x.p,f.parts.at("x").data(),d.input.buffer_bytes,r);
            hip_check(hipStreamSynchronize(r.stream));
            auto submit=[&](size_t c) {
                d.weights=weights[c]->w;
                api_check(rows<=8?strix_gemv(r.context,&d,&r.s):strix_gemm(r.context,&d,&r.s));
            };
            auto ref=f.floats("ref"); ref.resize(size_t(rows)*f.n);
            double checksum=0;
            uint64_t output_fnv=0;
            auto check=[&](const char* phase) {
                std::vector<float> actual(ref.size());
                // Check every clone, including the last output after the sweep.
                for(size_t c=0;c<clones;++c) {
                    submit(c);
                    hip_check(hipMemcpyAsync(actual.data(),y.p,d.output.buffer_bytes,hipMemcpyDeviceToHost,r.stream));
                    hip_check(hipStreamSynchronize(r.stream));
                    compare(std::string("CHECK phase=")+phase+" rows="+std::to_string(rows)+" clone="+std::to_string(c),actual,ref);
                    checksum=std::accumulate(actual.begin(),actual.end(),0.0);
                    // FNV-1a over the output bits: equal values mean bit-identical outputs across builds.
                    output_fnv=1469598103934665603ull;
                    for(float v:actual) {
                        uint32_t bits; std::memcpy(&bits,&v,4);
                        for(int b=0;b<4;++b) output_fnv=(output_fnv^((bits>>(8*b))&255u))*1099511628211ull;
                    }
                }
            };
            check("before");
            auto event=measure(r,clones,true,submit);
            auto wall=measure(r,clones,false,submit);
            // Consume the output of the final timed sweep before another submit.
            std::vector<float> final(ref.size());
            hip_check(hipMemcpyAsync(final.data(),y.p,d.output.buffer_bytes,hipMemcpyDeviceToHost,r.stream));
            hip_check(hipStreamSynchronize(r.stream)); compare("CHECK phase=final_timed rows="+std::to_string(rows),final,ref);
            check("after");
            const size_t minimal=payload+size_t(rows)*(f.i+f.n)*4;
            std::cout<<"RESULT mode=cold api="<<(rows<=8?"gemv":"gemm")<<" rows="<<rows
                     <<" clones="<<clones<<" cloned_bytes="<<clones*packed<<" minimal_bytes="<<minimal
                     <<" workspace_bytes="<<bytes<<" gpu_allocation_bound="<<gpu_bound<<" rotations="<<rotations;
            print_stats("event",event); print_stats("wall",wall);
            std::cout<<" compressed_GBs="<<packed/event.median/1000<<" useful_TOPS="
                     <<2.0*rows*f.i*f.n/event.median/1e6<<" checksum="<<checksum
                     <<" output_fnv="<<std::hex<<output_fnv<<std::dec<<"\n";
        }
    }
    read_baseline(r,f,clones);
}
}
int main(int argc,char** argv) {
    try {
        if(argc<2) throw std::runtime_error("Usage: api_linear_bench FIXTURE...");
        std::cout<<std::setprecision(10)<<std::unitbuf;
        Runtime r;
        hipDeviceProp_t p{}; hip_check(hipGetDeviceProperties(&p,0));
        std::cout<<"HARDWARE multiprocessors="<<p.multiProcessorCount<<" clockRate_kHz="<<p.clockRate
                 <<" memoryClockRate_kHz="<<p.memoryClockRate<<" totalGlobalMem="<<p.totalGlobalMem
                 <<" event_rotation_amortized=1 wall_sync_each_call=1\n";
        for(int k=1;k<argc;++k) run(r,argv[k]);
        r.close(); std::cout<<"PASS complete_operator\n"; return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL "<<e.what()<<"\n"; return 1; }
}
