// Bounded complete-operator workload for profiling, not a timing benchmark.
#define main roofline_contract_harness_main
#include "../tests/gpu_runtime.hip"
#undef main
#include <memory>

int main(int argc,char** argv) {
    try {
        if(argc!=3) throw std::runtime_error("Usage: roofline_probe FIXTURE ROWS(1|8)");
        const std::string argument=argv[2];
        if(argument!="1" && argument!="8") throw std::runtime_error("Rows must be exactly 1 or 8");
        const int rows=std::stoi(argument);
        Fixture f(argv[1]);
        if(f.rows<uint32_t(rows)) throw std::runtime_error("Insufficient fixture rows");
        for(const char* key:{"q","core","scaled","inh","a","old"}) f.parts.erase(key);
        const size_t packed=f.parts.at("trellis").size();
        const size_t clones=((256ull<<20)+packed-1)/packed;
        if(clones>256) throw std::runtime_error("Profiling is restricted to large matrices");
        Runtime r;
        std::vector<std::unique_ptr<Weight>> weights;
        for(size_t k=0;k<clones;++k) weights.emplace_back(new Weight(r,f));
        Device x(size_t(rows)*f.i*4),y(size_t(rows)*f.n*4);
        upload(x.p,f.parts.at("x").data(),size_t(rows)*f.i*4,r);
        auto d=desc(weights[0]->w,x.p,y.p,rows,f,false,1,0);
        uint64_t need=0;api_check(strix_matmul_workspace_bytes(r.context,&d,&need));
        const size_t allocation=clones*(packed+2ull*(f.i+f.n)+(f.bias?4ull*f.n:0)+4*255)+d.input.buffer_bytes+d.output.buffer_bytes+need;
        if(allocation>=(1ull<<30)) throw std::runtime_error("Explicit allocation limit exceeded");
        Device workspace(need);d.workspace=workspace.p;d.workspace_bytes=need;
        auto submit=[&](size_t clone) {
            d.weights=weights[clone]->w;
            hip_check(hipMemsetAsync(y.p,0xff,d.output.buffer_bytes,r.stream));
            api_check(strix_gemv(r.context,&d,&r.s));
        };
        auto ref=f.floats("ref");ref.resize(size_t(rows)*f.n);
        auto check=[&](const char* phase) {
            std::vector<float> actual(ref.size());
            for(size_t clone=0;clone<clones;++clone) {
                submit(clone);
                hip_check(hipMemcpyAsync(actual.data(),y.p,d.output.buffer_bytes,hipMemcpyDeviceToHost,r.stream));
                hip_check(hipStreamSynchronize(r.stream));
                compare(std::string("PROFILE_CHECK phase=")+phase+" clone="+std::to_string(clone),actual,ref);
            }
        };
        check("before");
        for(int repetition=0;repetition<3;++repetition)
            for(size_t clone=0;clone<clones;++clone) submit(clone);
        hip_check(hipStreamSynchronize(r.stream));
        std::vector<float> final(ref.size());
        hip_check(hipMemcpyAsync(final.data(),y.p,d.output.buffer_bytes,hipMemcpyDeviceToHost,r.stream));
        hip_check(hipStreamSynchronize(r.stream));compare("PROFILE_CHECK last_submitted",final,ref);
        check("after");
        std::cout<<"PROFILE_WORKLOAD rows="<<rows<<" clones="<<clones<<" packed_bytes="<<packed
                 <<" explicit_allocation_bound="<<allocation<<" repetitions=3 submission_count="<<5*clones
                 <<" profiler_overhead_included=1 performance_claim=0\n";
        for(auto& w:weights) w->close();
        r.close();std::cout<<"PASS profiling_workload\n";return 0;
    } catch(const std::exception& e) {std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
