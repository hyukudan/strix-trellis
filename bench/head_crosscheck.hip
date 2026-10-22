// Internal frozen four-split workspace inspection, not a public workspace ABI.
// The build extracts our existing helper prefix, before its main. Its nested
// runtime-main rename prevents renaming the entire benchmark by a macro.
#include "head_crosscheck_helpers.hpp"

namespace {
void dump(Runtime& r,const fs::path& path,const void* device,size_t bytes) {
    std::vector<unsigned char> host(bytes);
    hip_check(hipMemcpyAsync(host.data(),device,bytes,hipMemcpyDeviceToHost,r.stream));
    hip_check(hipStreamSynchronize(r.stream));
    std::ofstream out(path,std::ios::binary);
    out.write(reinterpret_cast<const char*>(host.data()),host.size());
    if(!out) throw std::runtime_error("Dump failed: "+path.string());
}
void crosscheck_rows(Runtime& r,const HeadData& f,strix_weights w,int rows,const fs::path& out) {
    Guarded x(size_t(rows)*f.i*4,r),y(size_t(rows)*f.n*4,r);
    upload(x.p,f.x.data(),x.bytes,r);
    hip_check(hipMemsetAsync(y.p,0xff,y.bytes,r.stream));
    strix_matmul_desc d;strix_matmul_desc_init(&d);d.weights=w;
    d.input=tensor(x.p,rows,f.i,false);d.output=tensor(y.p,rows,f.n,false);
    d.beta=0;
    uint64_t need=0;api_check(strix_matmul_workspace_bytes(r.context,&d,&need));
    const size_t abytes=size_t(rows)*f.i*2,aligned=(abytes+255)&~size_t(255);
    if(need!=aligned+4ull*rows*f.n*4) throw std::runtime_error("Unexpected four-split workspace layout");
    Guarded workspace(need,r);d.workspace=workspace.p;d.workspace_bytes=need;
    // All poison/upload/submit/read operations use the same explicit stream.
    hip_check(hipMemsetAsync(workspace.p,0xff,need,r.stream));
    hip_check(hipStreamSynchronize(r.stream));
    const auto start=Clock::now();
    api_check(strix_gemv(r.context,&d,&r.s));
    hip_check(hipStreamSynchronize(r.stream));
    const double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    check_output(r,f,y,rows,"full_crosscheck_dump");
    x.check(r);y.check(r);workspace.check(r);
    std::vector<float> input(size_t(rows)*f.i);
    hip_check(hipMemcpyAsync(input.data(),x.p,x.bytes,hipMemcpyDeviceToHost,r.stream));
    hip_check(hipStreamSynchronize(r.stream));
    if(std::memcmp(input.data(),f.x.data(),x.bytes)) throw std::runtime_error("Input changed");
    const std::string prefix="r"+std::to_string(rows);
    dump(r,out/(prefix+"-a.f16"),workspace.p,abytes);
    dump(r,out/(prefix+"-y.f32"),y.p,y.bytes);
    std::cout<<"HEAD_FULL_DUMP rows="<<rows<<" values="<<size_t(rows)*f.n
             <<" workspace_bytes="<<need<<" splits=4 beta=0 poisoned_y=1 poisoned_workspace=1"
             <<" input_unchanged=1 canaries=PASS single_submit_wall_ms="<<ms<<'\n';
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=3) throw std::runtime_error("Usage: head_crosscheck fixture-directory NEW-output-directory");
        std::cout<<std::unitbuf<<std::setprecision(12);
        HeadData f(argv[1]);
        if(!f.actual_model_values) throw std::runtime_error("Real shared EAQ required");
        if(!fs::create_directory(argv[2])) throw std::runtime_error("Output directory already exists");
        Runtime r;PreparedHead weights(r,f);
        for(int rows=1;rows<=8;++rows) crosscheck_rows(r,f,weights.w,rows,argv[2]);
        weights.close();r.close();
        std::cout<<"PASS full dumps ready; independent comparison still required\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
