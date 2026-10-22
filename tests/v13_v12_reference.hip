// Standalone C ABI consumer of the frozen V12 archive. ROOT lease only.
#include "grouped_direct_runtime_qa_helpers.hpp"
namespace {
void save(const fs::path& p,const void* data,size_t n) {
    std::ofstream f(p,std::ios::binary);f.write(static_cast<const char*>(data),n);
    if(!f)throw std::runtime_error("Reference output write failed");
}
void dump(Runtime& r,strix_weights w,unsigned i,unsigned n,const void* input,
          const fs::path& path) {
    Device x(8ull*i*4),y(8ull*n*4);upload(x.p,input,8ull*i*4,r);
    strix_matmul_desc d;strix_matmul_desc_init(&d);d.weights=w;
    d.input=tensor(x.p,8,i,false);d.output=tensor(y.p,8,n,false);
    uint64_t need=0;api_check(strix_matmul_workspace_bytes(r.context,&d,&need));
    Device ws(need);d.workspace=ws.p;d.workspace_bytes=need;
    api_check(strix_gemv(r.context,&d,&r.s));
    std::vector<unsigned char> out(8ull*n*4),a(8ull*i*2);
    hip_check(hipMemcpyAsync(out.data(),y.p,out.size(),hipMemcpyDeviceToHost,r.stream));
    hip_check(hipMemcpyAsync(a.data(),ws.p,a.size(),hipMemcpyDeviceToHost,r.stream));
    hip_check(hipStreamSynchronize(r.stream));
    save(path.string()+"-y.f32",out.data(),out.size());save(path.string()+"-a.f16",a.data(),a.size());
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=3)throw std::runtime_error("Usage: v12_reference SCRATCH_ROOT NEW_OUTPUT_DIRECTORY (ROOT lease only)");
        fs::path root=fs::canonical(argv[1]),out=argv[2];
        if(!fs::create_directory(out))throw std::runtime_error("Output directory exists");
        Runtime r;
        const char* names[]={"layers.0.attn.wq_b.stgpu","layers.0.attn.wo_b.stgpu","layers.0.ffn.shared_experts.w2.stgpu"};
        for(unsigned k=0;k<3;++k){Fixture f((root/"gpu-dense-prefill-tests"/names[k]).string());
            qa_weight_reserve=f.parts.at("trellis").size()+2ull*(f.i+f.n)+1024;
            Weight w(r,f);dump(r,w.w,f.i,f.n,f.parts.at("x").data(),out/std::to_string(k));w.close();}
        HeadData f(root/"gpu-head-eaq-ref-v2");qa_weight_reserve=f.blob_bytes+1024;
        PreparedHead w(r,f);dump(r,w.w,f.i,f.n,f.x.data(),out/"head");w.close();r.close();
        std::cout<<"PASS standalone_frozen_V12_reference\n";return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
