// Standalone frozen V14b C ABI reference. ROOT lease only; never linked to V15.
#include "runtime_v15_case.hpp"
int main(int argc,char** argv){try{
    require(argc==3,"Usage: v13_control INPUT_DIRECTORY NEW_OUTPUT_DIRECTORY");fs::path inputs=argv[1],out=argv[2];require(fs::create_directory(out),"Output exists");Runtime r;
    for(int id=0;id<2;++id)for(int bias=0;bias<2;++bias){Fixture f((inputs/(std::to_string(id)+".stgpu")).string());setbias(f,bias);qa_weight_reserve=f.parts.at("trellis").size()+2ull*(f.i+f.n)+4ull*f.n+1024;Weight w(r,f);
        for(int rows:{1,8})for(int ep=0;ep<8;++ep)for(bool gemm:{false,true}){if(rows==8 && gemm)continue;Case c(r,f,w.w,rows);c.reset(epilogues[ep][0],epilogues[ep][1]);c.submit(gemm);save(out/(tag(id,bias,rows,ep,gemm)+"-y"),get(r,c.y.p,c.y.bytes));save(out/(tag(id,bias,rows,ep,gemm)+"-ws"),get(r,c.ws.p,c.ws.bytes));c.intact();}w.close();host_budget();
    }r.close();std::cout<<"PASS V14b separate process\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
