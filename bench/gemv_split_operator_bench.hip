// Complete three-launch device operator; manual splits, no production policy.
#define main split_operator_unused_runtime_main
#include "../tests/gpu_runtime.hip"
#undef main
#include "../src/gemv_combined.hpp"
#include <chrono>
#include <memory>
#include <iomanip>

static size_t live_bytes=0, peak_bytes=0;
struct SplitGuarded {
    Device allocation;
    unsigned char* p;
    size_t bytes;
    SplitGuarded(size_t n, Runtime& r):allocation(n+512),
        p(static_cast<unsigned char*>(allocation.p)+256),bytes(n) {
        live_bytes+=n+512; peak_bytes=std::max(peak_bytes,live_bytes);
        if(live_bytes>=size_t(1)<<30) throw std::runtime_error("GPU budget exceeded");
        hip_check(hipMemsetAsync(allocation.p,0xa5,n+512,r.stream));
    }
    ~SplitGuarded(){live_bytes-=bytes+512;}
    void guards(Runtime& r) const {
        unsigned char a[256],b[256];
        hip_check(hipMemcpyAsync(a,allocation.p,256,hipMemcpyDeviceToHost,r.stream));
        hip_check(hipMemcpyAsync(b,p+bytes,256,hipMemcpyDeviceToHost,r.stream));
        hip_check(hipStreamSynchronize(r.stream));
        for(int k=0;k<256;++k) if(a[k]!=0xa5 || b[k]!=0xa5)
            throw std::runtime_error("Canary changed");
    }
    std::vector<unsigned char> read(Runtime& r) const {
        std::vector<unsigned char> v(bytes);
        hip_check(hipMemcpyAsync(v.data(),p,bytes,hipMemcpyDeviceToHost,r.stream));
        hip_check(hipStreamSynchronize(r.stream)); return v;
    }
    void contents(Runtime& r,const void* ref) const {
        auto v=read(r);
        if(std::memcmp(v.data(),ref,bytes)) throw std::runtime_error("Read-only content changed");
    }
};
static bool golden(const std::vector<unsigned char>& raw,const std::vector<float>& ref,
                   double& maxabs,double& nrms) {
    double e2=0,r2=0,mr=0; maxabs=0;
    for(size_t k=0;k<ref.size();++k) {
        float v;std::memcpy(&v,raw.data()+k*4,4);
        if(!std::isfinite(v)||!std::isfinite(ref[k])) {maxabs=nrms=INFINITY;return false;}
        double e=double(v)-ref[k]; maxabs=std::max(maxabs,std::abs(e));
        mr=std::max(mr,std::abs(double(ref[k])));e2+=e*e;r2+=double(ref[k])*ref[k];
    }
    nrms=std::sqrt(e2/std::max(r2,1e-30));
    return maxabs<=5e-4*std::max(1.0,mr) && nrms<=3e-5;
}
struct SplitTuple {int s,mode;bool pass=true;std::vector<double> samples;};
static void run_fixture(Runtime& r,const std::string& path,
                        const std::chrono::steady_clock::time_point& start) {
    Fixture f(path);
    // The existing parser checks all section sizes. Release dense Q/checkpoints now.
    for(auto it=f.parts.begin();it!=f.parts.end();) {
        const auto& k=it->first;
        if(k!="trellis"&&k!="suh"&&k!="svh"&&k!="bias"&&k!="x"&&k!="ref"&&k!="a")
            it=f.parts.erase(it);else ++it;
    }
    if(f.rows<8 || f.cb!=2 || (f.bits!=5&&f.bits!=6)) throw std::runtime_error("Fixture domain");
    const auto& packed=f.parts.at("trellis");
    size_t clones=((size_t(256)<<20)+packed.size()-1)/packed.size();
    std::vector<std::unique_ptr<SplitGuarded>> weights;
    for(size_t c=0;c<clones;++c) {
        weights.emplace_back(new SplitGuarded(packed.size(),r));
        upload(weights.back()->p,packed.data(),packed.size(),r);
    }
    SplitGuarded u(f.i*2,r),v(f.n*2,r),b(f.bias?f.n*4:0,r);
    upload(u.p,f.parts.at("suh").data(),u.bytes,r);
    upload(v.p,f.parts.at("svh").data(),v.bytes,r);
    upload(b.p,f.parts.at("bias").data(),b.bytes,r);
    auto check_weights=[&] {
        for(auto& w:weights){w->guards(r);w->contents(r,packed.data());}
        u.guards(r);v.guards(r);b.guards(r);
        u.contents(r,f.parts.at("suh").data());v.contents(r,f.parts.at("svh").data());
        if(b.bytes)b.contents(r,f.parts.at("bias").data());
    };
    check_weights();
    std::cout<<"FIXTURE path="<<path<<" I="<<f.i<<" N="<<f.n<<" bits="<<f.bits
        <<" clones="<<clones<<" packed_resident="<<clones*packed.size()
        <<" scales=shared bias=shared X_A_partial_Y=shared_per_rows\n";
    for(int rows:{4,8}) {
        SplitGuarded x(size_t(rows)*f.i*4,r),a(size_t(rows)*f.i*2,r),y(size_t(rows)*f.n*4,r);
        upload(x.p,f.parts.at("x").data(),x.bytes,r);
        std::vector<std::unique_ptr<SplitGuarded>> partial;
        std::vector<SplitTuple> tuples;
        for(int s:{1,2,4,8,16,32}) {
            partial.emplace_back(new SplitGuarded(size_t(s)*rows*f.n*4,r));
            for(int mode:{0,3})tuples.push_back({s,mode,true,{}});
        }
        std::vector<float> ref=f.floats("ref");ref.resize(size_t(rows)*f.n);
        std::vector<std::vector<unsigned char>> baseline(6),basepartial(6);
        auto launch=[&](int t,size_t c) {
            strix_trellis::DeviceWeights w{reinterpret_cast<const uint32_t*>(weights[c]->p),
                reinterpret_cast<const uint16_t*>(u.p),reinterpret_cast<const uint16_t*>(v.p),
                f.bias?reinterpret_cast<const float*>(b.p):nullptr,int(f.i),int(f.n),int(f.bits),int(f.cb)};
            strix_trellis::gpu_test_input_transform<<<(size_t(rows)*(f.i/128)+3)/4,128,0,r.stream>>>(
                reinterpret_cast<const float*>(x.p),w.suh,reinterpret_cast<_Float16*>(a.p),f.i,rows);
            hip_check(hipGetLastError());
            hip_check(strix_trellis_combined::launch(w,reinterpret_cast<_Float16*>(a.p),
                reinterpret_cast<float*>(partial[t/2]->p),rows,rows,tuples[t].s,r.stream,tuples[t].mode));
            strix_trellis::gpu_test_output_transform<<<(size_t(rows)*(f.n/128)+3)/4,128,0,r.stream>>>(
                reinterpret_cast<const float*>(partial[t/2]->p),w.svh,w.bias,
                reinterpret_cast<float*>(y.p),f.n,rows,tuples[t].s,1.0f,0.0f);
            hip_check(hipGetLastError());
        };
        auto guards=[&] {
            x.guards(r);a.guards(r);y.guards(r);
            for(auto& p:partial)p->guards(r);
            x.contents(r,f.parts.at("x").data());
        };
        auto validate=[&](const char* phase) {
            for(int t=0;t<12;++t) {
                bool pass=true;double maxabs=0,nrms=0;
                for(size_t c=0;c<clones;++c) {
                    hip_check(hipMemsetAsync(y.p,0xff,y.bytes,r.stream));
                    hip_check(hipMemsetAsync(partial[t/2]->p,0xff,partial[t/2]->bytes,r.stream));
                    launch(t,c);auto got=y.read(r);
                    double e,n;pass=golden(got,ref,e,n)&&pass;maxabs=std::max(maxabs,e);nrms=std::max(nrms,n);
                    if(t%2==0&&c==0&&std::string(phase)=="before") baseline[t/2]=got;
                    pass=(got==baseline[t/2])&&pass;
                    auto p=partial[t/2]->read(r);
                    if(t%2==0&&c==0&&std::string(phase)=="before")basepartial[t/2]=p;
                    pass=(p==basepartial[t/2])&&pass;
                    for(size_t k=0;k<p.size();k+=4){float z;std::memcpy(&z,p.data()+k,4);pass=std::isfinite(z)&&pass;}
                    a.contents(r,f.parts.at("a").data());
                }
                tuples[t].pass=tuples[t].pass&&pass;
                std::cout<<"VALID phase="<<phase<<" rows="<<rows<<" S="<<tuples[t].s<<" mode="<<tuples[t].mode
                    <<" clones="<<clones<<" pass="<<pass<<" maxabs="<<maxabs<<" nrms="<<nrms<<'\n';
                guards();
            }
        };
        guards();validate("before");
        hipEvent_t begin,end;hip_check(hipEventCreate(&begin));hip_check(hipEventCreate(&end));
        // One discarded full rotation per tuple, followed by eleven alternating sweeps.
        for(int sweep=-1;sweep<11;++sweep) {
            for(int j=0;j<12;++j) {
                int t=(sweep%2==0)?11-j:j;
                if(!tuples[t].pass)continue;
                if(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>145) {
                    std::cout<<"OMISSION path="<<path<<" rows="<<rows<<" sweep="<<sweep<<" reason=internal145s\n";
                    throw std::runtime_error("Internal 145s budget");
                }
                hip_check(hipEventRecord(begin,r.stream));
                for(size_t c=0;c<clones;++c)launch(t,c);
                hip_check(hipEventRecord(end,r.stream));hip_check(hipEventSynchronize(end));
                float ms=0;hip_check(hipEventElapsedTime(&ms,begin,end));
                // Validate the complete last timed Y before any next GPU submission.
                auto got=y.read(r);
                if(got!=baseline[t/2])throw std::runtime_error("Last timed Y differs");
                if(sweep>=0) {
                    double us=double(ms)*1000/clones;tuples[t].samples.push_back(us);
                    std::cout<<"SAMPLE rows="<<rows<<" S="<<tuples[t].s<<" mode="<<tuples[t].mode
                        <<" sweep="<<sweep<<" us="<<us<<" last_Y_exact=1\n";
                }
            }
        }
        hip_check(hipEventDestroy(begin));hip_check(hipEventDestroy(end));
        validate("after");guards();
        for(auto& t:tuples) {
            auto q=t.samples;std::sort(q.begin(),q.end());
            std::cout<<"RESULT rows="<<rows<<" S="<<t.s<<" mode="<<t.mode<<" pass="<<t.pass<<" samples="<<q.size();
            if(q.size()==11)std::cout<<" min_us="<<q[0]<<" median_us="<<q[5]<<" p90_us="<<q[9];
            std::cout<<'\n';
        }
    }
    check_weights();std::cout<<"FIXTURE_PASS full_clones_readback=before_after canaries=PASS gpu_peak_bytes="<<peak_bytes<<'\n';
}
int main(int argc,char** argv) {
    std::cout<<std::setprecision(10)<<std::unitbuf;
    try {
        auto start=std::chrono::steady_clock::now();Runtime r;
        for(int k=1;k<argc;++k)run_fixture(r,argv[k],start);
        r.close();std::cout<<"PASS complete_operator gpu_peak_bytes="<<peak_bytes<<'\n';return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
