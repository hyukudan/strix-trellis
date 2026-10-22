// Isolated controls and correctness gates; reuses only our fixture/lifecycle code.
#define main blocked_unused_prefill_main
#include "prefill_split_bench.hip"
#undef main
#include "../src/wmma_blocked_submit.hpp"
#include "../src/gemv_lds.hpp"
#include "../src/trellis_optimized.hpp"
#include <array>
#include <chrono>
#include <filesystem>

namespace experiment {
namespace bk = strix_trellis::blocked;
constexpr size_t limit=1ull<<30;
constexpr std::array<int,5> split_list={1,3,4,16,32};
constexpr std::array<int,6> row_list={1,8,17,64,128,256};
size_t skipped=0,cases=0,partial_values=0;
double max_partial_error=0,max_partial_ratio=0;
auto campaign_start=std::chrono::steady_clock::now();
bool expired() {return std::chrono::duration<double>(std::chrono::steady_clock::now()-campaign_start).count()>145;}

void require(bool ok,const std::string& message) { if(!ok) throw std::runtime_error(message); }
template<class T> std::vector<T> read_file(const std::string& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    require(bool(f),"Cannot open "+path);
    auto bytes=f.tellg();require(bytes>=0 && bytes<512*1024*1024 && size_t(bytes)%sizeof(T)==0,"Invalid oracle size");
    std::vector<T> v(size_t(bytes)/sizeof(T));f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()),bytes);require(bool(f),"Truncated oracle");return v;
}
void exact(const void* device,const void* ref,size_t bytes,hipStream_t stream,const std::string& label) {
    std::vector<unsigned char> got(bytes);
    hip_check(hipMemcpyAsync(got.data(),device,bytes,hipMemcpyDeviceToHost,stream));
    hip_check(hipStreamSynchronize(stream));
    if(std::memcmp(got.data(),ref,bytes)) {
        if(label=="Exhaustive mul1 low/high/slow") {
            std::ofstream actual_file("build/exhaustive-actual.bin",std::ios::binary),expected_file("build/exhaustive-expected.bin",std::ios::binary);
            actual_file.write(reinterpret_cast<const char*>(got.data()),bytes);
            expected_file.write(static_cast<const char*>(ref),bytes);
        }
        if(label.find("checkpoint")!=std::string::npos) {
            std::ofstream actual_file("build/checkpoint-actual.bin",std::ios::binary),expected_file("build/checkpoint-expected.bin",std::ios::binary);
            actual_file.write(reinterpret_cast<const char*>(got.data()),bytes);
            expected_file.write(static_cast<const char*>(ref),bytes);
        }
        size_t k=0;while(k<bytes && got[k]==static_cast<const unsigned char*>(ref)[k]) ++k;
        throw std::runtime_error(label+" byte="+std::to_string(k));
    }
}
void finite(const std::vector<float>& data) {
    for(float x:data) require(std::isfinite(x),"Nonfinite partial");
}
struct Weights {
    Guarded packed,u,v,bias;
    strix_trellis::DeviceWeights w;
    Weights(const Fixture& f,hipStream_t s,size_t clones=1):packed(clones*f.parts.at("trellis").size()),u(2*f.i),v(2*f.n),bias(f.bias?4*f.n:0),
        w{reinterpret_cast<uint32_t*>(packed.p),reinterpret_cast<uint16_t*>(u.p),reinterpret_cast<uint16_t*>(v.p),f.bias?reinterpret_cast<float*>(bias.p):nullptr,int(f.i),int(f.n),int(f.bits),int(f.cb)} {
        for(size_t c=0;c<clones;++c) copy(packed.p+c*f.parts.at("trellis").size(),f.parts.at("trellis"),s);
        copy(u.p,f.parts.at("suh"),s);copy(v.p,f.parts.at("svh"),s);
        if(f.bias) copy(bias.p,f.parts.at("bias"),s);
        hip_check(hipStreamSynchronize(s));
    }
    void check() {packed.check();u.check();v.check();bias.check();}
};
struct Variant {
    int m,b,decoder; // decoder -1: original WMMA; -2: CURRENT GEMV/LDS entry.
    std::string name() const {
        if(decoder==-1) return "original-M"+std::to_string(m);
        if(decoder==-2) return "current-GEMV-LDS";
        return "M"+std::to_string(m)+"-BK"+std::to_string(b)+(decoder?"-fast":"-slow");
    }
};
void launch(const Variant& v,const strix_trellis::DeviceWeights& w,const _Float16* a,float* p,int rows,int splits,hipStream_t s) {
    if(v.decoder==-1) hip_check(strix_trellis::launch_wmma_reuse_variant(w,a,p,rows,splits,s,v.m==64?-65:16));
    else if(v.decoder==-2) hip_check(strix_trellis::launch_gemv_lds_core(w,a,p,rows,rows,splits,s));
    else hip_check(bk::launch(w,a,p,rows,splits,s,v.m,v.b,static_cast<bk::Decoder>(v.decoder)));
}
std::vector<float> download(const Guarded& p,hipStream_t s) {
    std::vector<float> v(p.bytes/4);
    hip_check(hipMemcpyAsync(v.data(),p.p,p.bytes,hipMemcpyDeviceToHost,s));
    hip_check(hipStreamSynchronize(s));return v;
}
void input(const Fixture& f,const Weights& w,const Guarded& x,Guarded& a,int rows,hipStream_t s) {
    strix_trellis::prefill_test_input_transform<<<(size_t(rows)*(f.i/128)+3)/4,128,0,s>>>(reinterpret_cast<const float*>(x.p),w.w.suh,reinterpret_cast<_Float16*>(a.p),f.i,rows);
    hip_check(hipGetLastError());
}
void output(const Fixture& f,const Weights& w,const Guarded& p,Guarded& y,int rows,int splits,float alpha,float beta,hipStream_t s) {
    strix_trellis::prefill_test_output_transform<<<(size_t(rows)*(f.n/128)+3)/4,128,0,s>>>(reinterpret_cast<const float*>(p.p),w.w.svh,w.w.bias,reinterpret_cast<float*>(y.p),f.n,rows,splits,alpha,beta);
    hip_check(hipGetLastError());
}

__global__ void exhaustive(uint16_t* out) {
    unsigned state=blockIdx.x*blockDim.x+threadIdx.x;
    if(state>=65536) return;
    uint32_t pair=__builtin_bit_cast(uint32_t,strix_trellis::optimized::mul1_half2(state,65535-state));
    out[3*state]=uint16_t(pair);
    out[3*state+1]=uint16_t(pair>>16);
    out[3*state+2]=__builtin_bit_cast(uint16_t,_Float16(strix_trellis::codebook_value(state,2)));
}
void decode_gate(const std::string& dir,hipStream_t s) {
    auto lut=read_file<uint16_t>(dir+"/mul1-lut.bin");require(lut.size()==65536,"Invalid LUT");
    std::vector<uint16_t> expected(3*65536);
    for(size_t j=0;j<65536;++j) {expected[3*j]=lut[j];expected[3*j+1]=lut[65535-j];expected[3*j+2]=lut[j];}
    Guarded d(expected.size()*2);exhaustive<<<256,256,0,s>>>(reinterpret_cast<uint16_t*>(d.p));hip_check(hipGetLastError());
    exact(d.p,expected.data(),d.bytes,s,"Exhaustive mul1 low/high/slow");d.check();
    std::cout<<"GATE exhaustive_states=65536 low_high_slow=PASS\n";
}

// Re-exercise the documented 512 exact impulses and one mixed-sign case in
// every selected sub and m16. Both operands replicate in the two semiwaves.
__global__ void native_probe(float* out,int c,int selected_sub,int selected_m) {
    using H16=_Float16 __attribute__((ext_vector_type(16)));
    using F8=float __attribute__((ext_vector_type(8)));
    int test=blockIdx.x,lane=threadIdx.x&31,wave=threadIdx.x/32,col=lane%16;
    F8 acc[4]={};
    for(int sub=0;sub<c;++sub) for(int m=0;m<4;++m) {
        H16 a={},b={};
        if(sub==selected_sub && m==selected_m) for(int k=0;k<16;++k) {
            int ar=col*16+k,bc=k*16+col;
            a[k]=test<256 ? _Float16(ar==test?2:0) : test<512 ? _Float16(col==k) : _Float16((ar*7%13)-6);
            b[k]=test<256 ? _Float16(k==col) : test<512 ? _Float16(bc==test-256?-3:0) : _Float16((bc*11%17)-8);
        }
        acc[m]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b,acc[m]);
    }
    for(int m=0;m<4;++m) for(int j=0;j<8;++j)
        out[(((size_t(test)*4+wave)*4+m)*16+2*j+lane/16)*16+col]=acc[m][j];
}
void native_gate(const std::string& dir,hipStream_t s) {
    Guarded d(513*4*4*256*4);
    double maxabs=0,maxratio=0;
    // The reference is an analytic expression for impulses. Mixed-sign integer
    // reference/sumabs are only 256 dots of length16, prepared once below by
    // closed-form tables saved by --prepare (no CPU matmul during the turn).
    auto oracle=read_file<double>(dir+"/native.bin");require(oracle.size()==512,"Bad native oracle");
    for(int c:{1,2,4}) for(int sub=0;sub<c;++sub) for(int selected=0;selected<4;++selected) {
        native_probe<<<513,128,0,s>>>(reinterpret_cast<float*>(d.p),c,sub,selected);hip_check(hipGetLastError());
        auto got=download(d,s);
        for(int test=0;test<513;++test) for(int wave=0;wave<4;++wave) for(int m=0;m<4;++m) for(int r=0;r<16;++r) for(int n=0;n<16;++n) {
            double expected=0,sumabs=0;
            if(m==selected) {
                if(test<256) expected=(r*16+n==test)?2:0;
                else if(test<512) expected=(r*16+n==test-256)?-3:0;
                else {expected=oracle[r*16+n];sumabs=oracle[256+r*16+n];}
            }
            double actual=got[(((size_t(test)*4+wave)*4+m)*16+r)*16+n];
            double error=std::abs(actual-expected);require(std::isfinite(actual),"Native nonfinite");
            if(test<512 || m!=selected) require(error==0,"Native impulse/map mismatch");
            else require(error<=16*std::ldexp(1.0,-24)*sumabs,"Native mixed envelope");
            maxabs=std::max(maxabs,error);if(sumabs) maxratio=std::max(maxratio,error/sumabs);
        }
        d.check();
    }
    std::cout<<"GATE native_512_impulses=PASS sub_m16_combinations=28 all_waves=4 maxabs="<<maxabs<<" max_error_sumabs="<<maxratio<<"\n";
}
void prepare_native(const std::string& dir) {
    // This branch is invoked outside the GPU turn and makes no HIP calls.
    std::vector<double> data(512);
    for(int r=0;r<16;++r) for(int n=0;n<16;++n) for(int k=0;k<16;++k) {
        int p=(((r*16+k)*7%13)-6)*(((k*16+n)*11%17)-8);
        data[r*16+n]+=p;data[256+r*16+n]+=std::abs(p);
    }
    std::ofstream f(dir+"/native.bin",std::ios::binary);f.write(reinterpret_cast<char*>(data.data()),data.size()*8);
    require(bool(f),"Cannot write native CPU oracle");
}

void negative_gate(const Fixture& f,const Weights& w,const Guarded& a,hipStream_t s) {
    Guarded p(32ull*f.rows*f.n*4);
    auto reject=[&](const strix_trellis::DeviceWeights& weights,const _Float16* x,float* y,int rows,int splits,hipStream_t stream,int m,int b,int decoder) {
        require(bk::launch(weights,x,y,rows,splits,stream,m,b,static_cast<bk::Decoder>(decoder))!=hipSuccess,"Invalid launch accepted");
        std::unique_ptr<const bk::Prevalidated> t;
        require(bk::prevalidate(weights,x,y,rows,splits,stream,m,b,static_cast<bk::Decoder>(decoder),false,t)!=hipSuccess && !t,"Invalid token accepted");
        hip_check(hipGetLastError());
    };
    auto* x=reinterpret_cast<_Float16*>(a.p);auto* y=reinterpret_cast<float*>(p.p);
    for(int m:{0,32,128,-65}) reject(w.w,x,y,f.rows,1,s,m,16,0);
    for(int b:{0,8,48,128}) reject(w.w,x,y,f.rows,1,s,16,b,0);
    for(int d:{-1,2,17}) reject(w.w,x,y,f.rows,1,s,16,16,d);
    for(int rows:{0,-1,1025}) reject(w.w,x,y,rows,1,s,16,16,0);
    for(int splits:{0,-1,33}) reject(w.w,x,y,f.rows,splits,s,16,16,0);
    reject(w.w,x,y,f.rows,1,nullptr,16,16,0);
    reject(w.w,nullptr,y,f.rows,1,s,16,16,0);
    reject(w.w,reinterpret_cast<_Float16*>(a.p+1),y,f.rows,1,s,16,16,0);
    reject(w.w,x,reinterpret_cast<float*>(p.p+2),f.rows,1,s,16,16,0);
    reject(w.w,x,reinterpret_cast<float*>(a.p),f.rows,1,s,16,16,0);
    auto bad=w.w;bad.bits=4;reject(bad,x,y,f.rows,1,s,16,16,0);
    bad=w.w;bad.codebook=1;reject(bad,x,y,f.rows,1,s,16,16,0);
    bad=w.w;bad.input=127;reject(bad,x,y,f.rows,1,s,16,16,0);
    bad=w.w;bad.output=127;reject(bad,x,y,f.rows,1,s,16,16,0);
    bad=w.w;bad.trellis=reinterpret_cast<uint32_t*>(const_cast<unsigned char*>(f.parts.at("trellis").data()));
    reject(bad,x,y,f.rows,1,s,16,16,0);
    reject(w.w,reinterpret_cast<const _Float16*>(f.parts.at("a").data()),y,f.rows,1,s,16,16,0);
    Device managed(0);hip_check(hipMallocManaged(&managed.p,a.bytes));
    reject(w.w,static_cast<const _Float16*>(managed.p),y,f.rows,1,s,16,16,0);
    auto sentinel=std::vector<unsigned char>(p.bytes,0xa5);exact(p.p,sentinel.data(),p.bytes,s,"Rejected launch changed output");p.check();
    // Positive two-byte alignment contract (no packed activation loads).
    Guarded offset_a(a.bytes+2);
    hip_check(hipMemcpyAsync(offset_a.p+2,a.p,a.bytes,hipMemcpyDeviceToDevice,s));
    hip_check(bk::launch(w.w,x,y,f.rows,3,s,16,64,bk::Decoder::fast));
    std::vector<float> aligned(3ull*f.rows*f.n);
    hip_check(hipMemcpyAsync(aligned.data(),y,aligned.size()*4,hipMemcpyDeviceToHost,s));hip_check(hipStreamSynchronize(s));
    hip_check(bk::launch(w.w,reinterpret_cast<const _Float16*>(offset_a.p+2),y,f.rows,3,s,16,64,bk::Decoder::fast));
    exact(y,aligned.data(),aligned.size()*4,s,"Two-byte aligned A contract");offset_a.check();p.check();
    std::cout<<"GATE negative_arguments=PASS two_byte_A=PASS\n";
}
void checkpoints(const Fixture& f,const Weights& w,Guarded& a,hipStream_t s) {
    const auto& cpu_a=f.parts.at("a");const auto& cpu_q=f.parts.at("q");
    size_t count=0;
    for(int m:{16,64}) for(int block:{16,32,64}) for(int decoder:{0,1}) {
        int c=block/16;Guarded ad(c*m*16*2),qd(c*4*256*2);
        // Include starts that are not C-aligned and lengths0/1/2/3/4.
        for(int base=0;base<int(f.i/16);++base) for(int valid=0;valid<=std::min(c,int(f.i/16)-base);++valid)
        for(int fr:{0,int(f.rows)-1}) for(int fn:{0,int(f.n/16)-1}) {
            hip_check(bk::checkpoint(w.w,reinterpret_cast<_Float16*>(a.p),reinterpret_cast<uint16_t*>(ad.p),reinterpret_cast<uint16_t*>(qd.p),f.rows,s,m,block,static_cast<bk::Decoder>(decoder),fr,fn,base,valid));
            std::vector<unsigned char> ar(ad.bytes),qr(qd.bytes);
            for(int sub=0;sub<c;++sub) {
                for(int row=0;row<m;++row) for(int k=0;k<16;++k) if(sub<valid && fr+row<int(f.rows))
                    std::memcpy(ar.data()+((sub*m+row)*16+k)*2,cpu_a.data()+(size_t(fr+row)*f.i+size_t(base+sub)*16+k)*2,2);
                for(int wave=0;wave<4;++wave) for(int r=0;r<16;++r) for(int n=0;n<16;++n) if(sub<valid && fn+wave<int(f.n/16))
                    std::memcpy(qr.data()+(((sub*4+wave)*16+r)*16+n)*2,cpu_q.data()+(size_t((base+sub)*16+r)*f.n+(fn+wave)*16+n)*2,2);
            }
            std::string id=" M="+std::to_string(m)+" BK="+std::to_string(block)+" decoder="+std::to_string(decoder)+" base="+std::to_string(base)+" valid="+std::to_string(valid)+" row="+std::to_string(fr)+" n16="+std::to_string(fn);
            exact(ad.p,ar.data(),ar.size(),s,"A sub/row/padding checkpoint"+id);
            exact(qd.p,qr.data(),qr.size(),s,"Logical Q sub checkpoint"+id);ad.check();qd.check();++count;
        }
    }
    std::cout<<"GATE fixture_Q_A_checkpoints="<<count<<" exact=PASS\n";
}
struct Oracle {
    std::vector<unsigned char> data;
    size_t matrix,impulse_bytes;
    explicit Oracle(const std::string& path,const Fixture& f):data(read_file<unsigned char>(path)),matrix(size_t(f.rows)*f.n),impulse_bytes(size_t(f.rows)*f.i*2) {
        require(data.size()>=20 && !std::memcmp(data.data(),"STBK0001",8),"Bad partial oracle magic");
        require(le(data.data()+8,4)==f.i && le(data.data()+12,4)==f.n && le(data.data()+16,4)==f.rows,"Bad partial oracle shape");
        require(data.size()==20+impulse_bytes+2*56*matrix*16,"Bad partial oracle length");
    }
    void compare_partial(const std::vector<float>& got,const Fixture& f,int rows,int splits,int kind,bool numerical) const {
        int preceding=0;for(int n:split_list) {if(n==splits) break;preceding+=n;}
        size_t offset=20+impulse_bytes+size_t(kind*56+preceding)*matrix*16;
        double worst=0,ratio=0;
        for(int split=0;split<splits;++split) for(size_t j=0;j<size_t(rows)*f.n;++j) {
            double ref=0,sumabs=0;
            std::memcpy(&ref,data.data()+offset+size_t(split)*matrix*16+j*8,8);
            std::memcpy(&sumabs,data.data()+offset+size_t(split)*matrix*16+matrix*8+j*8,8);
            double actual=got[size_t(split)*rows*f.n+j],error=std::abs(actual-ref);
            require(std::isfinite(actual),"Nonfinite partial oracle comparison");
            int length=(int64_t(f.i/16)*(split+1)/splits-int64_t(f.i/16)*split/splits)*16;
            // The empirical native mixed-sign envelope is documented for small
            // normal cases, not an IEEE guarantee for arbitrary underflow.
            // Numerical range/subnormal fixtures use the unchanged full-layer
            // fixture gate, exact A, and exact staging/decoder partial parity.
            // Still record their independent partial errors without inventing
            // a new absolute tolerance. Raw impulses have normal operands.
            double tolerance=16*std::ldexp(1.0,-24)*sumabs;
            if(kind || !numerical) require(error<=tolerance,"Independent partial oracle exceeded native envelope");
            if(!length) require(actual==0 && !std::signbit(actual),"Empty split is not +0");
            worst=std::max(worst,error);if(sumabs) ratio=std::max(ratio,error/sumabs);
        }
        max_partial_error=std::max(max_partial_error,worst);max_partial_ratio=std::max(max_partial_ratio,ratio);
    }
};

void validate_fixture(const std::string& path,const std::string& dir,hipStream_t s) {
    Fixture f(path);require(f.cb==2 && (f.bits==5 || f.bits==6),"Unsupported fixture");
    const bool small=size_t(f.i)*f.n<=65536;
    const bool numerical=std::filesystem::path(path).filename().string().find("numerical-")==0;
    std::cout<<"VALIDATE path="<<path<<" I="<<f.i<<" N="<<f.n<<" rows="<<f.rows<<" independent_partial="<<small<<" numerical_fixture="<<numerical<<" native_partial_envelope="<<!numerical<<"\n";
    Weights w(f,s);Guarded x(size_t(f.rows)*f.i*4),a(size_t(f.rows)*f.i*2);
    copy(x.p,f.parts.at("x"),s);input(f,w,x,a,f.rows,s);
    exact(a.p,f.parts.at("a").data(),a.bytes,s,"Full input RN16/H/RN16");
    std::unique_ptr<Oracle> oracle;
    if(small) {
        oracle.reset(new Oracle(dir+"/"+std::filesystem::path(path).filename().string()+".oracle",f));
        negative_gate(f,w,a,s);checkpoints(f,w,a,s);
    }
    // Expanded diagnostics are host-only. No expanded Q ever used for timing.
    for(const char* key:{"q","core","scaled","inh"}) f.parts.erase(key);
    for(int kind=0;kind<(small?2:1);++kind) {
        if(kind) hip_check(hipMemcpyAsync(a.p,oracle->data.data()+20,a.bytes,hipMemcpyHostToDevice,s));
        for(int rows:row_list) if(rows<=int(f.rows)) for(int splits:split_list) {
            size_t pbytes=size_t(splits)*rows*f.n*4;
            size_t gpu_bound=f.parts.at("trellis").size()+2ull*(f.i+f.n)+4ull*f.bias*f.n+x.bytes+a.bytes+pbytes+size_t(rows)*f.n*4+16*guard;
            if(gpu_bound>=limit) {++skipped;std::cout<<"SKIP rows="<<rows<<" splits="<<splits<<" gpu_bound="<<gpu_bound<<" reason=one_GiB\n";continue;}
            Guarded p(pbytes),y(size_t(rows)*f.n*4);
            std::vector<float> ref=f.floats("ref"),old=f.floats("old");ref.resize(size_t(rows)*f.n);old.resize(ref.size());
            for(int m:{16,64}) {
                std::vector<float> baseline;
                for(int decoder:{0,1}) for(int block:{16,32,64}) {
                    Variant v{m,block,decoder};
                    launch(v,w.w,reinterpret_cast<_Float16*>(a.p),reinterpret_cast<float*>(p.p),rows,splits,s);
                    auto got=download(p,s);finite(got);
                    if(block==16 && decoder==0) baseline=got;
                    else require(got.size()==baseline.size() && !std::memcmp(got.data(),baseline.data(),pbytes),"Staging/decoder partial bits differ "+v.name());
                    if(oracle) oracle->compare_partial(got,f,rows,splits,kind,numerical);
                    // New token must replay the checked entry bit for bit.
                    std::unique_ptr<const bk::Prevalidated> token;
                    hip_check(bk::prevalidate(w.w,reinterpret_cast<_Float16*>(a.p),reinterpret_cast<float*>(p.p),
                        rows,splits,s,m,block,static_cast<bk::Decoder>(decoder),false,token));
                    hip_check(token->submit());
                    exact(p.p,got.data(),pbytes,s,"Nondeterministic partial "+v.name());
                    partial_values+=got.size();
                    if(!kind) for(auto ab:std::array<std::pair<float,float>,3>{{{1,0},{-0.75f,0.25f},{0,-0.5f}}}) {
                        hip_check(hipMemcpyAsync(y.p,old.data(),y.bytes,hipMemcpyHostToDevice,s));
                        output(f,w,p,y,rows,splits,ab.first,ab.second,s);
                        auto actual=download(y,s),expected=ref;
                        for(size_t j=0;j<ref.size();++j) {volatile float av=ab.first*ref[j],bv=ab.second*old[j];expected[j]=av+bv;}
                        // Existing fixture tolerances, unchanged.
                        compare("LAYER "+v.name()+" M="+std::to_string(rows)+" S="+std::to_string(splits)+" alpha="+std::to_string(ab.first)+" beta="+std::to_string(ab.second),actual,expected);
                    }
                    p.check();y.check();++cases;
                }
                // Original WMMA differs in M16 A padding/Q XOR; independent
                // layer references, not an assumed identical layout control.
                if(!kind) {
                    launch({m,16,-1},w.w,reinterpret_cast<_Float16*>(a.p),reinterpret_cast<float*>(p.p),rows,splits,s);
                    output(f,w,p,y,rows,splits,1,0,s);compare("LAYER original M="+std::to_string(rows),download(y,s),ref);p.check();y.check();
                }
            }
            if(rows<=8 && !kind) {
                launch({16,0,-2},w.w,reinterpret_cast<_Float16*>(a.p),reinterpret_cast<float*>(p.p),rows,splits,s);
                output(f,w,p,y,rows,splits,1,0,s);compare("LAYER current GEMV/LDS",download(y,s),ref);p.check();y.check();
            }
        }
    }
    x.check();a.check();
    std::cout<<"VALIDATE_DONE path="<<path<<" cases="<<cases<<" complete_partial_values="<<partial_values<<" skipped="<<skipped<<" max_partial_abs="<<max_partial_error<<" max_partial_error_sumabs="<<max_partial_ratio<<"\n";
}


// Full operator storage/lifetime preflight, outside all timing. Core ranges
// and device/stream were checked for each token by the original validator.
void preflight_operator(const Fixture& f,const Weights& w,Guarded& x,Guarded& a,
    Guarded& p,Guarded& y,int rows,int splits,hipStream_t s) {
    int device=-1;hip_check(hipGetDevice(&device));
    for(auto pair:std::vector<std::pair<const void*,size_t>>{
        {x.p,x.bytes},{a.p,a.bytes},{p.p,p.bytes},{y.p,y.bytes},
        {w.w.suh,size_t(f.i)*2},{w.w.svh,size_t(f.n)*2}}) {
        hipPointerAttribute_t attr{};hip_check(hipPointerGetAttributes(&attr,pair.first));
        require(attr.type==hipMemoryTypeDevice && !attr.isManaged && attr.device==device,"Operator storage device");
        void* base=nullptr;size_t bytes=0;
        hip_check(hipMemGetAddressRange(&base,&bytes,const_cast<void*>(pair.first)));
        size_t offset=reinterpret_cast<uintptr_t>(pair.first)-reinterpret_cast<uintptr_t>(base);
        require(offset<=bytes && pair.second<=bytes-offset,"Operator storage range");
    }
    if(f.bias) {
        hipPointerAttribute_t attr{};hip_check(hipPointerGetAttributes(&attr,w.w.bias));
        require(attr.type==hipMemoryTypeDevice && !attr.isManaged && attr.device==device,"Bias device");
        void* base=nullptr;size_t bytes=0;hip_check(hipMemGetAddressRange(&base,&bytes,const_cast<float*>(w.w.bias)));
        size_t offset=reinterpret_cast<uintptr_t>(w.w.bias)-reinterpret_cast<uintptr_t>(base);
        require(offset<=bytes && size_t(f.n)*4<=bytes-offset,"Bias range");
    }
    require(x.bytes==size_t(rows)*f.i*4 && a.bytes==size_t(rows)*f.i*2 &&
        y.bytes==size_t(rows)*f.n*4 && p.bytes==size_t(splits)*rows*f.n*4,"Operator dimensions");
    hip_check(hipStreamSynchronize(s));x.check();a.check();p.check();y.check();
    std::cout<<"GATE operator_preflight=PASS rows="<<rows<<" splits="<<splits<<"\n";
}

// Alternate all candidates against the same baseline, sweep by sweep, so
// temporal drift is not confounded with a fixed baseline-first ordering.
void performance(const std::string& path,hipStream_t s,bool confirm,int search_split) {
    if(expired()) {++skipped;std::cout<<"SKIP_TIME path="<<path<<" budget_seconds=145\n";return;}
    Fixture f(path);require(f.cb==2 && (f.bits==5 || f.bits==6),"Unsupported performance fixture");
    const size_t packed=f.parts.at("trellis").size(),clones=(cold_bytes+packed-1)/packed;
    std::cout<<"PERF_FIXTURE path="<<path<<" bytes="<<packed<<" clones="<<clones<<" independent_compressed_bytes="<<clones*packed<<"\n";
    for(const char* key:{"q","core","scaled","inh"}) f.parts.erase(key);
    Weights w(f,s,clones);
    for(int rows:{64,128,256}) if(rows<=int(f.rows)) {
        Guarded x(size_t(rows)*f.i*4),a(size_t(rows)*f.i*2),y(size_t(rows)*f.n*4);
        copy(x.p,f.parts.at("x"),s,x.bytes);input(f,w,x,a,rows,s);
        exact(a.p,f.parts.at("a").data(),a.bytes,s,"Performance A checkpoint");
        auto ref=f.floats("ref");ref.resize(size_t(rows)*f.n);
        // Same-split comparison first. Bounded search includes1/4/16; no
        // best-split claim until an independent confirmation campaign.
        for(int splits: {4}) {
            Variant selected{};bool found=false;
            if(confirm) {
                std::ifstream selections("build/selection.tsv");require(bool(selections),"Missing confirmation selection");
                std::string selected_path;int rm=0,rs=0,tm=0,b=0,d=0;
                while(selections>>selected_path>>rm>>rs>>tm>>b>>d)
                    if(selected_path==path && rm==rows && rs==splits) {selected={tm,b,d};found=true;}
                if(!found) continue;
            } else if(splits!=search_split) continue;
            if(expired()) {++skipped;std::cout<<"SKIP_TIME path="<<path<<" rows="<<rows<<" splits="<<splits<<" budget_seconds=145\n";return;}
            size_t pbytes=size_t(splits)*rows*f.n*4;
            size_t bound=clones*packed+2ull*(f.i+f.n)+4ull*f.bias*f.n+x.bytes+a.bytes+y.bytes+pbytes+16*guard;
            if(bound>=limit) {++skipped;std::cout<<"SKIP_PERF rows="<<rows<<" splits="<<splits<<" gpu_bound="<<bound<<"\n";continue;}
            Guarded p(pbytes);
            std::vector<Variant> variants;
            variants.push_back({rows<=8?16:64,16,rows<=8?-2:-1});
            if(confirm) {
                // Preserve an identically staged BK16 control for the selected
                // decoder, in addition to the original/current baseline.
                variants.push_back({selected.m,16,selected.decoder});
                if(selected.b!=16) variants.push_back(selected);
            } else {
                if(rows<=8) variants.push_back({16,16,-1});
                for(int m: {64}) {
                    if(rows<=8 && m==64) continue;
                    for(int decoder:{0,1}) for(int block:{16,32,64}) variants.push_back({m,block,decoder});
                }
            }
            // Tokens are immutable and bound to one clone/tuple. Every buffer
            // remains owned by this scope until all checks/timings finish.
            std::vector<std::vector<std::unique_ptr<const bk::Prevalidated>>> tokens(variants.size());
            for(size_t vi=0;vi<variants.size();++vi) for(size_t c=0;c<clones;++c) {
                auto weights=w.w;
                weights.trellis=reinterpret_cast<uint32_t*>(w.packed.p+c*packed);
                const auto& v=variants[vi];
                std::unique_ptr<const bk::Prevalidated> t;
                hip_check(bk::prevalidate(weights,reinterpret_cast<_Float16*>(a.p),reinterpret_cast<float*>(p.p),
                    rows,splits,s,v.m,v.b,static_cast<bk::Decoder>(std::max(0,v.decoder)),v.decoder==-1,t));
                tokens[vi].push_back(std::move(t));
            }
            preflight_operator(f,w,x,a,p,y,rows,splits,s);
            w.check();
            auto submit=[&](const Variant& v,size_t clone,bool complete,bool checked=false) {
                if(complete) input(f,w,x,a,rows,s);
                size_t vi=size_t(&v-variants.data());
                if(checked) {
                    auto weights=w.w;weights.trellis=reinterpret_cast<uint32_t*>(w.packed.p+clone*packed);
                    launch(v,weights,reinterpret_cast<_Float16*>(a.p),reinterpret_cast<float*>(p.p),rows,splits,s);
                } else hip_check(tokens[vi][clone]->submit());
                if(complete) output(f,w,p,y,rows,splits,1,0,s);
            };
            auto check=[&](const Variant& v,bool all) {
                for(size_t clone=0;clone<(all?clones:1);++clone) {
                    submit(v,clone,true);compare("PERF_CHECK "+v.name(),download(y,s),ref);
                }
                p.check();y.check();a.check();x.check();
                w.check();
            };
            // Diagnostic partial comparisons cover every candidate before
            // timing; no expanded global Q or second GPU partial is retained.
            std::vector<float> staging_ref;
            for(const auto& v:variants) {
                launch(v,w.w,reinterpret_cast<_Float16*>(a.p),reinterpret_cast<float*>(p.p),rows,splits,s);
                auto checked=download(p,s);
                submit(v,0,false);auto got=download(p,s);finite(got);
                require(checked==got && !std::memcmp(checked.data(),got.data(),pbytes),"Checked/token bits differ");
                if(v.decoder==0 && v.b==16) staging_ref=got;
                if(v.decoder>=0) require(staging_ref.size()==got.size() && !std::memcmp(staging_ref.data(),got.data(),pbytes),"Performance partial mismatch");
                check(v,true);
            }
            std::vector<float>().swap(staging_ref);
            for(bool complete:{false,true}) {
                std::vector<std::vector<double>> times(variants.size());
                for(const auto& v:variants) for(size_t c=0;c<clones;++c) submit(v,c,complete);
                hip_check(hipStreamSynchronize(s));
                hipEvent_t start{},stop{};hip_check(hipEventCreate(&start));hip_check(hipEventCreate(&stop));
                std::array<std::vector<float>,2> final_core_reference;
                auto sweep=[&](size_t v,int rep,bool record) {
                    hip_check(hipEventRecord(start,s));
                    for(size_t c=0;c<clones;++c) submit(variants[v],(c+rep)%clones,complete);
                    hip_check(hipEventRecord(stop,s));hip_check(hipEventSynchronize(stop));
                    float ms=0;hip_check(hipEventElapsedTime(&ms,start,stop));
                    if(record) times[v].push_back(ms*1000.0/clones);
                    if(complete) compare("TIMED_OUTPUT "+variants[v].name(),download(y,s),ref);
                    else if(rep==10) {
                        const auto& expected=final_core_reference[v==0?0:1];
                        require(expected.size()*4==pbytes,"Missing timed partial reference");
                        exact(p.p,expected.data(),pbytes,s,"Final timed core partial differs "+variants[v].name());
                    }
                    p.check();y.check();a.check();x.check();
                };
                // Eleven resident rotations per path, baseline alternates on
                // either side of each candidate. Baseline gets11 per pair.
                for(size_t v=1;v<variants.size();++v) {
                    if(expired()) {
                        hip_check(hipEventDestroy(start));hip_check(hipEventDestroy(stop));
                        ++skipped;std::cout<<"SKIP_TIME path="<<path<<" rows="<<rows<<" splits="<<splits<<" remaining_pairs="<<variants.size()-v<<"\n";return;
                    }
                    times[0].clear();times[v].clear();
                    if(!complete) {
                        submit(variants[0],0,false);final_core_reference[0]=download(p,s);finite(final_core_reference[0]);
                        submit(variants[v],0,false);final_core_reference[1]=download(p,s);finite(final_core_reference[1]);
                    }
                    for(int rep=0;rep<11;++rep) {
                        if(rep&1) {sweep(v,rep,true);sweep(0,rep,true);}
                        else {sweep(0,rep,true);sweep(v,rep,true);}
                    }
                    // Revalidate before publishing any timings for this pair.
                    check(variants[0],true);check(variants[v],true);
                    for(size_t idx:{size_t(0),v}) {
                        auto sorted=times[idx];std::sort(sorted.begin(),sorted.end());
                        std::cout<<"RESULT path="<<path<<" mode="<<(complete?"device_operator":"core")<<" C_API=0 launcher=prevalidated candidate="<<variants[idx].name()<<" pair="<<variants[v].name()<<" rows="<<rows<<" splits="<<splits<<" gpu_bound="<<bound
                            <<" cloned_bytes="<<clones*packed<<" workspace_bytes="<<a.bytes+pbytes<<" core_partial_bytes="<<pbytes
                            <<" logical_compressed_reread_bytes="<<packed*size_t((rows+variants[idx].m-1)/variants[idx].m)
                            <<" min_us="<<sorted[0]<<" median_us="<<sorted[5]<<" p90_us="<<sorted[9]<<" useful_TOPS="<<2.0*rows*f.i*f.n/sorted[5]/1e6<<" compressed_GBs="<<packed/sorted[5]/1000<<" samples=";
                        for(double t:times[idx]) std::cout<<t<<",";
                        std::cout<<" correctness=PASS canaries=PASS\n";
                    }
                    // A fixed diagnostic, not a candidate search: exact same
                    // M64/BK32/fast kernel through the checked and token paths.
                    // Event intervals include host feeding gaps in both paths.
                    if(variants[v].b==32 && variants[v].decoder==1) {
                        std::array<std::vector<double>,2> diagnostic;
                        auto run=[&](int checked,int rep) {
                            hip_check(hipEventRecord(start,s));
                            for(size_t c=0;c<clones;++c) submit(variants[v],(c+rep)%clones,complete,checked);
                            hip_check(hipEventRecord(stop,s));hip_check(hipEventSynchronize(stop));
                            float ms=0;hip_check(hipEventElapsedTime(&ms,start,stop));
                            diagnostic[checked].push_back(ms*1000.0/clones);
                            if(complete) compare("DIAGNOSTIC_OUTPUT",download(y,s),ref);
                            else exact(p.p,final_core_reference[1].data(),pbytes,s,"Diagnostic partial");
                            p.check();y.check();a.check();x.check();w.check();
                        };
                        for(int rep=0;rep<11;++rep) {
                            if(rep&1) {run(1,rep);run(0,rep);}
                            else {run(0,rep);run(1,rep);}
                        }
                        check(variants[v],true);
                        for(int checked:{0,1}) {
                            auto sorted=diagnostic[checked];std::sort(sorted.begin(),sorted.end());
                            std::cout<<"DIAGNOSTIC path="<<path<<" mode="<<(complete?"device_operator":"core")
                                <<" candidate="<<variants[v].name()<<" rows="<<rows<<" splits="<<splits
                                <<" launcher="<<(checked?"checked":"prevalidated")<<" min_us="<<sorted[0]
                                <<" median_us="<<sorted[5]<<" p90_us="<<sorted[9]<<" samples=";
                            for(double t:diagnostic[checked]) std::cout<<t<<",";
                            std::cout<<" correctness=PASS canaries=PASS\n";
                        }
                    }
                }
                hip_check(hipEventDestroy(start));hip_check(hipEventDestroy(stop));
            }
        }
    }
}
}
int main(int argc,char** argv) {
    try {
        using namespace experiment;
        std::cout<<std::unitbuf<<std::setprecision(10);
        require(argc>=3,"Usage: wmma_blocked_bench --prepare|--validate|--search|--confirm ORACLE_DIR [FIXTURE...]");
        std::string mode=argv[1],dir=argv[2];
        if(mode=="--prepare") {require(argc==3,"Unexpected prepare arguments");prepare_native(dir);std::cout<<"PASS CPU_ONLY native oracle\n";return 0;}
        require((mode=="--validate" || mode=="--search" || mode=="--confirm") && argc>=4,"Unknown mode or missing fixtures");
        Stream stream;
        hipDeviceProp_t p{};hip_check(hipGetDeviceProperties(&p,0));
        require(std::string(p.gcnArchName).find("gfx1151")==0 && p.warpSize==32,"Requires gfx1151 wave32");
        std::cout<<"HARDWARE name="<<p.name<<" arch="<<p.gcnArchName<<" wave="<<p.warpSize<<" multiprocessors="<<p.multiProcessorCount<<" nominal_gpu_kHz="<<p.clockRate<<" nominal_mem_kHz="<<p.memoryClockRate<<" totalGlobalMem="<<p.totalGlobalMem<<"\n";
        if(mode=="--validate") {decode_gate(dir,stream.s);native_gate(dir,stream.s);}
        campaign_start=std::chrono::steady_clock::now();
        if(mode=="--validate") for(int j=3;j<argc;++j) validate_fixture(argv[j],dir,stream.s);
        else if(mode=="--search") for(int split:{4}) for(int j=3;j<argc;++j) performance(argv[j],stream.s,false,split);
        else for(int j=3;j<argc;++j) performance(argv[j],stream.s,false,4);
        hip_check(hipStreamSynchronize(stream.s));
        std::cout<<"PASS mode="<<mode<<" cases="<<cases<<" skipped="<<skipped<<" partial_values="<<partial_values<<"\n";return 0;
    } catch(const std::exception& e) {std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
}
