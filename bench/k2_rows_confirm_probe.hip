// Private qualification. The eight-field V11 ABI is the only archive boundary.
#include "../src/gemv_k2_rows.hpp"
#include "../src/gemv_lds.hpp"
#include "../src/gemv_k2.hpp"
#define K2_ROWS_HELPERS_ONLY
#define k2_rows_private k2_rows_probe_helpers
#include "../src/gemv_k2_rows.hip"
#undef k2_rows_private
#undef K2_ROWS_HELPERS_ONLY
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>
#include <cstddef>
#include <chrono>
using namespace strix_trellis;
using namespace strix_trellis::k2_rows_private;
static_assert(sizeof(DeviceWeights)==48 && offsetof(DeviceWeights,input)==32 && offsetof(DeviceWeights,codebook)==44);
namespace strix_trellis {
__global__ void input_transform(const float*,const uint16_t*,_Float16*,int,int);
__global__ void output_transform(const float*,const uint16_t*,const float*,float*,int,int,int,float,float);
}
static void ck(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
static void require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
struct Buffer {
    unsigned char *base,*p; size_t size;
    static inline size_t live=0,peak=0;
    explicit Buffer(size_t n):size(n){live+=n+512;peak=std::max(peak,live);require(live<(size_t(1)<<30),"GPU budget");ck(hipMalloc(&base,n+512));p=base+256;ck(hipMemset(base,0xa5,n+512));}
    ~Buffer(){(void)hipFree(base);live-=size+512;}
    k2_rows_private::OwnedRange range() const{return {p,size};}
    void upload(const void* src,size_t n){require(n<=size,"upload size");ck(hipMemcpy(p,src,n,hipMemcpyHostToDevice));}
    std::vector<unsigned char> get(){std::vector<unsigned char> v(size);ck(hipMemcpy(v.data(),p,size,hipMemcpyDeviceToHost));return v;}
    void guards(){unsigned char v[256];for(auto* ptr:{base,p+size}){ck(hipMemcpy(v,ptr,256,hipMemcpyDeviceToHost));for(auto c:v)require(c==0xa5,"outer canary");}}
};
static uint32_t random_word(){static uint32_t s=0x732ad031;s^=s<<13;s^=s>>17;s^=s<<5;return s;}
static double host_half(uint16_t b){_Float16 h;memcpy(&h,&b,2);return double(h);}
static uint16_t book(uint32_t s){uint32_t h=s*0x83dcd12du;int sum=0;for(int j=0;j<4;++j)sum+=(h>>(8*j))&255;_Float16 v=_Float16(((1024+sum)*1774-1329*2048)/262144.0);uint16_t b;memcpy(&b,&v,2);return b;}
static uint16_t state(const uint32_t* p,int pos){uint16_t s=0;for(int j=0;j<16;++j){int b=(2*(pos+1)-16+j+512)%512;s=uint16_t((s<<1)|((p[b/32]>>(31-b%32))&1));}return s;}
static double weight(const uint32_t* p,int I,int N,int i,int col){(void)I;int r=i&15,c=col&15;int pos=((c&7)<<5)|((r&6)<<2)|(r&1)|((r&8)>>2)|((c&8)>>1);return host_half(book(state(p+(size_t(i/16)*(N/16)+col/16)*16,pos)));}
__global__ void books_probe(uint32_t* out){uint32_t s=blockIdx.x*256+threadIdx.x,v[8],h[8];for(int j=0;j<8;++j)v[j]=((j&1)?65535-s:s)|(uint32_t(0xa55a+j)<<16);k2_rows_probe_helpers::staged(v,h);for(int j=0;j<8;++j)out[s*12+j]=h[j];for(int j=0;j<4;++j)out[s*12+8+j]=__builtin_bit_cast(uint32_t,k2_rows_probe_helpers::sad_pair(h[2*j],h[2*j+1]));}
__global__ void reader_probe(const uint32_t* p,uint16_t* out){int l=threadIdx.x,index=(498+16*l)/32-16;auto* b=reinterpret_cast<const unsigned char*>(p+blockIdx.x*16);k2_rows_probe_helpers::RawPacked r{k2_rows_probe_helpers::word_load(b,index<0?index+16:index),k2_rows_probe_helpers::word_load(b,index+1>=16?index+1-16:index+1)};uint32_t s[8];k2_rows_probe_helpers::states(r,s);for(int j=0;j<8;++j)out[blockIdx.x*256+l*8+j]=uint16_t(s[j]);}
static void exhaustive(){Buffer h(65536*12*4);books_probe<<<256,256>>>((uint32_t*)h.p);ck(hipGetLastError());auto raw=h.get();auto* v=(uint32_t*)raw.data();for(uint32_t s=0;s<65536;++s){for(int j=0;j<8;++j)require(v[s*12+j]==((j&1)?65535-s:s)*0x83dcd12du,"hash contaminated");for(int j=0;j<4;++j)require(v[s*12+8+j]==(uint32_t(book(s))|(uint32_t(book(65535-s))<<16)),"paired half bits");}h.guards();std::vector<uint32_t> p(769*16);for(int b=0;b<512;++b)p[b*16+b/32]=1u<<(b%32);for(size_t j=512*16;j<p.size();++j)p[j]=random_word();Buffer dp(p.size()*4),dy(769*256*2);dp.upload(p.data(),dp.size);reader_probe<<<769,32>>>((uint32_t*)dp.p,(uint16_t*)dy.p);ck(hipGetLastError());auto y=dy.get();auto* s=(uint16_t*)y.data();for(int t=0;t<769;++t)for(int j=0;j<256;++j)require(s[t*256+j]==state(p.data()+t*16,j),"reader basis/random");dp.guards();dy.guards();puts("EXHAUSTIVE PASS hashes=524288 halves=524288 reader_states=196864");}
static void r1_identity() {
    constexpr int I=80,N=48,S=7;
    std::vector<uint32_t> b(size_t(I)*N/16);for(auto& v:b)v=random_word();
    std::vector<_Float16> a(I);for(int i=0;i<I;++i)a[i]=_Float16((i%13-6)/32.);
    Buffer B(b.size()*4),A(I*2),P(S*N*4);B.upload(b.data(),B.size);A.upload(a.data(),A.size);
    DeviceWeights w{(uint32_t*)B.p,nullptr,nullptr,nullptr,I,N,2,2};
    ck(launch_gemv_lds_variant(w,(_Float16*)A.p,(float*)P.p,1,1,S,nullptr,0));auto ref=P.get();
    ck(launch_gemv_k2_core(w,(_Float16*)A.p,(float*)P.p,1,S,nullptr));require(P.get()==ref,"unchanged R1 V1 identity only");
    B.guards();A.guards();P.guards();puts("R1_IDENTITY PASS no timing");
}
static size_t small_cases=0, original_cases=0, added_cases=0;
static unsigned chunks_seen[3][3]={},tails_seen[3][3]={};
static double worst_partial_ratio=0,worst_nrms=0,worst_y_relative=0;
static void partial_check(float got,double exact,double absolute) {
    double ratio=std::abs(double(got)-exact)/std::max(1.,absolute);
    worst_partial_ratio=std::max(worst_partial_ratio,ratio);
    require(std::isfinite(got)&&ratio<=2e-6,"all FP64 partials bound");
}
static void small() {
    const uint16_t special[]={0,0x8000,1,0x8001,0x3ff,0x83ff,0x7bff,0xfbff,0x3800,0xb400,0x3555,0x0400,0x8400};
    for(int I:{16,80,128,176,272,112,240})for(int N:{16,32,48,80})for(int R:{2,4,8}) {
        int T=R+3,K=I/16;
        std::vector<uint32_t> b(size_t(I)*N/16);for(auto& v:b)v=random_word();
        std::vector<uint16_t> a(size_t(T)*I+2);
        for(size_t j=0;j<a.size();++j)a[j]=special[(j*7+j/I*3)%13];
        Buffer B(b.size()*4),A(a.size()*2);B.upload(b.data(),B.size);A.upload(a.data(),A.size);
        auto originalB=B.get(),originalA=A.get();
        DeviceWeights w{(uint32_t*)B.p,nullptr,nullptr,nullptr,I,N,2,2};
        // Independent bit reader + exact half dyadic arithmetic, all columns/parent rows.
        size_t stride=size_t(K+1)*N;
        std::vector<double> prefix(size_t(T)*stride),absolute(prefix.size());
        for(int t=0;t<T;++t)for(int k=0;k<K;++k)for(int c=0;c<N;++c) {
            size_t before=size_t(t)*stride+size_t(k)*N+c,after=before+N;
            double sum=prefix[before],ab=absolute[before];
            for(int i=k*16;i<(k+1)*16;++i){double v=host_half(a[size_t(t)*I+i])*weight(b.data(),I,N,i,c);sum+=v;ab+=std::abs(v);}
            prefix[after]=sum;absolute[after]=ab;
        }
        for(int S=1;S<=32;++S)for(int layout=0;layout<3;++layout) {
            int total=layout==0?R:T,offset=layout==2?1:0;
            Buffer P(size_t(S)*total*N*4+64);
            auto* ap=(_Float16*)A.p+size_t(offset)*I;auto* pp=(float*)P.p+size_t(offset)*N;
            Plan plan;require(make_diagnostic(plan,w,ap,pp,R,total,offset,S,B.range(),A.range(),P.range()),"valid diagnostic factory");
            Plan invalid;
            require(!make_diagnostic(invalid,w,(_Float16*)A.p+I,(float*)P.p+N,R,R,1,S,B.range(),A.range(),P.range()),"reject offset1 total_rows Rows");
            require(!make_diagnostic(invalid,w,ap,pp,R,total,-1,S,B.range(),A.range(),P.range()),"reject negative offset");
            require(!make_diagnostic(invalid,w,ap,pp,R,total,offset,S,{B.p,B.size-1},A.range(),P.range()),"reject packed capacity");
            require(!make_diagnostic(invalid,w,ap,pp,R,total,offset,S,B.range(),A.range(),A.range()),"reject alias");
            require(!make_diagnostic(invalid,w,ap+1,pp,R,total,offset,S,B.range(),A.range(),P.range()),"reject offset2");
            std::vector<unsigned char> control;
            for(int m=0;m<3;++m) {
                ck(hipMemset(P.p,0xa5,P.size));
                if(m==0)ck(launch_gemv_lds_core(w,ap,pp,R,total,S,nullptr));else submit(plan,m==1?Mode::Global:Mode::LDS8,nullptr);
                ck(hipGetLastError());auto raw=P.get();
                for(size_t j=0;j<raw.size();++j) {
                    size_t element=j/4,row=(element/N)%size_t(total);
                    bool written=element<size_t(S)*total*N && row>=size_t(offset)&&row<size_t(offset+R);
                    if(!written)require(raw[j]==0xa5,"every padding byte canary");
                }
                if(m==0)control=raw;else require(raw==control,"all small partial bytes V11");
                for(int s=0;s<S;++s)for(int r=0;r<R;++r)for(int c=0;c<N;++c) {
                    int begin=K*s/S,end=K*(s+1)/S;size_t base=size_t(r+offset)*stride+c;
                    double exact=prefix[base+size_t(end)*N]-prefix[base+size_t(begin)*N];
                    double ab=absolute[base+size_t(end)*N]-absolute[base+size_t(begin)*N];
                    float got;memcpy(&got,raw.data()+((size_t(s)*total+r+offset)*N+c)*4,4);partial_check(got,exact,ab);
                    if(begin==end){uint32_t bits;memcpy(&bits,&got,4);require(bits==0,"empty split positive zero");}
                }
                P.guards();
            }
            // Count only launches whose every output and canary passed above.
            int ri=R==2?0:R==4?1:2;
            for(int m=0;m<3;++m)for(int split=0;split<S;++split) {
                int begin=K*split/S,end=K*(split+1)/S;
                for(int base=begin;base<end;base+=8) {
                    int length=std::min(8,end-base);
                    chunks_seen[m][ri]|=1u<<length;tails_seen[m][ri]|=1u<<(length%4);
                }
            }
            ++small_cases;if(I==112||I==240)++added_cases;else ++original_cases;
        }
        // The dispatcher retains the immutable V11 path for two-byte X alignment.
        Buffer P(size_t(4)*T*N*4);auto* off=(_Float16*)A.p+1;
        ck(launch_gemv_lds_core(w,off,(float*)P.p,R,T,4,nullptr));auto reference=P.get();
        Plan rejected;
        require(!make_performance(rejected,w,off,(float*)P.p,R,T,0,4,B.range(),A.range(),P.range()),"offset2 dispatcher fallback");
        ck(launch_gemv_lds_core(w,off,(float*)P.p,R,T,4,nullptr));require(reference==P.get(),"offset2 V11 fallback bytes");
        require(B.get()==originalB && A.get()==originalA,"small immutable inputs all bytes");
        B.guards();A.guards();P.guards();
        printf("SMALL PASS I=%d N=%d R=%d S=1..32 layouts=3\n",I,N,R);
    }
    require(original_cases==5760 && added_cases==2304 && small_cases==8064,"exact old and added case counts");
    for(int m=0;m<3;++m)for(int ri=0;ri<3;++ri) {
        require(chunks_seen[m][ri]==510 && tails_seen[m][ri]==15,"physical chunks1..8 tails0..3");
        printf("COVERAGE PASS mode=%d R=%d chunk_mask=%u tail_mask=%u original_cases=5760 added_cases=2304\n",m,2<<ri,chunks_seen[m][ri],tails_seen[m][ri]);
    }
    printf("SMALL_ALL PASS cases=%zu modes=3 partial_ratio=%.12g\n",small_cases,worst_partial_ratio);
}
struct Fixture {
    std::map<std::string,std::vector<unsigned char>> v;int I,N;
    explicit Fixture(const char* path) {
        std::ifstream f(path,std::ios::binary);char magic[8];uint32_t ver,count;
        f.read(magic,8);f.read((char*)&ver,4);f.read((char*)&count,4);
        require(f&&memcmp(magic,"STGPU001",8)==0&&ver==1&&count<64,"fixture header");
        for(uint32_t j=0;j<count;++j) {
            char tag[9]={};uint64_t n;f.read(tag,8);f.read((char*)&n,8);require(f&&n<256*1024*1024,"section length");
            // Expanded Q and obsolete R1 p64 never enter the benchmark address space.
            if(std::string(tag)=="q"||std::string(tag)=="p64"||std::string(tag)=="pabs")f.seekg(n,std::ios::cur);
            else {auto& b=v[tag];b.resize(n);f.read((char*)b.data(),n);}
            require(bool(f),"section read");
        }
        auto* meta=(uint32_t*)v.at("meta").data();I=meta[0];N=meta[1];require(meta[2]==2&&meta[3]==2&&meta[5]==8,"fixture contract");
    }
};
struct Oracle {
    std::vector<double> exact,absolute;
    Oracle(const char* path,int I,int N) {
        std::ifstream f(path,std::ios::binary);char magic[8];uint32_t dims[4];f.read(magic,8);f.read((char*)dims,16);
        require(f&&memcmp(magic,"K2RFP641",8)==0&&dims[0]==uint32_t(I)&&dims[1]==uint32_t(N)&&dims[2]==4&&dims[3]==8,"new allRows FP64 header");
        exact.resize(size_t(4)*8*N);absolute.resize(exact.size());
        f.read((char*)exact.data(),exact.size()*8);f.read((char*)absolute.data(),absolute.size()*8);require(bool(f),"allRows FP64 data");
    }
};
static void cpu_y_check(const std::vector<unsigned char>& raw,Fixture& f,int R,float alpha,float beta) {
    auto* cpu=(float*)f.v.at("ref").data();auto* old=(float*)f.v.at("old").data();
    double e2=0,r2=0,maxerr=0,maxref=1;
    for(int j=0;j<R*f.N;++j) {
        float got;memcpy(&got,raw.data()+j*4,4);float ref=cpu[j]*alpha;if(beta!=0)ref+=old[j]*beta;
        require(std::isfinite(got),"finite full operator including beta0 poison");
        double err=double(got)-double(ref);e2+=err*err;r2+=double(ref)*ref;
        maxerr=std::max(maxerr,std::abs(err));maxref=std::max(maxref,std::abs(double(ref)));
    }
    double nrms=std::sqrt(e2/std::max(r2,1e-300));worst_nrms=std::max(worst_nrms,nrms);worst_y_relative=std::max(worst_y_relative,maxerr/maxref);
    require(nrms<=3e-5 && maxerr<=5e-4*maxref,"all CPU Y NRMS and maxabs");
}
static void large(const char* path,const char* reference_path) {
    Fixture f(path);int I=f.I,N=f.N;Oracle oracle(reference_path,I,N);
    constexpr size_t C=26;size_t bytes=size_t(I)*N/4;require(C*bytes>=size_t(256)*1024*1024,"resident packed >=256MiB");
    Buffer B(C*bytes),A(size_t(8)*I*2+4),X(size_t(8)*I*4),SU(I*2),SV(N*2),bias(f.v.at("bias").size()),P(size_t(4)*8*N*4),Y(size_t(8)*N*4);
    for(size_t c=0;c<C;++c)ck(hipMemcpy(B.p+c*bytes,f.v.at("trellis").data(),bytes,hipMemcpyHostToDevice));
    A.upload(f.v.at("a").data(),size_t(8)*I*2);X.upload(f.v.at("x").data(),X.size);SU.upload(f.v.at("suh").data(),SU.size);SV.upload(f.v.at("svh").data(),SV.size);bias.upload(f.v.at("bias").data(),bias.size);
    auto originalA=A.get(),originalX=X.get(),originalSU=SU.get(),originalSV=SV.get(),originalBias=bias.get();
    auto check_inputs=[&]() {
        auto packed=B.get();
        for(size_t c=0;c<C;++c)require(memcmp(packed.data()+c*bytes,f.v.at("trellis").data(),bytes)==0,"immutable packed every byte all clones");
        require(A.get()==originalA&&X.get()==originalX&&SU.get()==originalSU&&SV.get()==originalSV&&bias.get()==originalBias,"immutable operator inputs every byte");
        for(Buffer* b:{&B,&A,&X,&SU,&SV,&bias,&P,&Y})b->guards();
    };
    check_inputs();
    DeviceWeights w{(uint32_t*)B.p,(uint16_t*)SU.p,(uint16_t*)SV.p,bias.size?(float*)bias.p:nullptr,I,N,2,2};
    // Owned allocations of finalization buffers are mutually disjoint and disjoint from core owners.
    std::vector<OwnedRange> owners={B.range(),A.range(),X.range(),SU.range(),SV.range(),bias.range(),P.range(),Y.range()};
    for(size_t i=0;i<owners.size();++i)for(size_t j=0;j<i;++j)require(disjoint(owners[i],owners[j]),"all finalization owned ranges disjoint");
    for(int R:{2,4,8}) {
        // Large parent views use the same fixture rows at an already shifted origin.
        for(int layout=0;layout<3;++layout) {
            int total=layout?R+3:R,offset=layout==2?1:0;
            Buffer parentA(size_t(total)*I*2+4),parentP(size_t(4)*total*N*4+64);
            ck(hipMemcpy(parentA.p+size_t(offset)*I*2,f.v.at("a").data(),size_t(R)*I*2,hipMemcpyHostToDevice));
            auto immutable=parentA.get();auto* av=(_Float16*)parentA.p+size_t(offset)*I;auto* pv=(float*)parentP.p+size_t(offset)*N;
            Plan view;require(make_performance(view,w,av,pv,R,total,offset,4,B.range(),parentA.range(),parentP.range()),"large parent view");
            std::vector<unsigned char> baseline;
            for(int m=0;m<3;++m) {
                ck(hipMemset(parentP.p,0xa5,parentP.size));
                if(m==0)ck(launch_gemv_lds_core(w,av,pv,R,total,4,nullptr));else submit(view,m==1?Mode::Global:Mode::LDS8,nullptr);
                auto raw=parentP.get();if(m==0)baseline=raw;else require(raw==baseline,"large parent all bytes V11");
                for(size_t j=0;j<raw.size();++j) {
                    size_t elem=j/4,row=elem/size_t(N)%size_t(total);
                    bool written=elem<size_t(4)*total*N && row>=size_t(offset)&&row<size_t(offset+R);
                    if(!written)require(raw[j]==0xa5,"large parent all padding bytes");
                }
                for(int split=0;split<4;++split)for(int r=0;r<R;++r)for(int c=0;c<N;++c) {
                    float got;size_t out=(size_t(split)*total+offset+r)*N+c,ref=(size_t(split)*8+r)*N+c;
                    memcpy(&got,raw.data()+out*4,4);partial_check(got,oracle.exact[ref],oracle.absolute[ref]);
                }
                require(parentA.get()==immutable,"large parent immutable A");parentA.guards();parentP.guards();
            }
        }
        // Every clone gets a prevalidated token; no factory or memory operation during submit.
        std::array<Plan,C> plans;
        for(size_t c=0;c<C;++c) {
            w.trellis=(uint32_t*)(B.p+c*bytes);
            require(make_performance(plans[c],w,(_Float16*)A.p,(float*)P.p,R,R,0,4,{B.p+c*bytes,bytes},A.range(),P.range()),"large private factory");
        }
        auto issue=[&](int m,size_t c,bool full,float alpha=1,float beta=0) {
            if(full)input_transform<<<(size_t(R)*(I/128)+3)/4,128>>>((float*)X.p,w.suh,(_Float16*)A.p,I,R);
            if(m==0){DeviceWeights control=w;control.trellis=(uint32_t*)(B.p+c*bytes);ck(launch_gemv_lds_core(control,(_Float16*)A.p,(float*)P.p,R,R,4,nullptr));}
            else submit(plans[c],m==1?Mode::Global:Mode::LDS8,nullptr);
            if(full)output_transform<<<(size_t(R)*(N/128)+3)/4,128>>>((float*)P.p,w.svh,w.bias,(float*)Y.p,N,R,4,alpha,beta);
        };
        auto check_partial=[&](const std::vector<unsigned char>& raw) {
            for(int s=0;s<4;++s)for(int r=0;r<R;++r)for(int c=0;c<N;++c) {
                size_t j=(size_t(s)*R+r)*N+c,ref=(size_t(s)*8+r)*N+c;float got;memcpy(&got,raw.data()+4*j,4);
                partial_check(got,oracle.exact[ref],oracle.absolute[ref]);
            }
            for(size_t j=size_t(4)*R*N*4;j<raw.size();++j)require(raw[j]==0xa5,"large P unused bytes");
        };
        ck(hipMemset(P.p,0xa5,P.size));ck(hipMemset(Y.p,0xa5,Y.size));issue(0,0,false);auto pref=P.get();check_partial(pref);
        for(auto ab:std::vector<std::array<float,2>>{{1,0},{.75f,0},{1,.5f},{-.5f,-.25f}}) {
            std::vector<unsigned char> control;
            for(int m=0;m<3;++m) {
                ck(hipMemset(Y.p,0xa5,Y.size));if(ab[1]==0)ck(hipMemset(Y.p,0xff,size_t(R)*N*4));else Y.upload(f.v.at("old").data(),size_t(R)*N*4);
                issue(m,0,true,ab[0],ab[1]);auto raw=Y.get();if(m==0)control=raw;else require(raw==control,"full Y bytes alpha beta V11");
                cpu_y_check(raw,f,R,ab[0],ab[1]);require(A.get()==originalA,"exact A half all stages");require(P.get()==pref,"full P bytes V11");
                for(size_t j=size_t(R)*N*4;j<raw.size();++j)require(raw[j]==0xa5,"large Y unused bytes");
            }
        }
        for(bool full:{false,true}) {
            ck(hipMemset(Y.p,0xa5,Y.size));issue(0,0,full);auto yref=Y.get();
            auto validate=[&]() {auto raw=P.get();require(raw==pref,"all clone P bytes V11");check_partial(raw);if(full){auto y=Y.get();require(y==yref,"all clone Y bytes V11");cpu_y_check(y,f,R,1,0);}require(A.get()==originalA,"all A bytes unchanged");P.guards();Y.guards();};
            check_inputs();
            for(size_t c=0;c<C;++c)for(int m=0;m<3;++m){issue(m,c,full);validate();}
            check_inputs();
            hipEvent_t begin,end;ck(hipEventCreate(&begin));ck(hipEventCreate(&end));std::array<std::vector<double>,3> samples;
            for(int sweep=-1;sweep<11;++sweep)for(int order=0;order<3;++order) {
                int m=(sweep&1)?2-order:order;
                auto host_begin=std::chrono::steady_clock::now();
                ck(hipEventRecord(begin));for(size_t c=0;c<C;++c)issue(m,c,full);ck(hipEventRecord(end));ck(hipEventSynchronize(end));
                double host_us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-host_begin).count()/C;
                if(sweep>=0)printf("HOST_SAMPLE I=%d N=%d R=%d S=4 protocol=%s mode=%d sweep=%d us=%.12g\n",I,N,R,full?"operator":"core",m,sweep,host_us);
                float ms;ck(hipEventElapsedTime(&ms,begin,end));
                if(sweep>=0){double us=ms*1000./C;samples[m].push_back(us);printf("SAMPLE I=%d N=%d R=%d S=4 protocol=%s mode=%d sweep=%d us=%.12g\n",I,N,R,full?"operator":"core",m,sweep,us);}
                if(sweep==10){validate();check_inputs();}
            }
            for(int m=0;m<3;++m){auto s=samples[m];std::sort(s.begin(),s.end());printf("RESULT I=%d N=%d R=%d S=4 protocol=%s mode=%d clones=26 resident_bytes=%zu min_us=%.12g median_us=%.12g p90_us=%.12g\n",I,N,R,full?"operator":"core",m,C*bytes,s[0],s[5],s[9]);}
            ck(hipEventDestroy(begin));ck(hipEventDestroy(end));
            check_inputs();
            for(size_t c=0;c<C;++c)for(int m=0;m<3;++m){issue(m,c,full);validate();}
            check_inputs();
        }
        printf("LARGE_ROWS PASS I=%d N=%d R=%d all_splits_columns=1 all_clones_before_after_last=1\n",I,N,R);
    }
    auto allB=B.get();for(size_t c=0;c<C;++c)require(memcmp(allB.data()+c*bytes,f.v.at("trellis").data(),bytes)==0,"immutable packed every byte every clone");
    require(A.get()==originalA&&X.get()==originalX&&SU.get()==originalSU&&SV.get()==originalSV&&bias.get()==originalBias,"immutable operator inputs every byte");
    for(Buffer* b:{&B,&A,&X,&SU,&SV,&bias,&P,&Y})b->guards();
    printf("LARGE PASS I=%d N=%d partial_ratio=%.12g nrms=%.12g maxabs_relative=%.12g gpu_peak_bytes=%zu\n",I,N,worst_partial_ratio,worst_nrms,worst_y_relative,Buffer::peak);
}
int main(int argc,char** argv) {
    try {
        setvbuf(stdout,nullptr,_IOLBF,0);require(argc==5,"two fixtures and two fresh FP64 references required");
        hipDeviceProp_t prop{};ck(hipGetDeviceProperties(&prop,0));require(prop.warpSize==32&&!strncmp(prop.gcnArchName,"gfx1151",7),"gfx1151 wave32");
        printf("DEVICE name=%s arch=%s wave=%d\n",prop.name,prop.gcnArchName,prop.warpSize);
        exhaustive();r1_identity();small();large(argv[1],argv[3]);large(argv[2],argv[4]);puts("PASS k2_rows_probe");return 0;
    }catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
