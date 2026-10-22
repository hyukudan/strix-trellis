// Shared C ABI-only QA. This header never includes private runtime definitions.
#pragma once
#include "v15_helpers.hpp"
#include <filesystem>
#include <sys/resource.h>
namespace fs=std::filesystem;
static void require(bool v,const std::string& why){if(!v)throw std::runtime_error(why);}
static std::vector<unsigned char> get(Runtime& r,const void* p,size_t n){std::vector<unsigned char> v(n);hip_check(hipMemcpyAsync(v.data(),p,n,hipMemcpyDeviceToHost,r.stream));hip_check(hipStreamSynchronize(r.stream));return v;}
static void save(const fs::path& p,const std::vector<unsigned char>& v){std::ofstream o(p,std::ios::binary);o.write(reinterpret_cast<const char*>(v.data()),v.size());require(bool(o),"Write failed");}
static std::vector<unsigned char> readbytes(const fs::path& p){std::ifstream f(p,std::ios::binary|std::ios::ate);require(bool(f),"Missing "+p.string());auto n=f.tellg();require(n>=0 && n<(128ll<<20),"File bound");std::vector<unsigned char> v(static_cast<size_t>(n));f.seekg(0);f.read(reinterpret_cast<char*>(v.data()),v.size());require(bool(f),"Read failed");return v;}
static bool poison(const std::vector<unsigned char>& v){return std::all_of(v.begin(),v.end(),[](unsigned char b){return b==255;});}
static void finite32(const unsigned char* p,size_t n){require(n%4==0,"F32 size");for(size_t k=0;k<n;k+=4){float x;std::memcpy(&x,p+k,4);require(std::isfinite(x),"Nonfinite workspace P");}}
struct Guarded {
    Device allocation;unsigned char* p;size_t bytes;
    Guarded(size_t n,Runtime& r):allocation(n+512),p(static_cast<unsigned char*>(allocation.p)+256),bytes(n){hip_check(hipMemsetAsync(allocation.p,0xa5,n+512,r.stream));}
    void check(Runtime& r){auto a=get(r,allocation.p,256),b=get(r,p+bytes,256);for(auto v:{a,b})require(std::all_of(v.begin(),v.end(),[](unsigned char x){return x==0xa5;}),"Canary changed");}
};
static constexpr float epilogues[][2]={{1,0},{1,-0.0f},{-0.75f,0},{0,0},{-0.0f,-0.0f},{-0.75f,-0.5f},{1.25f,0.25f},{0,-0.5f}};
static void setbias(Fixture& f,bool bias){f.bias=bias;f.parts["bias"].resize(bias?f.n*4:0);if(bias)for(unsigned k=0;k<f.n;++k){float b=(int(k%29)-14)*0.00390625f;std::memcpy(f.parts["bias"].data()+k*4,&b,4);}}
static std::string tag(int id,int bias,int rows,int ep,bool gemm){return std::to_string(id)+"-b"+std::to_string(bias)+"-r"+std::to_string(rows)+"-e"+std::to_string(ep)+(gemm?"-gemm":"-gemv");}
struct Case {
    Runtime& r;Fixture& f;Guarded x,y,ws;strix_matmul_desc d{};int rows;size_t aligned;
    static size_t need(Runtime& r,Fixture& f,strix_weights w,int rows,void* x,void* y){auto d=desc(w,x,y,rows,f,false,1,0);uint64_t n=0;api_check(strix_matmul_workspace_bytes(r.context,&d,&n));return n;}
    Case(Runtime& rt,Fixture& fixture,strix_weights w,int m):r(rt),f(fixture),x(size_t(m)*f.i*4,r),y(size_t(m)*f.n*4,r),ws(need(r,f,w,m,x.p,y.p),r),rows(m),aligned((size_t(m)*f.i*2+255)&~size_t(255)){
        d=desc(w,x.p,y.p,m,f,false,1,0);d.workspace=ws.p;d.workspace_bytes=ws.bytes;
        require(ws.bytes==aligned+size_t(4)*m*f.n*4,"V13 workspace/splits changed");upload(x.p,f.parts.at("x").data(),x.bytes,r);reset(1,0);
    }
    void reset(float alpha,float beta){d.alpha=alpha;d.beta=beta;hip_check(hipMemsetAsync(ws.p,255,ws.bytes,r.stream));if(beta==0)hip_check(hipMemsetAsync(y.p,255,y.bytes,r.stream));else upload(y.p,f.parts.at("old").data(),y.bytes,r);}
    void submit(bool gemm=false,const strix_stream* stream=nullptr){api_check(gemm?strix_gemm(r.context,&d,stream?stream:&r.s):strix_gemv(r.context,&d,stream?stream:&r.s));}
    void intact(){require(get(r,x.p,x.bytes)==std::vector<unsigned char>(f.parts.at("x").begin(),f.parts.at("x").begin()+x.bytes),"X changed");x.check(r);y.check(r);ws.check(r);}
    void oracle(const std::vector<double>& base,const std::vector<float>* previous=nullptr){auto raw=get(r,y.p,y.bytes);double e2=0,r2=0,ma=0,mr=0;for(size_t k=0;k<raw.size()/4;++k){float value,old=0,bias=0;std::memcpy(&value,raw.data()+k*4,4);if(d.beta!=0){if(previous)old=previous->at(k);else std::memcpy(&old,f.parts.at("old").data()+k*4,4);}if(f.bias)std::memcpy(&bias,f.parts.at("bias").data()+(k%f.n)*4,4);double ref=double(d.alpha)*(base.at(k)+bias);if(d.beta!=0)ref+=double(d.beta)*old;require(std::isfinite(value) && std::isfinite(ref),"Oracle nonfinite");double e=std::abs(value-ref);ma=std::max(ma,e);mr=std::max(mr,std::abs(ref));e2+=e*e;r2+=ref*ref;}require(ma<=5e-4*std::max(1.,mr) && std::sqrt(e2/std::max(r2,1e-30))<=3e-5,"Full FP64 tolerance");std::cout<<"FP64 values="<<raw.size()/4<<" maxabs="<<ma<<" NRMS="<<std::sqrt(e2/std::max(r2,1e-30))<<'\n';}
};
static void host_budget(){rusage u{};require(getrusage(RUSAGE_SELF,&u)==0 && uint64_t(u.ru_maxrss)*1024<(2ull<<30),"Host bound");std::cout<<"HOST_PEAK="<<uint64_t(u.ru_maxrss)*1024<<" GPU_EXPLICIT_PEAK="<<qa_peak<<'\n';}
