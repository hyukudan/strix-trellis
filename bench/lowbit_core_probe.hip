// Private core and three-stage DEVICE operator; no C API, MoE or expanded Q.
#include "gemv_lowbit.hpp"
#include "gemv_lds.hpp"
#include "transforms.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
using namespace strix_trellis;
namespace {
void require(bool ok,const std::string& msg) {if(!ok)throw std::runtime_error(msg);}
void ck(hipError_t e) {require(e==hipSuccess,hipGetErrorString(e));}
size_t live_bytes=0,peak_bytes=0;
double max_partial_ratio=0,max_output_abs=0,max_output_nrms=0;
uint64_t partial_values_checked=0,output_values_checked=0;
struct Device {
    void* p=nullptr;size_t bytes;
    explicit Device(size_t n):bytes(n) {
        require(live_bytes+n<(1ull<<30),"GPU memory budget");ck(hipMalloc(&p,n));
        live_bytes+=n;peak_bytes=std::max(live_bytes,peak_bytes);
    }
    ~Device(){if(p){(void)hipFree(p);live_bytes-=bytes;}}
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
    template<class T>T* as(size_t byte=0){return reinterpret_cast<T*>(static_cast<char*>(p)+byte);}
};
using Bytes=std::vector<unsigned char>;
struct File {
    std::map<std::string,Bytes> parts;
    explicit File(const std::string& path) {
        std::ifstream f(path,std::ios::binary);require(bool(f),"open "+path);
        char magic[8];uint32_t count;f.read(magic,8);f.read(reinterpret_cast<char*>(&count),4);
        require(!memcmp(magic,"LOWBIT01",8)&&count<32,"fixture header");
        for(uint32_t j=0;j<count;++j) {
            char tag[17]={};uint64_t n;f.read(tag,16);f.read(reinterpret_cast<char*>(&n),8);
            require(n<(1ull<<28),"fixture section bound");auto& v=parts[tag];v.resize(n);
            f.read(reinterpret_cast<char*>(v.data()),n);require(bool(f),"fixture payload");
        }
    }
    template<class T>std::vector<T> get(const std::string& key)const {
        const auto& b=parts.at(key);require(b.size()%sizeof(T)==0,"typed section");
        std::vector<T> v(b.size()/sizeof(T));memcpy(v.data(),b.data(),b.size());return v;
    }
    void upload(const std::string& key,Device& d,size_t offset=0) const {
        const auto& b=parts.at(key);require(offset+b.size()<=d.bytes,"upload bound");
        ck(hipMemcpy(d.as<char>(offset),b.data(),b.size(),hipMemcpyHostToDevice));
    }
};
template<class T>std::vector<T> download(Device& d,size_t count,size_t byte=0) {
    require(byte+count*sizeof(T)<=d.bytes,"download bound");std::vector<T> v(count);
    ck(hipMemcpy(v.data(),d.as<T>(byte),count*sizeof(T),hipMemcpyDeviceToHost));return v;
}
void bytes_equal(const void* a,const void* b,size_t n,const std::string& msg) {
    require(!memcmp(a,b,n),msg);
}
void cpu_partial(const std::vector<float>& actual,const std::vector<float>& ref,
                 const std::vector<double>& sumabs) {
    require(actual.size()==ref.size()&&ref.size()==sumabs.size(),"partial dimensions");
    partial_values_checked+=ref.size();
    for(size_t j=0;j<ref.size();++j) {
        max_partial_ratio=std::max(max_partial_ratio,std::abs(double(actual[j])-ref[j])/std::max(1.,sumabs[j]));
        if(!(std::isfinite(actual[j])&&std::abs(double(actual[j])-ref[j])<=2e-6*std::max(1.,sumabs[j])))
            throw std::runtime_error("CPU partial bound column "+std::to_string(j));
    }
}
void cpu_output(const std::vector<float>& a,const std::vector<float>& b) {
    double maxerr=0,maxref=0,err2=0,ref2=0;
    require(a.size()==b.size(),"output dimensions");
    for(size_t j=0;j<a.size();++j) {
        require(std::isfinite(a[j])&&std::isfinite(b[j]),"nonfinite output");
        double d=double(a[j])-b[j];maxerr=std::max(maxerr,std::abs(d));maxref=std::max(maxref,std::abs(double(b[j])));
        err2+=d*d;ref2+=double(b[j])*b[j];
    }
    double nrms=sqrt(err2/std::max(ref2,1e-30));output_values_checked+=a.size();
    max_output_abs=std::max(max_output_abs,maxerr);max_output_nrms=std::max(max_output_nrms,nrms);
    require(maxerr<=5e-4*std::max(1.,maxref)&&nrms<=3e-5,"full CPU output gate");
}
std::vector<int> modes(int k) {return k==3?std::vector<int>{-1,0,1,2}:std::vector<int>{-1,3,4};}
void core(const DeviceWeights& w,const _Float16* a,float* y,int total,int s,int mode) {
    if(mode<0)ck(launch_gemv_lds_core(w,a,y,1,total,s,nullptr));
    else ck(strix_trellis_lowbit::launch(w,a,y,1,total,s,mode));
}
void exhaustive(const std::string& root) {
    for(int k:{3,4}) {
        File f(root+"/reader"+std::to_string(k)+".lbc");auto expected=f.get<uint32_t>("states");
        int count=int(expected.size()/256);Device p(f.parts.at("rings").size()),out(expected.size()*4);
        f.upload("rings",p);
        for(bool coop:{false,true}) {
            ck(strix_trellis_lowbit::probe(k,coop,p.as<uint32_t>(),out.as<uint32_t>(),count));
            auto actual=download<uint32_t>(out,expected.size());
            bytes_equal(actual.data(),expected.data(),actual.size()*4,"GPU reader exhaustive");
            std::cout<<"CHECK reader K="<<k<<" coop="<<coop<<" states="<<actual.size()<<" PASS\n";
        }
    }
    File f(root+"/hash.lbc");auto lut=f.get<uint16_t>("lut");size_t count=65536*8;
    Device hashes(count*8*4),pairs(count*4*4);
    ck(strix_trellis_lowbit::probe_hash(hashes.as<uint32_t>(),pairs.as<uint32_t>()));
    auto h=download<uint32_t>(hashes,count*8);auto p=download<uint32_t>(pairs,count*4);
    for(size_t index=0;index<count;++index) {
        for(int j=0;j<8;++j) {
            uint32_t state=(index+j*8191)&65535;
            require(h[index*8+j]==state*uint32_t(0x83dcd12d),"GPU V1 contaminated hash");
        }
        for(int j=0;j<4;++j) {
            uint32_t a=(index+2*j*8191)&65535,b=(index+(2*j+1)*8191)&65535;
            require(p[index*4+j]==(uint32_t(lut[a])|(uint32_t(lut[b])<<16)),"GPU half2 exhaustive");
        }
    }
    std::cout<<"CHECK hash_contaminated="<<count*8<<" half2_pairs="<<count*4<<" PASS\n";
}
void small(const File& f,const std::string& name) {
    auto m=f.get<uint32_t>("meta");int i=m[0],n=m[1],k=m[2],s=m[3];
    Device p(f.parts.at("packed").size()),a(i*2+4);f.upload("packed",p);
    DeviceWeights w{p.as<uint32_t>(),nullptr,nullptr,nullptr,i,n,k,2};
    auto ref=f.get<float>("partial");auto bound=f.get<double>("sumabs");
    for(int offset:{0,2})for(int total:{1,3}) {
        int row=total==3?1:0;size_t count=size_t(s)*total*n+32;
        Device y(count*4);std::vector<uint32_t> fill(count,0x4f123456),baseline;
        f.upload("a",a,offset);
        for(int mode:modes(k)) {
            ck(hipMemcpy(y.p,fill.data(),count*4,hipMemcpyHostToDevice));
            core(w,a.as<_Float16>(offset),y.as<float>((16+row*n)*4),total,s,mode);
            auto raw=download<uint32_t>(y,count);auto actual=download<float>(y,count);
            std::vector<float> active;
            for(size_t index=0;index<count;++index) {
                bool valid=index>=16&&index<16+size_t(s)*total*n&&((index-16)/n)%total==size_t(row);
                if(valid)active.push_back(actual[index]);else require(raw[index]==fill[index],"protected gap/canary");
            }
            cpu_partial(active,ref,bound);
            if(mode<0)baseline=raw;else bytes_equal(raw.data(),baseline.data(),count*4,"small v9 bytes");
        }
    }
    std::cout<<"CHECK small="<<name<<" offset=0,2 total_rows=1,3 row_offset=0,1 PASS\n";
}
struct Events {
    hipEvent_t a,b;
    Events(){ck(hipEventCreate(&a));ck(hipEventCreate(&b));}
    ~Events(){(void)hipEventDestroy(a);(void)hipEventDestroy(b);}
};
void large(const File& f,const std::string& name) {
    auto m=f.get<uint32_t>("meta");int i=m[0],n=m[1],k=m[2],s=m[3];
    const size_t packed=f.parts.at("packed").size(),clones=((256ull<<20)+packed-1)/packed;
    const size_t pc=size_t(s)*n;
    Device p(packed*clones),a(i*2),x(i*4),suh(i*2),svh(n*2),bias(n*4),partial(pc*clones*4),y(n*clones*4);
    for(size_t c=0;c<clones;++c)f.upload("packed",p,c*packed);
    f.upload("a",a);f.upload("x",x);f.upload("suh",suh);f.upload("svh",svh);f.upload("bias",bias);
    auto pref=f.get<float>("partial"),outref=f.get<float>("output"),old=f.get<float>("old"),biasref=f.get<float>("bias");
    auto bound=f.get<double>("sumabs");auto aref=f.get<uint16_t>("a");
    std::vector<float> baseline_partial(pc*clones),baseline_y(n*clones);
    auto submit=[&](size_t c,int mode,bool full,float alpha,float beta,bool with_bias) {
        DeviceWeights w{p.as<uint32_t>(c*packed),suh.as<uint16_t>(),svh.as<uint16_t>(),with_bias?bias.as<float>():nullptr,i,n,k,2};
        if(full) {
            input_transform<<<(i/128+3)/4,128>>>(x.as<float>(),w.suh,a.as<_Float16>(),i,1);ck(hipGetLastError());
        }
        core(w,a.as<_Float16>(),partial.as<float>(c*pc*4),1,s,mode);
        if(full) {
            output_transform<<<(n/128+3)/4,128>>>(partial.as<float>(c*pc*4),w.svh,w.bias,y.as<float>(c*n*4),n,1,s,alpha,beta);ck(hipGetLastError());
        }
    };
    auto check=[&](int mode,bool full,float alpha,float beta,bool with_bias,bool save) {
        auto got=download<float>(partial,pc*clones);
        for(size_t c=0;c<clones;++c) {
            std::vector<float> one(got.begin()+c*pc,got.begin()+(c+1)*pc);cpu_partial(one,pref,bound);
        }
        if(mode<0&&save)baseline_partial=got;
        else if(mode>=0)bytes_equal(got.data(),baseline_partial.data(),got.size()*4,"large v9 partial bytes");
        if(full) {
            auto aa=download<uint16_t>(a,i);bytes_equal(aa.data(),aref.data(),i*2,"input checkpoint");
            auto yy=download<float>(y,n*clones);std::vector<float> expected(n);
            for(int j=0;j<n;++j)expected[j]=((with_bias?outref[j]:outref[j]-biasref[j])*alpha)+(beta?beta*old[j]:0.f);
            for(size_t c=0;c<clones;++c)cpu_output(std::vector<float>(yy.begin()+c*n,yy.begin()+(c+1)*n),expected);
            if(mode<0&&save)baseline_y=yy;
            else if(mode>=0)bytes_equal(yy.data(),baseline_y.data(),yy.size()*4,"full v9 bytes");
        }
    };
    auto correctness=[&](const char* phase) {
        for(int variant=0;variant<4;++variant) {
            bool full=variant>0,with_bias=variant!=3;
            float alpha=variant==2?.75f:variant==3?-.5f:1.f,beta=variant==2?-.25f:0.f;
            for(int mode:modes(k)) {
                for(size_t c=0;c<clones;++c) {
                    if(beta)f.upload("old",y,c*n*4);
                    else ck(hipMemset(y.as<float>(c*n*4),0xff,n*4)); // beta0 poison
                    submit(c,mode,full,alpha,beta,with_bias);
                }
                check(mode,full,alpha,beta,with_bias,true);
            }
        }
        std::cout<<"CHECK large="<<name<<" phase="<<phase<<" clones="<<clones<<" all_columns alpha_beta_bias=4 PASS\n";
    };
    correctness("before");
    struct Mode {int candidate;bool full;};std::vector<Mode> schedule;
    for(bool full:{false,true})for(int mode:modes(k))schedule.push_back({mode,full});
    // Reestablish baseline for the timed full mode (alpha1/beta0/bias).
    for(size_t c=0;c<clones;++c)submit(c,-1,true,1,0,true);
    check(-1,true,1,0,true,true);
    Events events;std::map<std::pair<int,bool>,std::vector<double>> records;
    for(int sweep=-1;sweep<11;++sweep) {
        auto order=schedule;if(sweep>=0&&(sweep&1))std::reverse(order.begin(),order.end());
        for(auto mode:order) {
            ck(hipEventRecord(events.a));
            for(size_t c=0;c<clones;++c)submit(c,mode.candidate,mode.full,1,0,true);
            ck(hipEventRecord(events.b));ck(hipEventSynchronize(events.b));float ms=0;
            ck(hipEventElapsedTime(&ms,events.a,events.b));
            // Consume all clone results from this exact final timed rotation.
            check(mode.candidate,mode.full,1,0,true,false);
            if(sweep>=0) {
                double us=double(ms)*1000/clones;records[{mode.candidate,mode.full}].push_back(us);
                std::cout<<"RAW case="<<name<<" sweep="<<sweep<<" candidate="<<mode.candidate<<" stage="<<(mode.full?"full":"core")<<" us="<<us<<" checked_all_clones=1\n";
            }
        }
    }
    correctness("after");
    for(auto& item:records) {
        auto times=item.second;std::sort(times.begin(),times.end());
        std::cout<<"RESULT case="<<name<<" candidate="<<item.first.first<<" stage="<<(item.first.second?"full":"core")
            <<" min_us="<<times[0]<<" median_us="<<times[5]<<" p90_us="<<times[9]
            <<" clones="<<clones<<" packed_resident="<<packed*clones<<" PASS\n";
    }
}
}
int main(int argc,char** argv) {
    try {
        require(argc==2,"Usage: lowbit_core_probe FIXTURE_DIRECTORY");
        std::cout<<std::setprecision(10)<<std::unitbuf;
        hipDeviceProp_t p{};ck(hipGetDeviceProperties(&p,0));
        std::cout<<"HARDWARE name="<<p.name<<" arch="<<p.gcnArchName<<" CUs="<<p.multiProcessorCount
            <<" totalGlobalMem="<<p.totalGlobalMem<<" clock_kHz="<<p.clockRate<<"\n";
        exhaustive(argv[1]);std::ifstream list(std::string(argv[1])+"/cases.txt");require(bool(list),"case list");
        std::string name;
        while(std::getline(list,name)) {File f(std::string(argv[1])+"/"+name);auto m=f.get<uint32_t>("meta");
            if(m[4])large(f,name);else small(f,name);}
        std::cout<<"PASS lowbit_core peak_device_bytes="<<peak_bytes<<" partial_values_checked="<<partial_values_checked
            <<" output_values_checked="<<output_values_checked<<" max_partial_ratio="<<max_partial_ratio
            <<" max_output_abs="<<max_output_abs<<" max_output_nrms="<<max_output_nrms<<"\n";
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<"\n";return 1;}
}
