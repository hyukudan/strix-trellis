#include "../bench/head_fixture.hpp"
#include <chrono>
#include <iostream>
#include <limits>
#include <string>

namespace fs=std::filesystem;
using strix_trellis::bench::HeadData;
namespace {
void require(bool value,const char* message) { if(!value)throw std::runtime_error(message); }
void put(std::vector<unsigned char>& bytes,size_t offset,uint64_t value,size_t length=4) {
    for(size_t i=0;i<length;++i)bytes.at(offset+i)=static_cast<unsigned char>(value>>(8*i));
}
void save(const fs::path& path,const void* data,size_t bytes) {
    std::ofstream out(path,std::ios::binary);out.write(static_cast<const char*>(data),bytes);
    require(bool(out),"Test fixture write failed");
}
struct Files {
    fs::path path;
    Files():path(fs::current_path()/ ("head-fixture-test-"+std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count()))) {
        require(fs::create_directory(path),"Fresh test directory required");
        std::vector<float> x(8*5120),expected(3*8*128);
        save(path/"input.f32",x.data(),x.size()*4);
        save(path/"expected.f32",expected.data(),expected.size()*4);
    }
    ~Files() {
        // Only remove the three exact files created by this test, then an empty directory.
        std::error_code error;
        for(const char* name:{"head.meta","input.f32","expected.f32"})fs::remove(path/name,error);
        fs::remove(path,error);
    }
};
std::vector<unsigned char> metadata(unsigned version,unsigned bits) {
    std::vector<unsigned char> b(version==1?100:104);
    std::memcpy(b.data(),"STHEAD01",8);
    for(auto entry:std::vector<std::pair<size_t,uint64_t>>{
            {8,version},{12,8},{16,5120},{20,129280},{24,bits},{28,2},{32,0x83dcd12d},
            {36,3},{88,0},{92,64640},{96,129152}})put(b,entry.first,entry.second);
    const uint64_t packed=uint64_t(5120)*129280*bits/8;
    for(auto entry:std::vector<std::pair<size_t,uint64_t>>{
            {40,packed+2*(5120+129280)},{48,packed},{56,packed},{64,packed+2*5120},
            {72,8*5120*4},{80,3*8*128*4}})put(b,entry.first,entry.second,8);
    if(version!=1)put(b,100,version==3?2:1);
    return b;
}
}
int main() {
    try {
        Files files;unsigned rejected=0;
        auto reject=[&](const std::vector<unsigned char>& bytes) {
            save(files.path/"head.meta",bytes.data(),bytes.size());
            bool failed=false;try{HeadData invalid(files.path);}catch(const std::exception&){failed=true;}
            require(failed,"Invalid head metadata accepted");++rejected;
        };
        for(auto v:std::vector<std::pair<unsigned,unsigned>>{{1,6},{2,6},{3,4},{3,5}}) {
            auto b=metadata(v.first,v.second);save(files.path/"head.meta",b.data(),b.size());
            HeadData data(files.path);
            require(data.bits==v.second && data.source_tag==(v.first==3?2:v.first==2?1:0),"Source/rate confused");
            require(data.actual_model_values==(v.first!=1),"Real/synthetic confused");
        }
        for(unsigned bits:{0u,1u,2u,3u,6u,7u,8u,0xffffffffu})reject(metadata(3,bits));
        reject(metadata(2,4));reject(metadata(2,5));reject(metadata(1,4));
        for(auto e:std::vector<std::pair<size_t,uint64_t>>{{8,4},{12,7},{16,0xffffffffu},
                {20,0xffffffffu},{28,1},{32,0},{36,4},{88,1},{92,0},{96,129280},{100,1}}) {
            auto b=metadata(3,4);put(b,e.first,e.second);reject(b);
        }
        for(size_t offset:{40,48,56,64,72,80}) {
            auto b=metadata(3,5);put(b,offset,~uint64_t(0),8);reject(b);
        }
        auto b=metadata(2,6);put(b,100,2);reject(b);
        b=metadata(3,4);b.pop_back();reject(b);
        b=metadata(3,5);b.push_back(0);reject(b);
        b=metadata(3,4);b[0]='X';reject(b);
        b=metadata(3,4);save(files.path/"head.meta",b.data(),b.size());
        std::vector<float> bad(8*5120);bad.back()=std::numeric_limits<float>::quiet_NaN();
        save(files.path/"input.f32",bad.data(),bad.size()*4);reject(b);
        std::cout<<"PASS head fixture: 4 valid schemas, "<<rejected<<" rejected corruptions\n";
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
