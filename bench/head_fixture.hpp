#pragma once
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
namespace strix_trellis::bench {
namespace fs=std::filesystem;
struct HeadData {
    bool actual_model_values=false;
    uint32_t source_tag=0;
    static uint64_t le(const unsigned char* p,int bytes) {
        uint64_t value=0;for(int k=0;k<bytes;++k)value|=uint64_t(p[k])<<(8*k);return value;
    }
    const char* source_kind() const {return source_tag==2?"ours-local-snapshot":(source_tag==1?"shared-eaq-v1":"synthetic");}
    uint32_t rows=0,i=0,n=0,bits=0,cb=0,marker=0;
    uint64_t blob_bytes=0,packed_bytes=0,suh_offset=0,svh_offset=0;
    std::vector<uint32_t> blocks;
    std::vector<float> x,expected;
    fs::path directory;
    static std::vector<unsigned char> read(const fs::path& p,size_t expected_bytes) {
        std::ifstream f(p,std::ios::binary|std::ios::ate);
        if(!f || f.tellg()!=std::streamoff(expected_bytes)) throw std::runtime_error("Invalid fixture size: "+p.string());
        std::vector<unsigned char> result(expected_bytes);f.seekg(0);
        f.read(reinterpret_cast<char*>(result.data()),result.size());
        if(!f) throw std::runtime_error("Fixture read failed");
        return result;
    }
    static std::vector<float> floats(const fs::path& p,size_t count) {
        auto raw=read(p,count*4);std::vector<float> result(count);
        std::memcpy(result.data(),raw.data(),raw.size());
        for(float v:result) if(!std::isfinite(v)) throw std::runtime_error("Nonfinite fixture value");
        return result;
    }
    explicit HeadData(const fs::path& path):directory(path) {
        const uint16_t endian=1;
        if(*reinterpret_cast<const unsigned char*>(&endian)!=1) throw std::runtime_error("Little endian host required");
        auto meta_bytes=fs::file_size(path/"head.meta");
        if(meta_bytes!=100 && meta_bytes!=104) throw std::runtime_error("Invalid head metadata size");
        auto raw=read(path/"head.meta",meta_bytes);const auto* p=raw.data();
        uint32_t version=le(p+8,4);
        if(std::memcmp(p,"STHEAD01",8) ||
           !((version==1 && meta_bytes==100) || (version==2 && meta_bytes==104 && le(p+100,4)==1) ||
             (version==3 && meta_bytes==104 && le(p+100,4)==2)))
            throw std::runtime_error("Invalid head metadata version/source");
        actual_model_values=version>=2;
        source_tag=version==3?2:(version==2?1:0);
        rows=le(p+12,4);i=le(p+16,4);n=le(p+20,4);bits=le(p+24,4);cb=le(p+28,4);marker=le(p+32,4);
        uint32_t count=le(p+36,4);
        if(rows!=8 || i!=5120 || n!=129280 || ((version<3 && bits!=6) || (version==3 && bits!=4 && bits!=5)) || cb!=2 || marker!=0x83DCD12Du || count!=3)
            throw std::runtime_error("Unexpected head geometry");
        blob_bytes=le(p+40,8);packed_bytes=le(p+48,8);suh_offset=le(p+56,8);svh_offset=le(p+64,8);
        if(packed_bytes!=uint64_t(i)*n*bits/8 || suh_offset!=packed_bytes ||
           svh_offset!=packed_bytes+2*i || blob_bytes!=packed_bytes+2ull*(i+n) ||
           le(p+72,8)!=uint64_t(rows)*i*4 || le(p+80,8)!=uint64_t(count)*rows*128*4)
            throw std::runtime_error("Invalid head extents");
        for(uint32_t b=0;b<count;++b) {
            uint32_t begin=le(p+88+4*b,4);
            if(begin%128 || begin>n-128 || std::find(blocks.begin(),blocks.end(),begin)!=blocks.end())
                throw std::runtime_error("Invalid sampled H128 block");
            blocks.push_back(begin);
        }
        x=floats(path/"input.f32",size_t(rows)*i);
        expected=floats(path/"expected.f32",size_t(count)*rows*128);
    }
};
} // namespace strix_trellis::bench
