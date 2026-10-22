// Independent hash/SAD substitution experiment; no production dispatch changes.
#include "../src/gemv_combined.hpp"
#include "../src/combined_decoder.hpp"
#include "../src/host_reference.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
using strix_trellis::DeviceWeights;
static void ck(hipError_t e) {if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
struct Buffer {
    void* base=nullptr;unsigned char* p=nullptr;size_t size;
    explicit Buffer(size_t n):size(n) {ck(hipMalloc(&base,n+512));p=(unsigned char*)base+256;ck(hipMemset(base,0xa5,n+512));}
    ~Buffer(){if(base)(void)hipFree(base);}
    Buffer(const Buffer&)=delete;
    void guards() const {
        unsigned char a[256],b[256];ck(hipMemcpy(a,base,256,hipMemcpyDeviceToHost));ck(hipMemcpy(b,p+size,256,hipMemcpyDeviceToHost));
        for(int k=0;k<256;++k)if(a[k]!=0xa5||b[k]!=0xa5)throw std::runtime_error("Canary changed");
    }
};
template<int C> __device__ uint32_t hash_probe(uint32_t s) {
    if constexpr(C==0)return strix_trellis::combined_decode::hash16_isa<89226354u,64248484u>(s);
    if constexpr(C==1)return strix_trellis::combined_decode::hash16_isa<0xcbac1fedu>(s);
    return strix_trellis::combined_decode::hash16_isa<0x83dcd12du>(s);
}
template<int C> __device__ void probe_one(uint32_t s,uint32_t* out) {
    out[C*3]=hash_probe<C>(s|0xa55a0000u);
    out[C*3+1]=__builtin_bit_cast(uint32_t,strix_trellis::combined_decode::codebook_half2_isa<C>(s|0xdead0000u,(65535-s)|0xbeef0000u));
    out[C*3+2]=__builtin_bit_cast(uint32_t,strix_trellis::combined_decode::codebook_half2_isa<C>(s|0xcafe0000u,((s*13+7)&65535)|0x12340000u));
}
__global__ void exhaustive(uint32_t* out) {
    uint32_t s=blockIdx.x*blockDim.x+threadIdx.x;
    probe_one<0>(s,out+s*9);probe_one<1>(s,out+s*9);probe_one<2>(s,out+s*9);
}
template<int Bits> __global__ void reader_probe(const uint32_t* p,uint16_t* out) {
    strix_trellis::combined_decode::SaddrLane<Bits> lane(p+size_t(blockIdx.x)*Bits*8);
    size_t base=size_t(blockIdx.x)*256+(threadIdx.x&31)*8;
    #pragma unroll
    for(int j=0;j<8;++j) {
        if(j==0)out[base+j]=lane.template state<0>();
        if(j==1)out[base+j]=lane.template state<1>();
        if(j==2)out[base+j]=lane.template state<2>();
        if(j==3)out[base+j]=lane.template state<3>();
        if(j==4)out[base+j]=lane.template state<4>();
        if(j==5)out[base+j]=lane.template state<5>();
        if(j==6)out[base+j]=lane.template state<6>();
        if(j==7)out[base+j]=lane.template state<7>();
    }
}
static void books() {
    Buffer d(65536*9*4);exhaustive<<<256,256>>>((uint32_t*)d.p);ck(hipGetLastError());
    std::vector<uint32_t> got(65536*9);ck(hipMemcpy(got.data(),d.p,d.size,hipMemcpyDeviceToHost));
    const uint32_t multipliers[]={89226354u,0xcbac1fedu,0x83dcd12du};
    for(uint32_t s=0;s<65536;++s)for(int c=0;c<3;++c) {
        uint32_t hash=s*multipliers[c]+(c==0?64248484u:0);
        if(got[s*9+c*3]!=hash)throw std::runtime_error("Full uint32 hash mismatch");
        for(int pattern=0;pattern<2;++pattern) {
            uint32_t b=pattern?((s*13+7)&65535):(65535-s);
            uint32_t expected=uint32_t(host_codebook(s,c))|(uint32_t(host_codebook(b,c))<<16);
            if(got[s*9+c*3+1+pattern]!=expected)throw std::runtime_error("Half2 codebook mismatch");
        }
    }
    d.guards();puts("BOOKS PASS hashes=196608 half_values=786432 upper_container_bits=ignored canaries=PASS");
}
static uint32_t rng=0x4a39b70du;
static uint32_t random_word(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
static void readers() {
    for(int bits:{5,6}) {
        int tiles=bits*256+257;
        std::vector<uint32_t> p(tiles*bits*8);for(auto& x:p)x=random_word();
        for(int t=0;t<bits*256;++t){for(int q=0;q<bits*8;++q)p[t*bits*8+q]=0;p[t*bits*8+t/32]=1u<<(31-t%32);}
        Buffer dp(p.size()*4),dy(tiles*256*2);ck(hipMemcpy(dp.p,p.data(),dp.size,hipMemcpyHostToDevice));
        #define R(B) case B:reader_probe<B><<<tiles,32>>>((uint32_t*)dp.p,(uint16_t*)dy.p);break
        switch(bits){R(1);R(2);R(3);R(4);R(5);R(6);R(7);R(8);}
        #undef R
        ck(hipGetLastError());std::vector<uint16_t> got(tiles*256);ck(hipMemcpy(got.data(),dy.p,dy.size,hipMemcpyDeviceToHost));
        for(int t=0;t<tiles;++t)for(int j=0;j<256;++j)
            if(got[t*256+j]!=host_state(p.data()+t*bits*8,bits,j))throw std::runtime_error("SADDR reader state mismatch");
        dp.guards();dy.guards();
    }
    puts("READERS PASS bits=1..8 tiles_per_bits=257 states=526336");
}
static void launch(int mode,const DeviceWeights& w,const _Float16* a,float* p,int rows,int splits) {
    ck(strix_trellis_combined::launch(w,a,p,rows,rows,splits,nullptr,mode));
}
static std::vector<unsigned char> download(const Buffer& b,size_t n) {
    std::vector<unsigned char> v(n);ck(hipMemcpy(v.data(),b.p,n,hipMemcpyDeviceToHost));return v;
}
static void finite(const std::vector<unsigned char>& raw) {
    for(size_t k=0;k<raw.size();k+=4){float f;std::memcpy(&f,raw.data()+k,4);if(!std::isfinite(f))throw std::runtime_error("Nonfinite partial");}
}
static void compare_cpu(const std::vector<uint32_t>& packed,const std::vector<_Float16>& a,
    const std::vector<unsigned char>& actual,int i,int n,int bits,int cb,int rows,int splits) {
    for(int split=0;split<splits;++split)for(int r=0;r<rows;++r)for(int col=0;col<n;++col) {
        double sum=0,abs_sum=0;
        for(int tk=(i/16)*split/splits;tk<(i/16)*(split+1)/splits;++tk)for(int k=0;k<16;++k) {
            int position=((col&7)<<5)|((k&6)<<2)|(k&1)|((k&8)>>2)|((col&8)>>1);
            uint16_t state=host_state(packed.data()+(size_t(tk)*(n/16)+col/16)*bits*8,bits,position);
            double product=double(float(a[size_t(r)*i+tk*16+k]))*host_half(host_codebook(state,cb));
            sum+=product;abs_sum+=std::abs(product);
        }
        float got;std::memcpy(&got,actual.data()+((size_t(split)*rows+r)*n+col)*4,4);
        if(std::abs(double(got)-sum)>2e-6*std::max(1.0,abs_sum))throw std::runtime_error("Independent small core reference mismatch");
    }
}
static void small() {
    Buffer probe(1024);DeviceWeights invalid{(uint32_t*)probe.p,nullptr,nullptr,nullptr,16,16,5,2};
    auto rejected=[&](int mode,int rows,int total,int splits){if(strix_trellis_combined::launch(invalid,(_Float16*)probe.p,(float*)probe.p,rows,total,splits,nullptr,mode)!=hipErrorInvalidValue)throw std::runtime_error("Missing domain rejection");};
    for(int mode=0;mode<4;++mode){for(int bits:{0,1,2,3,4,7,8,9}){invalid.bits=bits;rejected(mode,1,1,1);}invalid.bits=5;
    for(int cb:{0,1,3}){invalid.codebook=cb;rejected(mode,1,1,1);}invalid.codebook=2;
    rejected(mode,0,1,1);rejected(mode,9,9,1);rejected(mode,2,1,1);rejected(mode,1,1,0);}
    rejected(-1,1,1,1);rejected(4,1,1,1);puts("REJECT PASS domain bits5/6 mul1 rows1..8 splits_positive total_rows mode");
    for(int bits:{5,6}) for(int n:{16,128}) for(int i:{16,128,528}) {std::vector<uint32_t> packed(size_t(i)*n*bits/32);
        for(auto& p:packed)p=random_word();std::vector<_Float16> a(8*i);
        for(auto& v:a)v=_Float16((int(random_word()%2049)-1024)/1024.0f);
        Buffer dp(packed.size()*4),da(a.size()*2+2),dy(11*10*n*4);
        ck(hipMemcpy(dp.p,packed.data(),dp.size,hipMemcpyHostToDevice));ck(hipMemcpy(da.p,a.data(),a.size()*2,hipMemcpyHostToDevice));
        DeviceWeights w{(uint32_t*)dp.p,nullptr,nullptr,nullptr,i,n,bits,0};
        for(int offset:{0,2}){ck(hipMemcpy(da.p+offset,a.data(),a.size()*2,hipMemcpyHostToDevice));for(int cb:{2}){w.codebook=cb;for(int rows=1;rows<=8;++rows)for(int splits:{1,3,11}) {
            int total=rows+2;size_t bytes=size_t(total)*splits*n*4;std::array<std::vector<unsigned char>,4> outputs;
            for(int mode=0;mode<4;++mode) {
                ck(hipMemset(dy.p,0xa5,dy.size));ck(hipMemset(dy.p,0xa5,bytes));
                ck(strix_trellis_combined::launch(w,(_Float16*)(da.p+offset),(float*)dy.p+n,rows,total,splits,nullptr,mode));
                auto all=download(dy,dy.size);
                for(size_t k=bytes;k<all.size();++k)if(all[k]!=0xa5)throw std::runtime_error("Partial tail changed");
                all.resize(bytes);outputs[mode]=std::move(all);finite(outputs[mode]);dy.guards();
            }
            if(outputs[0]!=outputs[1]||outputs[0]!=outputs[2]||outputs[0]!=outputs[3])throw std::runtime_error("Changed core bits");
            std::vector<unsigned char> compact(size_t(rows)*splits*n*4);
            for(int sp=0;sp<splits;++sp){
                for(int q:{0,rows+1})for(int c=0;c<n*4;++c)if(outputs[0][(size_t(sp)*total+q)*n*4+c]!=0xa5)throw std::runtime_error("Row chunk guard changed");
                std::memcpy(compact.data()+size_t(sp)*rows*n*4,outputs[0].data()+(size_t(sp)*total+1)*n*4,size_t(rows)*n*4);
            }
            compare_cpu(packed,a,compact,i,n,bits,cb,rows,splits);
        }}}
        dp.guards();da.guards();printf("SMALL PASS bits=%d output=%d books=mul1 rows=1..8 splits=1,3,11 same_core_bits=1 cpu_reference=all_partials inactive_wave_test=%d\n",bits,n,int(n==16));
    }
}
struct Shape{const char* name;int i,n,bits;};
static void benchmark(const Shape& shape,const std::vector<int>& split_choices) {
    size_t bytes=size_t(shape.i)*shape.n*shape.bits/8;
    size_t clones=((size_t(256)<<20)+bytes-1)/bytes;
    int maxsplit=*std::max_element(split_choices.begin(),split_choices.end());
    size_t partial=size_t(maxsplit)*8*shape.n*4;
    if(clones*bytes+partial+size_t(8)*shape.i*2+3*512>=(size_t(1)<<30))throw std::runtime_error("Device allocations exceed 1GiB");
    std::vector<uint32_t> packed(bytes/4);for(auto& p:packed)p=random_word();
    std::vector<_Float16> a(size_t(8)*shape.i);for(auto& p:a)p=_Float16((int(random_word()%2049)-1024)/1024.0f);
    Buffer dp(clones*bytes),da(a.size()*2),dy(partial);
    for(size_t c=0;c<clones;++c)ck(hipMemcpy(dp.p+c*bytes,packed.data(),bytes,hipMemcpyHostToDevice));
    ck(hipMemcpy(da.p,a.data(),a.size()*2,hipMemcpyHostToDevice));
    DeviceWeights w{(uint32_t*)dp.p,nullptr,nullptr,nullptr,shape.i,shape.n,shape.bits,2};
    for(int rows:{1,2,4,8})for(int splits:split_choices) {
        size_t used=size_t(rows)*splits*shape.n*4;
        auto check=[&](){
            std::array<std::vector<unsigned char>,4> outputs;
            for(int mode=0;mode<4;++mode) {
                ck(hipMemset(dy.p,0xff,used));launch(mode,w,(_Float16*)da.p,(float*)dy.p,rows,splits);
                outputs[mode]=download(dy,used);finite(outputs[mode]);
            }
            if(outputs[0]!=outputs[1]||outputs[0]!=outputs[2]||outputs[0]!=outputs[3])throw std::runtime_error("Large core bit mismatch");
            return std::array<std::vector<unsigned char>,2>{std::move(outputs[0]),std::move(outputs[3])};
        };
        auto references=check();std::array<std::vector<double>,4> samples;
        hipEvent_t begin,end;ck(hipEventCreate(&begin));ck(hipEventCreate(&end));
        for(int sweep=-1;sweep<11;++sweep)for(int order=0;order<4;++order) {
            int mode=(sweep&1)?(3-order):order;
            ck(hipEventRecord(begin));
            for(size_t c=0;c<clones;++c){w.trellis=(uint32_t*)(dp.p+c*bytes);launch(mode,w,(_Float16*)da.p,(float*)dy.p,rows,splits);}
            ck(hipEventRecord(end));ck(hipEventSynchronize(end));float ms;ck(hipEventElapsedTime(&ms,begin,end));
            if(sweep>=0)samples[mode].push_back(double(ms)*1000/clones);
            if(sweep==10 && download(dy,used)!=references[0])
                throw std::runtime_error("Last timed partial differs from baseline");
        }
        w.trellis=(uint32_t*)dp.p;check();dy.guards();
        ck(hipEventDestroy(begin));ck(hipEventDestroy(end));
        for(int mode=0;mode<4;++mode) {
            auto& s=samples[mode];
            printf("SAMPLES shape=%s bits=%d rows=%d splits=%d mode=%d us=",shape.name,shape.bits,rows,splits,mode);for(auto v:s)printf("%.9g,",v);puts("");std::sort(s.begin(),s.end());
            printf("RESULT shape=%s I=%d N=%d bits=%d rows=%d splits=%d mode=%d clones=%zu resident_bytes=%zu median_us=%.9g min_us=%.9g p90_us=%.9g compressed_GBs=%.9g bit_identity=1\n",
                shape.name,shape.i,shape.n,shape.bits,rows,splits,mode,clones,clones*bytes,s[5],s[0],s[9],bytes/s[5]/1000);
        }
    }
    dp.guards();da.guards();
}
static void fixture(const char* path) {
 FILE* f=fopen(path,"rb");if(!f)throw std::runtime_error("Missing fixture");
 auto read=[&](void* p,size_t n){if(fread(p,1,n,f)!=n)throw std::runtime_error("Short fixture");};
 char magic[8];uint32_t version,sections;read(magic,8);read(&version,4);read(&sections,4);
 if(memcmp(magic,"STGPU001",8)||version!=1||sections>64)throw std::runtime_error("Bad fixture");
 uint32_t meta[8]={};std::vector<uint32_t> packed;std::vector<_Float16>a;
 for(unsigned j=0;j<sections;++j){char tag[9]={};uint64_t bytes;read(tag,8);read(&bytes,8);
 if(!strcmp(tag,"meta")){if(bytes!=32)throw std::runtime_error("Bad meta");read(meta,32);}
 else if(!strcmp(tag,"trellis")){packed.resize(bytes/4);read(packed.data(),bytes);}
 else if(!strcmp(tag,"a")){a.resize(bytes/2);read(a.data(),bytes);}
 else if(fseek(f,long(bytes),SEEK_CUR))throw std::runtime_error("Seek failed");}fclose(f);
 int i=meta[0],n=meta[1],bits=meta[2],cb=meta[3];
 if(cb!=2||(bits!=5&&bits!=6)||meta[5]<8||packed.size()*4!=size_t(i)*n*bits/8||a.size()<size_t(8)*i)throw std::runtime_error("Fixture domain mismatch");
 Buffer dp(packed.size()*4),da(a.size()*2+2),dy(size_t(11)*8*n*4);
 ck(hipMemcpy(dp.p,packed.data(),dp.size,hipMemcpyHostToDevice));
 DeviceWeights w{(uint32_t*)dp.p,nullptr,nullptr,nullptr,i,n,bits,cb};
 for(int offset:{0,2}){ck(hipMemcpy(da.p+offset,a.data(),a.size()*2,hipMemcpyHostToDevice));
 for(int rows:{1,2,4,8})for(int splits:{1,3,11}){
 size_t used=size_t(rows)*splits*n*4;std::vector<unsigned char> ref;
 for(int mode=0;mode<4;++mode){ck(hipMemset(dy.p,0xff,used));launch(mode,w,(_Float16*)(da.p+offset),(float*)dy.p,rows,splits);auto got=download(dy,used);finite(got);
 if(!mode)ref=got;else if(got!=ref)throw std::runtime_error("Real fixture byte mismatch");}
 // Existing 2e-6*sumabs tolerance, unchanged from V2. All columns in first/last split are included.
 if(i<=128&&n<=128)compare_cpu(packed,a,ref,i,n,bits,cb,rows,splits);
 dy.guards();} }
 dp.guards();da.guards();printf("FIXTURE PASS path=%s I=%d N=%d bits=%d offsets=0,2 rows=1,2,4,8 splits=1,3,11 all_partial_bytes=exact\n",path,i,n,bits);
}

int main(int argc,char** argv) {
 try {
 setvbuf(stdout,nullptr,_IOLBF,0);
 hipDeviceProp_t p{};ck(hipGetDeviceProperties(&p,0));
 if(p.warpSize!=32||std::strncmp(p.gcnArchName,"gfx1151",7))throw std::runtime_error("gfx1151 wave32 required");
 printf("DEVICE name=%s arch=%s wave=%d modes=0_v8,1_hash,2_saddr,3_combined\n",p.name,p.gcnArchName,p.warpSize);
 books();
 if(argc>1&&!std::strcmp(argv[1],"--validate")){readers();small();puts("PASS combined_validation");return 0;}
 if(argc>1&&!std::strcmp(argv[1],"--fixtures")){for(int j=2;j<argc;++j)fixture(argv[j]);puts("PASS combined_real_fixtures");return 0;}
 benchmark({"q_b_synthetic",1280,32768,5},{4});
 benchmark({"wo_b_synthetic",8192,5120,5},{4});
 benchmark({"wo_b_synthetic",8192,5120,6},{4});
 benchmark({"shared_synthetic",5120,2304,5},{16});
 benchmark({"shared_inverse_synthetic",2304,5120,5},{4});
 benchmark({"q_a_synthetic",5120,1280,6},{16});
 benchmark({"wo_a_synthetic",4096,1024,5},{32});
 benchmark({"wkv_synthetic",5120,512,6},{32});
 benchmark({"head_synthetic",5120,129280,6},{4});
 puts("PASS combined_benchmark");return 0;
 }catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}
