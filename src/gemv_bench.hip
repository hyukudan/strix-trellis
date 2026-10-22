// Cold rotation benchmark and independent bit-by-bit FP64 oracle. No transforms.
#include "gemv.hpp"
#include "trellis_optimized.hpp"
#include "host_reference.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cstring>
using namespace strix_trellis;
#define HIP(call) do {hipError_t e=(call);if(e!=hipSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(2);}} while(0)
static uint32_t rng=0x723f281a;
static uint32_t random_word(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
__global__ void read_contiguous(const uint4* p,size_t count,uint32_t* sink) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    uint32_t sum=0;
    for(;i<count;i+=size_t(gridDim.x)*blockDim.x){uint4 v=p[i];sum+=v.x+v.y+v.z+v.w;}
    __shared__ uint32_t sums[256];sums[threadIdx.x]=sum;__syncthreads();
    for(int s=128;s;s/=2){if(threadIdx.x<s)sums[threadIdx.x]+=sums[threadIdx.x+s];__syncthreads();}
    if(!threadIdx.x)sink[blockIdx.x]=sums[0];
}
__global__ void validate_decode(const uint32_t* p,uint16_t* states,uint16_t* values,int bits,int cb) {
    int t=threadIdx.x&31,base=blockIdx.x*256+t*8;
    #define DECODE(B) case B: {optimized::AlignedLane<B,true> tile(p+blockIdx.x*B*8); \
      states[base+0]=tile.template state<0>();states[base+1]=tile.template state<1>(); \
      states[base+2]=tile.template state<2>();states[base+3]=tile.template state<3>(); \
      states[base+4]=tile.template state<4>();states[base+5]=tile.template state<5>(); \
      states[base+6]=tile.template state<6>();states[base+7]=tile.template state<7>();break;}
    switch(bits){DECODE(1);DECODE(2);DECODE(3);DECODE(4);DECODE(5);DECODE(6);DECODE(7);DECODE(8);}
    #undef DECODE
    for(int j=0;j<8;j+=2){
        if(cb==2){__half2 h=optimized::mul1_half2(states[base+j],states[base+j+1]);uint32_t v=__builtin_bit_cast(uint32_t,h);values[base+j]=v;values[base+j+1]=v>>16;}
        else for(int d=0;d<2;++d)values[base+j+d]=__half_as_ushort(__float2half_rn(codebook_value(states[base+j+d],cb)));
    }
}
__global__ void exhaustive_books(uint16_t* out) {
    unsigned s=blockIdx.x*blockDim.x+threadIdx.x;
    for(int cb=0;cb<2;++cb)out[cb*65536+s]=__half_as_ushort(__float2half_rn(codebook_value(s,cb)));
    uint32_t v=__builtin_bit_cast(uint32_t,optimized::mul1_half2(s,65535-s));
    out[2*65536+s]=v;out[3*65536+s]=v>>16;
}
static void decode_checks(){
    uint16_t* d;HIP(hipMalloc(&d,4*65536*2));exhaustive_books<<<256,256>>>(d);
    std::vector<uint16_t> h(4*65536);HIP(hipMemcpy(h.data(),d,h.size()*2,hipMemcpyDeviceToHost));HIP(hipFree(d));
    for(int cb=0;cb<4;++cb)for(int s=0;s<65536;++s)
        if(h[cb*65536+s]!=host_codebook(cb==3?65535-s:s,std::min(cb,2))){fprintf(stderr,"codebook mismatch\n");exit(3);}
    for(int bits=1;bits<=8;++bits){
        const int tiles=257;std::vector<uint32_t> p(tiles*bits*8);for(auto& v:p)v=random_word();
        uint32_t* dp;uint16_t *ds,*dv;HIP(hipMalloc(&dp,p.size()*4));HIP(hipMalloc(&ds,tiles*256*2));HIP(hipMalloc(&dv,tiles*256*2));HIP(hipMemcpy(dp,p.data(),p.size()*4,hipMemcpyHostToDevice));
        std::vector<uint16_t> ss(tiles*256),vv(tiles*256);
        for(int cb=0;cb<3;++cb){validate_decode<<<tiles,32>>>(dp,ds,dv,bits,cb);HIP(hipMemcpy(ss.data(),ds,ss.size()*2,hipMemcpyDeviceToHost));HIP(hipMemcpy(vv.data(),dv,vv.size()*2,hipMemcpyDeviceToHost));
            for(int t=0;t<tiles;++t)for(int j=0;j<256;++j){auto s=host_state(p.data()+t*bits*8,bits,j);if(ss[t*256+j]!=s||vv[t*256+j]!=host_codebook(s,cb)){fprintf(stderr,"decode mismatch K%d cb%d\n",bits,cb);exit(3);}}
        }HIP(hipFree(dp));HIP(hipFree(ds));HIP(hipFree(dv));
    }
    puts("{\"validation\":\"all_65536_states_all_codebooks_both_mul1_halves_and_257_tiles_each_K1_to_8_cb0_to_2\",\"pass\":true}");
}
struct Shape{const char* name;int i,n,bits;};
static void check(const Shape& s,int cb,int rows,int splits,const std::vector<uint32_t>& packed,
    const std::vector<_Float16>& x,const DeviceWeights& w,const _Float16* dx,float* dy,bool full){
    int total=rows+2;size_t count=size_t(splits)*total*s.n;
    std::vector<float> result(count,1234567);HIP(hipMemcpy(dy,result.data(),count*4,hipMemcpyHostToDevice));
    HIP(launch_gemv_core(w,dx,dy+s.n,rows,total,splits,nullptr));
    HIP(hipMemcpy(result.data(),dy,count*4,hipMemcpyDeviceToHost));
    double err2=0,ref2=0,max_abs=0,max_norm=0;int checks=0;
    for(int sp=0;sp<splits;++sp)for(int c=0;c<s.n;++c)
        if(result[(size_t(sp)*total)*s.n+c]!=1234567||result[(size_t(sp)*total+rows+1)*s.n+c]!=1234567){fprintf(stderr,"chunk overwrite\n");exit(3);}
    int samples=full?s.n:32;
    for(int r=0;r<rows;++r)for(int sample=0;sample<samples;++sample){
        int c=full?sample:int((uint64_t(sample)*7919+11)%s.n);
        double sum=0,abs_sum=0;float got=0;
        for(int sp=0;sp<splits;++sp){
            double ref=0,ab=0;int begin=(s.i/16)*sp/splits,end=(s.i/16)*(sp+1)/splits;
            for(int k=begin*16;k<end*16;++k){
                int t=(c%16)%8*4+(k%8)/2;
                int j=(k&1)+((k%16)/8)*2+((c%16)/8)*4;
                size_t tile=size_t(k/16)*(s.n/16)+c/16;
                uint16_t state=host_state(packed.data()+tile*s.bits*8,s.bits,8*t+j);
                double product=double(x[size_t(r)*s.i+k])*host_half(host_codebook(state,cb));ref+=product;ab+=std::abs(product);
            }
            float v=result[(size_t(sp)*total+r+1)*s.n+c];
            double gamma=(2.0*s.i*std::ldexp(1.0,-24))/(1-2.0*s.i*std::ldexp(1.0,-24));
            if(!std::isfinite(v)||std::abs(double(v)-ref)>gamma*ab+1e-6){fprintf(stderr,"dot mismatch %s cb%d R%d S%d r%d c%d sp%d got%.9g ref%.17g\n",s.name,cb,rows,splits,r,c,sp,v,ref);exit(3);}
            sum+=ref;abs_sum+=ab;got+=v;
        }
        double error=double(got)-sum;err2+=error*error;ref2+=sum*sum;max_abs=std::max(max_abs,std::abs(error));max_norm=std::max(max_norm,std::abs(error)/std::max(abs_sum,1e-300));++checks;
    }
    printf("{\"correctness\":\"%s\",\"K\":%d,\"cb\":%d,\"rows\":%d,\"splits\":%d,\"full_outputs\":%s,\"outputs\":%d,\"max_abs\":%.9g,\"max_error_over_sum_abs\":%.9g,\"nrms\":%.9g,\"pass\":true}\n",s.name,s.bits,cb,rows,splits,full?"true":"false",checks,max_abs,max_norm,std::sqrt(err2/std::max(ref2,1e-300)));
}
static void shape(const Shape& s,bool small){
    size_t bytes=size_t(s.i)*s.n*s.bits/8;
    int clones=small?1:int(((size_t(256)<<20)+bytes-1)/bytes);
    std::vector<uint32_t> packed(bytes/4);for(auto& v:packed)v=random_word();
    std::vector<_Float16> x(size_t(8)*s.i);for(auto& v:x)v=_Float16((int(random_word()%2049)-1024)/1024.0f);
    uint32_t *dp,*sink;_Float16 *dx_alloc,*dx;float* dy;
    HIP(hipMalloc(&dp,bytes*clones));HIP(hipMalloc(&dx_alloc,x.size()*2+2));dx=dx_alloc+1;
    HIP(hipMalloc(&dy,size_t(32)*10*s.n*4));HIP(hipMalloc(&sink,1024*4));
    for(int c=0;c<clones;++c)HIP(hipMemcpy(reinterpret_cast<char*>(dp)+size_t(c)*bytes,packed.data(),bytes,hipMemcpyHostToDevice));
    HIP(hipMemcpy(dx,x.data(),x.size()*2,hipMemcpyHostToDevice));
    DeviceWeights w{dp,nullptr,nullptr,nullptr,s.i,s.n,s.bits,2};
    for(int cb=small?0:2;cb<3;++cb){w.codebook=cb;
        for(int rows:small?std::vector<int>{1,2,3,4,5,6,7,8}:std::vector<int>{1,2,4,8}){
            for(int splits:small?std::vector<int>{1,3}:std::vector<int>{1,4,16,32}){
                check(s,cb,rows,splits,packed,x,w,dx,dy,small);
                if(small)continue;
                hipEvent_t start,stop;HIP(hipEventCreate(&start));HIP(hipEventCreate(&stop));std::vector<float> times;
                // Each observation touches EVERY clone. Discard one entire resident rotation.
                for(int rep=-1;rep<11;++rep){HIP(hipEventRecord(start));for(int c=0;c<clones;++c){w.trellis=dp+size_t(c)*bytes/4;HIP(launch_gemv_core(w,dx,dy,rows,rows,splits,nullptr));}HIP(hipEventRecord(stop));HIP(hipEventSynchronize(stop));float ms;HIP(hipEventElapsedTime(&ms,start,stop));if(rep>=0)times.push_back(ms/clones);}
                w.trellis=dp;std::sort(times.begin(),times.end());
                size_t minimum=bytes+size_t(rows)*s.i*2+size_t(splits)*rows*s.n*4;
                printf("{\"bench\":\"%s\",\"I\":%d,\"N\":%d,\"K\":%d,\"rows\":%d,\"splits\":%d,\"clones\":%d,\"rotation_bytes\":%zu,\"sweeps\":11,\"compressed_bytes\":%zu,\"minimum_bytes\":%zu,\"min_ms\":%.9g,\"median_ms\":%.9g,\"p90_ms\":%.9g,\"compressed_GBs\":%.9g,\"minimum_GBs\":%.9g,\"sweep_ms\":[",s.name,s.i,s.n,s.bits,rows,splits,clones,bytes*clones,bytes,minimum,times[0],times[5],times[9],bytes/(times[5]*1e6),minimum/(times[5]*1e6));
                for(size_t j=0;j<times.size();++j)printf("%s%.9g",j?",":"",times[j]*clones);puts("]}");
                HIP(hipEventDestroy(start));HIP(hipEventDestroy(stop));
            }
        }
    }
    if(!small){hipEvent_t a,b;HIP(hipEventCreate(&a));HIP(hipEventCreate(&b));std::vector<float> times;
        for(int rep=-1;rep<11;++rep){HIP(hipEventRecord(a));for(int c=0;c<clones;++c)read_contiguous<<<1024,256>>>(reinterpret_cast<uint4*>(dp+size_t(c)*bytes/4),bytes/16,sink);HIP(hipEventRecord(b));HIP(hipEventSynchronize(b));float ms;HIP(hipEventElapsedTime(&ms,a,b));if(rep>=0)times.push_back(ms/clones);}
        std::vector<uint32_t> h(1024);HIP(hipMemcpy(h.data(),sink,h.size()*4,hipMemcpyDeviceToHost));uint32_t got=0,ref=0;for(auto v:h)got+=v;for(auto v:packed)ref+=v;if(got!=ref){fprintf(stderr,"read checksum mismatch\n");exit(3);}
        std::sort(times.begin(),times.end());printf("{\"bench\":\"contiguous_read\",\"shape\":\"%s\",\"clones\":%d,\"rotation_bytes\":%zu,\"min_ms\":%.9g,\"median_ms\":%.9g,\"p90_ms\":%.9g,\"compressed_GBs\":%.9g,\"checksum\":%u}\n",s.name,clones,bytes*clones,times[0],times[5],times[9],bytes/(times[5]*1e6),got);HIP(hipEventDestroy(a));HIP(hipEventDestroy(b));
    }
    HIP(hipFree(dp));HIP(hipFree(dx_alloc));HIP(hipFree(dy));HIP(hipFree(sink));
}
int main(int argc,char** argv){
    setvbuf(stdout,nullptr,_IOLBF,0);hipDeviceProp_t p{};HIP(hipGetDeviceProperties(&p,0));
    printf("{\"device\":\"%s\",\"arch\":\"%s\",\"warp\":%d,\"multiProcessorCount\":%d,\"clock_kHz\":%d,\"memory_clock_kHz\":%d,\"memory_bus_bits\":%d}\n",p.name,p.gcnArchName,p.warpSize,p.multiProcessorCount,p.clockRate,p.memoryClockRate,p.memoryBusWidth);
    decode_checks();for(int bits=1;bits<=8;++bits)shape({"small",128,128,bits},true);
    if(argc>1&&std::strcmp(argv[1],"--check-only")==0)return 0;
    for(auto s:std::vector<Shape>{{"wq_b",1280,32768,5},{"wo_b",8192,5120,5},{"shared",5120,2304,5},{"shared_inverse",2304,5120,5},{"wkv",5120,512,6}})shape(s,false);
    return 0;
}
