#include "trellis_optimized.hpp"
#include "host_reference.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <fstream>
#include <iterator>
#include <type_traits>
#define HIP(x) do { auto e=(x); if(e!=hipSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);} }while(0)
using namespace strix_trellis;
using namespace strix_trellis::optimized;
__device__ uint32_t half2_bits(__half2 h) { union { __half2 h; uint32_t u; } v; v.h=h; return v.u; }
__device__ uint32_t mix(uint32_t x, uint32_t y) { return ((x << 5) | (x >> 27)) ^ y; }
template<int B, int Variant> __device__ void decode(const uint32_t* data, uint32_t (&states)[8], uint32_t (&pairs)[4]) {
    if constexpr(Variant>=3) {
        using Reader = typename std::conditional<Variant==3, DirectLane<B>, AlignedLane<B, Variant==5>>::type;
        Reader tile(data);
        states[0]=tile.template state<0>(); states[1]=tile.template state<1>();
        states[2]=tile.template state<2>(); states[3]=tile.template state<3>();
        states[4]=tile.template state<4>(); states[5]=tile.template state<5>();
        states[6]=tile.template state<6>(); states[7]=tile.template state<7>();
    } else {
        WaveTile<B> tile(data);
        constexpr int G=B<=4?8:4;
        #pragma unroll
        for(int base=0;base<8;base+=G) {
            uint64_t w=tile.template group<G>((threadIdx.x&31)*8+base);
            #pragma unroll
            for(int j=0;j<G;++j) states[base+j]=uint32_t(w>>((G-1-j)*B))&65535u;
        }
    }
    #pragma unroll
    for(int j=0;j<4;++j) {
        if constexpr(Variant>=2) pairs[j]=half2_bits(mul1_half2(states[2*j],states[2*j+1]));
        else {
            uint32_t a,b;
            if constexpr(Variant==1) { a=__half_as_ushort(mul1_half(states[2*j])); b=__half_as_ushort(mul1_half(states[2*j+1])); }
            else { a=__half_as_ushort(__float2half_rn(codebook_value(states[2*j],2))); b=__half_as_ushort(__float2half_rn(codebook_value(states[2*j+1],2))); }
            pairs[j]=a|(b<<16);
        }
    }
}
template<int B,int V> __global__ void scan(const uint32_t* data,uint32_t* out,size_t tiles,uint16_t* states=nullptr,uint16_t* values=nullptr) {
    size_t t=(size_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    if(t>=tiles)return;
    uint32_t s[8],p[4];decode<B,V>(data+t*B*8,s,p);
    uint32_t sum=0;
    #pragma unroll
    for(int j=0;j<4;++j)sum=mix(sum,p[j]);
    if(states) {
        #pragma unroll
        for(int j=0;j<8;++j) {size_t i=t*256+(threadIdx.x&31)*8+j;states[i]=s[j];values[i]=p[j/2]>>(16*(j&1));}
    }
    for(int d=16;d;d/=2)sum^=__shfl_down(sum,d,32);
    if((threadIdx.x&31)==0)out[t]=sum;
}
__global__ void exhaustive(uint16_t* out) {
    uint32_t s=blockIdx.x*blockDim.x+threadIdx.x;
    if(s<65536) {out[s]=__half_as_ushort(mul1_half(s));uint32_t p=half2_bits(mul1_half2(s,65535-s));out[65536+s]=p;out[131072+s]=p>>16;}
}
__global__ void read_only(const uint4* data,uint32_t* out,size_t n) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;uint32_t s=0;
    for(;i<n;i+=size_t(gridDim.x)*blockDim.x){uint4 v=data[i];s^=v.x^v.y^v.z^v.w;}
    for(int d=16;d;d/=2)s^=__shfl_down(s,d,32);
    if((threadIdx.x&31)==0)out[(size_t(blockIdx.x)*blockDim.x+threadIdx.x)/32]=s;
}
template<int B,int V> void launch(const uint32_t* data,uint32_t* out,size_t n,uint16_t* s=nullptr,uint16_t* w=nullptr){scan<B,V><<<(n+7)/8,256>>>(data,out,n,s,w);HIP(hipGetLastError());}
template<int B> void dispatch(int v,const uint32_t* d,uint32_t* o,size_t n,uint16_t* s=nullptr,uint16_t* w=nullptr) {
    if(v==0)launch<B,0>(d,o,n,s,w);if(v==1)launch<B,1>(d,o,n,s,w);if(v==2)launch<B,2>(d,o,n,s,w);if(v==3)launch<B,3>(d,o,n,s,w);if(v==4)launch<B,4>(d,o,n,s,w);if(v==5)launch<B,5>(d,o,n,s,w);
}
static uint64_t hash(const std::vector<uint32_t>& v) {uint64_t h=14695981039346656037ull;for(uint32_t x:v){h^=x;h*=1099511628211ull;}return h;}
template<int B> void run(size_t target,int repeats,int first,int last) {
    size_t n=(target+32*B-1)/(32*B),bytes=n*32*B;
    std::vector<uint32_t> host(bytes/4);uint32_t rng=0x12345678;
    for(auto& x:host){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;x=rng;}
    uint32_t *data,*out;uint16_t *states,*values;
    HIP(hipMalloc(&data,bytes));HIP(hipMalloc(&out,n*4));HIP(hipMalloc(&states,257*512));HIP(hipMalloc(&values,257*512));
    HIP(hipMemcpy(data,host.data(),bytes,hipMemcpyHostToDevice));
    std::vector<uint32_t> baseline(n),got(n);
    hipEvent_t a,z;HIP(hipEventCreate(&a));HIP(hipEventCreate(&z));
    if(first>0 && first<6) { launch<B,0>(data,out,n);HIP(hipMemcpy(baseline.data(),out,n*4,hipMemcpyDeviceToHost)); }
    for(int v=first;v<=last;++v) {
        if(v<6){
            dispatch<B>(v,data,out,257,states,values);
            std::vector<uint16_t> s(257*256),w(s.size());
            HIP(hipMemcpy(s.data(),states,s.size()*2,hipMemcpyDeviceToHost));HIP(hipMemcpy(w.data(),values,w.size()*2,hipMemcpyDeviceToHost));
            for(size_t t=0;t<257;++t)for(int j=0;j<256;++j){unsigned ref=host_state(host.data()+t*B*8,B,j);if(s[t*256+j]!=ref||w[t*256+j]!=host_codebook(ref,2)){fprintf(stderr,"FAIL B%d V%d tile%zu pos%d\n",B,v,t,j);exit(2);}}
        }
        auto go=[&](){if(v==6)read_only<<<512,256>>>((const uint4*)data,out,bytes/16);else dispatch<B>(v,data,out,n);HIP(hipGetLastError());};
        go();HIP(hipDeviceSynchronize());std::vector<float> times;
        for(int r=0;r<repeats;++r){HIP(hipEventRecord(a));go();HIP(hipEventRecord(z));HIP(hipEventSynchronize(z));float ms;HIP(hipEventElapsedTime(&ms,a,z));times.push_back(ms);}
        std::sort(times.begin(),times.end());size_t outputs=v==6?4096:n;
        HIP(hipMemcpy(got.data(),out,outputs*4,hipMemcpyDeviceToHost));
        if(v==0)baseline=got;else if(v<6 && got!=baseline){fprintf(stderr,"FAIL whole-buffer checksum\n");exit(2);}
        auto copy=got;copy.resize(outputs);float ms=times[times.size()/2];
        printf("{\"bits\":%d,\"variant\":%d,\"compressed_bytes\":%zu,\"minimum_traffic_bytes\":%zu,\"repeats\":%d,\"median_ms\":%.6f,\"min_ms\":%.6f,\"max_ms\":%.6f,\"packed_GBs\":%.3f,\"minimum_traffic_GBs\":%.3f,\"checksum_fnv_u32\":\"%016llx\",\"validation\":\"257_tiles_states_fp16_and_full_checksum\"}\n",B,v,bytes,bytes+outputs*4,repeats,ms,times.front(),times.back(),double(bytes)/ms/1e6,double(bytes+outputs*4)/ms/1e6,(unsigned long long)hash(copy));fflush(stdout);
    }
    HIP(hipEventDestroy(a));HIP(hipEventDestroy(z));HIP(hipFree(states));HIP(hipFree(values));HIP(hipFree(data));HIP(hipFree(out));
}
static int fixture(const char* input,const char* ref_states,const char* ref_values) {
    auto read=[](const char* path){std::ifstream f(path,std::ios::binary);if(!f){fprintf(stderr,"Cannot read fixture %s\n",path);exit(2);}return std::vector<char>(std::istreambuf_iterator<char>(f),{});};
    auto h=read(input),rs=read(ref_states),rw=read(ref_values);
    if(h.empty()||h.size()%192||rs.size()!=h.size()/192*512||rw.size()!=rs.size())return 2;
    size_t n=h.size()/192;uint32_t *d,*out;uint16_t *st,*wt;
    HIP(hipMalloc(&d,h.size()));HIP(hipMalloc(&out,n*4));HIP(hipMalloc(&st,rs.size()));HIP(hipMalloc(&wt,rw.size()));
    HIP(hipMemcpy(d,h.data(),h.size(),hipMemcpyHostToDevice));
    for(int v=0;v<6;++v){dispatch<6>(v,d,out,n,st,wt);std::vector<char> gs(rs.size()),gw(rw.size());HIP(hipMemcpy(gs.data(),st,gs.size(),hipMemcpyDeviceToHost));HIP(hipMemcpy(gw.data(),wt,gw.size(),hipMemcpyDeviceToHost));if(gs!=rs||gw!=rw){fprintf(stderr,"FAIL real K6 variant %d\n",v);return 2;}printf("{\"fixture\":\"real_K6\",\"variant\":%d,\"tiles\":%zu,\"states_and_fp16\":\"bit_exact\"}\n",v,n);}
    HIP(hipFree(d));HIP(hipFree(out));HIP(hipFree(st));HIP(hipFree(wt));return 0;
}
int main(int argc,char** argv) {
    if(argc==5 && !strcmp(argv[1],"--fixture"))return fixture(argv[2],argv[3],argv[4]);
    int first=argc>3?atoi(argv[3]):0,last=argc>4?atoi(argv[4]):6;
    size_t mib=argc>1?strtoull(argv[1],nullptr,10):256;int reps=argc>2?atoi(argv[2]):11;
    if(first<0||last>6||first>last)return 2;
    if(mib<256||mib>2048||reps<11||reps>101)return 2;
    hipDeviceProp_t p;HIP(hipGetDeviceProperties(&p,0));
    printf("{\"device\":\"%s\",\"arch\":\"%s\",\"hip_multiprocessors_WGP\":%d,\"physical_CU_expected\":40,\"wave\":%d}\n",p.name,p.gcnArchName,p.multiProcessorCount,p.warpSize);
    if(p.warpSize!=32)return 2;
    uint16_t* d;HIP(hipMalloc(&d,3*65536*2));exhaustive<<<256,256>>>(d);HIP(hipGetLastError());
    std::vector<uint16_t> h(3*65536);HIP(hipMemcpy(h.data(),d,h.size()*2,hipMemcpyDeviceToHost));HIP(hipFree(d));
    for(unsigned s=0;s<65536;++s)if(h[s]!=host_codebook(s,2)||h[65536+s]!=host_codebook(s,2)||h[131072+s]!=host_codebook(65535-s,2)){fprintf(stderr,"FAIL exhaustive %u\n",s);return 2;}
    puts("{\"exhaustive_mul1_states\":65536,\"scalar_half_and_both_half2_lanes\":\"bit_exact\"}");fflush(stdout);
    run<5>(mib*1024*1024,reps,first,last);run<6>(mib*1024*1024,reps,first,last);run<2>(mib*1024*1024,reps,first,last);run<3>(mib*1024*1024,reps,first,last);run<4>(mib*1024*1024,reps,first,last);run<8>(mib*1024*1024,reps,first,last);
}
