// Standalone correctness and warm-cache timing harness; no expanded GPU weights.
#include "gemm.hpp"
#include "host_reference.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <cmath>
using namespace strix_trellis;
#define HIP(call) do { hipError_t e=(call); if(e!=hipSuccess) { std::fprintf(stderr,"%s: %s\n",#call,hipGetErrorString(e)); std::exit(1); } } while(0)
namespace strix_trellis {
hipError_t launch_wmma_reuse_variant(const DeviceWeights&, const _Float16*, float*, int, int, hipStream_t, int);
}
// Selectors: 0=production, 16/64/128=plain, -64=Q XOR, -65=Q XOR+A18.
static int tile_m=16;
static int tile_rows(int rows,int outputs,int bits,int cb) {
    if (tile_m == -65) return 64;
    if (tile_m) return std::abs(tile_m);
    return (rows>=256 && outputs>=2304) || (rows>=64 && outputs>=32768 && bits==5 && cb==2) ? 64 : 16;
}
static hipError_t selected(const DeviceWeights& w,const _Float16* a,float* p,int m,int s,hipStream_t stream) {
    return launch_wmma_reuse_variant(w,a,p,m,s,stream,tile_m);
}
static uint32_t rng=0x12345678;
static uint32_t next() { rng^=rng<<13; rng^=rng>>17; rng^=rng<<5; return rng; }
template<class T> static T* device(size_t n) { T* p; HIP(hipMalloc(&p,n*sizeof(T))); return p; }
static float weight(const std::vector<uint32_t>& p,int i,int o,int inputs,int outputs,int bits,int cb) {
    (void)inputs;
    int r=i%16,c=o%16;
    int lane=(c%8)*4+(r%8)/2;
    int j=(r%2)+2*(r/8)+4*(c/8);
    const uint32_t* tile=p.data()+(size_t(i/16)*(outputs/16)+o/16)*bits*8;
    return host_half(host_codebook(host_state(tile,bits,8*lane+j),cb));
}
struct Case {
    int m,i,o,b,cb;
    std::vector<uint32_t> packed;
    std::vector<_Float16> a;
    uint32_t* dp; _Float16* da;
    Case(int rows,int inputs,int outputs,int bits,int codebook,bool impulse=false):
        m(rows),i(inputs),o(outputs),b(bits),cb(codebook),packed(size_t(i)*o*b/32),a(size_t(m)*i) {
        for(auto& x:packed)x=next();
        for(int row=0;row<m;++row)for(int k=0;k<i;++k)
            a[size_t(row)*i+k]=impulse ? _Float16(k==row%i ? (row%2?-1.0f:1.0f):0.0f) : _Float16((int(next()%257)-128)/128.0f);
        dp=device<uint32_t>(packed.size()); da=device<_Float16>(a.size());
        HIP(hipMemcpy(dp,packed.data(),packed.size()*4,hipMemcpyHostToDevice));
        HIP(hipMemcpy(da,a.data(),a.size()*2,hipMemcpyHostToDevice));
    }
    ~Case(){HIP(hipFree(dp));HIP(hipFree(da));}
    DeviceWeights desc(){return {dp,nullptr,nullptr,nullptr,i,o,b,cb};}
};
static void check(Case& c,int splits,bool sampled) {
    size_t n=size_t(splits)*c.m*c.o;
    float* out=device<float>(n+64); HIP(hipMemset(out,0x7f,(n+64)*4));
    HIP(selected(c.desc(),c.da,out,c.m,splits,nullptr));
    std::vector<float> got(n+64); HIP(hipMemcpy(got.data(),out,(n+64)*4,hipMemcpyDeviceToHost));
    double maxerr=0,maxnorm=0,maxscalar=0; size_t checked=0;
    // Decode only selected columns on host; never materialize Q on GPU.
    std::vector<int> columns;
    if(sampled) { for(int j=0;j<16;++j)columns.push_back((j*997+31)%c.o); }
    else { for(int o=0;o<c.o;++o)columns.push_back(o); }
    std::vector<float> q(size_t(c.i)*columns.size());
    for(int k=0;k<c.i;++k)for(size_t col=0;col<columns.size();++col)
        q[size_t(k)*columns.size()+col]=weight(c.packed,k,columns[col],c.i,c.o,c.b,c.cb);
    for(int s=0;s<splits;++s)for(int r=0;r<c.m;++r) {
        if(sampled && r!=0 && r!=c.m/2 && r!=c.m-1)continue;
        int begin=(c.i/16)*s/splits*16,end=(c.i/16)*(s+1)/splits*16;
        for(size_t col=0;col<columns.size();++col) {
            double ref=0,scale=0; float scalar=0;
            for(int k=begin;k<end;++k) {
                double product=double(c.a[size_t(r)*c.i+k])*q[size_t(k)*columns.size()+col];
                ref+=product;scale+=std::abs(product);scalar=scalar+float(product);
            }
            float value=got[(size_t(s)*c.m+r)*c.o+columns[col]];
            double err=std::abs(double(value)-ref), norm=err/(scale+1e-30);
            double nu=2.0*(end-begin)*std::ldexp(1.0,-24);
            double tolerance=nu/(1-nu)*scale+1e-6;
            if(!std::isfinite(value)||err>tolerance) {
                std::fprintf(stderr,"FAIL tile_m=%d m=%d i=%d o=%d bits=%d cb=%d split=%d r=%d c=%d got=%.9g ref=%.17g err=%g bound=%g\n",tile_m,c.m,c.i,c.o,c.b,c.cb,s,r,columns[col],value,ref,err,tolerance);std::exit(2);
            }
            maxerr=std::max(maxerr,err);maxnorm=std::max(maxnorm,norm);maxscalar=std::max(maxscalar,std::abs(double(value)-scalar));++checked;
        }
    }
    for(size_t j=n;j<n+64;++j) { uint32_t bits;std::memcpy(&bits,&got[j],4);if(bits!=0x7f7f7f7f)std::exit(3); }
    HIP(selected(c.desc(),c.da,out,c.m,splits,nullptr));
    std::vector<float> again(n);HIP(hipMemcpy(again.data(),out,n*4,hipMemcpyDeviceToHost));
    if(std::memcmp(again.data(),got.data(),n*4))std::exit(4);
    std::printf("{\"test\":\"core\",\"dispatcher\":%s,\"a_stride\":%d,\"q_swizzle\":%s,\"tile_M\":%d,\"M\":%d,\"I\":%d,\"O\":%d,\"bits\":%d,\"cb\":%d,\"splits\":%d,\"checked\":%zu,\"sampled\":%s,\"max_abs\":%.9g,\"max_error_over_sumabs\":%.9g,\"max_vs_scalar_fp32\":%.9g,\"deterministic\":true}\n",tile_m==0?"true":"false",(tile_m==-65 || (tile_m==0&&tile_rows(c.m,c.o,c.b,c.cb)==64&&c.b==5&&c.cb==2))?18:16,(tile_m<0||(tile_m==0&&tile_rows(c.m,c.o,c.b,c.cb)==64))?"true":"false",tile_rows(c.m,c.o,c.b,c.cb),c.m,c.i,c.o,c.b,c.cb,splits,checked,sampled?"true":"false",maxerr,maxnorm,maxscalar);
    HIP(hipFree(out));
}
static void bench(Case& c) {
    float* out=device<float>(size_t(c.m)*c.o);
    auto launch=[&](){HIP(selected(c.desc(),c.da,out,c.m,1,nullptr));};
    for(int j=0;j<2;++j)launch();HIP(hipDeviceSynchronize());
    hipEvent_t start,end;HIP(hipEventCreate(&start));HIP(hipEventCreate(&end));
    std::vector<float> times;
    for(int j=0;j<11;++j){HIP(hipEventRecord(start));launch();HIP(hipEventRecord(end));HIP(hipEventSynchronize(end));float ms;HIP(hipEventElapsedTime(&ms,start,end));times.push_back(ms);}
    std::sort(times.begin(),times.end());
    size_t packed=c.packed.size()*4, logical_reads=packed*size_t((c.m+tile_rows(c.m,c.o,c.b,c.cb)-1)/tile_rows(c.m,c.o,c.b,c.cb)),decoded=size_t(c.i)*c.o*((c.m+tile_rows(c.m,c.o,c.b,c.cb)-1)/tile_rows(c.m,c.o,c.b,c.cb));
    std::printf("{\"bench\":\"fused_native_wmma_reuse\",\"dispatcher\":%s,\"a_stride\":%d,\"q_swizzle\":%s,\"tile_M\":%d,\"M\":%d,\"I\":%d,\"O\":%d,\"bits\":%d,\"cb\":%d,\"splits\":1,\"median_ms\":%.6f,\"min_ms\":%.6f,\"max_ms\":%.6f,\"useful_TOPS\":%.6f,\"packed_bytes\":%zu,\"logical_packed_read_bytes\":%zu,\"decoded_weights\":%zu,\"output_bytes\":%zu,\"cache\":\"warm_single_matrix\",\"warmups\":2,\"repetitions\":11}\n",tile_m==0?"true":"false",(tile_m==-65 || (tile_m==0&&tile_rows(c.m,c.o,c.b,c.cb)==64&&c.b==5&&c.cb==2))?18:16,(tile_m<0||(tile_m==0&&tile_rows(c.m,c.o,c.b,c.cb)==64))?"true":"false",tile_rows(c.m,c.o,c.b,c.cb),c.m,c.i,c.o,c.b,c.cb,times[5],times[0],times[10],2.0*c.m*c.i*c.o/times[5]/1e9,packed,logical_reads,decoded,size_t(c.m)*c.o*4);
    HIP(hipEventDestroy(start));HIP(hipEventDestroy(end));HIP(hipFree(out));
}
int main(int argc,char**) {
    hipDeviceProp_t prop; HIP(hipGetDeviceProperties(&prop,0));
    std::printf("{\"hardware\":\"%s\",\"arch\":\"%s\",\"warp\":%d,\"multiProcessorCount\":%d,\"seed\":305419896}\n",prop.name,prop.gcnArchName,prop.warpSize,prop.multiProcessorCount);
    for(int tm:{16,-64,-65,0}) {
        tile_m=tm;
        for(int b=1;b<=8;++b)for(int cb=0;cb<3;++cb)for(int rows:{1,17,32,65,129}) {
            Case c(rows,384,128,b,cb);for(int splits:{1,3,29})check(c,splits,false);
        }
        for(int cb=0;cb<3;++cb){Case c(129,128,256,5,cb,true);check(c,1,false);check(c,3,false);}
    }
    if(argc>1)for(int geometry=0;geometry<3;++geometry)for(int m:{16,64,256,512,1024}) {
        int i=geometry==0?1280:geometry==1?8192:5120;
        int o=geometry==0?32768:geometry==1?5120:2304;
        Case c(m,i,o,5,2);
        for(int tm:{16,-64,-65,0}) {tile_m=tm;check(c,1,true);bench(c);}
    }
    if(argc>1)for(int b=1;b<=8;++b)for(int cb=0;cb<3;++cb) {
        Case c(256,5120,2304,b,cb);
        for(int tm:{16,0}) {tile_m=tm;check(c,1,true);bench(c);}
    }
    std::puts("{\"result\":\"PASS\"}");
}
