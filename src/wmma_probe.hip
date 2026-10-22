// Independent small WMMA diagnostic; no large GEMM and no tolerance.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define HIP(call) do { auto s=(call); if(s!=hipSuccess) { std::fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(s)); std::exit(1); } } while(0)
using half16 = _Float16 __attribute__((ext_vector_type(16)));
using float8 = float __attribute__((ext_vector_type(8)));
template<int Mode>
__global__ void probe(const _Float16* a,const _Float16* b,float* out) {
    int lane=threadIdx.x, col=lane%16, off=blockIdx.x*256;
    half16 left,right;
    for(int k=0;k<16;++k) { left[k]=a[off+col*16+k]; right[k]=b[off+k*16+col]; }
    float8 acc={};
    if constexpr(Mode==0) acc=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(left,right,acc);
    if constexpr(Mode==1) asm volatile("v_wmma_f32_16x16x16_f16 %0, %1, %2, %3" : "=v"(acc) : "v"(left), "v"(right), "v"(acc));
    if constexpr(Mode==2) {
        // Isolate each K product, then add in ordinary FP32 (diagnostic only).
        for(int k=0;k<16;++k) {
            half16 one={}; one[k]=left[k]; float8 zero={};
            float8 part=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(one,right,zero);
            acc+=part;
        }
    }
    if constexpr(Mode==3) {
        for(int j=0;j<8;++j) {
            int row=2*j+lane/16; float sum=0;
            for(int k=0;k<16;++k) sum=fmaf(float(a[off+row*16+k]),float(b[off+k*16+col]),sum);
            acc[j]=sum;
        }
    }
    if constexpr(Mode==4) {
        // Separate product signs without changing input/output mapping.
        half16 ap={},an={},bp={},bn={};
        for(int k=0;k<16;++k) {
            ap[k]=left[k]>0?left[k]:_Float16(0); an[k]=left[k]<0?-left[k]:_Float16(0);
            bp[k]=right[k]>0?right[k]:_Float16(0); bn[k]=right[k]<0?-right[k]:_Float16(0);
        }
        float8 z={};
        float8 pp=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ap,bp,z);
        float8 nn=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(an,bn,z);
        float8 pn=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ap,bn,z);
        float8 np=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(an,bp,z);
        acc=(pp+nn)-(pn+np);
    }
    for(int j=0;j<8;++j) out[off+(2*j+lane/16)*16+col]=acc[j];
}
int main(int argc,char**) {
    constexpr int cases=513;
    std::vector<_Float16> a(cases*256),b(a.size());
    std::vector<float> legacy(a.size()),got(a.size());
    std::vector<int> ia(a.size()),ib(a.size()),exact(a.size());
    std::vector<double> ref(a.size());
    for(int c=0;c<cases;++c) {
        for(int i=0;i<16;++i) {a[c*256+i*16+i]=1; b[c*256+i*16+i]=1;}
        if(c<256) {std::fill(a.begin()+c*256,a.begin()+(c+1)*256,0); a[c*256+c]=2;}
        else if(c<512) {std::fill(b.begin()+c*256,b.begin()+(c+1)*256,0); b[c*256+c-256]=-3;}
        else for(int i=0;i<256;++i) {a[c*256+i]=_Float16((i*7%13)-6);b[c*256+i]=_Float16((i*11%17)-8);}
        for(int r=0;r<16;++r) for(int n=0;n<16;++n) for(int k=0;k<16;++k) legacy[c*256+r*16+n]+=float(a[c*256+r*16+k])*float(b[c*256+k*16+n]);
    }
    // Generate integer operands independently, without reading the half arrays.
    for(int c=0;c<cases;++c) for(int r=0;r<16;++r) for(int n=0;n<16;++n) {
        int p=c*256+r*16+n;
        ia[p]=c<256 ? (r*16+n==c ? 2:0) : c<512 ? int(r==n) : ((r*16+n)*7%13)-6;
        ib[p]=c<256 ? int(r==n) : c<512 ? (r*16+n==c-256 ? -3:0) : ((r*16+n)*11%17)-8;
        if(float(a[p])!=ia[p] || float(b[p])!=ib[p]) {std::fprintf(stderr,"input mismatch %d\n",p);return 2;}
    }
    for(int c=0;c<cases;++c) for(int k=0;k<16;++k) for(int r=0;r<16;++r) for(int n=0;n<16;++n) {
        int p=c*256+r*16+n;
        exact[p]+=ia[c*256+r*16+k]*ib[c*256+k*16+n];
        ref[p]+=double(ia[c*256+r*16+k])*double(ib[c*256+k*16+n]);
    }
    size_t host_bad=0;
    for(size_t p=0;p<exact.size();++p) {
        if(ref[p]!=exact[p]) return 3;
        if(legacy[p]!=exact[p]) {++host_bad; std::fprintf(stderr,"host_expected case=%zu row=%zu col=%zu legacy=%.9g int=%d double=%.17g\n",p/256,p%256/16,p%16,legacy[p],exact[p],ref[p]);}
    }
    std::printf("{\"gate\":\"host_oracle\",\"legacy_mismatches\":%zu,\"integer_double_mismatches\":0}\n",host_bad);
    if(argc>1) return host_bad?2:0; // CPU-only mode: never calls HIP.
    hipDeviceProp_t props; HIP(hipGetDeviceProperties(&props,0));
    std::printf("{\"device\":\"%s\",\"arch\":\"%s\",\"warp_size\":%d}\n",props.name,props.gcnArchName,props.warpSize);
    _Float16 *da,*db;float* dc;
    HIP(hipMalloc(&da,a.size()*2));HIP(hipMalloc(&db,b.size()*2));HIP(hipMalloc(&dc,got.size()*4));
    HIP(hipMemcpy(da,a.data(),a.size()*2,hipMemcpyHostToDevice));HIP(hipMemcpy(db,b.data(),b.size()*2,hipMemcpyHostToDevice));
    int failures=host_bad?1:0;
    bool raw_pass=false;
    std::vector<float> raw(got.size());
    const char* names[]={"builtin","inline_asm","isolated_k","scalar_f32","sign_separated"};
    for(int mode=0;mode<5;++mode) {
    #define RUN(M) if(mode==M) probe<M><<<cases,32>>>(da,db,dc)
    RUN(0);RUN(1);RUN(2);RUN(3);RUN(4);
    #undef RUN
    HIP(hipGetLastError());HIP(hipMemcpy(got.data(),dc,got.size()*4,hipMemcpyDeviceToHost));
    size_t bad[3]={},legacy_bad=0,raw_diff=0;
    double max_abs_error=0;
    if(mode==0) raw=got;
    for(size_t p=0;p<got.size();++p) {
        legacy_bad+=got[p]!=legacy[p];
        raw_diff+=got[p]!=raw[p];
        max_abs_error=std::max(max_abs_error,std::abs(double(got[p])-exact[p]));
        if(got[p]!=exact[p]) ++bad[p/256<256?0:p/256<512?1:2];
        if(mode==0 && p/256==512 && (got[p]!=legacy[p] || got[p]!=exact[p])) std::fprintf(stderr,"candidate0 case=512 row=%zu col=%zu got=%.9g legacy=%.9g int=%d double=%.17g\n",p%256/16,p%16,got[p],legacy[p],exact[p],ref[p]);
    }
    std::printf("{\"gate\":\"wmma_variant0\",\"mode\":%d,\"policy\":\"%s\",\"raw_differences\":%zu,\"max_abs_error\":%.17g,\"cases\":513,\"outputs\":131328,\"A_impulse_mismatches\":%zu,\"B_impulse_mismatches\":%zu,\"non_symmetric_mismatches\":%zu,\"legacy_mismatches\":%zu,\"exact\":%s}\n",mode,names[mode],raw_diff,max_abs_error,bad[0],bad[1],bad[2],legacy_bad,(bad[0]+bad[1]+bad[2])?"false":"true");
    if((mode==3 || mode==4) && (bad[0]+bad[1]+bad[2])) ++failures;
    if(mode==0) raw_pass=(bad[0]+bad[1]+bad[2])==0;
    if(mode==1 && raw_diff) ++failures;
    }
    HIP(hipFree(da));HIP(hipFree(db));HIP(hipFree(dc));
    std::printf("{\"gate\":\"sign_separated_exact_integer_gate\",\"pass\":%s,\"raw_single_wmma_pass\":%s}\n",failures?"false":"true",raw_pass?"true":"false");
    return failures?2:0;
}
