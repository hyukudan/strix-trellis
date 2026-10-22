// CPU-only checks of the integration boundary, independent of HIP discovery.
#include "../src/gemv_policy.hpp"
#include "../src/fused_both_policy.hpp"
#include "../src/fused_output_policy.hpp"
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
static void require(bool v) {if(!v) std::abort();}
int main() {
    using namespace strix_trellis;
    const void* aligned=reinterpret_cast<const void*>(256);
    for(int i:{-1,0,1280,2304,5120,8192,8193}) for(int n:{0,2304,5120,32768,129280})
      for(int bits=1;bits<=8;++bits) for(int cb=0;cb<3;++cb) for(int rows=0;rows<=9;++rows)
       for(bool prefer:{false,true}) for(bool cap:{false,true}) for(int bad=-1;bad<6;++bad) {
        const void* p[6]={aligned,aligned,aligned,aligned,aligned,aligned};
        if(bad>=0)p[bad]=reinterpret_cast<const void*>(257);
        FusedBothLayout layout{};
        bool got=fused_both_layout(i,n,bits,cb,rows,prefer,cap,p[0],p[1],p[2],p[3],p[4],p[5],layout);
        bool want=(i==2304 || i==8192) && n==5120 && bits==5 && cb==2 && rows==1 && !prefer && cap && bad<0;
        require(got==want);
        if(got)require(layout.blocks==40 && layout.static_bytes==2048 && layout.dynamic_bytes==size_t(i==8192?16384:5120));
        else require(!layout.blocks && !layout.dynamic_bytes && !layout.static_bytes);
       }
    for(int threads:{0,1023,1024}) for(size_t shared:{size_t(0),size_t(2048),size_t(2049)})
      for(size_t local:{size_t(0),size_t(4)}) for(int dynamic:{-1,0,16383,16384,63488})
       for(size_t available:{size_t(18431),size_t(18432),size_t(65536)})
        require(fused_both_image_limits(threads,shared,local,dynamic,available)==
            (threads>=1024 && shared==2048 && local==0 && dynamic>=16384 && available>=18432));

    for(int threads:{512,1023,1024,2048}) for(int dimension:{512,1023,1024})
        for(size_t shared:{size_t(24575),size_t(24576),size_t(65536)}) {
            require(fused_output_device_limits(threads,dimension,shared)==
                (threads>=1024 && dimension>=1024 && shared>=24576));
            for(size_t local:{size_t(0),size_t(4)})
                require(fused_output_image_limits(threads,shared,local,24576)==
                    (threads>=1024 && shared<=24576 && local==0));
        }
    require(fused_output_groups(1280,32768,5,2,8,4,false,true,true)==1);
    require(fused_output_groups(5120,129280,6,2,8,4,false,true,true)==2);
    require(fused_output_groups(8192,5120,5,2,8,4,false,true,true)==4);
    require(fused_output_groups(2304,5120,5,2,8,4,false,true,true)==4);
    for(int required:{256,512,1024}) for(int threads:{255,256,511,512,1023,1024})
        require(fused_output_image_limits(threads,6144,0,24576,required)==(threads>=required));
    for(int i:{1280,2304,5120,8192}) for(int n:{512,1280,2304,5120,32768,129280})
        for(int k=1;k<=8;++k) for(int cb=0;cb<3;++cb) for(int row=0;row<=9;++row)
            for(int split:{1,2,4,8,16,32}) {
                const bool expected=row==8 && split==4 && cb==2 &&
                    ((k==5 && ((i==1280 && n==32768) || ((i==8192 || i==2304) && n==5120))) ||
                     (k==6 && i==5120 && n==129280));
                const int groups=expected?(k==6?2:(i==1280?1:4)):0;
                require(fused_output_groups(i,n,k,cb,row,split,false,true,true)==groups);
                require(fused_output_candidate(i,n,k,cb,row,split,false,true,true)==expected);
                require(!fused_output_candidate(i,n,k,cb,row,split,true,true,true));
                require(!fused_output_candidate(i,n,k,cb,row,split,false,false,true));
                require(!fused_output_candidate(i,n,k,cb,row,split,false,true,false));
            }
    for(int i:{1280,8192}) for(int n:{5120,32768})
        for(int k=1;k<=8;++k) for(int cb=0;cb<3;++cb)
            for(int row=0;row<=9;++row) for(int split=1;split<=32;++split) {
                const bool expected=k==2 && cb==2 && row==1 && split==4 &&
                    ((i==1280 && n==32768) || (i==8192 && n==5120));
                require(pipeline_k2_candidate(i,n,k,cb,row,split,true)==expected);
                require(!pipeline_k2_candidate(i,n,k,cb,row,split,false));
            }
    require(!pipeline_k2_candidate(5120,129280,2,2,1,4,true));
    require(!pipeline_k2_candidate(5120,2304,2,2,1,4,true));
    struct Shape {int i,n,k,s;};
    const Shape measured[]={{1280,32768,5,4},{8192,5120,5,4},
        {5120,2304,5,16},{2304,5120,5,4},{5120,1280,6,16},
        {4096,1024,5,32},{5120,512,6,32},{5120,129280,6,4}};
    int selected=0;
    for(auto w:measured) for(int r=1;r<=8;++r) {
        const int m=exact_gemv_mode(w.i,w.n,w.k,2,r,w.s,true);
        require(m>=0 && m<=3);selected+=m!=0;
        require(!exact_gemv_mode(w.i,w.n,w.k,2,r,w.s,false));
        for(int cb=0;cb<2;++cb) require(!exact_gemv_mode(w.i,w.n,w.k,cb,r,w.s,true));
        require(!exact_gemv_mode(w.i+16,w.n,w.k,2,r,w.s,true));
        require(!exact_gemv_mode(w.i,w.n+16,w.k,2,r,w.s,true));
        require(!exact_gemv_mode(w.i,w.n,w.k,2,r,w.s+1,true));
        for(int k=1;k<=8;++k) if(k!=w.k)
            require(!exact_gemv_mode(w.i,w.n,k,2,r,w.s,true));
    }
    require(selected==52); // wo_a R4 now uses the measured S16 body instead.
    require(exact_gemv_mode(4096,1024,5,2,4,16,true)==4);
    require(exact_gemv_mode(4096,1024,5,2,8,16,true)==4);
    require(exact_gemv_mode(5120,2304,5,2,8,8,true)==4);
    require(!exact_gemv_mode(5120,2304,5,2,4,8,true));
    require(!exact_gemv_mode(4096,1024,5,2,5,16,true));
    for(auto w:measured) for(int r:{0,9,16})
        require(!exact_gemv_mode(w.i,w.n,w.k,2,r,w.s,true));
    for(int r=1;r<=8;++r) {
        require(measured_vector_splits(4096,1024,5,2,r)==((r==1 || r==4 || r==8)?16:32));
        require(measured_vector_splits(5120,2304,5,2,r)==(r==8?8:16));
        require(measured_vector_splits(4096,1024,6,2,r)==32);
        require(measured_vector_splits(5120,2304,5,1,r)==16);
    }
    std::puts("PASS exact-row guards, alignment fallback and measured split isolation");
}
