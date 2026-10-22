#include "../src/v14_core_policy.hpp"
#include <cassert>
using namespace strix_trellis;
int main() {
    int accepted_k2=0,accepted_wmma=0;
    const int shapes[][2]={{8192,5120},{1280,32768},{5120,2304},{2304,5120},{4096,1024},{5120,1280},{5120,129280},{8192,32768}};
    for(auto& shape:shapes) for(int bits=1;bits<=8;++bits) for(int cb=0;cb<3;++cb)
    for(int r=0;r<=513;++r) for(int s=0;s<=33;++s) {
        const int i=shape[0],n=shape[1];
        bool k=k2_multirow_candidate(i,n,bits,cb,r,r,s,true,true,true);
        bool expected=bits==2 && cb==2 && s==4 && (r==2 || r==4 || r==8) &&
            ((i==8192 && n==5120) || (i==1280 && n==32768));
        assert(k==expected);accepted_k2+=k;
        bool w=wmma_packed_candidate(i,n,bits,cb,r,s,64,true,true,true,true);
        bool e=false;
        if(cb==2 && (r==64 || r==128 || r==256)) {
            if(bits==6) e=i==5120 && n==1280 && r==256 && s==4;
            if(bits==5) {
                if(i==1280 && n==32768) e=s==1;
                if((i==8192 && n==5120) || (i==2304 && n==5120)) e=s==256/r;
                if(i==5120 && n==2304) e=r!=256 && s==512/r;
                if(i==4096 && n==1024) e=r==256 && s==1024/r;
            }
        }
        assert(w==e);assert(!(k&&w));accepted_wmma+=w;
        if(k) {
            assert(!k2_multirow_candidate(i,n,bits,cb,r,r+1,s,true,true,true));
            assert(!k2_multirow_candidate(i,n,bits,cb,r,r-1,s,true,true,true));
            for(int a=0;a<3;++a) assert(!k2_multirow_candidate(i,n,bits,cb,r,r,s,a!=0,a!=1,a!=2));
        }
        if(w) {
            assert(!wmma_packed_candidate(i,n,bits,cb,r,s,16,true,true,true,true));
            assert(!wmma_packed_candidate(i,n,bits,cb,r,s,64,false,true,true,true));
            for(int a=0;a<3;++a) assert(!wmma_packed_candidate(i,n,bits,cb,r,s,64,true,a!=0,a!=1,a!=2));
        }
    }
    assert(accepted_k2==6 && accepted_wmma==13);
}
