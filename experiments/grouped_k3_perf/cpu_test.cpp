// Executable CPU mocks: no HIP/HSA include, link or load.
#include "contract.hpp"
#include "submit_guard.hpp"
#include "heap_guard.hpp"
#include <cstdio>
#include <vector>
using namespace grouped_k3_perf;
static unsigned enqueues=0;
static int fail=-1;
static int old(void*,unsigned j){++enqueues;return int(j)==fail?713:0;}
static int now(void*){++enqueues;return fail==0?719:0;}
static bool rejects(void(*test)()){try{test();}catch(const std::runtime_error&){return true;}return false;}
static Region region_at(uintptr_t at,uint64_t size){return {at,size,0,size,1,0,true};}
static std::array<Descriptor,instances> make(unsigned variant) {
    std::array<Descriptor,instances> ds{};
    for(unsigned j=0;j<instances;++j){auto& d=ds[j];d.input=5120;d.output=2304;d.splits=16;
        uintptr_t base=0x100000000ull+uintptr_t(j)*0x1000000;
        d.regions[Packed]=region_at(base,packed_bytes);d.regions[X]=region_at(base+0x500000,5120*4);
        d.regions[Suh]=region_at(base+0x600000,5120*2);d.regions[Svh]=region_at(base+0x700000,2304*2);
        uintptr_t writer=base+0x800000+variant*0x100000;
        d.regions[Y]=region_at(writer,2304*4);
        Region a{writer+0x10000,157696,0,10240,1,0,true};d.regions[A]=a;a.offset=10240;a.capacity=147456;d.regions[P]=a;
    }return ds;
}
static std::array<Descriptor,instances> aa,bb;
static void range_test(){disjoint(aa,bb);}
static void g64(){clones(64);}
static void wrong_start(){start(4,16);}
static void missed_last(){Last l;for(unsigned v=0;v<2;++v)for(unsigned j=0;j<instances;++j)if(v!=1 || j!=63)l.mark(v,j);l.complete();}
static void require_prepare_reject(const std::array<Descriptor,instances>& d,unsigned g){
    Plan p;auto before=enqueues;require(!prepare(d.data(),g,{1,1,0,0,true},p),"mutant accepted");require(before==enqueues,"invalid group enqueued");
}
int main(){
    aa=make(0);bb=make(1);disjoint(aa,bb);
    {const auto real=aa[0].regions[Packed];require(authentic_certificate(real,&real,1),"registered cert rejected");
        auto forged=real;forged.capacity++;require(!authentic_certificate(forged,&real,1),"forged extent accepted");
        forged=real;forged.allocation=aa[1].regions[Packed].allocation;require(!authentic_certificate(forged,&real,1),"aliased raw B accepted as registered field");
        forged=real;forged.owner++;require(!authentic_certificate(forged,&real,1),"stale generation accepted");}
    for(unsigned g:selections){std::array<bool,instances> covered{};
        for(unsigned c=0;c<clones(g);++c)for(unsigned j=0;j<g;++j){auto k=start(g,c)+j;require(k<64 && !covered[k],"clone addressing");covered[k]=true;}
        for(bool b:covered)require(b,"pool coverage");
        for(unsigned c=0;c<clones(g);++c){Plan p;require(prepare(aa.data()+start(g,c),g,{1,1,0,0,true},p),"guard rejected valid contiguous A/P");}
        for(bool grouped:{false,true}){enqueues=0;fail=-1;require(dispatch({nullptr,old,now},grouped,g)==0 && enqueues==(grouped?1:g),"prepared dispatch count");}
        for(unsigned j=0;j<g;++j){enqueues=0;fail=int(j);require(dispatch({nullptr,old,now},false,g)==713 && enqueues==j+1,"first error must stop");}
    }
    fail=0;enqueues=0;require(dispatch({nullptr,old,now},true,32)==719 && enqueues==1,"grouped first error");
    require(clean_submit(),"unexpected submit prohibition failure");
    for(unsigned k=0;k<6;++k){
        require(!forbidden(SideEffect(k)),"outside submit should be allowed");
        {SubmitScope scope;require(forbidden(SideEffect(k)),"injected alloc/copy/query/sync/event/heap missed");}
        require(forbidden_calls[k]==1,"actual prohibition count");forbidden_calls[k]=0;
    }
    require(clean_submit(),"guard test cleanup");
    {SubmitScope scope;bool rejected=false;try{void* p=::operator new(32);::operator delete(p);}catch(const std::bad_alloc&){rejected=true;}require(rejected,"actual operator new inside submit allowed");}
    require(forbidden_calls[unsigned(SideEffect::heap)]==1,"heap observation");forbidden_calls[unsigned(SideEffect::heap)]=0;
    for(unsigned f=0;f<Fields;++f){if(f==Bias)continue;auto m=aa;m[31].regions[f].capacity--;require_prepare_reject(m,32);}
    {auto m=aa;m[31].regions[P].owner++;require_prepare_reject(m,32);}
    {auto m=aa;m[31].regions[A]=m[0].regions[A];require_prepare_reject(m,32);}
    {auto m=aa;m[31].regions[P]=m[0].regions[Packed];require_prepare_reject(m,32);}
    {auto m=aa;m[31].regions[Packed]=m[0].regions[Y];require_prepare_reject(m,32);}
    aa[63].regions[Y]=bb[0].regions[P];require(rejects(range_test),"cross clone/variant overlap missed");aa=make(0);
    aa[63].regions[Packed]=aa[0].regions[Packed];require(rejects(range_test),"cold pointer alias missed");aa=make(0);
    require(rejects(g64) && rejects(wrong_start) && rejects(missed_last),"clone/LAST negatives");
    Last last;for(unsigned v=0;v<2;++v)for(unsigned j=0;j<64;++j)last.mark(v,j);last.complete();
    // Shared ownership survives dropping pool/plan; mirrors the host lease mechanism.
    auto owner=std::make_shared<int>(7);std::weak_ptr<int> weak=owner;auto graph=owner;owner.reset();require(!weak.expired(),"lease lost");graph.reset();require(weak.expired(),"lease leaked");
    std::puts("PASS CPU dispatch/first-error/real-guard/pool64/clone/LAST/lifetime; native graph/resources/timing OPEN");
}
