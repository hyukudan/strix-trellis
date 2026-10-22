// Private host scheduling contract. No HIP/HSA headers or dependencies.
#pragma once
#include "../../reports/grouped_k3_private_guard.hpp"
#include <array>
#include <stdexcept>
#include <memory>
namespace grouped_k3_perf {
using namespace grouped_k3_review;
inline void require(bool ok,const char* why) { if(!ok)throw std::runtime_error(why); }
constexpr unsigned instances=64;
constexpr uint64_t packed_bytes=4423680;
constexpr unsigned selections[]={4,8,16,32};
inline bool same_certificate(const Region& a,const Region& b) {
    return a.allocation==b.allocation && a.allocation_bytes==b.allocation_bytes && a.offset==b.offset &&
        a.capacity==b.capacity && a.owner==b.owner && a.device==b.device && a.live==b.live;
}
inline bool authentic_certificate(const Region& r,const Region* registry,size_t count) {
    if(!r.allocation)return r.allocation_bytes==0 && r.capacity==0 && r.offset==0 && r.owner==0 && !r.live && r.device==-1;
    for(size_t j=0;j<count;++j)if(same_certificate(r,registry[j]))return true;
    return false;
}
inline unsigned clones(unsigned g) {
    require(g==4 || g==8 || g==16 || g==32,"fixed G only");return instances/g;
}
inline unsigned start(unsigned g,unsigned clone) {
    require(clone<clones(g),"clone bound");return clone*g;
}
struct Operations {
    void* state;
    int (*old_launch)(void*,unsigned);
    int (*new_launch)(void*);
};
// Used literally by the native and dependency-free CPU mock submissions.
inline int dispatch(const Operations& ops,bool grouped,unsigned g) {
    if(grouped)return ops.new_launch(ops.state);
    for(unsigned j=0;j<g;++j) {int e=ops.old_launch(ops.state,j);if(e)return e;}
    return 0;
}
struct Last {
    std::array<std::array<bool,instances>,2> checked{};
    void mark(unsigned variant,unsigned instance) {checked.at(variant).at(instance)=true;}
    void complete() const {for(const auto& v:checked)for(bool b:v)require(b,"missed LAST instance");}
};
inline void disjoint(const std::array<Descriptor,instances>& old,
                     const std::array<Descriptor,instances>& now) {
    // Entire pool, not just the currently submitted Batch. Read sharing allowed.
    const std::array<Descriptor,instances>* variants[]={&old,&now};
    for(unsigned v=0;v<2;++v)for(unsigned j=0;j<instances;++j)for(unsigned w=Y;w<Fields;++w) {
        const auto& a=(*variants[v])[j].regions[w];
        Interval aa{a.allocation+a.offset,a.capacity};
        for(unsigned vv=0;vv<2;++vv)for(unsigned k=0;k<instances;++k)for(unsigned f=0;f<Fields;++f) {
            if(v==vv && j==k && w==f)continue;
            const auto& b=(*variants[vv])[k].regions[f];
            require(!overlap(aa,{b.allocation+b.offset,b.capacity}),"pool writer overlaps read/write");
        }
    }
    for(unsigned j=0;j<instances;++j)for(unsigned k=0;k<j;++k) {
        const auto& a=old[j].regions[Packed];const auto& b=old[k].regions[Packed];
        require(!overlap({a.allocation+a.offset,packed_bytes},{b.allocation+b.offset,packed_bytes}),"packed cold ranges alias");
    }
}
}
