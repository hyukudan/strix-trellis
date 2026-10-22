// Private link-boundary observations. No HIP/HSA dependency, allocation or IO.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>

namespace grouped_k3_math {
enum class AttrPhase { idle, creation, math, prepare, submit, capture, replay, reuse, cleanup };
template<class Attributes, size_t Capacity=128, size_t ContextCapacity=8>
struct AttributeTrace {
    struct Event {
        uint64_t creation=0;
        const void* context=nullptr;
        const void* function=nullptr;
        AttrPhase phase=AttrPhase::idle;
        int result=0;
        bool valid=false;
        Attributes attributes{};
    };
    struct Context {
        const void* handle=nullptr;
        size_t begin=0;
        bool admitted=false;
    };
    std::array<Event,Capacity> events{};
    std::array<Context,ContextCapacity> contexts{};
    size_t calls=0, stored=0, creations=0;
    uint64_t current=0;
    AttrPhase phase=AttrPhase::idle;
    std::array<size_t,9> regions{}, deltas{};
    int first_error=0;
    bool failed=false, overflow=false;

    uint64_t reserve() noexcept {
        if(creations==ContextCapacity) {failed=overflow=true;return 0;}
        current=++creations;contexts[current-1].begin=stored;
        phase=AttrPhase::creation;return current;
    }
    void associate(uint64_t id,const void* handle) noexcept {
        if(!id || id>creations || !handle) {failed=true;return;}
        contexts[id-1].handle=handle;
        for(size_t j=contexts[id-1].begin;j<stored;++j)
            if(events[j].creation==id)events[j].context=handle;
    }
    void record(const void* function,int result,const Attributes* attributes) noexcept {
        ++calls;
        if(result && !first_error)first_error=result;
        if(result || !current || (phase!=AttrPhase::creation && phase!=AttrPhase::math))failed=true;
        if(stored==Capacity) {failed=overflow=true;return;}
        auto& e=events[stored++];e.creation=current;e.function=function;e.phase=phase;e.result=result;
        if(current && current<=creations)e.context=contexts[current-1].handle;
        e.valid=result==0 && attributes;
        if(e.valid)e.attributes=*attributes; // Failed HIP calls never expose output bytes.
        else failed=true;
    }
    bool admit(uint64_t id,const void* const (&math)[3]) noexcept {
        if(!id || id>creations) {failed=true;return false;}
        auto& c=contexts[id-1];bool ok=c.handle && !c.admitted && stored-c.begin==7;
        for(size_t j=0;j<3;++j) {
            ok=ok && math[j];
            for(size_t k=0;k<j;++k)ok=ok && math[j]!=math[k];
        }
        if(stored-c.begin==7)for(size_t j=0;j<7;++j) {
            const auto& e=events[c.begin+j];
            ok=ok && e.creation==id && e.context==c.handle && e.function && e.valid && !e.result &&
                e.phase==(j<4?AttrPhase::creation:AttrPhase::math);
            if(j>=4)ok=ok && e.function==math[j-4];
            for(size_t k=0;k<j;++k)ok=ok && e.function!=events[c.begin+k].function;
        }
        if(!ok)failed=true;
        c.admitted=ok && !failed;phase=AttrPhase::idle;return c.admitted;
    }
    bool complete() const noexcept {
        if(failed || overflow || !creations || calls!=stored || calls!=7*creations)return false;
        for(size_t j=0;j<creations;++j)if(!contexts[j].admitted)return false;
        return true;
    }
    // A zero-query region nests safely and includes error/cleanup exits.
    struct Region {
        AttributeTrace& trace;
        AttrPhase previous;
        uint64_t previous_context;
        size_t before;
        Region(AttributeTrace& t,AttrPhase p,uint64_t id):trace(t),previous(t.phase),
            previous_context(t.current),before(t.calls) {t.phase=p;t.current=id;}
        ~Region() noexcept {
            ++trace.regions[static_cast<unsigned>(trace.phase)];
            trace.deltas[static_cast<unsigned>(trace.phase)]+=trace.calls-before;
            if(trace.calls!=before)trace.failed=true;
            trace.phase=previous;trace.current=previous_context;
        }
    };
};
} // namespace grouped_k3_math
