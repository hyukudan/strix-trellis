// CPU-only test of the actual wrappers forwarding standard allocation failures.
#include "../heap_guard.hpp"
#include <cstdio>
#include <limits>
struct HandlerException {};
static unsigned calls=0;
static void handler(){++calls;throw HandlerException{};}
int main(){
    void* p=::operator new(0);::operator delete(p);
    p=::operator new[](0);::operator delete[](p);
    auto previous=std::set_new_handler(handler);
    volatile std::size_t impossible=std::numeric_limits<std::size_t>::max();
    bool exact=false;try{p=::operator new(impossible);::operator delete(p);}catch(const HandlerException&){exact=true;}
    std::set_new_handler(previous);
    if(!exact || calls!=1 || !grouped_k3_perf::clean_submit())return 1;
    std::puts("PASS CPU real new_handler and exact exception forwarding; zero-size ordinary/array allocation outside Submit");
}
