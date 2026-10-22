#include <cstddef>
extern "C" void* tp_dso_allocate(size_t n){return new unsigned char[n];}
extern "C" void tp_dso_release(void* p){delete[] static_cast<unsigned char*>(p);}
