#include "dense_r1_device.hpp"
namespace strix_trellis::dense_r1 {
const void* function_i4096_s4(const DenseR1Launch& a) { return rows<4096,4>(a); }
}
