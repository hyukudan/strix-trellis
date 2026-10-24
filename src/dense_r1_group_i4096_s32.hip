#include "dense_r1_device.hpp"
namespace strix_trellis::dense_r1 {
const void* group_i4096_s32(const DenseR1Launch& a) { return rows<4096,32,true>(a); }
}
