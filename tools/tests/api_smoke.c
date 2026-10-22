#include "strix_trellis.h"

_Static_assert(STRIX_TRELLIS_ABI_VERSION == 1, "Unexpected ABI");
_Static_assert(STRIX_TRANSPORT_EXL3_CANONICAL != STRIX_TRANSPORT_INVALID,
               "Transport must be explicit");
int main(void) {
    strix_matmul_desc desc = {0};
    desc.struct_size = (uint32_t)sizeof(desc);
    desc.abi_version = STRIX_TRELLIS_ABI_VERSION;
    return desc.abi_version != 1;
}
