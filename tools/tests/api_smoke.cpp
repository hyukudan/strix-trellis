#include "strix_trellis.h"
#include <type_traits>

static_assert(std::is_standard_layout<strix_grouped_moe_desc>::value, "C ABI");
static_assert(std::is_standard_layout<strix_exl3_transport_desc>::value, "C ABI");
int main() {
    strix_stream stream{};
    stream.struct_size = sizeof(stream);
    stream.abi_version = STRIX_TRELLIS_ABI_VERSION;
    return stream.hip_stream != nullptr;
}
