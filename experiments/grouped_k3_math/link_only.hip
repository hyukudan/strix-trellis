// LINK INSPECTION ONLY. This binary must never be executed during preparation.
#include "kernels.hip"
#include "../../src/backend.hpp"
#include "../../include/strix_trellis.h"
// Force the actual pinned V13 archive into the link, without executing it.
auto* volatile original_linear = &strix_trellis::launch_linear;
auto* volatile original_context = &strix_context_create;
int main() { return 125; }
