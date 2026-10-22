/* Private experiment only. No public symbols or ABI changes. */
#ifndef TP_DECODE_DESCRIPTOR_H
#define TP_DECODE_DESCRIPTOR_H
#include "../../include/strix_trellis.h"
typedef enum tp_numerics { TP_LOGICAL_K16_V1 = 1 } tp_numerics;
typedef struct tp_desc {
    uint32_t struct_size, abi_version;
    strix_matmul_desc linear;
    tp_numerics numerics;
    uint32_t logical_splits;
    uint64_t reserved[4];
} tp_desc;
#if defined(__cplusplus)
#define TP_ASSERT static_assert
#else
#define TP_ASSERT _Static_assert
#endif
TP_ASSERT(sizeof(strix_tensor_desc)==96,"tensor layout");
TP_ASSERT(sizeof(strix_matmul_desc)==264,"linear layout");
TP_ASSERT(sizeof(tp_desc)==312,"private layout");
TP_ASSERT(offsetof(tp_desc,linear)==8,"linear offset");
TP_ASSERT(offsetof(tp_desc,numerics)==272,"numerics offset");
TP_ASSERT(offsetof(tp_desc,logical_splits)==276,"split offset");
TP_ASSERT(offsetof(tp_desc,reserved)==280,"reserved offset");
#undef TP_ASSERT
#endif
