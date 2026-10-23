#ifndef STRIX_TRELLIS_H
#define STRIX_TRELLIS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* C ABI v1. See docs/API.md for the complete contract.
 * All descriptors must be zero-initialized, then set struct_size and abi_version.
 * Reserved fields must remain zero. Dimensions/strides are in elements; buffer
 * sizes and transport offsets are in bytes. No ownership is transferred. */
#define STRIX_TRELLIS_ABI_VERSION UINT32_C(1)
#define STRIX_TRELLIS_VERSION_MAJOR 0
#define STRIX_TRELLIS_VERSION_MINOR 3
#define STRIX_TRELLIS_VERSION_PATCH 0
#define STRIX_TRELLIS_TRANSPORT_VERSION UINT32_C(1)
#define STRIX_TRELLIS_WORKSPACE_ALIGNMENT UINT32_C(256)
#define STRIX_EXL3_MARKER_3INST UINT32_C(0)
#define STRIX_EXL3_MARKER_MCG UINT32_C(0xCBAC1FED)
#define STRIX_EXL3_MARKER_MUL1 UINT32_C(0x83DCD12D)

typedef enum strix_status {
    STRIX_SUCCESS = 0,
    STRIX_ERROR_INVALID_ARGUMENT = 1,
    STRIX_ERROR_ABI_MISMATCH = 2,
    STRIX_ERROR_UNSUPPORTED_FORMAT = 3,
    STRIX_ERROR_UNSUPPORTED_DEVICE = 4,
    STRIX_ERROR_INSUFFICIENT_BUFFER = 5,
    STRIX_ERROR_OUT_OF_MEMORY = 6,
    STRIX_ERROR_BACKEND = 7,
    STRIX_ERROR_NOT_IMPLEMENTED = 8
} strix_status;

typedef enum strix_dtype {
    STRIX_DTYPE_INVALID = 0,
    STRIX_DTYPE_F16 = 1,
    STRIX_DTYPE_BF16 = 2,
    STRIX_DTYPE_F32 = 3
} strix_dtype;

/* Host-native descriptors, not serialized structs or a safetensors parser.
 * Blob payloads are little endian. Prepare completes all uploads before return. */
typedef enum strix_transport_format {
    STRIX_TRANSPORT_INVALID = 0,
    STRIX_TRANSPORT_EXL3_CANONICAL = 1
} strix_transport_format;

typedef enum strix_codebook {
    STRIX_CODEBOOK_3INST = 0,
    STRIX_CODEBOOK_MCG = 1,
    STRIX_CODEBOOK_MUL1 = 2
} strix_codebook;

typedef struct strix_exl3_metadata_v1 {
    uint32_t struct_size;
    uint32_t transport_version;
    uint64_t rows;    /* N = output channels. */
    uint64_t columns; /* I = input channels. Both positive multiples of 128. */
    uint32_t bits_x2; /* Even values 2..16: integer bits 1..8. */
    strix_codebook codebook;
    uint32_t marker;  /* Exact matching STRIX_EXL3_MARKER_* constant. */
    uint32_t flags;   /* Zero. */
    uint64_t trellis_offset, trellis_bytes;
    uint64_t suh_offset, suh_bytes; /* Little-endian F16 [columns]. */
    uint64_t svh_offset, svh_bytes; /* Little-endian F16 [rows]. */
    uint64_t bias_offset, bias_bytes; /* Little-endian F32 [rows]; both zero if absent. */
    uint64_t reserved[4];
} strix_exl3_metadata_v1;

typedef struct strix_exl3_transport_desc {
    uint32_t struct_size;
    uint32_t abi_version;
    strix_transport_format format;
    uint32_t transport_version;
    const void *data;
    uint64_t data_bytes;
    const void *metadata;
    uint64_t metadata_bytes;
    uint64_t rows;
    uint64_t columns;
    uint64_t reserved[4];
} strix_exl3_transport_desc;

/* Borrowed, independent host regions. No contiguous staging blob is required.
 * Payload encodings match transport v1; host pointer alignment is unrestricted.
 * Every region must remain readable and unchanged until prepare returns. */
typedef struct strix_host_region {
    const void *data;
    uint64_t bytes;
} strix_host_region;

typedef struct strix_exl3_parts_desc {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t transport_version;
    uint32_t bits_x2;
    uint64_t rows;
    uint64_t columns;
    strix_codebook codebook;
    uint32_t marker;
    strix_host_region trellis;
    strix_host_region suh;
    strix_host_region svh;
    strix_host_region bias; /* NULL/zero when absent; otherwise F32 [rows]. */
    uint64_t reserved[4];
} strix_exl3_parts_desc;

/* Internal layouts are opaque and may change independently of the transport.
 * No internal layout identifiers are promised by this header yet. */
typedef struct strix_context_impl *strix_context;
typedef struct strix_weights_impl *strix_weights;
typedef struct strix_linear_plan_impl *strix_linear_plan;
typedef struct strix_tp_dense_plan_impl *strix_tp_dense_plan;
typedef struct strix_tp_moe_plan_impl *strix_tp_moe_plan;
#define STRIX_OPERATION_GEMV UINT32_C(0)
#define STRIX_OPERATION_GEMM UINT32_C(1)
typedef struct strix_stream {
    uint32_t struct_size;
    uint32_t abi_version;
    void *hip_stream; /* Borrowed hipStream_t cast to void*. NULL = default. */
    uint64_t reserved[2];
} strix_stream;

typedef struct strix_context_desc {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t device_ordinal;
    uint32_t flags; /* Must be zero. */
    uint64_t reserved[4];
} strix_context_desc;

/* Contiguous row-major device tensors. Only F32 rank 1 or rank 2 is supported.
 * Buffers must belong to the context device and survive stream completion. */
typedef struct strix_tensor_desc {
    uint32_t struct_size;
    uint32_t abi_version;
    void *data;
    uint64_t buffer_bytes;
    strix_dtype dtype;
    uint32_t rank;
    uint64_t shape[2];
    uint64_t strides[2];
    uint64_t reserved[4];
} strix_tensor_desc;

/* output = alpha * Linear(input) + beta * output, including bias inside Linear.
 * The effective conventional W has [N,I]; transport Q is packed in [I,N] order.
 * GEMV: [I] -> [N] or [B,I] -> [B,N], B=0..8. GEMM: rank 2, B=0..INT_MAX.
 * B=0 validates descriptors but accesses no device buffers and submits no work.
 * Workspace is caller-owned, 256-byte aligned; query before submission.
 * beta=0 never reads old output. Zero-initialized alpha means zero, not one. */
typedef struct strix_matmul_desc {
    uint32_t struct_size;
    uint32_t abi_version;
    strix_weights weights;
    strix_tensor_desc input;
    strix_tensor_desc output;
    float alpha;
    float beta;
    void *workspace;
    uint64_t workspace_bytes;
    uint64_t reserved[4];
} strix_matmul_desc;

/* Grouped MoE consumes pre-routed, disjoint input/output slices per expert.
 * Routing, scatter and top-k reduction are caller responsibilities. The host
 * group array is read during submission; tensor buffers live through completion.
 * Groups share the workspace supplied here; group workspaces must be NULL/zero. */
typedef struct strix_grouped_moe_desc {
    uint32_t struct_size;
    uint32_t abi_version;
    const strix_matmul_desc *groups;
    uint64_t group_count;
    void *workspace;
    uint64_t workspace_bytes;
    uint64_t reserved[4];
} strix_grouped_moe_desc;

/* Experimental TP APIs, enabled by STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP.
 * They do not change the dispatch of gemv/gemm or grouped_moe. Unsupported
 * builds return NOT_IMPLEMENTED. See docs/TP_RANK.md for qualification status.
 * TP dense accepts F32 row views: last stride 1, row stride >= local width.
 * It freezes a matmul descriptor with NULL/zero workspace and R=1..8. */
#define STRIX_TP_TOP_K UINT32_C(6)
#define STRIX_TP_MAX_ROWS UINT32_C(8)
#define STRIX_TP_MAX_EXPERTS UINT32_C(48)
#define STRIX_TP_MAX_BANK_EXPERTS UINT32_C(256)

/* The exact union of routed experts in a window. IDs need not be contiguous.
 * Gate/up have [N=1152,I=5120], down [N=5120,I=1152], K2 or K3 mul1.
 * All three weights may choose K2/K3 independently. No ownership transfers. */
typedef struct strix_tp_expert_desc {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t expert_id;
    uint32_t flags; /* Zero. */
    strix_weights gate;
    strix_weights up;
    strix_weights down;
    uint64_t reserved[2];
} strix_tp_expert_desc;

/* Host route IDs and scores are row-major [R,6], with six distinct IDs per row.
 * IDs and scores are copied at plan creation, as is the exact expert union.
 * Recreate the plan when routing, scores, weights, shapes or addresses change.
 * X/Y are F32 [R,5120], R=1..8, with unit channel stride; padded row strides
 * are accepted. Workspace is F32 [R,6,1152], caller-owned, 256-byte aligned.
 * Gate/up + SiLU is one grouped launch; down + ordered top-6 is one launch.
 * output = alpha * sum_slot(score * down(SiLU(gate(X))*up(X))) + beta*output.
 * No normalization is applied to scores; they must be finite and nonnegative.
 * Exactly zero scores skip their down projection and contribute +0.
 * Output is a local TP contribution. Inter-rank reduction is caller-owned;
 * a down bias must be present on only one contributing rank. */
typedef struct strix_tp_moe_desc {
    uint32_t struct_size;
    uint32_t abi_version;
    strix_tensor_desc input;
    strix_tensor_desc output;
    const strix_tp_expert_desc *experts;
    uint64_t expert_count;
    const uint32_t *route_ids;
    uint64_t route_ids_count;
    const float *route_scores;
    uint64_t route_scores_count;
    void *workspace;
    uint64_t workspace_bytes;
    float alpha;
    float beta;
    uint64_t reserved[4];
} strix_tp_moe_desc;

/* Persistent expert bank with routes generated/updated on the device.
 * Bank positions are router IDs: experts[i].expert_id must equal i. The bank
 * may contain 6..256 experts; a window selects at most R*6 distinct experts.
 * route_ids is device U32 [R,6], route_scores is device F32 [R,6], contiguous.
 * Each row must contain six distinct in-range IDs and finite nonnegative
 * scores. Invalid device values produce NaN outputs, never out-of-range bank
 * accesses. The API validates allocation spans, not device tensor contents.
 * Route contents may change in place between ordered submits or graph replays,
 * as may X/Y. Pointers, dimensions, bank and scalars remain frozen. The kernel
 * discovers the window union itself, without a routing copy or extra launch.
 * All tensor/workspace/operator semantics match strix_tp_moe_desc. */
typedef struct strix_tp_moe_device_desc {
    uint32_t struct_size;
    uint32_t abi_version;
    strix_tensor_desc input;
    strix_tensor_desc output;
    const strix_tp_expert_desc *experts; /* Host bank, copied at creation. */
    uint64_t expert_count;
    const uint32_t *route_ids; /* Device memory. */
    uint64_t route_ids_bytes;
    const float *route_scores; /* Device memory. */
    uint64_t route_scores_bytes;
    void *workspace;
    uint64_t workspace_bytes;
    float alpha;
    float beta;
    uint64_t reserved[4];
} strix_tp_moe_device_desc;

uint32_t strix_get_abi_version(void);
const char *strix_status_string(strix_status status);
/* Set headers, nested F32 tensor headers, alpha=1, beta=0; shapes remain unset. */
void strix_matmul_desc_init(strix_matmul_desc *desc);
strix_status strix_context_create(const strix_context_desc *desc, strix_context *out);
/* Destroy only after all work using the handle has completed. */
strix_status strix_context_destroy(strix_context context);
strix_status strix_weights_prepare(strix_context context,
    const strix_exl3_transport_desc *transport, const strix_stream *stream,
    strix_weights *out);
/* Same device layout and completion guarantee as prepare, without a host blob. */
strix_status strix_weights_prepare_parts(strix_context context,
    const strix_exl3_parts_desc *parts, const strix_stream *stream,
    strix_weights *out);
strix_status strix_weights_destroy(strix_weights weights);
strix_status strix_matmul_workspace_bytes(strix_context context,
    const strix_matmul_desc *desc, uint64_t *out_bytes);
strix_status strix_gemv(strix_context context, const strix_matmul_desc *desc,
    const strix_stream *stream);
strix_status strix_gemm(strix_context context, const strix_matmul_desc *desc,
    const strix_stream *stream);
/* Validate once and freeze the descriptor, buffers and borrowed stream.
 * Create outside stream capture. Only tensor CONTENTS may change subsequently.
 * Buffers/stream must stay live and unchanged in extent through plan destruction
 * and stream completion; reallocation at the same address is not permitted.
 * Submit checks the calling thread's device but performs no pointer, allocation
 * range, stream or descriptor queries. Retained weights cannot be destroyed.
 * Captured graphs borrow the same memory: preserve it through their last replay
 * and graph destruction, even after plan destruction. Serialize plan/weight
 * lifecycle changes externally, as for context and weight lifecycle operations.
 * Destroy does not synchronize; the caller must complete submitted GPU work. */
strix_status strix_linear_plan_create(strix_context context,
    const strix_matmul_desc *desc, const strix_stream *stream,
    uint32_t operation, strix_linear_plan *out);
strix_status strix_linear_plan_submit(strix_linear_plan plan);
strix_status strix_linear_plan_destroy(strix_linear_plan plan);
strix_status strix_grouped_moe_workspace_bytes(strix_context context,
    const strix_grouped_moe_desc *desc, uint64_t *out_bytes);
strix_status strix_grouped_moe(strix_context context,
    const strix_grouped_moe_desc *desc, const strix_stream *stream);

/* Creation is outside capture, validates resources and retains weights.
 * Submit performs no allocation, copy, synchronization or pointer/stream query.
 * All plan buffers, weights and stream must outlive completion and all captured
 * graph replays. Serialize plan lifecycle operations externally. */
strix_status strix_tp_dense_plan_create(strix_context context,
    const strix_matmul_desc *desc, const strix_stream *stream,
    strix_tp_dense_plan *out);
/* Explicit row tiling for occupancy/reuse comparisons. 0 uses all R rows;
 * otherwise rows_per_block must be 1..R. Every variant still launches once. */
strix_status strix_tp_dense_plan_create_tiled(strix_context context,
    const strix_matmul_desc *desc, const strix_stream *stream,
    uint32_t rows_per_block, strix_tp_dense_plan *out);
strix_status strix_tp_dense_plan_submit(strix_tp_dense_plan plan);
strix_status strix_tp_dense_plan_destroy(strix_tp_dense_plan plan);
void strix_tp_moe_desc_init(strix_tp_moe_desc *desc);
void strix_tp_moe_device_desc_init(strix_tp_moe_device_desc *desc);
strix_status strix_tp_moe_workspace_bytes(strix_context context,
    const strix_tp_moe_desc *desc, uint64_t *out_bytes);
strix_status strix_tp_moe_plan_create(strix_context context,
    const strix_tp_moe_desc *desc, const strix_stream *stream,
    strix_tp_moe_plan *out);
/* This creator uploads the bank once. Subsequent route changes require only
 * ordinary stream-ordered writes to the existing device routing tensors.
 * Required workspace bytes = R * 6 * 1152 * sizeof(float). */
strix_status strix_tp_moe_plan_create_device(strix_context context,
    const strix_tp_moe_device_desc *desc, const strix_stream *stream,
    strix_tp_moe_plan *out);
/* Explicit split3 variant. ABI 1 descriptors and handles are unchanged.
 * Workspace: H[R,6,1152] followed by P[R,6,4,5120], F32, 256-byte aligned;
 * minimum 519168*R bytes, or 642048*R with the pretransformed pipeline build.
 * Query the required size; no memset or arrival counters are required.
 * submit: A->B->C; the optional pipeline expands A to prepare/core/finish
 * and B to prepare/core (six kernel nodes in total). The optional fused
 * full-submit epilogue prepares down in A, using five nodes; staged calls
 * still prepare caller-provided H independently. Prepared owner maps add
 * 8192 bytes per workspace. The cooperative experimental build submits full
 * as one occupancy-admitted kernel, with the same staged entry points.
 * Adaptive rows can split each core by device-resolved row count. Immutable
 * plans omit empty classes using creation-time routing metadata. Parallel
 * finish expands C into route epilogues and an ordered combine, reusing dead
 * activation scratch without changing H, P or workspace size.
 * Query workspace size from the loaded library for every optional build.
 * The original creators continue to select the two-kernel 0.3 implementation.
 * All existing lifetime, routing, alias and enqueue-error contracts apply. */
strix_status strix_tp_moe_split_workspace_bytes(strix_context context,
    const strix_tp_moe_desc *desc, uint64_t *out_bytes);
strix_status strix_tp_moe_plan_create_split(strix_context context,
    const strix_tp_moe_desc *desc, const strix_stream *stream,
    strix_tp_moe_plan *out);
strix_status strix_tp_moe_plan_create_device_split(strix_context context,
    const strix_tp_moe_device_desc *desc, const strix_stream *stream,
    strix_tp_moe_plan *out);
/* Explicit new identity: K2 mul1 affine weights, rotated A8 groups of32,
 * integer group accumulation, fixed RN32 group/S4 reduction and deterministic
 * SiLU. This is NOT the original S4/DOT2ACC arithmetic. Only K2/mul1 weights
 * are admitted. Build with STRIX_TRELLIS_TP_MOE_A8R (OFF by default), then
 * query split workspace size from that library. Original creators never
 * select A8R, even in an A8R-enabled build. Existing stream/lifetime rules
 * and staged entry points apply; down_reference_core rejects A8R plans. */
strix_status strix_tp_moe_plan_create_a8r(strix_context context,
    const strix_tp_moe_desc *desc, const strix_stream *stream,
    strix_tp_moe_plan *out);
strix_status strix_tp_moe_plan_create_device_a8r(strix_context context,
    const strix_tp_moe_device_desc *desc, const strix_stream *stream,
    strix_tp_moe_plan *out);
/* Diagnostic staged B and C entry points require a split3 plan. P is consumed
 * directly from the workspace, allowing independent S4 checkpoint validation. */
strix_status strix_tp_moe_plan_down_core(strix_tp_moe_plan plan);
strix_status strix_tp_moe_plan_down_finish(strix_tp_moe_plan plan);
/* Qualification control: original two-phase S4 down core, exporting P only. */
strix_status strix_tp_moe_plan_down_reference_core(strix_tp_moe_plan plan);
strix_status strix_tp_moe_plan_submit(strix_tp_moe_plan plan);
/* Staged entry points share the same frozen plan. down reads the hidden
 * workspace from the latest gate_up (or caller-provided equivalent values).
 * Enforce completion/order using the plan's stream, as for ordinary HIP work. */
strix_status strix_tp_moe_plan_gate_up(strix_tp_moe_plan plan);
strix_status strix_tp_moe_plan_down(strix_tp_moe_plan plan);
/* Destroy only outside capture and after completion AND graph destruction:
 * the owned device routing table is referenced by captured kernel arguments. */
strix_status strix_tp_moe_plan_destroy(strix_tp_moe_plan plan);

/* Submission status covers validation/enqueue only. Asynchronous device errors
 * and synchronization are handled by the caller through HIP. Non-NULL stream
 * descriptors are mandatory; no implicit synchronization is promised. */
#ifdef __cplusplus
}
#endif
#endif
