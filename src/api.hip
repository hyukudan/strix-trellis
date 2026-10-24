#include "strix_trellis.h"
#include "backend.hpp"
#include "gemv_fused_both.hpp"
#include "fused_both_policy.hpp"
#include "fused_output_policy.hpp"

#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>

struct strix_context_impl {
    int device;
    size_t weights_count = 0;
    bool fused_output_capable = false;
    bool fused_both_capable = false;
};
struct strix_weights_impl {
    strix_context owner;
    strix_trellis::DeviceWeights device{};
    void* allocation = nullptr;
    size_t allocation_bytes = 0;
    size_t plan_refs = 0;
};
struct strix_linear_plan_impl {
    strix_context owner;
    strix_matmul_desc desc;
    hipStream_t stream;
    int rows;
    bool prefer_wmma;
};

namespace {
constexpr uint64_t alignment = STRIX_TRELLIS_WORKSPACE_ALIGNMENT;
template<class E> uint32_t enum_bits(const E& value) {
    // C clients may supply arbitrary enum bytes. Do not load an invalid C++ enum.
    static_assert(sizeof(E) == sizeof(uint32_t), "Standard enum ABI required");
    uint32_t raw = 0;
    std::memcpy(&raw, &value, sizeof(raw));
    return raw;
}
template<class T> strix_status header(const T* d) {
    if (!d) return STRIX_ERROR_INVALID_ARGUMENT;
    if (d->struct_size != sizeof(T) || d->abi_version != STRIX_TRELLIS_ABI_VERSION)
        return STRIX_ERROR_ABI_MISMATCH;
    for (auto v : d->reserved) if (v) return STRIX_ERROR_INVALID_ARGUMENT;
    return STRIX_SUCCESS;
}
strix_status hip_status(hipError_t e) {
    if (e == hipSuccess) return STRIX_SUCCESS;
    if (e == hipErrorOutOfMemory) return STRIX_ERROR_OUT_OF_MEMORY;
    return STRIX_ERROR_BACKEND;
}
bool multiply(uint64_t a, uint64_t b, uint64_t& out) {
    if (b && a > UINT64_MAX / b) return false;
    out = a * b;
    return true;
}
bool address(const void* p, uint64_t bytes) {
    return bytes <= SIZE_MAX && bytes <= UINTPTR_MAX - reinterpret_cast<uintptr_t>(p);
}
bool overlap(const void* a, uint64_t an, const void* b, uint64_t bn) {
    if (!an || !bn) return false;
    const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x < y + bn && y < x + an;
}
strix_status context_check(strix_context c) {
    if (!c) return STRIX_ERROR_INVALID_ARGUMENT;
    int device = -1;
    auto e = hipGetDevice(&device);
    if (e != hipSuccess) return hip_status(e);
    return device == c->device ? STRIX_SUCCESS : STRIX_ERROR_INVALID_ARGUMENT;
}
strix_status stream_check(strix_context c, const strix_stream* s, hipStream_t& stream) {
    auto status = header(s);
    if (status != STRIX_SUCCESS) return status;
    stream = reinterpret_cast<hipStream_t>(s->hip_stream);
    if (!stream) return STRIX_SUCCESS;
    // Require an explicit live stream on this device, or NULL for its default.
    hipDevice_t device = -1;
    unsigned flags = 0;
    if (hipStreamGetDevice(stream, &device) != hipSuccess ||
        hipStreamGetFlags(stream, &flags) != hipSuccess || device != c->device)
        return STRIX_ERROR_INVALID_ARGUMENT;
    return STRIX_SUCCESS;
}
strix_status device_buffer(strix_context c, const void* p, uint64_t bytes, uint64_t align) {
    if (!bytes) return STRIX_SUCCESS;
    if (!p || !address(p, bytes) || reinterpret_cast<uintptr_t>(p) % align)
        return STRIX_ERROR_INVALID_ARGUMENT;
    hipPointerAttribute_t attr{};
    if (hipPointerGetAttributes(&attr, p) != hipSuccess ||
        attr.type != hipMemoryTypeDevice || attr.device != c->device || attr.isManaged)
        return STRIX_ERROR_INVALID_ARGUMENT;
    hipDeviceptr_t base{};
    size_t size = 0;
    if (hipMemGetAddressRange(&base, &size, const_cast<void*>(p)) != hipSuccess)
        return STRIX_ERROR_INVALID_ARGUMENT;
    const auto start = reinterpret_cast<uintptr_t>(base), at = reinterpret_cast<uintptr_t>(p);
    if (at < start || at - start > size || bytes > size - (at - start))
        return STRIX_ERROR_INSUFFICIENT_BUFFER;
    return STRIX_SUCCESS;
}

struct Region { uint64_t offset, bytes; };
strix_status metadata_check(const strix_exl3_metadata_v1& m, uint64_t rows, uint64_t columns) {
    if (m.struct_size != sizeof(m) || m.transport_version != STRIX_TRELLIS_TRANSPORT_VERSION)
        return STRIX_ERROR_ABI_MISMATCH;
    for (auto v : m.reserved) if (v) return STRIX_ERROR_INVALID_ARGUMENT;
    if (m.flags) return STRIX_ERROR_INVALID_ARGUMENT;
    if (m.rows != rows || m.columns != columns || !m.rows || !m.columns ||
        m.rows > INT_MAX || m.columns > INT_MAX || m.rows % 128 || m.columns % 128)
        return STRIX_ERROR_UNSUPPORTED_FORMAT;
    if (m.bits_x2 < 2 || m.bits_x2 > 16 || m.bits_x2 % 2)
        return STRIX_ERROR_UNSUPPORTED_FORMAT;
    uint32_t marker = 0;
    switch (enum_bits(m.codebook)) {
    case STRIX_CODEBOOK_3INST: marker = STRIX_EXL3_MARKER_3INST; break;
    case STRIX_CODEBOOK_MCG: marker = STRIX_EXL3_MARKER_MCG; break;
    case STRIX_CODEBOOK_MUL1: marker = STRIX_EXL3_MARKER_MUL1; break;
    default: return STRIX_ERROR_UNSUPPORTED_FORMAT;
    }
    if (marker != m.marker) return STRIX_ERROR_UNSUPPORTED_FORMAT;
    const uint16_t endian = 1;
    if (*reinterpret_cast<const unsigned char*>(&endian) != 1)
        return STRIX_ERROR_UNSUPPORTED_FORMAT;
    uint64_t packed = 0;
    if (!multiply(m.rows, m.columns, packed) || !multiply(packed / 8, m.bits_x2 / 2, packed))
        return STRIX_ERROR_UNSUPPORTED_FORMAT;
    if (m.trellis_bytes != packed || m.suh_bytes != m.columns * 2 ||
        m.svh_bytes != m.rows * 2 || (m.bias_bytes && m.bias_bytes != m.rows * 4) ||
        (!m.bias_bytes && m.bias_offset)) return STRIX_ERROR_UNSUPPORTED_FORMAT;
    return STRIX_SUCCESS;
}
strix_status transport_check(const strix_exl3_transport_desc* d, strix_exl3_metadata_v1& m) {
    auto s = header(d);
    if (s != STRIX_SUCCESS) return s;
    if (enum_bits(d->format) != STRIX_TRANSPORT_EXL3_CANONICAL ||
        d->transport_version != STRIX_TRELLIS_TRANSPORT_VERSION)
        return STRIX_ERROR_UNSUPPORTED_FORMAT;
    if (!d->metadata || d->metadata_bytes != sizeof(m)) return STRIX_ERROR_INVALID_ARGUMENT;
    std::memcpy(&m, d->metadata, sizeof(m));
    s = metadata_check(m, d->rows, d->columns);
    if (s != STRIX_SUCCESS) return s;
    if (!d->data || !address(d->data, d->data_bytes)) return STRIX_ERROR_INVALID_ARGUMENT;
    const Region regions[] = {{m.trellis_offset, m.trellis_bytes}, {m.suh_offset, m.suh_bytes},
        {m.svh_offset, m.svh_bytes}, {m.bias_offset, m.bias_bytes}};
    for (size_t i = 0; i < 4; ++i) {
        const auto r = regions[i];
        if (r.offset > d->data_bytes || r.bytes > d->data_bytes - r.offset)
            return STRIX_ERROR_INSUFFICIENT_BUFFER;
        const uint64_t a = i == 1 || i == 2 ? 2 : 4;
        if (r.bytes && r.offset % a) return STRIX_ERROR_UNSUPPORTED_FORMAT;
        for (size_t j = 0; j < i; ++j)
            if (r.bytes && regions[j].bytes && r.offset < regions[j].offset + regions[j].bytes &&
                regions[j].offset < r.offset + r.bytes) return STRIX_ERROR_INVALID_ARGUMENT;
    }
    return STRIX_SUCCESS;
}

strix_status parts_check(const strix_exl3_parts_desc* d, strix_exl3_metadata_v1& m) {
    auto s = header(d);
    if (s != STRIX_SUCCESS) return s;
    if (d->transport_version != STRIX_TRELLIS_TRANSPORT_VERSION)
        return STRIX_ERROR_UNSUPPORTED_FORMAT;
    // Check borrowed ranges before any allocation or upload. Sources need not be
    // adjacent or aligned, but overlap would make their format ambiguous.
    const strix_host_region regions[] = {d->trellis, d->suh, d->svh, d->bias};
    uint64_t offsets[4]{}, total = 0;
    for (size_t i = 0; i < 4; ++i) {
        const auto& r = regions[i];
        if (!r.bytes) {
            if (r.data || i != 3) return STRIX_ERROR_INVALID_ARGUMENT;
            continue;
        }
        if (!r.data || !address(r.data, r.bytes)) return STRIX_ERROR_INVALID_ARGUMENT;
        for (size_t j = 0; j < i; ++j)
            if (overlap(r.data, r.bytes, regions[j].data, regions[j].bytes))
                return STRIX_ERROR_INVALID_ARGUMENT;
        if (total > UINT64_MAX - 3) return STRIX_ERROR_UNSUPPORTED_FORMAT;
        total = (total + 3) & ~UINT64_C(3);
        offsets[i] = total;
        if (r.bytes > UINT64_MAX - total) return STRIX_ERROR_UNSUPPORTED_FORMAT;
        total += r.bytes;
    }
    m = {};
    m.struct_size = sizeof(m);
    m.transport_version = d->transport_version;
    m.rows = d->rows; m.columns = d->columns; m.bits_x2 = d->bits_x2;
    std::memcpy(&m.codebook, &d->codebook, sizeof(m.codebook));
    m.marker = d->marker;
    m.trellis_offset = offsets[0]; m.trellis_bytes = d->trellis.bytes;
    m.suh_offset = offsets[1]; m.suh_bytes = d->suh.bytes;
    m.svh_offset = offsets[2]; m.svh_bytes = d->svh.bytes;
    m.bias_offset = offsets[3]; m.bias_bytes = d->bias.bytes;
    return metadata_check(m, d->rows, d->columns);
}

strix_status prepare_regions(strix_context c, const strix_exl3_metadata_v1& m,
                             const strix_host_region* regions, hipStream_t hs,
                             strix_weights* out) {
    auto w = new (std::nothrow) strix_weights_impl;
    if (!w) return STRIX_ERROR_OUT_OF_MEMORY;
    w->owner = c;
    w->device.input = static_cast<int>(m.columns);
    w->device.output = static_cast<int>(m.rows);
    w->device.bits = m.bits_x2 / 2;
    w->device.codebook = static_cast<int>(enum_bits(m.codebook));
    w->device.fused_output_capable = c->fused_output_capable;
    w->device.fused_both_capable = c->fused_both_capable;
    size_t offsets[4]{};
    uint64_t total = 0;
    for (size_t i = 0; i < 4; ++i) {
        if (total > SIZE_MAX - (alignment - 1)) { delete w; return STRIX_ERROR_OUT_OF_MEMORY; }
        total = (total + alignment - 1) & ~(alignment - 1);
        offsets[i] = static_cast<size_t>(total);
        if (regions[i].bytes > SIZE_MAX - total) { delete w; return STRIX_ERROR_OUT_OF_MEMORY; }
        total += regions[i].bytes;
    }
    w->allocation_bytes = static_cast<size_t>(total);
    auto e = hipMalloc(&w->allocation, w->allocation_bytes);
    if (e != hipSuccess) { delete w; return hip_status(e); }
    auto base = static_cast<unsigned char*>(w->allocation);
    w->device.trellis = reinterpret_cast<const uint32_t*>(base + offsets[0]);
    w->device.suh = reinterpret_cast<const uint16_t*>(base + offsets[1]);
    w->device.svh = reinterpret_cast<const uint16_t*>(base + offsets[2]);
    w->device.bias = m.bias_bytes ? reinterpret_cast<const float*>(base + offsets[3]) : nullptr;
    for (size_t i = 0; i < 4; ++i) {
        if (!regions[i].bytes) continue;
        e = hipMemcpyAsync(base + offsets[i], regions[i].data,
                           static_cast<size_t>(regions[i].bytes), hipMemcpyHostToDevice, hs);
        if (e != hipSuccess) break;
    }
    // Drain successful earlier uploads even on failure, before returning host ownership.
    const auto completion = hipStreamSynchronize(hs);
    if (e != hipSuccess || completion != hipSuccess) {
        (void)hipFree(w->allocation);
        delete w;
        return hip_status(e != hipSuccess ? e : completion);
    }
    ++c->weights_count;
    *out = w;
    return STRIX_SUCCESS;
}

struct Call { int rows; uint64_t input_bytes, output_bytes, workspace_bytes; };
strix_status tensor_check(const strix_tensor_desc& t, uint64_t width, uint32_t rank,
                          uint64_t rows, uint64_t& bytes) {
    auto s = header(&t);
    if (s != STRIX_SUCCESS) return s;
    if (enum_bits(t.dtype) != STRIX_DTYPE_F32 || t.rank != rank) return STRIX_ERROR_UNSUPPORTED_FORMAT;
    if (rank == 1) {
        if (t.shape[0] != width || t.shape[1] || t.strides[0] != 1 || t.strides[1])
            return STRIX_ERROR_UNSUPPORTED_FORMAT;
    } else if (t.shape[0] != rows || t.shape[1] != width ||
               t.strides[0] != width || t.strides[1] != 1) return STRIX_ERROR_UNSUPPORTED_FORMAT;
    if (!multiply(rows, width, bytes) || !multiply(bytes, sizeof(float), bytes) ||
        !address(t.data, bytes)) return STRIX_ERROR_INVALID_ARGUMENT;
    if (bytes > t.buffer_bytes) return STRIX_ERROR_INSUFFICIENT_BUFFER;
    if (bytes && (!t.data || reinterpret_cast<uintptr_t>(t.data) % alignof(float)))
        return STRIX_ERROR_INVALID_ARGUMENT;
    return STRIX_SUCCESS;
}
// Query validates host descriptors/aliasing without touching device tensor memory.
strix_status call_check(strix_context c, const strix_matmul_desc* d, Call& call) {
    auto s = header(d);
    if (s != STRIX_SUCCESS) return s;
    if (!d->weights || d->weights->owner != c || !std::isfinite(d->alpha) || !std::isfinite(d->beta))
        return STRIX_ERROR_INVALID_ARGUMENT;
    const auto& w = d->weights->device;
    const auto rank = d->input.rank;
    if (rank != 1 && rank != 2) return STRIX_ERROR_UNSUPPORTED_FORMAT;
    uint64_t rows = rank == 1 ? 1 : d->input.shape[0];
    if (rows > INT_MAX) return STRIX_ERROR_UNSUPPORTED_FORMAT;
    call.rows = static_cast<int>(rows);
    s = tensor_check(d->input, w.input, rank, rows, call.input_bytes);
    if (s != STRIX_SUCCESS) return s;
    s = tensor_check(d->output, w.output, rank, rows, call.output_bytes);
    if (s != STRIX_SUCCESS) return s;
    call.workspace_bytes = rows ? strix_trellis::linear_workspace_bytes(w, call.rows) : 0;
    if (call.workspace_bytes == SIZE_MAX) return STRIX_ERROR_UNSUPPORTED_FORMAT;
    if (overlap(d->input.data, call.input_bytes, d->output.data, call.output_bytes) ||
        overlap(d->input.data, call.input_bytes, d->weights->allocation, d->weights->allocation_bytes) ||
        overlap(d->output.data, call.output_bytes, d->weights->allocation, d->weights->allocation_bytes))
        return STRIX_ERROR_INVALID_ARGUMENT;
    return STRIX_SUCCESS;
}
strix_status storage_check(strix_context c, const strix_matmul_desc& d, const Call& call,
                          void* workspace, uint64_t bytes) {
    if (bytes < call.workspace_bytes) return STRIX_ERROR_INSUFFICIENT_BUFFER;
    if (!address(workspace, bytes) || (bytes && (!workspace ||
        reinterpret_cast<uintptr_t>(workspace) % alignment))) return STRIX_ERROR_INVALID_ARGUMENT;
    if (overlap(workspace, bytes, d.input.data, call.input_bytes) ||
        overlap(workspace, bytes, d.output.data, call.output_bytes) ||
        overlap(workspace, bytes, d.weights->allocation, d.weights->allocation_bytes))
        return STRIX_ERROR_INVALID_ARGUMENT;
    if (!call.rows) return STRIX_SUCCESS;
    auto s = device_buffer(c, d.input.data, call.input_bytes, alignof(float));
    if (s != STRIX_SUCCESS) return s;
    s = device_buffer(c, d.output.data, call.output_bytes, alignof(float));
    if (s != STRIX_SUCCESS) return s;
    return device_buffer(c, workspace, bytes, alignment);
}
strix_status submit(strix_context c, const strix_matmul_desc* d, const strix_stream* stream, bool gemv) {
    auto s = context_check(c);
    if (s != STRIX_SUCCESS) return s;
    hipStream_t hs{};
    s = stream_check(c, stream, hs);
    if (s != STRIX_SUCCESS) return s;
    Call call{};
    s = call_check(c, d, call);
    if (s != STRIX_SUCCESS) return s;
    if ((gemv && call.rows > 8) || (!gemv && d->input.rank != 2))
        return STRIX_ERROR_UNSUPPORTED_FORMAT;
    s = storage_check(c, *d, call, d->workspace, d->workspace_bytes);
    if (s != STRIX_SUCCESS || !call.rows) return s;
    return hip_status(strix_trellis::launch_linear(d->weights->device,
        static_cast<const float*>(d->input.data), static_cast<float*>(d->output.data),
        call.rows, d->workspace, hs, d->alpha, d->beta, !gemv));
}
strix_status grouped_check(strix_context c, const strix_grouped_moe_desc* d, uint64_t& maximum) {
    auto s = header(d);
    if (s != STRIX_SUCCESS) return s;
    if (d->group_count > SIZE_MAX / sizeof(strix_matmul_desc) ||
        (d->group_count && !d->groups) ||
        !address(d->groups, d->group_count * sizeof(strix_matmul_desc)))
        return STRIX_ERROR_INVALID_ARGUMENT;
    maximum = 0;
    for (uint64_t i = 0; i < d->group_count; ++i) {
        const auto& g = d->groups[i];
        if (g.workspace || g.workspace_bytes) return STRIX_ERROR_INVALID_ARGUMENT;
        Call a{};
        s = call_check(c, &g, a);
        if (s != STRIX_SUCCESS) return s;
        if (a.workspace_bytes > maximum) maximum = a.workspace_bytes;
        // No input/output slice sharing, including cross-group read/write hazards.
        for (uint64_t j = 0; j < i; ++j) {
            Call b{};
            s = call_check(c, &d->groups[j], b);
            if (s != STRIX_SUCCESS) return s;
            const auto& h = d->groups[j];
            if (overlap(g.input.data, a.input_bytes, h.input.data, b.input_bytes) ||
                overlap(g.input.data, a.input_bytes, h.output.data, b.output_bytes) ||
                overlap(g.output.data, a.output_bytes, h.input.data, b.input_bytes) ||
                overlap(g.output.data, a.output_bytes, h.output.data, b.output_bytes) ||
                overlap(g.input.data, a.input_bytes, h.weights->allocation, h.weights->allocation_bytes) ||
                overlap(h.input.data, b.input_bytes, g.weights->allocation, g.weights->allocation_bytes) ||
                overlap(g.output.data, a.output_bytes, h.weights->allocation, h.weights->allocation_bytes) ||
                overlap(h.output.data, b.output_bytes, g.weights->allocation, g.weights->allocation_bytes))
                return STRIX_ERROR_INVALID_ARGUMENT;
        }
    }
    return STRIX_SUCCESS;
}
} // namespace

extern "C" {
uint32_t strix_get_abi_version(void) { return STRIX_TRELLIS_ABI_VERSION; }
const char* strix_status_string(strix_status s) {
    switch (s) {
    case STRIX_SUCCESS: return "success";
    case STRIX_ERROR_INVALID_ARGUMENT: return "invalid argument";
    case STRIX_ERROR_ABI_MISMATCH: return "ABI mismatch";
    case STRIX_ERROR_UNSUPPORTED_FORMAT: return "unsupported format";
    case STRIX_ERROR_UNSUPPORTED_DEVICE: return "unsupported device";
    case STRIX_ERROR_INSUFFICIENT_BUFFER: return "insufficient buffer";
    case STRIX_ERROR_OUT_OF_MEMORY: return "out of memory";
    case STRIX_ERROR_BACKEND: return "HIP/backend error";
    case STRIX_ERROR_NOT_IMPLEMENTED: return "not implemented";
    default: return "unknown status";
    }
}
void strix_matmul_desc_init(strix_matmul_desc* d) {
    if (!d) return;
    *d = {};
    d->struct_size = sizeof(*d);
    d->abi_version = STRIX_TRELLIS_ABI_VERSION;
    d->alpha = 1.0f;
    d->input.struct_size = d->output.struct_size = sizeof(strix_tensor_desc);
    d->input.abi_version = d->output.abi_version = STRIX_TRELLIS_ABI_VERSION;
    d->input.dtype = d->output.dtype = STRIX_DTYPE_F32;
}
strix_status strix_context_create(const strix_context_desc* d, strix_context* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = nullptr;
    auto s = header(d);
    if (s != STRIX_SUCCESS) return s;
    if (d->flags || d->device_ordinal < 0) return STRIX_ERROR_INVALID_ARGUMENT;
    hipDeviceProp_t props{};
    auto e = hipGetDeviceProperties(&props, d->device_ordinal);
    if (e == hipErrorInvalidDevice) return STRIX_ERROR_UNSUPPORTED_DEVICE;
    if (e != hipSuccess) return hip_status(e);
    if (std::strncmp(props.gcnArchName, "gfx1151", 7) ||
        (props.gcnArchName[7] && props.gcnArchName[7] != ':') || props.warpSize != 32)
        return STRIX_ERROR_UNSUPPORTED_DEVICE;
    int active_device=-1;
    e=hipGetDevice(&active_device);
    if(e!=hipSuccess) return hip_status(e);
    bool fused_output_capable=false;
    if(active_device==d->device_ordinal && strix_trellis::fused_output_device_limits(
            props.maxThreadsPerBlock,props.maxThreadsDim[0],props.sharedMemPerBlock)) {
        e=strix_trellis::query_fused_output_resources(d->device_ordinal,
            props.sharedMemPerBlock,&fused_output_capable);
        if(e!=hipSuccess) return hip_status(e);
    }
    bool fused_both_capable=false;
    if(active_device==d->device_ordinal && strix_trellis::fused_both_device_limits(
            props.maxThreadsPerBlock,props.maxThreadsDim[0],props.sharedMemPerBlock)) {
        e=strix_trellis::query_fused_both_resources(props.sharedMemPerBlock,&fused_both_capable);
        if(e!=hipSuccess) return hip_status(e);
    }
    auto c = new (std::nothrow) strix_context_impl;
    if (!c) return STRIX_ERROR_OUT_OF_MEMORY;
    c->device = d->device_ordinal;
    c->fused_output_capable=fused_output_capable;
    c->fused_both_capable=fused_both_capable;
    *out = c;
    return STRIX_SUCCESS;
}
strix_status strix_context_destroy(strix_context c) {
    if (!c) return STRIX_ERROR_INVALID_ARGUMENT;
    if (c->weights_count) return STRIX_ERROR_INVALID_ARGUMENT;
    delete c;
    return STRIX_SUCCESS;
}
strix_status strix_weights_prepare(strix_context c, const strix_exl3_transport_desc* d,
                                   const strix_stream* stream, strix_weights* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = nullptr;
    auto s = context_check(c);
    if (s != STRIX_SUCCESS) return s;
    hipStream_t hs{};
    s = stream_check(c, stream, hs);
    if (s != STRIX_SUCCESS) return s;
    strix_exl3_metadata_v1 m{};
    s = transport_check(d, m);
    if (s != STRIX_SUCCESS) return s;
    const auto base = static_cast<const unsigned char*>(d->data);
    const strix_host_region regions[] = {
        {base + m.trellis_offset, m.trellis_bytes}, {base + m.suh_offset, m.suh_bytes},
        {base + m.svh_offset, m.svh_bytes},
        {m.bias_bytes ? base + m.bias_offset : nullptr, m.bias_bytes}};
    return prepare_regions(c, m, regions, hs, out);
}
strix_status strix_weights_prepare_parts(strix_context c, const strix_exl3_parts_desc* d,
                                         const strix_stream* stream, strix_weights* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = nullptr;
    auto s = context_check(c);
    if (s != STRIX_SUCCESS) return s;
    hipStream_t hs{};
    s = stream_check(c, stream, hs);
    if (s != STRIX_SUCCESS) return s;
    strix_exl3_metadata_v1 m{};
    s = parts_check(d, m);
    if (s != STRIX_SUCCESS) return s;
    const strix_host_region regions[] = {d->trellis, d->suh, d->svh, d->bias};
    return prepare_regions(c, m, regions, hs, out);
}
strix_status strix_weights_destroy(strix_weights w) {
    if (!w || w->plan_refs) return STRIX_ERROR_INVALID_ARGUMENT;
    auto s = context_check(w->owner);
    if (s != STRIX_SUCCESS) return s;
    const auto e = hipFree(w->allocation);
    if (e != hipSuccess) return hip_status(e);
    --w->owner->weights_count;
    delete w;
    return STRIX_SUCCESS;
}
strix_status strix_matmul_workspace_bytes(strix_context c, const strix_matmul_desc* d, uint64_t* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = 0;
    auto s = context_check(c);
    if (s != STRIX_SUCCESS) return s;
    Call call{};
    s = call_check(c, d, call);
    if (s == STRIX_SUCCESS) *out = call.workspace_bytes;
    return s;
}
strix_status strix_gemv(strix_context c, const strix_matmul_desc* d, const strix_stream* stream) {
    return submit(c, d, stream, true);
}
strix_status strix_gemm(strix_context c, const strix_matmul_desc* d, const strix_stream* stream) {
    return submit(c, d, stream, false);
}
strix_status strix_linear_plan_create(strix_context c, const strix_matmul_desc* d,
                                      const strix_stream* stream, uint32_t operation,
                                      strix_linear_plan* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = nullptr;
    if (operation != STRIX_OPERATION_GEMV && operation != STRIX_OPERATION_GEMM)
        return STRIX_ERROR_INVALID_ARGUMENT;
    auto s = context_check(c);
    if (s != STRIX_SUCCESS) return s;
    hipStream_t hs{};
    s = stream_check(c, stream, hs);
    if (s != STRIX_SUCCESS) return s;
    Call call{};
    s = call_check(c, d, call);
    if (s != STRIX_SUCCESS) return s;
    const bool wmma = operation == STRIX_OPERATION_GEMM;
    if ((!wmma && call.rows > 8) || (wmma && d->input.rank != 2))
        return STRIX_ERROR_UNSUPPORTED_FORMAT;
    s = storage_check(c, *d, call, d->workspace, d->workspace_bytes);
    if (s != STRIX_SUCCESS) return s;
    if (d->weights->plan_refs == SIZE_MAX) return STRIX_ERROR_OUT_OF_MEMORY;
    auto plan = new (std::nothrow) strix_linear_plan_impl{c, *d, hs, call.rows, wmma};
    if (!plan) return STRIX_ERROR_OUT_OF_MEMORY;
    ++d->weights->plan_refs;
    *out = plan;
    return STRIX_SUCCESS;
}
strix_status strix_linear_plan_submit(strix_linear_plan p) {
    if (!p) return STRIX_ERROR_INVALID_ARGUMENT;
    const auto s = context_check(p->owner);
    if (s != STRIX_SUCCESS || !p->rows) return s;
    const auto& d = p->desc;
    return hip_status(strix_trellis::launch_linear(d.weights->device,
        static_cast<const float*>(d.input.data), static_cast<float*>(d.output.data),
        p->rows, d.workspace, p->stream, d.alpha, d.beta, p->prefer_wmma));
}
strix_status strix_linear_plan_destroy(strix_linear_plan p) {
    if (!p) return STRIX_ERROR_INVALID_ARGUMENT;
    --p->desc.weights->plan_refs;
    delete p;
    return STRIX_SUCCESS;
}
strix_status strix_grouped_moe_workspace_bytes(strix_context c, const strix_grouped_moe_desc* d,
                                              uint64_t* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out = 0;
    auto s = context_check(c);
    if (s != STRIX_SUCCESS) return s;
    uint64_t maximum = 0;
    s = grouped_check(c, d, maximum);
    if (s == STRIX_SUCCESS) *out = maximum;
    return s;
}
strix_status strix_grouped_moe(strix_context c, const strix_grouped_moe_desc* d,
                              const strix_stream* stream) {
    auto s = context_check(c);
    if (s != STRIX_SUCCESS) return s;
    hipStream_t hs{};
    s = stream_check(c, stream, hs);
    if (s != STRIX_SUCCESS) return s;
    uint64_t maximum = 0;
    s = grouped_check(c, d, maximum);
    if (s != STRIX_SUCCESS) return s;
    if (d->workspace_bytes < maximum) return STRIX_ERROR_INSUFFICIENT_BUFFER;
    if (!address(d->workspace, d->workspace_bytes) ||
        (d->workspace_bytes && (!d->workspace || reinterpret_cast<uintptr_t>(d->workspace) % alignment)))
        return STRIX_ERROR_INVALID_ARGUMENT;
    // Validate every group and shared workspace before enqueueing the first group.
    for (uint64_t i = 0; i < d->group_count; ++i) {
        Call call{};
        s = call_check(c, &d->groups[i], call);
        if (s != STRIX_SUCCESS) return s;
        s = storage_check(c, d->groups[i], call, d->workspace, d->workspace_bytes);
        if (s != STRIX_SUCCESS) return s;
    }
    for (uint64_t i = 0; i < d->group_count; ++i) {
        const auto& g = d->groups[i];
        const int rows = g.input.rank == 1 ? 1 : static_cast<int>(g.input.shape[0]);
        if (!rows) continue;
        s = hip_status(strix_trellis::launch_linear(g.weights->device,
            static_cast<const float*>(g.input.data), static_cast<float*>(g.output.data), rows,
            d->workspace, hs, g.alpha, g.beta, rows > 8));
        if (s != STRIX_SUCCESS) return s;
    }
    return STRIX_SUCCESS;
}
} // extern "C"

#include "tp_api.inc"
#include "tp_split_api.inc"
#include "dense_r1_api.inc"
#include "dense_r1_group_api.inc"
