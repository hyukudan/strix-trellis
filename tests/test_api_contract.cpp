// CPU-only contract test. The generated hip/hip_runtime.h includes the first
// branch of this file; no HIP library or GPU is used. See docs/API.md.
#ifdef STRIX_API_FAKE_HIP
#ifndef STRIX_API_FAKE_HIP_TYPES
#define STRIX_API_FAKE_HIP_TYPES
#include <cstddef>
using hipError_t = int;
constexpr hipError_t hipSuccess = 0, hipErrorOutOfMemory = 1, hipErrorInvalidDevice = 2, hipErrorInvalidValue = 3;
struct hipFuncAttributes { int maxThreadsPerBlock; size_t sharedSizeBytes, localSizeBytes; int maxDynamicSharedSizeBytes=63488; };
using hipStream_t = void*;
using hipDevice_t = int;
using hipDeviceptr_t = void*;
constexpr int hipMemoryTypeDevice = 1, hipMemcpyHostToDevice = 1;
struct hipDeviceProp_t { char gcnArchName[256]; int warpSize; int maxThreadsPerBlock;
    int maxThreadsDim[3]; size_t sharedMemPerBlock; };
struct hipPointerAttribute_t { int type, device, isManaged; };
hipError_t hipGetDevice(int*);
hipError_t hipGetDeviceProperties(hipDeviceProp_t*, int);
hipError_t hipStreamGetDevice(hipStream_t, hipDevice_t*);
hipError_t hipStreamGetFlags(hipStream_t, unsigned*);
hipError_t hipPointerGetAttributes(hipPointerAttribute_t*, const void*);
hipError_t hipMemGetAddressRange(hipDeviceptr_t*, size_t*, hipDeviceptr_t);
hipError_t hipMalloc(void**, size_t);
hipError_t hipFree(void*);
hipError_t hipMemcpyAsync(void*, const void*, size_t, int, hipStream_t);
hipError_t hipMemsetAsync(void*, int, size_t, hipStream_t);
hipError_t hipStreamSynchronize(hipStream_t);
#endif
#else
#include "../src/api.hip"
#include "../src/launch_policy.hpp"
#include "../src/v14_core_policy.hpp"
#include "../src/fused_output_resources.hpp"
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>
#include <vector>

namespace fake {
std::map<uintptr_t, size_t> buffers;
int device = 0, wave = 32, allocs = 0, copies = 0, syncs = 0, launches = 0;
int fail_copy = 0, fail_sync = 0, fail_launch = 0;
int copy_failure_at = -1;
int pointer_queries = 0, range_queries = 0, stream_queries = 0;
std::vector<const void*> copy_sources;
const float* last_input = nullptr;
float* last_output = nullptr;
void* last_workspace = nullptr;
hipStream_t last_stream = nullptr;
int last_rows = -1;
int fail_get_device = 0, probe_error = 0, probes = 0, probe_device = -1;
int max_threads = 1024, max_dimension = 1024;
size_t shared_bytes = 65536;
bool probe_supported = true;
int both_attributes=0,both_error=0,both_dynamic=63488;
bool both_supported=true;
int direct_routes=0,fallback_routes=0;
int attribute_calls=0, attribute_error_index=-1, insufficient_index=-1;
const void* functions[4]={reinterpret_cast<void*>(4),reinterpret_cast<void*>(8),
    reinterpret_cast<void*>(12),reinterpret_cast<void*>(16)};
bool wmma = false;
float alpha = 0, beta = 0;
const char* arch = "gfx1151:sramecc-";
auto find(const void* p) {
    auto i = buffers.upper_bound(reinterpret_cast<uintptr_t>(p));
    if (i == buffers.begin()) return buffers.end();
    --i;
    return reinterpret_cast<uintptr_t>(p) - i->first < i->second ? i : buffers.end();
}
}
hipError_t hipGetDevice(int* d) { *d = fake::device; return fake::fail_get_device?99:0; }
hipError_t hipGetDeviceProperties(hipDeviceProp_t* p, int d) {
    if (d != 0) return hipErrorInvalidDevice;
    std::strcpy(p->gcnArchName, fake::arch); p->warpSize = fake::wave;
    p->maxThreadsPerBlock=fake::max_threads;p->maxThreadsDim[0]=fake::max_dimension;
    p->sharedMemPerBlock=fake::shared_bytes;return 0;
}
hipError_t hipStreamGetDevice(hipStream_t s, hipDevice_t* d) {
    ++fake::stream_queries;
    *d = s == reinterpret_cast<void*>(1) ? 0 : 1; return 0;
}
hipError_t hipStreamGetFlags(hipStream_t, unsigned* f) { ++fake::stream_queries; *f = 0; return 0; }
hipError_t hipPointerGetAttributes(hipPointerAttribute_t* a, const void* p) {
    ++fake::pointer_queries;
    if (fake::find(p) == fake::buffers.end()) return 99;
    *a = {hipMemoryTypeDevice, 0, 0}; return 0;
}
hipError_t hipMemGetAddressRange(hipDeviceptr_t* p, size_t* n, hipDeviceptr_t at) {
    ++fake::range_queries;
    auto i = fake::find(at); if (i == fake::buffers.end()) return 99;
    *p = reinterpret_cast<void*>(i->first); *n = i->second; return 0;
}
hipError_t hipMalloc(void** p, size_t n) {
    // Keep fake allocations apart: out-of-bounds tests must not depend on malloc adjacency.
    *p = std::aligned_alloc(256, ((n + 255) & ~size_t(255)) + 4096);
    if (!*p) return hipErrorOutOfMemory;
    fake::buffers[reinterpret_cast<uintptr_t>(*p)] = n; ++fake::allocs; return 0;
}
hipError_t hipFree(void* p) { fake::buffers.erase(reinterpret_cast<uintptr_t>(p)); std::free(p); return 0; }
hipError_t hipMemcpyAsync(void* d, const void* s, size_t n, int, hipStream_t) {
    ++fake::copies; fake::copy_sources.push_back(s);
    if (fake::fail_copy || fake::copies == fake::copy_failure_at) return 99;
    std::memcpy(d, s, n); return 0;
}
hipError_t hipStreamSynchronize(hipStream_t) { ++fake::syncs; return fake::fail_sync ? 99 : 0; }
namespace strix_trellis {
hipError_t query_fused_output_resources(int device,size_t shared,bool* supported) {
    *supported=false;++fake::probes;fake::probe_device=device;
    assert(device==fake::device && shared==fake::shared_bytes);
    if(fake::probe_error) return fake::probe_error;
    return probe_fused_output_images<hipError_t,hipFuncAttributes>(device,shared,supported,
        hipSuccess,hipErrorInvalidValue,fake::functions,hipGetDevice,
        [](hipFuncAttributes* a,const void* f) {
            const int i=fake::attribute_calls++%4;
            assert(f==fake::functions[i]);
            if(i==fake::attribute_error_index) return 99;
            const int required[4]={1024,1024,256,512};
            *a={required[i],size_t(required[i])*24,0};
            if(!fake::probe_supported || i==fake::insufficient_index) a->maxThreadsPerBlock=0;
            return 0;
        });
}
hipError_t query_fused_both_resources(size_t shared,bool* supported) {
    return probe_fused_both_image<hipError_t,hipFuncAttributes>(shared,supported,
        hipSuccess,hipErrorInvalidValue,reinterpret_cast<void*>(20),
        [](hipFuncAttributes* a,const void* f) {
            assert(f==reinterpret_cast<void*>(20));++fake::both_attributes;
            if(fake::both_error)return fake::both_error;
            *a={1024,2048,0,fake::both_dynamic};
            if(!fake::both_supported)a->maxThreadsPerBlock=0;
            return 0;
        });
}
size_t linear_workspace_bytes(const DeviceWeights&, int rows) { return size_t(rows) * 1024; }
hipError_t launch_linear(const DeviceWeights& weights, const float* input, float* output, int rows, void* workspace, hipStream_t stream,
                        float a, float b, bool w) {
    fake::last_input = input; fake::last_output = output; fake::last_rows = rows;
    fake::last_workspace = workspace; fake::last_stream = stream;
    FusedBothLayout layout{};
    if(fused_both_layout(weights.input,weights.output,weights.bits,weights.codebook,
        rows,w,weights.fused_both_capable,weights.trellis,input,weights.suh,weights.svh,weights.bias,output,layout))++fake::direct_routes;
    else ++fake::fallback_routes;
    ++fake::launches; fake::alpha = a; fake::beta = b; fake::wmma = w;
    return fake::fail_launch ? 99 : 0;
}
}
template<class T> T descriptor() {
    T d{}; d.struct_size = sizeof(T); d.abi_version = STRIX_TRELLIS_ABI_VERSION; return d;
}
int main() {
    // Initializers remain available when experimental kernels are not built.
    {
        strix_tp_moe_desc host;
        strix_tp_moe_device_desc device;
        std::memset(&host, 0xa5, sizeof(host)); std::memset(&device, 0xa5, sizeof(device));
        strix_tp_moe_desc_init(&host); strix_tp_moe_device_desc_init(&device);
        strix_tp_moe_desc_init(nullptr); strix_tp_moe_device_desc_init(nullptr);
        assert(host.struct_size == sizeof(host) && host.abi_version == 1 && host.alpha == 1 && host.beta == 0);
        assert(device.struct_size == sizeof(device) && device.abi_version == 1 && device.alpha == 1 && device.beta == 0);
        for (const auto* tensor : {&host.input, &host.output, &device.input, &device.output}) {
            assert(tensor->struct_size == sizeof(*tensor) && tensor->abi_version == 1);
            assert(tensor->dtype == STRIX_DTYPE_F32 && tensor->rank == 2 && !tensor->data && !tensor->buffer_bytes);
            assert(!tensor->shape[0] && !tensor->shape[1] && !tensor->strides[0] && !tensor->strides[1]);
            for (auto value : tensor->reserved) assert(value == 0);
        }
        assert(!host.experts && !host.expert_count && !host.route_ids && !host.route_ids_count &&
               !host.route_scores && !host.route_scores_count && !host.workspace && !host.workspace_bytes);
        assert(!device.experts && !device.expert_count && !device.route_ids && !device.route_ids_bytes &&
               !device.route_scores && !device.route_scores_bytes && !device.workspace && !device.workspace_bytes);
        for (auto value : host.reserved) assert(value == 0);
        for (auto value : device.reserved) assert(value == 0);
    }
#ifndef STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP
    // Disabled optional APIs clear outputs and cannot accidentally submit work,
    // probe HIP metadata or require valid runtime objects to report availability.
    {
        const int allocs = fake::allocs, copies = fake::copies, syncs = fake::syncs, launches = fake::launches;
        const int pointers = fake::pointer_queries, ranges = fake::range_queries, streams = fake::stream_queries;
        const int probes = fake::probes, attributes = fake::attribute_calls, both = fake::both_attributes;
        auto dense = reinterpret_cast<strix_tp_dense_plan>(uintptr_t(1));
        assert(strix_tp_dense_plan_create(nullptr, nullptr, nullptr, &dense) == STRIX_ERROR_NOT_IMPLEMENTED && !dense);
        dense = reinterpret_cast<strix_tp_dense_plan>(uintptr_t(1));
        assert(strix_tp_dense_plan_create_tiled(nullptr, nullptr, nullptr, UINT32_MAX, &dense) == STRIX_ERROR_NOT_IMPLEMENTED && !dense);
        assert(strix_tp_dense_plan_create(nullptr, nullptr, nullptr, nullptr) == STRIX_ERROR_INVALID_ARGUMENT);
        assert(strix_tp_dense_plan_create_tiled(nullptr, nullptr, nullptr, 0, nullptr) == STRIX_ERROR_INVALID_ARGUMENT);
        assert(strix_tp_dense_plan_submit(nullptr) == STRIX_ERROR_NOT_IMPLEMENTED);
        assert(strix_tp_dense_plan_destroy(nullptr) == STRIX_ERROR_NOT_IMPLEMENTED);
        uint64_t bytes = UINT64_MAX;
        assert(strix_tp_moe_workspace_bytes(nullptr, nullptr, &bytes) == STRIX_ERROR_NOT_IMPLEMENTED && !bytes);
        assert(strix_tp_moe_workspace_bytes(nullptr, nullptr, nullptr) == STRIX_ERROR_INVALID_ARGUMENT);
        auto moe = reinterpret_cast<strix_tp_moe_plan>(uintptr_t(1));
        assert(strix_tp_moe_plan_create(nullptr, nullptr, nullptr, &moe) == STRIX_ERROR_NOT_IMPLEMENTED && !moe);
        moe = reinterpret_cast<strix_tp_moe_plan>(uintptr_t(1));
        assert(strix_tp_moe_plan_create_device(nullptr, nullptr, nullptr, &moe) == STRIX_ERROR_NOT_IMPLEMENTED && !moe);
        assert(strix_tp_moe_plan_create(nullptr, nullptr, nullptr, nullptr) == STRIX_ERROR_INVALID_ARGUMENT);
        assert(strix_tp_moe_plan_create_device(nullptr, nullptr, nullptr, nullptr) == STRIX_ERROR_INVALID_ARGUMENT);
        assert(strix_tp_moe_plan_submit(nullptr) == STRIX_ERROR_NOT_IMPLEMENTED);
        assert(strix_tp_moe_plan_gate_up(nullptr) == STRIX_ERROR_NOT_IMPLEMENTED);
        assert(strix_tp_moe_plan_down(nullptr) == STRIX_ERROR_NOT_IMPLEMENTED);
        assert(strix_tp_moe_plan_destroy(nullptr) == STRIX_ERROR_NOT_IMPLEMENTED);
        assert(fake::allocs == allocs && fake::copies == copies && fake::syncs == syncs && fake::launches == launches);
        assert(fake::pointer_queries == pointers && fake::range_queries == ranges && fake::stream_queries == streams);
        assert(fake::probes == probes && fake::attribute_calls == attributes && fake::both_attributes == both);
    }
#endif
    // Exercise the actual immutable WMMA tile and A-padding policy on current headers.
    for(int rows:{64,128,256}) {
        strix_trellis::DeviceWeights q{nullptr,nullptr,nullptr,nullptr,8192,5120,5,2,false};
        assert(strix_trellis::wmma_tile_rows(q,rows)==64);
        bool pad18=(q.bits==5 && q.codebook==2) || strix_trellis::medium_prefill_candidate(q,rows);
        assert(pad18);
        assert(strix_trellis::wmma_packed_candidate(q.input,q.output,q.bits,q.codebook,rows,256/rows,
            strix_trellis::wmma_tile_rows(q,rows),pad18,true,true,true));
        if(rows==256)assert(!strix_trellis::medium_prefill_candidate(q,rows));
    }

    // Independent new flag, real context creation, exactly one distinct probe.
    auto both_cd=descriptor<strix_context_desc>();
    for(int active:{0,1}) for(int dynamic:{-1,0,16383,16384,63488}) {
        fake::device=active;fake::both_dynamic=dynamic;
        int before=fake::both_attributes;strix_context ctx=nullptr;
        assert(strix_context_create(&both_cd,&ctx)==STRIX_SUCCESS);
        assert(ctx->fused_both_capable==(active==0 && dynamic>=16384));
        assert(fake::both_attributes-before==(active==0?1:0));
        assert(strix_context_destroy(ctx)==STRIX_SUCCESS);
    }
    fake::device=0;fake::both_dynamic=63488;
    fake::both_error=99;strix_context ctx=nullptr;
    assert(strix_context_create(&both_cd,&ctx)==STRIX_ERROR_BACKEND && !ctx);
    fake::both_error=0;
    fake::shared_bytes=18432;
    int old=fake::probes, fresh=fake::both_attributes;
    assert(strix_context_create(&both_cd,&ctx)==STRIX_SUCCESS);
    assert(!ctx->fused_output_capable && ctx->fused_both_capable);
    assert(fake::probes==old && fake::both_attributes==fresh+1);
    assert(strix_context_destroy(ctx)==STRIX_SUCCESS);fake::shared_bytes=65536;
    fake::both_supported=false;
    assert(strix_context_create(&both_cd,&ctx)==STRIX_SUCCESS);
    assert(ctx->fused_output_capable && !ctx->fused_both_capable);
    assert(strix_context_destroy(ctx)==STRIX_SUCCESS);fake::both_supported=true;
    fake::attribute_calls=0;
    // Exercise the production probe algorithm, including no-device-access guards,
    // active ordinal recheck, every per-image limit and every later error.
    for(int ordinal:{0,1,3}) for(int active:{0,1,3})
        for(int fail=-1;fail<4;++fail) for(int bad=-1;bad<4;++bad)
            for(int kind=0;kind<3;++kind) {
                int calls=0,gets=0;bool supported=true;
                auto get=[&](int* d){++gets;*d=active;return 0;};
                auto attr=[&](hipFuncAttributes* a,const void* f){
                    int i=calls++;assert(f==fake::functions[i]);
                    if(i==fail)return 99;
                    const int t[4]={1024,1024,256,512};*a={t[i],24576,0};
                    if(i==bad){if(kind==0)--a->maxThreadsPerBlock;if(kind==1)++a->sharedSizeBytes;if(kind==2)++a->localSizeBytes;}
                    return 0;};
                auto probe=[&](bool* out){return strix_trellis::probe_fused_output_images<hipError_t,hipFuncAttributes>(
                    ordinal,24576,out,hipSuccess,hipErrorInvalidValue,fake::functions,get,attr);};
                assert(probe(nullptr)==hipErrorInvalidValue && gets==0 && calls==0);
                int result=probe(&supported);
                if(active!=ordinal){assert(result==0 && !supported && calls==0);}
                else if(fail>=0){assert(result==99 && !supported && calls==fail+1);}
                else {assert(result==0 && supported==(bad<0) && calls==4);}
            }
    strix_trellis::DeviceWeights empty{},aggregate{nullptr,nullptr,nullptr,nullptr,128,256,6,2};
    assert(!empty.fused_output_capable && !aggregate.fused_output_capable);
    static_assert(sizeof(strix_trellis::DeviceWeights)==56);
    static_assert(offsetof(strix_trellis::DeviceWeights,fused_output_capable)==48);
    static_assert(offsetof(strix_trellis::DeviceWeights,codebook)==44);
    static_assert(offsetof(strix_trellis::DeviceWeights,fused_both_capable)==49);
    assert(!empty.fused_both_capable && !aggregate.fused_both_capable);
    assert(strix_get_abi_version() == 1);
    assert(std::strcmp(strix_status_string(STRIX_ERROR_BACKEND), "HIP/backend error") == 0);
    auto cd = descriptor<strix_context_desc>();
    strix_context c = nullptr;
    cd.reserved[0] = 1; assert(strix_context_create(&cd, &c) == STRIX_ERROR_INVALID_ARGUMENT);
    cd.reserved[0] = 0; cd.struct_size--; assert(strix_context_create(&cd, &c) == STRIX_ERROR_ABI_MISMATCH);
    cd.struct_size++; fake::wave = 64; assert(strix_context_create(&cd, &c) == STRIX_ERROR_UNSUPPORTED_DEVICE);
    fake::wave = 32; fake::arch = "gfx11510"; assert(strix_context_create(&cd, &c) == STRIX_ERROR_UNSUPPORTED_DEVICE);
    fake::arch = "gfx1151";
    // Resource limits disable only the optional route. They never poison a context.
    for(int field=0;field<3;++field) {
        const int before=fake::probes;
        if(field==0)fake::max_threads=1023;
        if(field==1)fake::max_dimension=512;
        if(field==2)fake::shared_bytes=24575;
        assert(strix_context_create(&cd,&c)==STRIX_SUCCESS && !c->fused_output_capable);
        assert(fake::probes==before);assert(strix_context_destroy(c)==STRIX_SUCCESS);
        fake::max_threads=1024;fake::max_dimension=1024;fake::shared_bytes=65536;
    }
    fake::device=1;const int inactive_probes=fake::probes;
    assert(strix_context_create(&cd,&c)==STRIX_SUCCESS && !c->fused_output_capable);
    assert(fake::probes==inactive_probes);assert(strix_context_destroy(c)==STRIX_SUCCESS);fake::device=0;
    fake::fail_get_device=1;assert(strix_context_create(&cd,&c)==STRIX_ERROR_BACKEND && !c);fake::fail_get_device=0;
    fake::probe_error=99;assert(strix_context_create(&cd,&c)==STRIX_ERROR_BACKEND && !c);fake::probe_error=0;
    for(int fail=0;fail<4;++fail) {
        for(int insufficient=-1;insufficient<fail;++insufficient) {
            fake::attribute_calls=0;fake::attribute_error_index=fail;
            fake::insufficient_index=insufficient;
            assert(strix_context_create(&cd,&c)==STRIX_ERROR_BACKEND && !c);
            assert(fake::attribute_calls==fail+1);
        }
    }
    fake::attribute_error_index=-1;
    for(int insufficient=0;insufficient<4;++insufficient) {
        fake::attribute_calls=0;fake::insufficient_index=insufficient;
        assert(strix_context_create(&cd,&c)==STRIX_SUCCESS && !c->fused_output_capable);
        assert(fake::attribute_calls==4);assert(strix_context_destroy(c)==STRIX_SUCCESS);
    }
    fake::attribute_calls=0;fake::insufficient_index=-1;
    fake::probe_error=hipErrorOutOfMemory;assert(strix_context_create(&cd,&c)==STRIX_ERROR_OUT_OF_MEMORY && !c);fake::probe_error=0;
    fake::probe_supported=false;
    strix_context fallback=nullptr;
    assert(strix_context_create(&cd,&fallback)==STRIX_SUCCESS && !fallback->fused_output_capable);
    fake::probe_supported=true;
    assert(strix_context_create(&cd,&c)==STRIX_SUCCESS && c->fused_output_capable && fake::probe_device==0);
    assert(!fallback->fused_output_capable);
    const int creation_probes=fake::probes,creation_attributes=fake::attribute_calls,creation_both=fake::both_attributes;
    assert(c->fused_both_capable);
    auto stream = descriptor<strix_stream>();
    strix_exl3_metadata_v1 m{};
    m.struct_size = sizeof(m); m.transport_version = 1; m.rows = 256; m.columns = 128;
    m.bits_x2 = 12; m.codebook = STRIX_CODEBOOK_MUL1; m.marker = STRIX_EXL3_MARKER_MUL1;
    m.trellis_bytes = m.rows * m.columns * 6 / 8;
    m.suh_offset = m.trellis_bytes; m.suh_bytes = m.columns * 2;
    m.svh_offset = m.suh_offset + m.suh_bytes; m.svh_bytes = m.rows * 2;
    m.bias_offset = m.svh_offset + m.svh_bytes; m.bias_bytes = m.rows * 4;
    std::vector<unsigned char> blob(m.rows * m.columns + m.suh_bytes + m.svh_bytes + m.bias_bytes, 0x3c);
    auto t = descriptor<strix_exl3_transport_desc>();
    t.format = STRIX_TRANSPORT_EXL3_CANONICAL; t.transport_version = 1;
    t.rows = m.rows; t.columns = m.columns; t.data = blob.data(); t.data_bytes = blob.size();
    t.metadata = &m; t.metadata_bytes = sizeof(m);
    strix_weights w = nullptr;
    const auto original = m;
    m.suh_offset = UINT64_MAX; assert(strix_weights_prepare(c, &t, &stream, &w) == STRIX_ERROR_INSUFFICIENT_BUFFER);
    m = original; m.svh_offset = m.suh_offset; assert(strix_weights_prepare(c, &t, &stream, &w) == STRIX_ERROR_INVALID_ARGUMENT);
    m = original; m.bits_x2 = 3; assert(strix_weights_prepare(c, &t, &stream, &w) == STRIX_ERROR_UNSUPPORTED_FORMAT);
    m = original; m.marker ^= 1; assert(strix_weights_prepare(c, &t, &stream, &w) == STRIX_ERROR_UNSUPPORTED_FORMAT);
    m = original; m.rows = 128; assert(strix_weights_prepare(c, &t, &stream, &w) == STRIX_ERROR_UNSUPPORTED_FORMAT);
    m = original; m.reserved[0] = 1; assert(strix_weights_prepare(c, &t, &stream, &w) == STRIX_ERROR_INVALID_ARGUMENT);
    m = original; fake::fail_copy = 1; auto buffers = fake::buffers.size();
    assert(strix_weights_prepare(c, &t, &stream, &w) == STRIX_ERROR_BACKEND && !w);
    assert(fake::buffers.size() == buffers); fake::fail_copy = 0;
    fake::fail_sync = 1; assert(strix_weights_prepare(c, &t, &stream, &w) == STRIX_ERROR_BACKEND && !w);
    fake::fail_sync = 0;
    // Every advertised integer rate/codebook and absent bias reaches preparation.
    for (uint32_t bits = 1; bits <= 8; ++bits) {
        for (int cb = 0; cb <= 2; ++cb) {
            m = original; m.bits_x2 = 2 * bits; m.codebook = static_cast<strix_codebook>(cb);
            m.marker = cb == 0 ? STRIX_EXL3_MARKER_3INST : cb == 1 ? STRIX_EXL3_MARKER_MCG : STRIX_EXL3_MARKER_MUL1;
            m.trellis_bytes = m.rows * m.columns * bits / 8;
            m.suh_offset = m.trellis_bytes; m.svh_offset = m.suh_offset + m.suh_bytes;
            m.bias_bytes = m.bias_offset = 0;
            strix_weights trial = nullptr;
            assert(strix_weights_prepare(c, &t, &stream, &trial) == STRIX_SUCCESS);
            assert(trial->device.bits == static_cast<int>(bits) && trial->device.codebook == cb && !trial->device.bias);
            assert(strix_weights_destroy(trial) == STRIX_SUCCESS);
        }
    }
    m = original;
    assert(strix_weights_prepare(c, &t, &stream, &w) == STRIX_SUCCESS);
    assert(w->device.fused_output_capable && w->device.fused_both_capable);
    strix_weights fallback_weights=nullptr;
    assert(strix_weights_prepare(fallback,&t,&stream,&fallback_weights)==STRIX_SUCCESS);
    assert(!fallback_weights->device.fused_output_capable);
    assert(strix_weights_destroy(fallback_weights)==STRIX_SUCCESS);
    assert(w->device.bits == 6 && w->device.input == 128 && w->device.output == 256);
    assert(std::memcmp(w->device.svh, blob.data() + m.svh_offset, m.svh_bytes) == 0);
    // Borrow independent mmap-compatible ranges; no combined blob or source copy.
    std::vector<unsigned char> packed(m.trellis_bytes + 1, 0x17),
        left(m.suh_bytes + 1, 0x29), right(m.svh_bytes + 1, 0x43), bias(m.bias_bytes + 1, 0x65);
    auto parts = descriptor<strix_exl3_parts_desc>();
    parts.transport_version = 1; parts.rows = m.rows; parts.columns = m.columns;
    parts.bits_x2 = m.bits_x2; parts.codebook = m.codebook; parts.marker = m.marker;
    parts.trellis = {packed.data() + 1, m.trellis_bytes};
    parts.suh = {left.data() + 1, m.suh_bytes}; parts.svh = {right.data() + 1, m.svh_bytes};
    parts.bias = {bias.data() + 1, m.bias_bytes}; // Unaligned host pointers are supported.
    strix_weights part_weights = nullptr;
    const int region_copies = fake::copies, region_syncs = fake::syncs, region_allocs = fake::allocs;
    assert(strix_weights_prepare_parts(c, &parts, &stream, &part_weights) == STRIX_SUCCESS);
    assert(fake::copies == region_copies + 4 && fake::syncs == region_syncs + 1 && fake::allocs == region_allocs + 1);
    const void* destinations[] = {part_weights->device.trellis, part_weights->device.suh,
                                 part_weights->device.svh, part_weights->device.bias};
    const strix_host_region sources[] = {parts.trellis, parts.suh, parts.svh, parts.bias};
    for (size_t i = 0; i < 4; ++i) {
        assert(fake::copy_sources[size_t(region_copies) + i] == sources[i].data);
        assert(std::memcmp(destinations[i], sources[i].data, sources[i].bytes) == 0);
    }
    assert(strix_weights_destroy(part_weights) == STRIX_SUCCESS);
    const auto good_parts = parts;
    auto reject_parts = [&](strix_status expected) {
        const int before_alloc = fake::allocs, before_copy = fake::copies, before_sync = fake::syncs;
        part_weights = reinterpret_cast<strix_weights>(uintptr_t(1));
        assert(strix_weights_prepare_parts(c, &parts, &stream, &part_weights) == expected && !part_weights);
        assert(fake::allocs == before_alloc && fake::copies == before_copy && fake::syncs == before_sync);
    };
    parts.struct_size--; reject_parts(STRIX_ERROR_ABI_MISMATCH); parts = good_parts;
    parts.reserved[2] = 1; reject_parts(STRIX_ERROR_INVALID_ARGUMENT); parts = good_parts;
    parts.transport_version = 2; reject_parts(STRIX_ERROR_UNSUPPORTED_FORMAT); parts = good_parts;
    parts.trellis.bytes--; reject_parts(STRIX_ERROR_UNSUPPORTED_FORMAT); parts = good_parts;
    parts.columns = UINT64_MAX; reject_parts(STRIX_ERROR_UNSUPPORTED_FORMAT); parts = good_parts;
    parts.bits_x2 = 5; reject_parts(STRIX_ERROR_UNSUPPORTED_FORMAT); parts = good_parts;
    parts.marker ^= 1; reject_parts(STRIX_ERROR_UNSUPPORTED_FORMAT); parts = good_parts;
    parts.suh.data = parts.trellis.data; reject_parts(STRIX_ERROR_INVALID_ARGUMENT); parts = good_parts;
    parts.suh.data = nullptr; reject_parts(STRIX_ERROR_INVALID_ARGUMENT); parts = good_parts;
    parts.suh.data = reinterpret_cast<void*>(UINTPTR_MAX - 1); reject_parts(STRIX_ERROR_INVALID_ARGUMENT); parts = good_parts;
    parts.bias.bytes = 0; reject_parts(STRIX_ERROR_INVALID_ARGUMENT); parts = good_parts;
    for (uint32_t raw : {UINT32_MAX, 0x10002u}) {
        std::memcpy(&parts.codebook, &raw, sizeof(raw));
        reject_parts(STRIX_ERROR_UNSUPPORTED_FORMAT);
        std::memcpy(&parts.codebook, &good_parts.codebook, sizeof(raw));
    }
    parts.bias = {nullptr, 0};
    assert(strix_weights_prepare_parts(c, &parts, &stream, &part_weights) == STRIX_SUCCESS);
    assert(!part_weights->device.bias);
    assert(strix_weights_destroy(part_weights) == STRIX_SUCCESS);
    parts = good_parts;
    const auto held_buffers = fake::buffers.size();
    const int failed_syncs = fake::syncs;
    fake::copy_failure_at = fake::copies + 3;
    assert(strix_weights_prepare_parts(c, &parts, &stream, &part_weights) == STRIX_ERROR_BACKEND && !part_weights);
    assert(fake::buffers.size() == held_buffers && fake::syncs == failed_syncs + 1);
    fake::copy_failure_at = -1;
    fake::fail_sync = 1;
    assert(strix_weights_prepare_parts(c, &parts, &stream, &part_weights) == STRIX_ERROR_BACKEND && !part_weights);
    assert(fake::buffers.size() == held_buffers); fake::fail_sync = 0;
    assert(strix_context_destroy(c) == STRIX_ERROR_INVALID_ARGUMENT);
    void *x, *y, *ws; hipMalloc(&x, 16384); hipMalloc(&y, 16384); hipMalloc(&ws, 32768);
    strix_matmul_desc d; strix_matmul_desc_init(&d);
    assert(d.alpha == 1 && d.beta == 0);
    d.weights = w; d.input.data = x; d.output.data = y;
    d.input.rank = d.output.rank = 2;
    d.input.shape[0] = d.output.shape[0] = 2;
    d.input.shape[1] = 128; d.output.shape[1] = 256;
    d.input.strides[0] = 128; d.output.strides[0] = 256;
    d.input.strides[1] = d.output.strides[1] = 1;
    d.input.buffer_bytes = d.output.buffer_bytes = 16384;
    uint64_t need = 0; assert(strix_matmul_workspace_bytes(c, &d, &need) == STRIX_SUCCESS && need == 2048);
    assert(strix_gemv(c, &d, &stream) == STRIX_ERROR_INSUFFICIENT_BUFFER);
    d.workspace = ws; d.workspace_bytes = need;
    int allocs = fake::allocs, copies = fake::copies, syncs = fake::syncs;
    d.alpha = 0.25f; d.beta = -0.5f;
    assert(strix_gemv(c, &d, &stream) == STRIX_SUCCESS && !fake::wmma);
    assert(fake::alpha == d.alpha && fake::beta == d.beta);
    assert(strix_gemm(c, &d, &stream) == STRIX_SUCCESS && fake::wmma);
    assert(fake::allocs == allocs && fake::copies == copies && fake::syncs == syncs);
    auto good = d;
    // Plans freeze descriptors and avoid repeated device/stream metadata queries.
    stream.hip_stream = reinterpret_cast<void*>(1);
    strix_linear_plan plan = nullptr, gemm_plan = nullptr;
    assert(strix_linear_plan_create(c, &d, &stream, STRIX_OPERATION_GEMV, &plan) == STRIX_SUCCESS);
    assert(strix_linear_plan_create(c, &d, &stream, STRIX_OPERATION_GEMM, &gemm_plan) == STRIX_SUCCESS);
    assert(w->plan_refs == 2 && strix_weights_destroy(w) == STRIX_ERROR_INVALID_ARGUMENT);
    const int pp = fake::pointer_queries, pr = fake::range_queries, ps = fake::stream_queries;
    allocs = fake::allocs; copies = fake::copies; syncs = fake::syncs;
    d.alpha = 99; d.input.data = nullptr; d.output.data = nullptr; d.workspace = nullptr;
    stream.hip_stream = reinterpret_cast<void*>(2); // Mutating caller descriptors cannot retarget a plan.
    for (int i = 0; i < 8; ++i) {
        assert(strix_linear_plan_submit(plan) == STRIX_SUCCESS && !fake::wmma);
        assert(fake::alpha == good.alpha && fake::beta == good.beta);
        assert(fake::last_input == good.input.data && fake::last_output == good.output.data);
        assert(fake::last_workspace == good.workspace && fake::last_rows == 2);
        assert(fake::last_stream == reinterpret_cast<void*>(1));
    }
    assert(strix_linear_plan_submit(gemm_plan) == STRIX_SUCCESS && fake::wmma);
    assert(fake::pointer_queries == pp && fake::range_queries == pr && fake::stream_queries == ps);
    assert(fake::allocs == allocs && fake::copies == copies && fake::syncs == syncs);
    const int plan_launches = fake::launches;
    fake::device = 1; assert(strix_linear_plan_submit(plan) == STRIX_ERROR_INVALID_ARGUMENT);
    assert(fake::launches == plan_launches); fake::device = 0;
    fake::fail_launch = 1; assert(strix_linear_plan_submit(plan) == STRIX_ERROR_BACKEND); fake::fail_launch = 0;
    assert(strix_linear_plan_destroy(plan) == STRIX_SUCCESS && w->plan_refs == 1);
    assert(strix_weights_destroy(w) == STRIX_ERROR_INVALID_ARGUMENT);
    assert(strix_linear_plan_destroy(gemm_plan) == STRIX_SUCCESS && w->plan_refs == 0);
    assert(strix_linear_plan_submit(nullptr) == STRIX_ERROR_INVALID_ARGUMENT);
    assert(strix_linear_plan_destroy(nullptr) == STRIX_ERROR_INVALID_ARGUMENT);
    d = good; stream.hip_stream = nullptr;
    assert(strix_linear_plan_create(c, &d, &stream, UINT32_MAX, &plan) == STRIX_ERROR_INVALID_ARGUMENT && !plan);
    d.output.data = d.input.data;
    assert(strix_linear_plan_create(c, &d, &stream, STRIX_OPERATION_GEMV, &plan) == STRIX_ERROR_INVALID_ARGUMENT && !plan);
    d = good;
    // Reject arbitrary C enum bytes under UBSan as well as in optimized builds.
    const uint32_t f32 = STRIX_DTYPE_F32;
    for (int output : {0, 1}) for (uint32_t raw : {UINT32_MAX, 0x10003u}) {
        auto& dtype = output ? d.output.dtype : d.input.dtype;
        std::memcpy(&dtype, &raw, sizeof(raw));
        const int before = fake::launches;
        assert(strix_gemv(c, &d, &stream) == STRIX_ERROR_UNSUPPORTED_FORMAT);
        assert(strix_gemm(c, &d, &stream) == STRIX_ERROR_UNSUPPORTED_FORMAT);
        assert(strix_linear_plan_create(c, &d, &stream, STRIX_OPERATION_GEMV, &plan) == STRIX_ERROR_UNSUPPORTED_FORMAT && !plan);
        assert(fake::launches == before);
        std::memcpy(&dtype, &f32, sizeof(f32));
    }
    for(float scalar:{std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){
        for(int field=0;field<2;++field){d=good;if(field)d.beta=scalar;else d.alpha=scalar;
            const int before=fake::launches;
            assert(strix_gemv(c,&d,&stream)==STRIX_ERROR_INVALID_ARGUMENT && fake::launches==before);}
    }
    d=good;
    d.output.data = x; assert(strix_gemm(c, &d, &stream) == STRIX_ERROR_INVALID_ARGUMENT);
    d = good; d.workspace = y; assert(strix_gemm(c, &d, &stream) == STRIX_ERROR_INVALID_ARGUMENT);
    d = good; d.workspace = static_cast<char*>(ws) + 4; assert(strix_gemm(c, &d, &stream) == STRIX_ERROR_INVALID_ARGUMENT);
    d = good; d.input.data = const_cast<uint32_t*>(w->device.trellis); assert(strix_gemm(c, &d, &stream) == STRIX_ERROR_INVALID_ARGUMENT);
    d = good; d.input.dtype = STRIX_DTYPE_BF16; assert(strix_gemm(c, &d, &stream) == STRIX_ERROR_UNSUPPORTED_FORMAT);
    d = good; d.alpha = std::numeric_limits<float>::infinity(); assert(strix_gemm(c, &d, &stream) == STRIX_ERROR_INVALID_ARGUMENT);
    d = good; d.output.reserved[3] = 1; assert(strix_gemm(c, &d, &stream) == STRIX_ERROR_INVALID_ARGUMENT);
    d = good; d.input.strides[1] = 2; assert(strix_gemm(c, &d, &stream) == STRIX_ERROR_UNSUPPORTED_FORMAT);
    d = good; d.input.buffer_bytes = 1; assert(strix_gemm(c, &d, &stream) == STRIX_ERROR_INSUFFICIENT_BUFFER);
    d = good; d.input.data = static_cast<char*>(x) + 16380; assert(strix_gemm(c, &d, &stream) == STRIX_ERROR_INSUFFICIENT_BUFFER);
    d = good; stream.hip_stream = reinterpret_cast<void*>(2); assert(strix_gemm(c, &d, &stream) == STRIX_ERROR_INVALID_ARGUMENT);
    stream.hip_stream = nullptr; fake::device = 1; assert(strix_gemm(c, &d, &stream) == STRIX_ERROR_INVALID_ARGUMENT); fake::device = 0;
    d = good; d.input.shape[0] = d.output.shape[0] = 9;
    assert(strix_gemv(c, &d, &stream) == STRIX_ERROR_UNSUPPORTED_FORMAT);
    d = good; d.input.rank = d.output.rank = 1;
    d.input.shape[0] = 128; d.output.shape[0] = 256; d.input.shape[1] = d.output.shape[1] = 0;
    d.input.strides[0] = d.output.strides[0] = 1; d.input.strides[1] = d.output.strides[1] = 0;
    assert(strix_gemv(c, &d, &stream) == STRIX_SUCCESS);
    assert(strix_gemm(c, &d, &stream) == STRIX_ERROR_UNSUPPORTED_FORMAT);
    d = good; d.input.shape[0] = d.output.shape[0] = 0; d.input.data = d.output.data = nullptr;
    d.input.buffer_bytes = d.output.buffer_bytes = 0; d.workspace = nullptr; d.workspace_bytes = 0;
    int launches = fake::launches;
    assert(strix_gemm(c, &d, &stream) == STRIX_SUCCESS && fake::launches == launches);
    assert(strix_gemv(c, &d, &stream) == STRIX_SUCCESS);
    assert(strix_matmul_workspace_bytes(c, &d, &need) == STRIX_SUCCESS && need == 0);
    const int empty_queries = fake::pointer_queries;
    assert(strix_linear_plan_create(c, &d, &stream, STRIX_OPERATION_GEMM, &plan) == STRIX_SUCCESS);
    assert(strix_linear_plan_submit(plan) == STRIX_SUCCESS && fake::launches == launches);
    assert(fake::pointer_queries == empty_queries);
    assert(strix_linear_plan_destroy(plan) == STRIX_SUCCESS);
    d.input.dtype = STRIX_DTYPE_F16; assert(strix_gemm(c, &d, &stream) == STRIX_ERROR_UNSUPPORTED_FORMAT);
    strix_matmul_desc groups[2] = {good, good};
    for (auto& g : groups) { g.workspace = nullptr; g.workspace_bytes = 0; }
    groups[1].input.data = static_cast<char*>(x) + 4096;
    groups[1].output.data = static_cast<char*>(y) + 8192;
    groups[1].input.shape[0] = groups[1].output.shape[0] = 4;
    auto gd = descriptor<strix_grouped_moe_desc>(); gd.groups = groups; gd.group_count = 2;
    gd.workspace = ws; gd.workspace_bytes = 4096;
    assert(strix_grouped_moe_workspace_bytes(c, &gd, &need) == STRIX_SUCCESS && need == 4096);
    assert(strix_grouped_moe(c, &gd, &stream) == STRIX_SUCCESS);
    groups[1].output.data = groups[0].input.data;
    launches = fake::launches; assert(strix_grouped_moe(c, &gd, &stream) == STRIX_ERROR_INVALID_ARGUMENT);
    assert(fake::launches == launches);
    groups[1].output.data = static_cast<char*>(y) + 8192;
    groups[1].input.dtype = STRIX_DTYPE_F16;
    assert(strix_grouped_moe(c, &gd, &stream) == STRIX_ERROR_UNSUPPORTED_FORMAT && fake::launches == launches);
    gd.group_count = 0; gd.groups = nullptr; gd.workspace = nullptr; gd.workspace_bytes = 0;
    assert(strix_grouped_moe(c, &gd, &stream) == STRIX_SUCCESS);
    fake::fail_launch = 1; assert(strix_gemm(c, &good, &stream) == STRIX_ERROR_BACKEND); fake::fail_launch = 0;
    // Actual API context -> prepare -> weights flag -> guarded backend route.
    // The backend itself is mocked; source ordering and HIP ELF are checked separately.
    auto tm=original;tm.rows=5120;tm.columns=8192;tm.bits_x2=10;
    tm.trellis_bytes=uint64_t(tm.rows)*tm.columns*5/8;
    tm.suh_offset=tm.trellis_bytes;tm.suh_bytes=tm.columns*2;
    tm.svh_offset=tm.suh_offset+tm.suh_bytes;tm.svh_bytes=tm.rows*2;
    tm.bias_offset=tm.bias_bytes=0;
    std::vector<unsigned char> target_blob(size_t(tm.svh_offset+tm.svh_bytes),0x3c);
    auto tt=t;tt.rows=tm.rows;tt.columns=tm.columns;tt.data=target_blob.data();
    tt.data_bytes=target_blob.size();tt.metadata=&tm;
    strix_weights target=nullptr;
    fake::both_error=99; // Must be irrelevant after successful context creation.
    assert(strix_weights_prepare(c,&tt,&stream,&target)==STRIX_SUCCESS);
    assert(target->device.fused_both_capable && target->device.fused_output_capable);
    void* tx=nullptr;void* ty=nullptr;void* tws=nullptr;
    hipMalloc(&tx,size_t(tm.columns)*8*4);hipMalloc(&ty,size_t(tm.rows)*8*4);hipMalloc(&tws,8192);
    strix_matmul_desc route{};strix_matmul_desc_init(&route);route.weights=target;
    route.input=good.input;route.output=good.output;
    route.input.data=tx;route.input.shape[0]=1;route.input.shape[1]=tm.columns;
    route.input.strides[0]=tm.columns;route.input.buffer_bytes=size_t(tm.columns)*8*4;
    route.output.data=ty;route.output.shape[0]=1;route.output.shape[1]=tm.rows;
    route.output.strides[0]=tm.rows;route.output.buffer_bytes=size_t(tm.rows)*8*4;
    route.workspace=tws;route.workspace_bytes=8192;
    int direct=fake::direct_routes,fallback_count=fake::fallback_routes;
    assert(strix_gemv(c,&route,&stream)==STRIX_SUCCESS && fake::direct_routes==direct+1);
    target->device.fused_both_capable=false;
    assert(strix_gemv(c,&route,&stream)==STRIX_SUCCESS && fake::fallback_routes==fallback_count+1);
    target->device.fused_both_capable=true;
    assert(strix_gemm(c,&route,&stream)==STRIX_SUCCESS && fake::fallback_routes==fallback_count+2);
    fake::fail_launch=1;direct=fake::direct_routes;fallback_count=fake::fallback_routes;
    assert(strix_gemv(c,&route,&stream)==STRIX_ERROR_BACKEND);
    assert(fake::direct_routes==direct+1 && fake::fallback_routes==fallback_count);
    fake::fail_launch=0;fake::both_error=0;
    assert(strix_weights_destroy(target)==STRIX_SUCCESS);
    hipFree(tx);hipFree(ty);hipFree(tws);
    assert(strix_weights_destroy(w) == STRIX_SUCCESS);
    assert(strix_context_destroy(c) == STRIX_SUCCESS);
    assert(strix_context_destroy(fallback)==STRIX_SUCCESS);
    assert(fake::both_attributes==creation_both);
    assert(fake::probes==creation_probes && fake::attribute_calls==creation_attributes);
    hipFree(x); hipFree(y); hipFree(ws); assert(fake::buffers.empty());
    std::cout << "API CPU contract checks passed (mock HIP/backend, no GPU).\n";
    return 0;
}
#endif
