// Included after TP/split validation helpers; no runtime environment dispatch.
#ifdef STRIX_TRELLIS_ENABLE_EXPERIMENTAL_TP
#include "dense_r1_policy.hpp"
namespace dense_r1_api {
using namespace strix_trellis;
strix_status validate(strix_context c, const strix_matmul_desc* d, unsigned tile,
                      unsigned identity, tp_split_api::Validated& v) {
    auto s = header(d);
    if (s != STRIX_SUCCESS) return s;
    if (identity > STRIX_DENSE_R1_LEGACY || !d->weights || d->weights->owner != c ||
        !std::isfinite(d->alpha) || !std::isfinite(d->beta)) return STRIX_ERROR_INVALID_ARGUMENT;
    const auto& w = d->weights->device;
    const unsigned rank = d->input.rank;
    const uint64_t rows = rank == 1 ? 1 : d->input.shape[0];
    if ((rank != 1 && rank != 2) || rows < 1 || rows > 8 ||
        !dense_r1_geometry(w.input,w.output) || (w.bits != 4 && w.bits != 5) || w.codebook != 2)
        return STRIX_ERROR_UNSUPPORTED_FORMAT;
    if (tile > rows) return STRIX_ERROR_INVALID_ARGUMENT;
    if (!tile) tile = unsigned(rows);
    uint64_t xs=0,ys=0;
    s = tp_api::tensor(d->input,w.input,rank,rows,v.xb,xs); if(s != STRIX_SUCCESS) return s;
    s = tp_api::tensor(d->output,w.output,rank,rows,v.yb[0],ys); if(s != STRIX_SUCCESS) return s;
    v.x=d->input.data; v.y[0]=d->output.data; v.weights[0]=d->weights;
    v.launch.member[0]={w,static_cast<const float*>(v.x),static_cast<float*>(v.y[0]),
        xs,ys,int(rows),d->alpha,d->beta,int(tile)};
    v.partial_bytes[0]=size_t(dense_r1_splits(w.input,w.output,identity != 0))*rows*w.output*4;
    return tp_split_api::aliases(v);
}
}
extern "C" strix_status strix_tp_dense_r1_workspace_bytes(strix_context c,
    const strix_matmul_desc* d, uint32_t tile, uint32_t identity, uint64_t* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out=0; auto s=context_check(c); if(s != STRIX_SUCCESS) return s;
    tp_split_api::Validated v; s=dense_r1_api::validate(c,d,tile,identity,v);
    if(s == STRIX_SUCCESS) *out=v.bytes;
    return s;
}
extern "C" strix_status strix_tp_dense_plan_create_r1(strix_context c,
    const strix_matmul_desc* d, const strix_stream* stream, uint32_t tile,
    uint32_t identity, strix_tp_dense_plan* out) {
    if (!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out=nullptr; auto s=context_check(c); if(s != STRIX_SUCCESS) return s;
    hipStream_t hs{}; s=stream_check(c,stream,hs); if(s != STRIX_SUCCESS) return s;
    tp_split_api::Validated v; s=dense_r1_api::validate(c,d,tile,identity,v); if(s != STRIX_SUCCESS) return s;
    if (d->workspace_bytes < v.bytes) return STRIX_ERROR_INSUFFICIENT_BUFFER;
    if (!d->workspace || !address(d->workspace,d->workspace_bytes) ||
        reinterpret_cast<uintptr_t>(d->workspace)%alignment ||
        overlap(d->workspace,d->workspace_bytes,v.x,v.xb) ||
        overlap(d->workspace,d->workspace_bytes,v.y[0],v.yb[0]) ||
        tp_api::touches_weights(d->workspace,d->workspace_bytes,d->weights)) return STRIX_ERROR_INVALID_ARGUMENT;
    s=device_buffer(c,v.x,v.xb,4); if(s != STRIX_SUCCESS) return s;
    s=device_buffer(c,v.y[0],v.yb[0],4); if(s != STRIX_SUCCESS) return s;
    s=device_buffer(c,d->workspace,d->workspace_bytes,alignment); if(s != STRIX_SUCCESS) return s;
    strix_trellis::DenseR1Launch a{};
    a.projection=v.launch.member[0];
    a.partial=static_cast<float*>(d->workspace);
    a.counters=reinterpret_cast<uint32_t*>(static_cast<char*>(d->workspace)+v.counter_offset);
    a.splits=strix_trellis::dense_r1_splits(a.projection.weights.input,a.projection.weights.output,identity != 0);
    const auto schedule=strix_trellis::dense_r1_schedule(a.projection.weights.input,
        a.projection.weights.output,a.projection.weights.bits,a.projection.rows_per_block,a.splits);
    a.columns_per_block=schedule.columns;
    a.prefetch=schedule.prefetch;
    a.outputs_per_wave=schedule.outputs_per_wave;
    const bool use_split=schedule.use_splitk04 &&
        strix_trellis::tp_split_geometry(a.projection.weights.input,a.projection.weights.output) &&
        a.splits==strix_trellis::tp_split_count(a.projection.weights.input,a.projection.weights.output);
    strix_trellis::TpSplitLaunch split{};
    split.member[0]=a.projection;split.partial[0]=a.partial;split.counters=a.counters;
    hipDeviceProp_t props{}; s=tp_api::properties(c,props); if(s != STRIX_SUCCESS) return s;
    bool supported=false;
    auto e=use_split ? strix_trellis::query_tp_split_resources(split,props.sharedMemPerBlock,&supported) :
        strix_trellis::query_dense_r1_resources(a,props.sharedMemPerBlock,&supported);
    if(e != hipSuccess) return hip_status(e);
    if(!supported) return STRIX_ERROR_UNSUPPORTED_DEVICE;
    auto p=new(std::nothrow) strix_tp_dense_plan_impl;
    if(!p) return STRIX_ERROR_OUT_OF_MEMORY;
    e=hipMemsetAsync(a.counters,0,v.counter_bytes,hs);
    const auto completion=hipStreamSynchronize(hs);
    if(e != hipSuccess || completion != hipSuccess) { delete p; return hip_status(e != hipSuccess ? e : completion); }
    *p={c,d->weights,a.projection,hs}; p->r1=a; p->use_r1=!use_split;p->use_split=use_split;p->split=split;
    ++p->weights->plan_refs; *out=p; return STRIX_SUCCESS;
}
#else
extern "C" strix_status strix_tp_dense_r1_workspace_bytes(strix_context,
    const strix_matmul_desc*,uint32_t,uint32_t,uint64_t* out) {
    if(!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out=0; return STRIX_ERROR_NOT_IMPLEMENTED;
}
extern "C" strix_status strix_tp_dense_plan_create_r1(strix_context,
    const strix_matmul_desc*,const strix_stream*,uint32_t,uint32_t,strix_tp_dense_plan* out) {
    if(!out) return STRIX_ERROR_INVALID_ARGUMENT;
    *out=nullptr; return STRIX_ERROR_NOT_IMPLEMENTED;
}
#endif
