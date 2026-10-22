#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gpu-site/scratch-trellis"
{
 hostname; TZ=:/etc/localtime date -Is
 /opt/rocm/core-10.1/bin/hipcc --version
 sha256sum build-hip/strix_trellis_gpu_runtime_fused_v6 build-hip/libstrix_trellis_fused_v6.a build-hip/strix_trellis_engine_example_v6 build-hip/strix_trellis_api_linear_bench_fused_v6 build-hip/strix_trellis_graph_linear_bench_fused_v6 runtime-v6-source/*
} > runtime-v6.meta
./build-hip/strix_trellis_gpu_runtime_fused_v6 --route all runtime-fixtures/*.stgpu gpu-dense-prefill-tests/*.stgpu gpu-numerical-tests/*.stgpu
./build-hip/strix_trellis_engine_example_v6 integration-plain/wkv input.f32 integration-result-plain-v6.f32 0
./build-hip/strix_trellis_engine_example_v6 integration-bias/wkv input.f32 integration-result-bias-v6.f32 0
python3 - <<'PY'
import struct,math,hashlib,json
from pathlib import Path
for mode in ['plain','bias']:
 p=Path(f'integration-result-{mode}-v6.f32');raw=p.read_bytes();ref=Path(f'expected-{mode}.f32').read_bytes()
 a=struct.unpack('<512f',raw);b=struct.unpack('<512f',ref)
 assert all(math.isfinite(v) for v in a)
 maxabs=max(abs(x-y) for x,y in zip(a,b));maxref=max(abs(v) for v in b)
 nrmse=math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/max(sum(x*x for x in b),1e-30))
 assert maxabs<=5e-4*max(1,maxref) and nrmse<=3e-5,(mode,maxabs,nrmse)
 print(json.dumps(dict(integration=mode,maxabs=maxabs,maxref=maxref,nrmse=nrmse,sha256=hashlib.sha256(raw).hexdigest(),passed=True)))
PY

./build-hip/strix_trellis_api_linear_bench_fused_v6 \
 gpu-dense-prefill-tests/layers.0.attn.wq_b.stgpu \
 gpu-dense-prefill-tests/layers.0.attn.wo_b.stgpu \
 gpu-dense-prefill-tests/layers.0.ffn.shared_experts.w1.stgpu \
 gpu-dense-prefill-tests/layers.0.ffn.shared_experts.w2.stgpu \
 gpu-dense-prefill-tests/layers.0.ffn.shared_experts.w3.stgpu \
 gpu-dense-prefill-tests/layers.0.attn.wq_a.stgpu \
 gpu-dense-prefill-tests/layers.0.attn.wo_a.slice.0.stgpu \
 runtime-fixtures/eaq-wkv-k6-mul1.stgpu > api-linear-v6.log

./build-hip/strix_trellis_graph_linear_bench_fused_v6 \
 gpu-dense-prefill-tests/layers.0.attn.wq_b.stgpu \
 gpu-dense-prefill-tests/layers.0.attn.wo_b.stgpu \
 gpu-dense-prefill-tests/layers.0.ffn.shared_experts.w1.stgpu \
 gpu-dense-prefill-tests/layers.0.ffn.shared_experts.w2.stgpu \
 gpu-dense-prefill-tests/layers.0.ffn.shared_experts.w3.stgpu \
 gpu-dense-prefill-tests/layers.0.attn.wq_a.stgpu \
 gpu-dense-prefill-tests/layers.0.attn.wo_a.slice.0.stgpu \
 runtime-fixtures/eaq-wkv-k6-mul1.stgpu > hip-graphs-v6.log
