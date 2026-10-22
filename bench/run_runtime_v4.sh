#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gpu-site/scratch-trellis"
{
 hostname; TZ=:/etc/localtime date -Is
 /opt/rocm/core-10.1/bin/hipcc --version
 sha256sum build-hip/strix_trellis_gpu_runtime_lds_v4 build-hip/libstrix_trellis_lds_v4.a build-hip/strix_trellis_engine_example_v4 runtime-v4-source/*
} > runtime-v4.meta
./build-hip/strix_trellis_gpu_runtime_lds_v4 --route all runtime-fixtures/*.stgpu gpu-dense-prefill-tests/*.stgpu
./build-hip/strix_trellis_engine_example_v4 integration-plain/wkv input.f32 integration-result-plain-v4.f32 0
./build-hip/strix_trellis_engine_example_v4 integration-bias/wkv input.f32 integration-result-bias-v4.f32 0
python3 - <<'PY'
import struct,math,hashlib,json
from pathlib import Path
for mode in ['plain','bias']:
 p=Path(f'integration-result-{mode}-v4.f32');raw=p.read_bytes();ref=Path(f'expected-{mode}.f32').read_bytes()
 a=struct.unpack('<512f',raw);b=struct.unpack('<512f',ref)
 assert all(math.isfinite(v) for v in a)
 maxabs=max(abs(x-y) for x,y in zip(a,b));maxref=max(abs(v) for v in b)
 nrmse=math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/max(sum(x*x for x in b),1e-30))
 assert maxabs<=5e-4*max(1,maxref) and nrmse<=3e-5,(mode,maxabs,nrmse)
 print(json.dumps(dict(integration=mode,maxabs=maxabs,maxref=maxref,nrmse=nrmse,sha256=hashlib.sha256(raw).hexdigest(),passed=True)))
PY
