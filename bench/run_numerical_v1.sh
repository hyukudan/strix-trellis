#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gpu-site/scratch-trellis"
{
 hostname; TZ=:/etc/localtime date -Is
 sha256sum build-hip/strix_trellis_gpu_runtime_lds_v4 gpu-numerical-tests/*.stgpu gpu-numerical-tests/manifest.json
} > numerical-v1.meta
./build-hip/strix_trellis_gpu_runtime_lds_v4 --route all gpu-numerical-tests/*.stgpu
