#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gpu-site/scratch-trellis"
{
 hostname; TZ=:/etc/localtime date -Is
 sha256sum build-hip/strix_trellis_gpu_runtime_vector_v1 build-hip/libstrix_trellis.a runtime-vector-v1-source/backend.hip runtime-vector-v1-source/transforms.hpp runtime-vector-v1-source/api.hip runtime-vector-v1-source/gpu_runtime.hip
} > runtime-v1.meta
exec ./build-hip/strix_trellis_gpu_runtime_vector_v1 --route all runtime-fixtures/*.stgpu
