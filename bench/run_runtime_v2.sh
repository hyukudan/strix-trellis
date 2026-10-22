#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gpu-site/scratch-trellis"
{
 hostname; TZ=:/etc/localtime date -Is
 sha256sum build-hip/strix_trellis_gpu_runtime_wmma_v2 build-hip/libstrix_trellis_wmma_v2.a runtime-v2-source/*
} > runtime-v2.meta
exec ./build-hip/strix_trellis_gpu_runtime_wmma_v2 --route all runtime-fixtures/*.stgpu
