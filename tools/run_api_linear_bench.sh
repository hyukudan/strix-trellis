#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/api-linear" ]] || trellis_die 'Expected dedicated api-linear subtree'
if [[ ${1:-} == --payload ]]; then
    [[ $# == 1 ]] || trellis_die 'Unexpected payload arguments'
    date --iso-8601=seconds
    hostname
    /opt/rocm/core-10.1/bin/hipcc --version
    sha256sum bench/api_linear_bench.hip tests/gpu_runtime.hip include/strix_trellis.h src/*.hip src/*.hpp build/api_linear_bench tools/*.sh
    base="$HOME/gpu-site/scratch-trellis"
    fixtures=(
      "$base/gpu-dense-prefill-tests/layers.0.attn.wq_b.stgpu"
      "$base/gpu-dense-prefill-tests/layers.0.attn.wo_b.stgpu"
      "$base/gpu-dense-prefill-tests/layers.0.ffn.shared_experts.w1.stgpu"
      "$base/gpu-dense-prefill-tests/layers.0.ffn.shared_experts.w2.stgpu"
      "$base/gpu-dense-prefill-tests/layers.0.ffn.shared_experts.w3.stgpu"
      "$base/gpu-dense-prefill-tests/layers.0.attn.wq_a.stgpu"
      "$base/gpu-dense-prefill-tests/layers.0.attn.wo_a.slice.0.stgpu"
      "$base/runtime-fixtures/eaq-wkv-k6-mul1.stgpu"
    )
    sha256sum "${fixtures[@]}"
    exec "$TRELLIS_SOURCE/build/api_linear_bench" "${fixtures[@]}"
fi
[[ $# == 0 ]] || trellis_die 'No arguments expected'
exec bash "$TRELLIS_SOURCE/tools/run_gpu_turn.sh" --seconds 180 \
    --maintenance-check "$TRELLIS_SOURCE/tools/check_maintenance.sh" \
    --queue-arg strix-trellis-api-linear -- "$TRELLIS_SOURCE/tools/run_api_linear_bench.sh" --payload
