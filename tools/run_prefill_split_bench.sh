#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/prefill-splits" ]] || trellis_die 'Expected dedicated prefill-splits subtree'
if [[ ${1:-} == --payload ]]; then
    [[ $# == 1 ]] || trellis_die 'Unexpected payload arguments'
    date --iso-8601=seconds
    hostname
    /opt/rocm/core-10.1/bin/hipcc --version
    sha256sum bench/prefill_split_bench.hip src/gemm.hip src/*.hpp build/prefill_split_bench tools/run_prefill_split_bench.sh tools/run_gpu_turn.sh tools/check_maintenance.sh tools/scratch_paths.sh
    base="$HOME/gpu-site/scratch-trellis/gpu-dense-prefill-tests"
    fixtures=(
      "$base/layers.0.attn.wq_b.stgpu"
      "$base/layers.0.attn.wo_b.stgpu"
      "$base/layers.0.ffn.shared_experts.w1.stgpu"
      "$base/layers.0.ffn.shared_experts.w2.stgpu"
      "$base/layers.0.ffn.shared_experts.w3.stgpu"
      "$base/layers.0.attn.wq_a.stgpu"
      "$base/layers.0.attn.wo_a.slice.0.stgpu"
    )
    sha256sum "${fixtures[@]}"
    exec "$TRELLIS_SOURCE/build/prefill_split_bench" "${fixtures[@]}"
fi
[[ $# == 0 ]] || trellis_die 'No arguments expected'
exec bash "$TRELLIS_SOURCE/tools/run_gpu_turn.sh" --seconds 180 \
    --maintenance-check "$TRELLIS_SOURCE/tools/check_maintenance.sh" \
    --queue-arg strix-trellis-prefill-splits -- "$TRELLIS_SOURCE/tools/run_prefill_split_bench.sh" --payload
