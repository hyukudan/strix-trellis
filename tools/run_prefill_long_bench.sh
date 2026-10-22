#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/prefill-long" ]] || trellis_die 'Expected dedicated prefill-long subtree'
if [[ ${1:-} == --payload ]]; then
    [[ $# == 1 ]] || trellis_die 'Unexpected payload arguments'
    date --iso-8601=seconds
    hostname
    /opt/rocm/core-10.1/bin/hipcc --version
    sha256sum source.tar bench/prefill_long_bench.hip include/strix_trellis.h src/*.hip src/*.hpp build/prefill_long_bench tools/run_prefill_long_bench.sh tools/run_gpu_turn.sh tools/check_maintenance.sh tools/scratch_paths.sh fixtures/*
    exec /usr/bin/time -v "$TRELLIS_SOURCE/build/prefill_long_bench" \
        "$TRELLIS_SOURCE/fixtures/layers.0.attn.wq_a.m512.stgpu" \
        "$TRELLIS_SOURCE/fixtures/layers.0.attn.wo_a.slice.0.m512.stgpu"
fi
[[ $# == 0 ]] || trellis_die 'No arguments expected'
exec bash "$TRELLIS_SOURCE/tools/run_gpu_turn.sh" --seconds 180 \
    --maintenance-check "$TRELLIS_SOURCE/tools/check_maintenance.sh" \
    --queue-arg strix-trellis-prefill-long -- "$TRELLIS_SOURCE/tools/run_prefill_long_bench.sh" --payload
