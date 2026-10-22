#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/gemm-reuse" ]] || trellis_die 'Expected dedicated gemm-reuse subtree'
if [[ ${1:-} == --payload ]]; then
    date --iso-8601=seconds
    hostname
    /opt/rocm/core-10.1/bin/hipcc --version
    sha256sum src/gemm.hip src/gemm_reuse_bench.hip src/gemm.hpp src/trellis_device.hpp src/host_reference.hpp build/gemm_reuse_bench
    exec "$TRELLIS_SOURCE/build/gemm_reuse_bench" --bench
fi
[[ $# == 0 ]] || trellis_die 'No arguments expected'
exec bash "$TRELLIS_SOURCE/tools/run_gpu_turn.sh" --seconds 180 \
    --maintenance-check "$TRELLIS_SOURCE/tools/check_maintenance.sh" \
    --queue-arg strix-trellis-gemm-reuse -- "$TRELLIS_SOURCE/tools/run_gemm_reuse_bench.sh" --payload
