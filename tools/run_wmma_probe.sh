#!/usr/bin/env bash
# Run only the small, separately compiled WMMA correctness diagnostic.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scratch_paths.sh"
trellis_require_own_source
if [[ ${1:-} == --payload ]]; then
    {
        hostname
        TZ=:/etc/localtime date -Is
        /opt/rocm/core-10.1/bin/hipcc --version
        echo 'nice -n 10 /opt/rocm/core-10.1/bin/hipcc -O3 --offload-arch=gfx1151 -std=c++17 src/wmma_probe.hip -o build/wmma_probe'
        sha256sum src/wmma_probe.hip build/wmma_probe tools/run_wmma_probe.sh
    } > build/run.meta
    exec build/wmma_probe
fi
exec bash tools/run_gpu_turn.sh --seconds 180 \
    --maintenance-check "$TRELLIS_SOURCE/tools/check_maintenance.sh" \
    --queue-arg ref-wmma-debug -- "$TRELLIS_SOURCE/tools/run_wmma_probe.sh" --payload
