#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/gemv-lds" || "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/gemv-lds/node1" || "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/gemv-lds/node2" ]] || trellis_die 'Expected gemv-lds subtree'
if [[ ${1:-} == --payload ]]; then
    date --iso-8601=seconds
    hostname
    /opt/rocm/core-10.1/bin/hipcc --version
    sha256sum src/gemv_lds.hip src/gemv_lds_bench.hip src/gemv_lds.hpp src/backend.hpp src/trellis_optimized.hpp src/trellis_device.hpp src/host_reference.hpp gemv_lds_bench
    for f in /sys/class/drm/card*/device/pp_dpm_sclk /sys/class/drm/card*/device/pp_dpm_mclk; do
        if [[ -r "$f" ]]; then echo "$f"; cat "$f"; fi
    done
    exec "$TRELLIS_SOURCE/gemv_lds_bench" "${@:2}"
fi
exec bash "$TRELLIS_SOURCE/tools/run_gpu_turn.sh" --seconds 180 \
    --maintenance-check "$TRELLIS_SOURCE/tools/check_maintenance.sh" \
    --queue-arg strix-trellis-gemv-lds -- "$TRELLIS_SOURCE/tools/run_gemv_lds_bench.sh" --payload "$@"
