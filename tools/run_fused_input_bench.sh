#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scratch_paths.sh"
trellis_require_own_source
case "$TRELLIS_SOURCE" in
    "$HOME/gpu-site/scratch-trellis/fused-input"|"$HOME/gpu-site/scratch-trellis/fused-input/node1"|"$HOME/gpu-site/scratch-trellis/fused-input/node2") ;;
    *) trellis_die 'Expected fused-input subtree' ;;
esac
if [[ ${1:-} == --payload ]]; then
    date --iso-8601=seconds
    hostname
    /opt/rocm/core-10.1/bin/hipcc --version
    sha256sum src/gemv_fused.* src/gemv_lds.* src/hadamard.hpp src/transforms.hpp src/trellis_optimized.hpp src/trellis_device.hpp src/backend.hpp src/host_reference.hpp bench/fused_input_bench.hip fused_input_bench
    for f in /sys/class/drm/card*/device/pp_dpm_sclk /sys/class/drm/card*/device/pp_dpm_mclk; do
        if [[ -r "$f" ]]; then echo "$f"; cat "$f"; fi
    done
    root="$HOME/gpu-site/scratch-trellis"
    sha256sum fixtures-large/*.stgpu fixtures-large/manifest.json tools/generate_large.py "$root"/runtime-fixtures/*.stgpu "$root"/gpu-dense-prefill-tests/*.stgpu
    exec "$TRELLIS_SOURCE/fused_input_bench" "$root"/runtime-fixtures/synthetic*.stgpu "$TRELLIS_SOURCE"/fixtures-large/*.stgpu --bench "$root"/runtime-fixtures/eaq-wkv-k6-mul1.stgpu "$root"/gpu-dense-prefill-tests/*.stgpu
fi
exec bash "$TRELLIS_SOURCE/tools/run_gpu_turn.sh" --seconds 180 \
    --maintenance-check "$TRELLIS_SOURCE/tools/check_maintenance.sh" \
    --queue-arg strix-trellis-fused-input -- "$TRELLIS_SOURCE/tools/run_fused_input_bench.sh" --payload
