#!/usr/bin/env bash
# Qualify existing V12 paths on one real K3 expert; no new kernel selection.
set -euo pipefail
source "$(dirname -- "$0")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/real-k3-v12" ]]
export TMPDIR="$TRELLIS_SOURCE/.tmp" XDG_CACHE_HOME="$TRELLIS_SOURCE/.cache"
export HIP_CACHE_DIR="$XDG_CACHE_HOME/hip" AMD_COMGR_CACHE_DIR="$XDG_CACHE_HOME/comgr"
sha256sum -c --quiet frozen.sha256
if (( $# )); then
    [[ $# == 1 && "$1" == --payload && ! -e runs/qualification-v1 ]]
    run="$TRELLIS_SOURCE/runs/qualification-v1"
    mkdir -p "$run"
    phase=metadata
    trap 'status=$?; if (( status )); then echo "FAILED phase=$phase status=$status" > "$run/failed.txt"; fi' EXIT
    { hostname; TZ=:/etc/localtime date -Is; cat frozen.sha256; } > "$run/meta.txt"
    fixtures=(data/*.stgpu)
    [[ ${#fixtures[@]} == 3 ]]
    phase=correctness
    bin/strix_trellis_gpu_runtime "${fixtures[@]}" > "$run/correctness.log" 2> "$run/correctness.stderr"
    phase=api
    bin/strix_trellis_api_linear_bench "${fixtures[@]}" > "$run/api.log" 2> "$run/api.stderr"
    phase=graphs
    bin/strix_trellis_graph_linear_bench "${fixtures[@]}" > "$run/graphs.log" 2> "$run/graphs.stderr"
    sha256sum -c --quiet frozen.sha256
    TZ=:/etc/localtime date -Is >> "$run/meta.txt"
    echo PASS_real_K3_V12 | tee "$run/passed.txt"
    exit
fi
[[ ! -e runs/qualification-v1 ]]
exec bash tools/run_gpu_turn.sh --seconds 180 --maintenance-check tools/check_maintenance.sh \
    --queue-arg strix-trellis-real-k3-v12 -- "$TRELLIS_SOURCE/tools/run_real_k3_v12.sh" --payload
