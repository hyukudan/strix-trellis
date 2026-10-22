#!/usr/bin/env bash
# Existing V12 kernels, new strict benchmark fixtures for OUR K4/K5 heads.
set -euo pipefail
source "$(dirname -- "$0")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/ours-head-v12" ]]
export TMPDIR="$TRELLIS_SOURCE/.tmp" XDG_CACHE_HOME="$TRELLIS_SOURCE/.cache"
export HIP_CACHE_DIR="$XDG_CACHE_HOME/hip" AMD_COMGR_CACHE_DIR="$XDG_CACHE_HOME/comgr"
sha256sum -c --quiet frozen.sha256
if (( $# )); then
    [[ $# == 1 && "$1" == --payload && ! -e runs/qualification-v1 ]]
    run="$TRELLIS_SOURCE/runs/qualification-v1";mkdir -p "$run"
    phase=metadata
    trap 'status=$?; if (( status )); then echo "FAILED phase=$phase status=$status" > "$run/failed.txt"; fi' EXIT
    { hostname; TZ=:/etc/localtime date -Is; cat frozen.sha256; } > "$run/meta.txt"
    for bits in 4 5; do
        phase="k$bits-full-dumps"
        bin/head_crosscheck "data/k$bits" "$run/k$bits-full" > "$run/k$bits-full.log" 2> "$run/k$bits-full.stderr"
        phase="k$bits-cpu"
        python3 tools/compare_ours_head_full.py "$run/k$bits-full" "reference/k$bits" > "$run/k$bits-cpu.json"
        phase="k$bits-api"
        bin/head_bench "data/k$bits" > "$run/k$bits-api.log" 2> "$run/k$bits-api.stderr"
    done
    phase=legacy-k6
    bin/head_crosscheck ../gpu-head-eaq-ref-v2 "$run/legacy-k6" > "$run/legacy-k6.log" 2> "$run/legacy-k6.stderr"
    for rows in {1..8}; do
        for stage in a.f16 y.f32; do
            cmp "$run/legacy-k6/r$rows-$stage" "../runtime-v12/runs/qualification-v1/head-full/r$rows-$stage"
        done
    done
    sha256sum -c --quiet frozen.sha256
    TZ=:/etc/localtime date -Is >> "$run/meta.txt"
    echo PASS_OURS_heads_K4_K5_V12 | tee "$run/passed.txt"
    exit
fi
[[ ! -e runs/qualification-v1 ]]
exec bash tools/run_gpu_turn.sh --seconds 180 --maintenance-check tools/check_maintenance.sh \
    --queue-arg strix-trellis-ours-heads-v12 -- "$TRELLIS_SOURCE/tools/run_ours_heads_v12.sh" --payload
