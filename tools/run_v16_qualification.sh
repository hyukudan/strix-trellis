#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/v16-qualification" ]]
root="$HOME/gpu-site/scratch-trellis"
runtime="$root/runtime-v16";qa="$root/runtime-v16-qa"
[[ "$(realpath -e "$runtime")" == "$runtime" && "$(realpath -e "$qa")" == "$qa" ]]
sha256sum -c --quiet frozen.sha256
sha256sum -c --quiet children.sha256
(cd "$runtime" && sha256sum -c --quiet frozen.sha256)
(cd "$qa" && sha256sum -c --quiet frozen.sha256)
(cd "$root/engine-v16" && sha256sum -c --quiet frozen.sha256)
if (( $# )); then
    [[ $# == 1 && "$1" == --payload && ! -e runs/review001 ]]
    mkdir -p runs/review001;run="$TRELLIS_SOURCE/runs/review001"
    trap 'status=$?; if (( status )); then echo "FAILED status=$status" > "$run/failed.txt"; fi' EXIT
    { hostname; TZ=:/etc/localtime date -Is; cat frozen.sha256 children.sha256; } > "$run/meta.txt"
    (cd "$runtime" && ROOT_GPU_LEASE=1 bash tools/runtime_v16/root-payload.sh review001) > "$run/payload.log" 2> "$run/payload.stderr"
    [[ -f "$runtime/runs/root-v16-review001/passed.txt" && ! -e "$runtime/runs/root-v16-review001/failed.txt" ]]
    TZ=:/etc/localtime date -Is >> "$run/meta.txt"
    echo PASS_v16_native_and_global | tee "$run/passed.txt"
    exit
fi
[[ ! -e runs/review001 && ! -e "$qa/root-v16-native-review001" && ! -e "$runtime/runs/root-v16-review001" ]]
exec bash tools/run_gpu_turn.sh --seconds 1180 --controller-seconds 3600 \
    --maintenance-check tools/check_maintenance.sh --queue-arg strix-trellis-v16-qualification \
    -- "$TRELLIS_SOURCE/tools/run_v16_qualification.sh" --payload
