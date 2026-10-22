#!/usr/bin/env bash
# One approved turn runs two frozen, sequential qualification payloads.
set -euo pipefail
source "$(dirname -- "$0")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/v13-native-head-bundle" ]]
root="$HOME/gpu-site/scratch-trellis"
native="$root/runtime-v13-qa";heads="$root/ours-head-v12"
[[ "$(realpath -e "$native")" == "$native" && "$(realpath -e "$heads")" == "$heads" ]]
sha256sum -c --quiet frozen.sha256
sha256sum -c --quiet children.sha256
(cd "$native" && sha256sum -c --quiet frozen.sha256)
(cd "$heads" && sha256sum -c --quiet frozen.sha256)
if (( $# )); then
    [[ $# == 1 && "$1" == --payload && ! -e runs/qualification-v1 ]]
    mkdir -p runs/qualification-v1;run="$TRELLIS_SOURCE/runs/qualification-v1"
    phase=metadata
    trap 'status=$?; if (( status )); then echo "FAILED phase=$phase status=$status" > "$run/failed.txt"; fi' EXIT
    { hostname; TZ=:/etc/localtime date -Is; cat frozen.sha256 children.sha256; } > "$run/meta.txt"
    # These child payload entry points are called exclusively inside THIS lease.
    # No child creates another request; each retains its own frozen source and logs.
    phase=v13-native
    (cd "$native" && timeout --signal=TERM --kill-after=3s 600s bash tools/run_v13_native_qa.sh --payload qualification-v1) > "$run/v13-native.log" 2> "$run/v13-native.stderr"
    phase=ours-heads
    (cd "$heads" && timeout --signal=TERM --kill-after=3s 180s bash tools/run_ours_heads_v12.sh --payload) > "$run/ours-heads.log" 2> "$run/ours-heads.stderr"
    TZ=:/etc/localtime date -Is >> "$run/meta.txt"
    echo PASS_v13_native_and_OURS_heads | tee "$run/passed.txt"
    exit
fi
[[ ! -e runs/qualification-v1 && ! -e "$native/runs/qualification-v1" && ! -e "$heads/runs/qualification-v1" ]]
exec bash tools/run_gpu_turn.sh --seconds 900 --maintenance-check tools/check_maintenance.sh \
    --queue-arg strix-trellis-v13-head-bundle -- "$TRELLIS_SOURCE/tools/run_v13_native_and_ours_heads.sh" --payload
