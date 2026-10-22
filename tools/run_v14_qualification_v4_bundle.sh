#!/usr/bin/env bash
# One central lease, two immutable and sequential qualification payloads.
set -euo pipefail
source "$(dirname -- "$0")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/v14-qualification-v4-bundle" ]]
root="$HOME/gpu-site/scratch-trellis"
native="$root/runtime-v14-qa"; global="$root/runtime-v14"
[[ "$(realpath -e "$native")" == "$native" && "$(realpath -e "$global")" == "$global" ]]
sha256sum -c --quiet frozen.sha256
sha256sum -c --quiet children.sha256
(cd "$native" && sha256sum -c --quiet frozen.sha256 && sha256sum -c --quiet complete-v4/frozen.sha256 && sha256sum -c --quiet complete-v4-handoff/frozen.sha256 && sha256sum -c --quiet complete-v4-handoff/children.sha256)
(cd "$global" && sha256sum -c --quiet frozen.sha256 && sha256sum -c --quiet qualification-v2/frozen.sha256)
if (( $# )); then
    [[ $# == 1 && "$1" == --payload && ! -e runs/qualification-v1 ]]
    mkdir -p runs/qualification-v1; run="$TRELLIS_SOURCE/runs/qualification-v1"
    phase=metadata
    trap 'status=$?; if (( status )); then echo "FAILED phase=$phase status=$status" > "$run/failed.txt"; fi' EXIT
    { hostname; TZ=:/etc/localtime date -Is; cat frozen.sha256 children.sha256; } > "$run/meta.txt"
    phase=v14-native
    (cd "$native" && timeout --signal=TERM --kill-after=3s 450s bash complete-v4/run.sh --payload root-v14-v4-qualification-v1) > "$run/native.log" 2> "$run/native.stderr"
    phase=v14-global
    (cd "$global" && timeout --signal=TERM --kill-after=3s 360s bash qualification-v2/run_runtime_v14.sh --payload qualification-v1) > "$run/global.log" 2> "$run/global.stderr"
    TZ=:/etc/localtime date -Is >> "$run/meta.txt"
    echo PASS_v14_native_and_global | tee "$run/passed.txt"
    exit
fi
[[ ! -e runs/qualification-v1 && ! -e "$native/complete-v4/runs/root-v14-v4-qualification-v1" && ! -e "$global/runs/qualification-v1" ]]
exec bash tools/run_gpu_turn.sh --seconds 900 --maintenance-check tools/check_maintenance.sh \
    --queue-arg strix-trellis-v14-v4-qualification -- "$TRELLIS_SOURCE/tools/run_v14_qualification_v4_bundle.sh" --payload
