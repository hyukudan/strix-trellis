#!/usr/bin/env bash
# Sequential native and global V14b qualification inside one central lease.
set -euo pipefail
source "$(dirname -- "$0")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/v14b-qualification-lq-bundle" ]]
root="$HOME/gpu-site/scratch-trellis"
native="$root/runtime-v14b-qa"; global="$root/runtime-v14b"
label=root-v14b-qualification-v1
[[ "$(realpath -e "$native")" == "$native" && "$(realpath -e "$global")" == "$global" ]]
sha256sum -c --quiet frozen.sha256
sha256sum -c --quiet children.sha256
(cd "$native" && sha256sum -c --quiet frozen.sha256 && sha256sum -c --quiet complete-v1/frozen.sha256)
(cd "$global" && sha256sum -c --quiet frozen.sha256)
(cd "$root/engine-v14b" && sha256sum -c --quiet frozen.sha256)
if (( $# )); then
    [[ $# == 1 && "$1" == --payload && ! -e runs/qualification-v1 ]]
    mkdir -p runs/qualification-v1;run="$TRELLIS_SOURCE/runs/qualification-v1"
    phase=metadata
    trap 'status=$?; if (( status )); then echo "FAILED phase=$phase status=$status" > "$run/failed.txt"; fi' EXIT
    { hostname; TZ=:/etc/localtime date -Is; cat frozen.sha256 children.sha256; } > "$run/meta.txt"
    phase=native
    (cd "$native" && timeout --signal=TERM --kill-after=3s 450s bash complete-v1/run.sh --payload "$label") > "$run/native.log" 2> "$run/native.stderr"
    phase=global
    (cd "$global" && timeout --signal=TERM --kill-after=3s 360s bash tools/run_runtime_v14b.sh --payload "$label") > "$run/global.log" 2> "$run/global.stderr"
    TZ=:/etc/localtime date -Is >> "$run/meta.txt"
    echo PASS_v14b_native_and_global | tee "$run/passed.txt"
    exit
fi
[[ ! -e runs/qualification-v1 && ! -e "$native/complete-v1/runs/$label" && ! -e "$global/runs/$label" ]]
exec bash tools/run_gpu_turn.sh --seconds 900 --controller-seconds 3600 --maintenance-check tools/check_maintenance.sh \
    --queue-arg strix-trellis-v14b-lq -- "$TRELLIS_SOURCE/tools/run_v14b_qualification_lq_bundle.sh" --payload
