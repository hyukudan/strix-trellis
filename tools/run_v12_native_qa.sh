#!/usr/bin/env bash
# Separate native integration gate; never run the HIP program outside the turn.
set -euo pipefail
source "$(dirname -- "$0")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/v12-runtime-qa" ]]
export TMPDIR="$TRELLIS_SOURCE/.tmp" XDG_CACHE_HOME="$TRELLIS_SOURCE/.cache"
export HIP_CACHE_DIR="$XDG_CACHE_HOME/hip" AMD_COMGR_CACHE_DIR="$XDG_CACHE_HOME/comgr"
sha256sum -c --quiet build/frozen.sha256
sha256sum -c --quiet build/linked-audit.sha256
sha256sum -c --quiet root-native.sha256
if (( $# )); then
    [[ $# == 1 && "$1" == --payload && ! -e runs/root-native-v1 ]]
    run="$TRELLIS_SOURCE/runs/root-native-v1"
    mkdir -p "$run"
    { hostname; TZ=:/etc/localtime date -Is; uname -r; cat root-native.sha256 build/frozen.sha256; } > "$run/meta.txt"
    trap 'status=$?; if (( status )); then echo "FAILED status=$status" > "$run/failed.txt"; fi' EXIT
    /usr/bin/time -v timeout --signal=TERM --kill-after=3s 145s \
        ./build/fused_runtime_qa "$HOME/gpu-site/scratch-trellis" "$TRELLIS_SOURCE/reference" > "$run/native.log" 2> "$run/native.stderr"
    sha256sum -c --quiet build/frozen.sha256
    sha256sum -c --quiet build/linked-audit.sha256
    TZ=:/etc/localtime date -Is >> "$run/meta.txt"
    echo PASS_v12_native_integration | tee "$run/passed.txt"
    exit
fi
[[ ! -e runs/root-native-v1 ]]
exec bash tools/run_gpu_turn.sh --seconds 180 --maintenance-check tools/check_maintenance.sh \
    --queue-arg strix-trellis-v12-native -- "$TRELLIS_SOURCE/tools/run_v12_native_qa.sh" --payload
