#!/usr/bin/env bash
# First mathematical admission only; no performance experiment is included.
set -euo pipefail
source "$(dirname -- "$0")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/grouped-k3-math-r1-admission" ]]
target="$HOME/gpu-site/scratch-trellis/grouped-k3-math-r1"
[[ "$(realpath -e "$target")" == "$target" ]]
sha256sum -c --quiet frozen.sha256
sha256sum -c --quiet children.sha256
(cd "$target" && sha256sum -c --quiet frozen.sha256)
if (( $# )); then
    [[ $# == 1 && "$1" == --payload && ! -e runs/qa001 && ! -e "$target/runs/physical-claimed" ]]
    mkdir -p runs/qa001;run="$TRELLIS_SOURCE/runs/qa001"
    trap 'status=$?; if (( status )); then echo "FAILED status=$status" > "$run/failed.txt"; fi' EXIT
    { hostname; TZ=:/etc/localtime date -Is; cat frozen.sha256 children.sha256; } > "$run/meta.txt"
    admission=$(sha256sum "$target/frozen.sha256");admission=${admission%% *}
    (cd "$target" && ROOT_GROUPED_K3_MATH_R1_ADMISSION_SHA256="$admission" \
        ROOT_GROUPED_K3_MATH_R1_LABEL=ROOT-grouped-k3-math-r1-qa001 \
        bash experiments/grouped_k3_math/native_payload.sh --inside-root-lease) > "$run/native-payload.log" 2> "$run/native-payload.stderr"
    grep -Fxq 'payload_exit=0 integrity_exit=0' "$target/runs/closed.txt"
    grep -Fxq 'PASS native mathematical QA only; no performance claim' "$target/runs/native.log"
    [[ $(grep -c '^PASS config geometry=' "$target/runs/native.log") == 14 ]]
    grep -Fq 'logical_creations=1 contexts_with_handle=1 contexts_admitted=1 calls=7 stored=7 overflow=0 failed=0 first_error=0 ' "$target/runs/native.log"
    for phase in prepare submit capture replay reuse cleanup; do
        grep -Eq "^attribute_zero_region phase=$phase regions=[1-9][0-9]* delta=0$" "$target/runs/native.log"
    done
    TZ=:/etc/localtime date -Is >> "$run/meta.txt"
    echo PASS_grouped_k3_math_native_only | tee "$run/passed.txt"
    exit
fi
[[ ! -e runs/qa001 && ! -e "$target/runs/physical-claimed" ]]
# 160s payload + <=10s maintenance + 5s termination grace stays below 180s.
exec bash tools/run_gpu_turn.sh --seconds 160 --controller-seconds 3600 \
    --maintenance-check tools/check_maintenance.sh --queue-arg strix-trellis-grouped-k3-math-r1 \
    -- "$TRELLIS_SOURCE/tools/run_grouped_k3_math_r1_admission.sh" --payload
