#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/tp-decode-qualification" ]]
root="$HOME/gpu-site/scratch-trellis"
child="$root/tp-decode-v1/host-r2"
[[ "$(realpath -e "$child")" == "$child" ]]
sha256sum -c --quiet frozen.sha256
sha256sum -c --quiet children.sha256
[[ ${1:-} =~ ^[12]$ ]]
campaign=$1
shift
run="$TRELLIS_SOURCE/runs/$(printf 'review%03d' "$campaign")"
if (( campaign == 2 )); then
    python3 -I -S -B -c 'import json;assert json.load(open("runs/campaign1-root-verified.json"))["state"]=="PHYSICAL_TP_CAMPAIGN1_FULL_PASS"'
fi
if [[ ${1:-} == --prequeue ]]; then
    [[ $# == 1 && ! -e "$run" && ! -e "$child/runs/STOP" &&
       ! -e "$child/runs/physical-$campaign-claimed" ]]
    (cd "$child" && python3 -I -S -B experiments/tp_decode/admission.py --prequeue)
    exit
fi
if [[ ${1:-} == --payload ]]; then
    [[ $# == 1 && ! -e "$run" ]]
    mkdir -p "$run"
    child_status=125
    finish() {
        local status=$? integrity
        trap - EXIT
        set +e
        (cd "$child" && sha256sum -c --quiet frozen.sha256) > "$run/integrity-after.log" 2>&1
        integrity=$?
        printf 'parent_exit=%d child_exit=%d integrity_exit=%d\n' "$status" "$child_status" "$integrity" > "$run/closed.txt"
        TZ=:/etc/localtime date -Is >> "$run/meta.txt"
        grep -F "strix-trellis-tp$campaign" "$HOME/gpu-site/.gpu-turn/log" | tail -8 > "$run/scheduler-after.txt"
        if (( status == 0 && integrity == 0 )); then
            echo PASS_TP_PARENT > "$run/passed.txt"
        else
            echo "FAILED parent=$status child=$child_status integrity=$integrity" > "$run/failed.txt"
        fi
        (( status != 0 || integrity == 0 )) || status=1
        exit "$status"
    }
    trap finish EXIT
    { hostname; TZ=:/etc/localtime date -Is; cat frozen.sha256 children.sha256; } > "$run/meta.txt"
    grep -F "strix-trellis-tp$campaign" "$HOME/gpu-site/.gpu-turn/log" | tail -8 > "$run/scheduler-before.txt" || true
    export ROOT_GPU_LEASE=1
    python3 -I -S -B tools/tp_decode_parent_receipt.py "$campaign" > "$run/receipt-label.txt"
    set +e
    (cd "$child" && bash experiments/tp_decode/root_payload.sh --inside-root-lease "$run/child-admission.json") > "$run/payload.log" 2> "$run/payload.stderr"
    child_status=$?
    set -e
    (( child_status == 0 )) || exit "$child_status"
    [[ "$(cat "$child/runs/ROOT-tp-decode-v1-$(printf 'qa%03d' "$campaign")/closed.txt")" == 'payload_exit=0 integrity_exit=0' ]]
    echo PASS_TP_NATIVE_AND_BENCH
    exit
fi
[[ $# == 0 && ! -e "$run" && ! -e "$child/runs/STOP" &&
   ! -e "$child/runs/physical-$campaign-claimed" ]]
(cd "$child" && python3 -I -S -B experiments/tp_decode/admission.py --prequeue)
exec bash tools/run_gpu_turn.sh --seconds 900 --controller-seconds 3600 \
    --maintenance-check tools/check_maintenance.sh --queue-arg "strix-trellis-tp$campaign" \
    -- "$TRELLIS_SOURCE/tools/run_tp_decode_qualification.sh" "$campaign" --payload
