#!/usr/bin/env bash
# One pending request, two fixed attempts, 180s lease envelope, 145s work.
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
[[ "$PWD" == "$HOME/gpu-site/scratch-trellis/k2-rows-core" ]]
export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr"
[[ -d "$TMPDIR" && -d "$AMD_COMGR_CACHE_DIR" ]]
safe() {
    local now minutes remaining
    now=$(TZ=:/etc/localtime date +%s)
    minutes=$(TZ=:/etc/localtime date +%H:%M)
    minutes=$((10#${minutes%:*}*60+10#${minutes#*:}))
    ((minutes<1260 || minutes>=1410))
    if ((minutes<1260)); then
        remaining=$(( $(TZ=:/etc/localtime date -d 'today 21:00' +%s) - now ))
        ((remaining>195))
    fi
    [[ ! -e "$HOME/gpu-site/.gpu-turn/maintenance" ]]
}
if [[ ${1:-} == --inside ]]; then
    [[ $# == 2 && ( $2 == 1 || $2 == 2 ) ]]
    safe
    sha256sum --check --quiet frozen.sha256
    mkdir "runs/$2"
    run="runs/$2"
    { hostname; TZ=:/etc/localtime date -Is; uname -a; cat build/compiler.txt; cat frozen.sha256; } > "$run/meta.txt"
    # Preserve failures and a closure marker even when the internal deadline fires.
    set +e
    /usr/bin/time -v timeout --signal=TERM --kill-after=3s 145s nice -n 10 \
        bin/k2_rows_probe data/wo_b.stgpu data/q_b.stgpu data/derived/wo_b.fp64 data/derived/q_b.fp64 > "$run/benchmark.log" 2> "$run/benchmark.stderr"
    result=$?
    sha256sum --check --quiet frozen.sha256
    integrity=$?
    { TZ=:/etc/localtime date -Is; echo "payload_exit=$result integrity_exit=$integrity"; } > "$run/closed.txt"
    cat "$run/benchmark.log"
    ((result==0 && integrity==0))
    exit
fi
[[ $# == 1 && ( $1 == 1 || $1 == 2 ) ]]
safe
sha256sum --check --quiet frozen.sha256
mkdir -p attempts runs
[[ ! -e attempts/pending ]]
mkdir "attempts/$1.request"
printf '%s\n' "$1" > attempts/pending
# The controller allows the queue to wait; never use a 175s outer deadline.
set +e
timeout --signal=TERM --kill-after=5s 1195s "$HOME/gpu-site/gpu-turn-any.sh" \
    "strix-trellis-k2-rows-core-$1" timeout --signal=TERM --kill-after=3s 177s bash "$PWD/tools/k2-rows-core/run_gpu_turn.sh" --inside "$1"
result=$?
set -e
printf 'controller_exit=%d\n' "$result" > "attempts/$1.controller-exit"
# A timed-out controller does not prove the queued request closed. Leave pending.
if (( result!=124 && result!=137 )); then rm -- attempts/pending; fi
exit "$result"
