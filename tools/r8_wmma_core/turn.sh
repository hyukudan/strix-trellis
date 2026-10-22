#!/usr/bin/env bash
# Exactly two physical payloads at most. Never execute a HIP binary outside here.
set -euo pipefail
cd "$(dirname -- "$0")/../.."
own="$HOME/gpu-site/scratch-trellis/r8-wmma-core"
[[ "$(pwd -P)" == "$own" ]]
export TMPDIR="$own/.tmp" XDG_CACHE_HOME="$own/.tmp/cache" HIP_CACHE_DIR="$own/.tmp/hip" AMD_COMGR_CACHE_DIR="$own/.tmp/comgr"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
read -ra process_stat < /proc/$$/stat
if (( process_stat[18]==0 ));then exec nice -n 10 bash "$0" "$@";fi
(( process_stat[18]==10 ))
safe() {
    local now minute remaining
    now=$(TZ=:/etc/localtime date +%s)
    minute=$(TZ=:/etc/localtime date -d "@$now" +%H:%M)
    minute=$((10#${minute%:*}*60+10#${minute#*:}))
    (( minute<1260 || minute>=1410 )) || { echo 'Local time veto' >&2;return 1; }
    if (( minute<1260 ));then
        remaining=$(( $(TZ=:/etc/localtime date -d 'today 21:00' +%s)-now ))
        (( remaining>195 )) || return 1
    fi
    [[ ! -e "$HOME/gpu-site/.gpu-turn/maintenance" ]]
}
campaign=${2:-${1:-}}
[[ "$campaign" == first || "$campaign" == confirm ]]
safe
sha256sum -c --quiet frozen.sha256
if [[ "$campaign" == confirm ]];then
    [[ -f runs/first/passed && -f selection.sha256 ]]
    sha256sum -c --quiet selection.sha256
fi
if [[ ${1:-} == --payload ]];then
    [[ $# == 2 && ! -e runs/$campaign/started ]]
    mkdir -p "runs/$campaign"
    (set -o noclobber; TZ=:/etc/localtime date -Is > "runs/$campaign/started")
    { hostname; TZ=:/etc/localtime date -Is; cat frozen.sha256; } > "runs/$campaign/meta.txt"
    status=0
    timeout --signal=TERM --kill-after=2s 145s /usr/bin/time -v -o "runs/$campaign/memory.txt" build/r8-driver "$own/data" > "runs/$campaign/native.log" 2> "runs/$campaign/native.stderr" || status=$?
    printf '%s\n' "$status" > "runs/$campaign/status"
    TZ=:/etc/localtime date -Is > "runs/$campaign/closed"
    (( status==0 )) || exit "$status"
    sha256sum -c --quiet frozen.sha256
    touch "runs/$campaign/passed"
    exit
fi
[[ $# == 1 && ! -e runs/$campaign/started ]]
printf '%s\n' "$$" > "build/controller-$campaign.pid"
status=0
timeout --signal=TERM --kill-after=5s 1195s "$HOME/gpu-site/gpu-turn-any.sh" strix-trellis-r8-wmma-core \
    timeout --signal=TERM --kill-after=5s 180s bash "$own/tools/r8_wmma_core/turn.sh" --payload "$campaign" || status=$?
printf '%s\n' "$status" > "build/controller-$campaign.status"
TZ=:/etc/localtime date -Is > "build/controller-$campaign.closed"
exit "$status"
