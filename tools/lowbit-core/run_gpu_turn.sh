#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ "$PWD" == "$HOME/gpu-site/scratch-trellis/lowbit-core" ]]
export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache"
export AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr"
check_safety() {
    local now minute remaining
    now=$(TZ=:/etc/localtime date +%s)
    minute=$(TZ=:/etc/localtime date +%H:%M)
    minute=$((10#${minute%:*}*60+10#${minute#*:}))
    (( minute<1260 || minute>=1410 )) || { echo 'VETO local 21:00-23:30'; exit 3; }
    if ((minute<1260)); then
        remaining=$(( $(TZ=:/etc/localtime date -d 'today 21:00' +%s)-now ))
        (( remaining>195 )) || { echo 'VETO overlap'; exit 3; }
    fi
    [[ -d "$HOME/gpu-site/.gpu-turn" && -r "$HOME/gpu-site/gpu-turn-any.sh" ]]
    [[ ! -e "$HOME/gpu-site/.gpu-turn/maintenance" ]]
    sha256sum -c --quiet frozen.sha256
}
if [[ ${1:-} == --inside ]]; then
    [[ $# == 2 && "$2" =~ ^(validation|confirmation)$ ]]
    check_safety
    run="runs/$2";mkdir -p "$run"
    { hostname; TZ=:/etc/localtime date -Is; cat frozen.sha256; } > "$run/meta.txt"
    /usr/bin/time -v timeout --signal=TERM --kill-after=3s 145s \
      nice -n 10 build/lowbit_core_probe fixtures > "$run/raw.log" 2> "$run/stderr.log"
    date -Is >> "$run/meta.txt"
    exit
fi
[[ $# == 1 && "$1" =~ ^(validation|confirmation)$ ]]
check_safety
mkdir -p runs
[[ ! -e runs/request-validation || "$1" == confirmation ]]
[[ ! -e runs/request-confirmation ]]
if [[ "$1" == confirmation ]]; then [[ -f runs/validation/raw.log ]]; fi
# Own markers only. At most one synchronous request; no scheduler state writes.
(set -C; echo "$1" > "runs/request-$1")
timeout --signal=TERM --kill-after=3s 175s "$HOME/gpu-site/gpu-turn-any.sh" \
    "strix-trellis-lowbit-$1" bash "$PWD/tools/lowbit-core/run_gpu_turn.sh" --inside "$1"
