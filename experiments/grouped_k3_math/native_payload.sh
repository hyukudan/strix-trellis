#!/usr/bin/env bash
# FUTURE ROOT-ONLY PAYLOAD. No queue/turn request or automatic retry exists here.
# ROOT must first review all source/CPU/ELF gates and use gpu-turn-any externally.
set -euo pipefail
[[ "$PWD" == "$HOME/gpu-site/scratch-trellis/grouped-k3-math-r1" ]]
[[ $# == 1 && "$1" == --inside-root-lease ]]
manifest=$(sha256sum frozen.sha256);manifest=${manifest%% *}
[[ ${ROOT_GROUPED_K3_MATH_R1_ADMISSION_SHA256:-UNAUTHORIZED} == "$manifest" ]]
export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" HIP_CACHE_DIR="$PWD/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr"
export TZ=:/etc/localtime
[[ ${ROOT_GROUPED_K3_MATH_R1_LABEL:-} =~ ^ROOT-grouped-k3-math-r1-[A-Za-z0-9_-]+$ ]]
export MAX_JOBS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
admission_window() {
    local clock minutes
    clock=$(date +%H:%M);minutes=$((10#${clock%:*}*60+10#${clock#*:}))
    ((minutes<1260 || minutes>=1410))
    if ((minutes<1260));then (( $(date -d 'today 21:00' +%s)-$(date +%s)>195 ));fi
    # Authoritative marker used by the shared gpu-turn.sh; read-only here.
    [[ ! -e "$HOME/gpu-site/.gpu-turn/maintenance" ]]
}
admission_window
sha256sum -c --quiet frozen.sha256
admission_window # Recheck after potentially slow integrity hashing, before HIP.
[[ ! -e runs/physical-claimed ]]
mkdir runs/physical-claimed
{ printf 'ROOT_label=%s\n' "$ROOT_GROUPED_K3_MATH_R1_LABEL";hostname;date -Is;uname -a;sha256sum frozen.sha256 bin/math_native_qa lib/libstrix_trellis.a; } > runs/meta.txt
set +e
/usr/bin/time -v timeout --signal=TERM --kill-after=3s 145s nice -n 10 bin/math_native_qa data/cpu > runs/native.log 2> runs/native.stderr
result=$?
sha256sum -c --quiet frozen.sha256
integrity=$?
printf 'payload_exit=%d integrity_exit=%d\n' "$result" "$integrity" > runs/closed.txt
((result==0 && integrity==0))
