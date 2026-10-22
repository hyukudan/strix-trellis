#!/usr/bin/env bash
# One physical request only; no automatic retries and no HIP executable discovery.
set -euo pipefail
[[ "$PWD" == "$HOME/gpu-site/scratch-trellis/grouped-k3-abi" ]]
export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" HIP_CACHE_DIR="$PWD/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr" MAX_JOBS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
label=strix-trellis-grouped-k3-abi
preflight() {
    local clock minutes now
    clock=$(date +%H:%M);minutes=$((10#${clock%:*}*60+10#${clock#*:}));now=$(date +%s)
    ((minutes<1260 || minutes>=1410))
    if ((minutes<1260));then (( $(date -d 'today 21:00' +%s)-now>195 ));fi
    [[ ! -e "$HOME/gpu-site/.gpu-turn/maintenance" ]]
    sha256sum --check --quiet frozen.sha256
    grep -q '^PASS linked trace' build/resources.json
}
if [[ ${1:-} == --check ]];then preflight;exit;fi
if [[ ${1:-} == --inside ]];then
    preflight
    mkdir attempts/physical-claimed
    [[ -z $(find runs -type f -print -quit) ]]
    { hostname;date -Is;uname -a;sha256sum frozen.sha256 bin/abi_pilot; } > runs/meta.txt
    set +e
    /usr/bin/time -v timeout --signal=TERM --kill-after=3s 145s nice -n 10 bin/abi_pilot > runs/native.log 2> runs/native.stderr
    result=$?
    sha256sum --check --quiet frozen.sha256
    integrity=$?
    { date -Is;printf 'payload_exit=%d integrity_exit=%d\n' "$result" "$integrity"; } > runs/closed.txt
    cat runs/native.log
    ((result==0 && integrity==0))
    exit
fi
[[ $# == 0 && ! -e attempts/pending && ! -e attempts/controller-started ]]
for node in node1 node2;do
    if [[ $(hostname) == "$node" ]];then
        preflight
    else
        ssh -o BatchMode=yes "$node" 'cd ~/gpu-site/scratch-trellis/grouped-k3-abi && export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" HIP_CACHE_DIR="$PWD/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr"; nice -n 10 bash experiments/grouped-k3-abi/run.sh --check'
    fi
done
mkdir attempts/pending
printf '%s\n' "$$" > attempts/controller-started
date -Is > attempts/queued.txt
set +e
GPU_TURN_MAX=180 timeout --signal=TERM --kill-after=5s 1195s "$HOME/gpu-site/gpu-turn-any.sh" "$label" timeout --signal=TERM --kill-after=3s 177s nice -n 10 bash "$PWD/experiments/grouped-k3-abi/run.sh" --inside
result=$?
printf 'controller_exit=%d\n' "$result" > attempts/controller-closed.txt
# Keep pending on timeout until own PID/holder/closed state is inspected manually.
if ((result!=124 && result!=137));then rmdir attempts/pending;fi
exit "$result"
