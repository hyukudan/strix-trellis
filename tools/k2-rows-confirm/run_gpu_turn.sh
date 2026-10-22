#!/usr/bin/env bash
# NEW authorization: at most two physical 180s requests, no automatic retries.
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ "$PWD" == "$HOME/gpu-site/scratch-trellis/k2-rows-confirm" ]]
export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" HIP_CACHE_DIR="$PWD/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr" MAX_JOBS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
if [[ ${1:-} == --inside ]]; then
    [[ $# == 3 && ( $2 == 1 || $2 == 2 ) && $3 == "strix-trellis-k2-rows-confirm-new-$2" ]]
    n=$2;label=$3;run="runs/$n"
    bash tools/k2-rows-confirm/preflight.sh
    # Parents and run directories already exist on BOTH nodes before freeze.
    [[ $(cat "attempts/$n/request-label") == "$label" ]]
    [[ -z $(find "$run" -mindepth 1 -maxdepth 1 -print -quit) ]]
    mkdir "attempts/$n/payload-claimed"
    printf '%s\n' "$label" > "$run/fresh-label.txt"
    { hostname; date -Is; uname -a; cat build/compiler.txt; sha256sum frozen.sha256 bin/k2_rows_confirm_probe; } > "$run/meta.txt"
    set +e
    /usr/bin/time -v timeout --signal=TERM --kill-after=3s 145s nice -n 10 bin/k2_rows_confirm_probe data/wo_b.stgpu data/q_b.stgpu data/derived/wo_b.fp64 data/derived/q_b.fp64 > "$run/benchmark.log" 2> "$run/benchmark.stderr"
    result=$?
    sha256sum --check --quiet frozen.sha256
    integrity=$?
    (cd ../k2-rows-core; sha256sum --check --quiet frozen.sha256)
    old_integrity=$?
    { date -Is; printf 'payload_exit=%d integrity_exit=%d old_integrity_exit=%d\n' "$result" "$integrity" "$old_integrity"; } > "$run/closed.txt"
    cat "$run/benchmark.log"
    ((result==0 && integrity==0 && old_integrity==0))
    exit
fi
[[ $# == 1 && ( $1 == 1 || $1 == 2 ) ]]
n=$1;label="strix-trellis-k2-rows-confirm-new-$n"
[[ ! -e attempts/pending ]]
if [[ $n == 2 ]]; then [[ -f attempts/1/verified-pass ]]; fi
for node in node1 node2; do
    ssh -o BatchMode=yes "$node" 'cd ~/gpu-site/scratch-trellis/k2-rows-confirm && bash tools/k2-rows-confirm/preflight.sh'
done
# Atomic fresh reservations on both mirrors; failure aborts before queue.
for node in node1 node2; do
    ssh -o BatchMode=yes "$node" bash -s -- "$n" "$label" <<'REMOTE'
set -euo pipefail
cd "$HOME/gpu-site/scratch-trellis/k2-rows-confirm"
export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" HIP_CACHE_DIR="$PWD/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr"
n=$1;label=$2
[[ ! -e attempts/$n/request-label && -z $(find "runs/$n" -mindepth 1 -maxdepth 1 -print -quit) ]]
(set -o noclobber; printf '%s\n' "$label" > "attempts/$n/request-label")
REMOTE
done
printf '%s\n' "$label" > attempts/pending
set +e
GPU_TURN_MAX=180 timeout --signal=TERM --kill-after=5s 1195s "$HOME/gpu-site/gpu-turn-any.sh" "$label" timeout --signal=TERM --kill-after=3s 177s bash "$PWD/tools/k2-rows-confirm/run_gpu_turn.sh" --inside "$n" "$label"
result=$?
set -e
printf 'controller_exit=%d\n' "$result" > "attempts/$n/controller-exit.txt"
# Timeout leaves pending: inspect our own remote run read-only until closure.
if ((result!=124 && result!=137)); then rm -- attempts/pending; fi
exit "$result"
