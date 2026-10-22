#!/usr/bin/env bash
# ROOT calls ONLY from a later externally acquired lease. No auto queue exists.
set -euo pipefail
[[ "$PWD" == "$HOME/gpu-site/scratch-trellis/grouped-k3-perf-r1/runs/host-r2" ]]
[[ $# == 1 && "$1" == --inside-root-lease ]]
export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" HIP_CACHE_DIR="$PWD/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr"
export LC_ALL=C
export TZ=:/etc/localtime OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=1
label=${ROOT_GROUPED_K3_PERF_LABEL:-BLOCKED}
[[ "$label" =~ ^ROOT-grouped-k3-perf-r1-[A-Za-z0-9_-]+$ ]]
[[ ! -e runs/STOP && ! -e "runs/$label" ]]
trap 'rc=$?; printf "payload_preflight_or_dispatch_exit=%s\n" "$rc" > "runs/failure-$label-$(date +%s%N).txt"; touch runs/STOP; exit "$rc"' ERR
receipt="runs/admission-$label/receipt.json"
python3 -I -S -B experiments/grouped_k3_perf/admission.py --receipt "$receipt" --manifest frozen.sha256
python3 -I -S -B -c 'import json,sys;assert json.load(open(sys.argv[1]))["label"]==sys.argv[2]' "$receipt" "$label"
window() {
    local clock minutes
    clock=$(date +%H:%M);minutes=$((10#${clock%:*}*60+10#${clock#*:}))
    ((minutes<1260 || minutes>=1410))
    if ((minutes<1260));then (( $(date -d 'today 21:00' +%s)-$(date +%s)>315 ));fi
    [[ ! -e "$HOME/gpu-site/.gpu-turn/maintenance" ]]
}
window
python3 -I -S -B experiments/grouped_k3_perf/provenance.py --manifest frozen.sha256
window
# No repeat of a failed physical run, even with another label.
campaign=$(python3 -I -S -B -c 'import json,sys;print(json.load(open(sys.argv[1]))["campaign"])' "$receipt")
physical=$(python3 -I -S -B -c 'import json,sys;print(json.load(open(sys.argv[1]))["physical_id"])' "$receipt")
mkdir "runs/physical-$physical-claimed"
mkdir "runs/campaign-$campaign-claimed"
mkdir "runs/$label"
mkdir -p "runs/$label/.tmp" "runs/$label/.cache/hip" "runs/$label/.cache/comgr"
export TMPDIR="$PWD/runs/$label/.tmp" XDG_CACHE_HOME="$PWD/runs/$label/.cache" HIP_CACHE_DIR="$PWD/runs/$label/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/runs/$label/.cache/comgr"
{ hostname;date -Is;uname -a;sha256sum frozen.sha256 bin/perf_native lib/libstrix_trellis.a;cat "$receipt"; } > "runs/$label/meta.txt"
set +e
# Preserve the raw exit and always run the post-callback, including native failure.
trap - ERR
/usr/bin/time -v timeout --signal=TERM --kill-after=3s 270s nice -n 10 bin/perf_native --inside-root-lease data/cpu > "runs/$label/native.log" 2> "runs/$label/native.stderr"
result=$?
python3 -I -S -B experiments/grouped_k3_perf/provenance.py --manifest frozen.sha256
integrity=$?
printf 'payload_exit=%d integrity_exit=%d\n' "$result" "$integrity" > "runs/$label/closed.txt"
if ((result!=0 || integrity!=0));then touch runs/STOP;exit 1;fi
printf "%s %s PASS\n" "$physical" "$label" > "runs/$label/child-passed.txt"
