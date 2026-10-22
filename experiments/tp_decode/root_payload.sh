#!/usr/bin/env bash
# ROOT ONLY, after a real outer 900s lease. Never queues, waits for or requests GPU.
set -euo pipefail
export TZ=:/etc/localtime
[[ "$PWD" == "$HOME/gpu-site/scratch-trellis/tp-decode-v1/host-r2" ]]
[[ $# == 2 && "$1" == --inside-root-lease && ${ROOT_GPU_LEASE:-0} == 1 ]]
trap 'touch runs/STOP' ERR
receipt=$2
python3 -I -S -B experiments/tp_decode/admission.py --receipt "$receipt"
label=$(python3 -I -S -B -c 'import json,sys;print(json.load(open(sys.argv[1]))["label"])' "$receipt")
campaign=$(python3 -I -S -B -c 'import json,sys;print(json.load(open(sys.argv[1]))["campaign"])' "$receipt")
[[ "$label" =~ ^ROOT-tp-decode-v1-[a-zA-Z0-9_-]+$ && ! -e "runs/$label" && ! -e runs/STOP ]]
mkdir "runs/physical-$campaign-claimed"
mkdir "runs/$label"
export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" HIP_CACHE_DIR="$PWD/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MAX_JOBS=1
sha256sum -c --quiet frozen.sha256
python3 -I -S -B experiments/tp_decode/admission.py --receipt "$receipt" > "runs/$label/admission.json"
{ hostname; date -Is; uname -a; cat "$receipt"; } > "runs/$label/meta.txt"
# Inner hard timeout is 850s. No automatic second physical run after ANY failure.
/usr/bin/time -v timeout --signal=TERM --kill-after=3s 850s nice -n 10 bin/tp_native --root-campaign "$PWD/fixtures" > "runs/$label/native.log" 2> "runs/$label/native.stderr"
grep -Fx 'PASS TP_PRIVATE_CAMPAIGN' "runs/$label/native.log"
sha256sum -c --quiet frozen.sha256
python3 -I -S -B experiments/tp_decode/admission.py --receipt "$receipt" --after > "runs/$label/admission-after.json"
printf '%s\n' 'payload_exit=0 integrity_exit=0' > "runs/$label/closed.txt"
