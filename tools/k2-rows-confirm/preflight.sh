#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ "$PWD" == "$HOME/gpu-site/scratch-trellis/k2-rows-confirm" ]]
export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" HIP_CACHE_DIR="$PWD/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr" MAX_JOBS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for d in .tmp .cache .cache/hip .cache/comgr runs/1 runs/2 attempts/1 attempts/2; do [[ -d $d ]]; done
now=$(date +%s); clock=$(date +%H:%M); minutes=$((10#${clock%:*}*60+10#${clock#*:}))
((minutes<1260 || minutes>=1410))
if ((minutes<1260)); then (( $(date -d 'today 21:00' +%s)-now>195 )); fi
[[ ! -e "$HOME/gpu-site/.gpu-turn/maintenance" ]]
sha256sum --check --quiet frozen.sha256
(cd ../k2-rows-core; sha256sum --check --quiet frozen.sha256)
# Read/recompute all Q, original F32 input stages and full [4,8,N] FP64.
# The existing references/provenance must compare exactly; write_fresh is idempotent.
nice -n 10 python3 -B tools/k2-rows-core/prep.py --fixtures data --out data/derived
sha256sum --check --quiet frozen.sha256
printf 'PREFLIGHT PASS node=%s date=%s frozen=%s\n' "$(hostname)" "$(date -Is)" "$(sha256sum frozen.sha256 | cut -d' ' -f1)"
