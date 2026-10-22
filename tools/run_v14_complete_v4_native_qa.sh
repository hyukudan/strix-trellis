#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../tools/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/runtime-v14-qa" ]]
export TMPDIR="$PWD/complete-v4/.tmp" XDG_CACHE_HOME="$PWD/complete-v4/.cache" HIP_CACHE_DIR="$PWD/complete-v4/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/complete-v4/.cache/comgr"
if [[ ${1:-} == --payload ]]; then
  [[ $# == 2 && "$2" =~ ^[a-z0-9][a-z0-9-]{0,31}$ && ! -e "complete-v4/runs/$2" ]]
  sha256sum -c --quiet frozen.sha256
  sha256sum -c --quiet complete-v4/frozen.sha256
  mkdir -p "complete-v4/runs/$2"
  timeout --signal=TERM --kill-after=5s 420s complete-v4/runtime_v14_qa "$HOME/gpu-site/scratch-trellis" > "complete-v4/runs/$2/native.log" 2> "complete-v4/runs/$2/native.stderr"
  tail -n 1 "complete-v4/runs/$2/native.log" | grep -Fx 'PASS native_runtime_v14 attributes=4 K2_all_partials=1 WMMA_tuples=16'
  echo PASS_native_runtime_v14 > "complete-v4/runs/$2/passed.txt"
  exit
fi
echo 'ROOT parent payload only: use --payload FRESH_LABEL' >&2
exit 2
