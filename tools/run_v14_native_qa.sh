#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/runtime-v14-qa" ]]
export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" HIP_CACHE_DIR="$PWD/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr"
if [[ ${1:-} == --payload ]]; then
  [[ $# == 2 && "$2" =~ ^[a-z0-9][a-z0-9-]{0,31}$ && ! -e "runs/$2" ]]
  sha256sum -c --quiet frozen.sha256
  mkdir -p "runs/$2"
  timeout --signal=TERM --kill-after=5s 420s build/runtime_v14_qa "$HOME/gpu-site/scratch-trellis" > "runs/$2/native.log" 2> "runs/$2/native.stderr"
  tail -n 1 "runs/$2/native.log" | grep -Fx 'PASS native_runtime_v14 attributes=4 K2_all_partials=1 WMMA_tuples=16'
  echo PASS_native_runtime_v14 > "runs/$2/passed.txt"
  exit
fi
[[ $# == 1 && "$1" =~ ^[a-z0-9][a-z0-9-]{0,31}$ && ! -e "runs/$1" ]]
sha256sum -c --quiet frozen.sha256
exec bash tools/run_gpu_turn.sh --seconds 600 --maintenance-check tools/check_maintenance.sh --queue-arg strix-trellis-runtime-v14-native -- "$TRELLIS_SOURCE/tools/run_v14_native_qa.sh" --payload "$1"
