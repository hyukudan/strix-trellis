#!/usr/bin/env bash
# ROOT-only entry point. Preparation scripts never invoke this runner.
set -euo pipefail
source "$(dirname -- "$0")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/runtime-v13-qa" ]]
export TMPDIR="$TRELLIS_SOURCE/.tmp" XDG_CACHE_HOME="$TRELLIS_SOURCE/.cache"
export HIP_CACHE_DIR="$XDG_CACHE_HOME/hip" AMD_COMGR_CACHE_DIR="$XDG_CACHE_HOME/comgr"
sha256sum -c --quiet frozen.sha256
if [[ ${1:-} == --payload ]]; then
    [[ $# == 2 && "$2" =~ ^[a-z0-9][a-z0-9-]{0,31}$ ]]
    run="$TRELLIS_SOURCE/runs/$2";[[ ! -e "$run" ]];mkdir -p "$run"
    trap 'status=$?; if (( status )); then echo "FAILED status=$status" > "$run/failed.txt"; fi' EXIT
    { hostname; TZ=:/etc/localtime date -Is; cat frozen.sha256; } > "$run/meta.txt"
    root="$HOME/gpu-site/scratch-trellis"
    timeout --signal=TERM --kill-after=3s 120s build/v12_reference "$root" "$run/v12" > "$run/v12.log" 2> "$run/v12.stderr"
    STRICT_CPU_HEAD=${STRICT_CPU_HEAD:-"$TRELLIS_SOURCE/reference/cpu/r8-y-fp64-core.f32"}
    args=("$root" "$TRELLIS_SOURCE/reference" "$run/v12")
    if [[ -n ${STRICT_CPU_HEAD:-} ]]; then args+=("$STRICT_CPU_HEAD"); fi
    /usr/bin/time -v timeout --signal=TERM --kill-after=3s 420s build/grouped_direct_runtime_qa "${args[@]}" > "$run/native.log" 2> "$run/native.stderr"
    sha256sum -c --quiet frozen.sha256
    if [[ -n ${STRICT_CPU_HEAD:-} ]]; then echo PASS_v13_native_full_CPU_and_CUDA; else echo PREPARED_CPU_HEAD_GATE_PENDING_native_CUDA_pass; fi | tee "$run/passed.txt"
    exit
fi
[[ $# == 1 && "$1" =~ ^[a-z0-9][a-z0-9-]{0,31}$ ]]
[[ ! -e "runs/$1" ]]
exec bash tools/run_gpu_turn.sh --seconds 600 --maintenance-check tools/check_maintenance.sh \
    --queue-arg strix-trellis-v13-native -- "$TRELLIS_SOURCE/tools/run_v13_native_qa.sh" --payload "$1"
