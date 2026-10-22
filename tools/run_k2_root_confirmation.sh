#!/usr/bin/env bash
# A separate, bounded root confirmation after one failed and one valid worker run.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/k2-core" ]]
export TMPDIR="$TRELLIS_SOURCE/.tmp" XDG_CACHE_HOME="$TRELLIS_SOURCE/.cache"
export AMD_COMGR_CACHE_DIR="$XDG_CACHE_HOME/comgr"
[[ -d "$TMPDIR" && -d "$AMD_COMGR_CACHE_DIR" ]]
sha256sum -c --quiet frozen.sha256
sha256sum -c --quiet root-confirmation.sha256
if [[ ${1:-} == --payload ]]; then
    [[ $# == 1 && ! -e runs/root-confirmation-v1 ]]
    mkdir -p runs/root-confirmation-v1
    run=runs/root-confirmation-v1
    { hostname; TZ=:/etc/localtime date -Is; uname -r; cat frozen.sha256 root-confirmation.sha256; } > "$run/meta.txt"
    /usr/bin/time -v timeout --signal=TERM --kill-after=3s 145s \
        nice -n 10 bin/k2_core_probe data/wo_b.stgpu data/q_b.stgpu > "$run/benchmark.log" 2> "$run/benchmark.stderr"
    sha256sum -c --quiet frozen.sha256
    TZ=:/etc/localtime date -Is >> "$run/meta.txt"
    echo PASS_root_k2_confirmation | tee "$run/passed.txt"
    exit
fi
[[ $# == 0 && ! -e runs/root-confirmation-v1 ]]
exec bash tools/run_gpu_turn.sh --seconds 180 --maintenance-check tools/check_maintenance.sh \
    --queue-arg strix-trellis-k2-root-confirm -- "$TRELLIS_SOURCE/tools/run_k2_root_confirmation.sh" --payload
