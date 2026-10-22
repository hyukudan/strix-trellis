#!/usr/bin/env bash
# Inventory and trace only this bounded child, inside the approved GPU lease.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/roofline-probe" ]] || trellis_die 'Expected roofline-probe subtree'
if [[ ${1:-} == --payload ]]; then
    [[ $# == 2 && "$2" =~ ^[a-z0-9][a-z0-9-]{0,31}$ ]] || trellis_die 'Invalid label'
    run="$TRELLIS_SOURCE/runs/$2"
    trellis_require_owned "$run"
    [[ ! -e "$run" ]] || trellis_die 'Existing run is never overwritten'
    sha256sum -c --quiet frozen.sha256
    mkdir -p "$run/trace" "$TRELLIS_SOURCE/.tmp" "$TRELLIS_SOURCE/.cache/comgr"
    export TMPDIR="$TRELLIS_SOURCE/.tmp" XDG_CACHE_HOME="$TRELLIS_SOURCE/.cache" AMD_COMGR_CACHE_DIR="$TRELLIS_SOURCE/.cache/comgr"
    { hostname; TZ=:/etc/localtime date -Is; uname -r; cat frozen.sha256; } > "$run/meta.txt"
    # Both inventory and all child execution are inside the acquired lease.
    set +e
    timeout --signal=TERM --kill-after=2s 30s /opt/rocm/core-10.1/bin/rocprofv3 --list-avail > "$run/available.txt" 2> "$run/available.stderr"
    inventory_status=$?
    printf 'inventory_status=%d\n' "$inventory_status" > "$run/status.txt"
    timeout --signal=TERM --kill-after=2s 90s /opt/rocm/core-10.1/bin/rocprofv3 --kernel-trace --stats --output-format csv --output-directory "$run/trace" --output-file profile -- \
      "$TRELLIS_SOURCE/roofline_probe" "$HOME/gpu-site/scratch-trellis/gpu-dense-prefill-tests/layers.0.attn.wo_b.stgpu" 8 > "$run/workload.log" 2> "$run/workload.stderr"
    trace_status=$?
    set -e
    printf 'trace_status=%d\n' "$trace_status" >> "$run/status.txt"
    TZ=:/etc/localtime date -Is >> "$run/meta.txt"
    cat "$run/status.txt"
    (( trace_status == 0 )) || exit "$trace_status"
    echo PASS_bounded_profile_trace
    exit
fi
[[ $# == 1 && "$1" =~ ^[a-z0-9][a-z0-9-]{0,31}$ && ! -e runs/$1 ]] || trellis_die 'Usage: NEW_RUN_LABEL'
sha256sum -c --quiet frozen.sha256
exec bash tools/run_gpu_turn.sh --seconds 180 --maintenance-check tools/check_maintenance.sh \
    --queue-arg strix-trellis-roofline -- "$TRELLIS_SOURCE/tools/run_roofline_probe.sh" --payload "$1"
