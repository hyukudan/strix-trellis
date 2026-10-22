#!/usr/bin/env bash
# Hardware counters for a fixed qualified child; no system profiler attachment.
set -euo pipefail
ulimit -c 0
source "$(dirname -- "${BASH_SOURCE[0]}")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/roofline-probe" ]] || trellis_die 'Expected roofline-probe subtree'
if [[ ${1:-} == --payload ]]; then
    [[ $# == 2 && "$2" =~ ^[a-z0-9][a-z0-9-]{0,31}$ ]] || trellis_die 'Invalid label'
    run="$TRELLIS_SOURCE/runs/$2"
    trellis_require_owned "$run"
    [[ ! -e "$run" ]] || trellis_die 'Existing run is never overwritten'
    sha256sum -c --quiet frozen.sha256
    sha256sum -c --quiet counters-v2-frozen.sha256
    mkdir -p "$run" "$TRELLIS_SOURCE/.tmp" "$TRELLIS_SOURCE/.cache/comgr"
    export TMPDIR="$TRELLIS_SOURCE/.tmp" XDG_CACHE_HOME="$TRELLIS_SOURCE/.cache" AMD_COMGR_CACHE_DIR="$TRELLIS_SOURCE/.cache/comgr"
    { hostname; TZ=:/etc/localtime date -Is; uname -r; cat counters-v2-frozen.sha256; cat frozen.sha256; } > "$run/meta.txt"
    for shape in wq_b wo_b; do
        fixture="$HOME/gpu-site/scratch-trellis/gpu-dense-prefill-tests/layers.0.attn.$shape.stgpu"
        for rows in 1 8; do
            for group in compute fetch write busy; do
                if [[ "$group" == compute ]]; then
                    counters=(SQ_INSTS_VALU_sum SQ_INSTS_SALU_sum SQ_INSTS_LDS_sum SQ_WAVES_sum)
                elif [[ "$group" == fetch ]]; then
                    counters=(FETCH_SIZE)
                elif [[ "$group" == write ]]; then
                    counters=(WRITE_SIZE)
                else
                    counters=(MemUnitBusy GRBM_GUI_ACTIVE)
                fi
                label="$shape-r$rows-$group"
                mkdir -p "$run/$label"
                printf '%s\n' "${counters[*]}" > "$run/$label/metrics.txt"
                set +e
                /opt/rocm/core-10.1/bin/rocprofv3-avail pmc-check "${counters[@]}" > "$run/$label/pmc-check.log" 2> "$run/$label/pmc-check.stderr"
                check_status=$?
                set -e
                if (( check_status != 0 )); then
                    printf '%s unsupported_preflight=%d\n' "$label" "$check_status" | tee -a "$run/status.txt"
                    continue
                fi
                set +e
                timeout --signal=TERM --kill-after=2s 60s /opt/rocm/core-10.1/bin/rocprofv3 \
                  --pmc "${counters[@]}" --kernel-include-regex gemv_ --output-format csv \
                  --output-directory "$run/$label" --output-file profile -- \
                  "$TRELLIS_SOURCE/roofline_probe" "$fixture" "$rows" > "$run/$label/workload.log" 2> "$run/$label/workload.stderr"
                status=$?
                set -e
                printf '%s status=%d\n' "$label" "$status" | tee -a "$run/status.txt"
                (( status == 0 )) || exit "$status"
            done
        done
    done
    TZ=:/etc/localtime date -Is >> "$run/meta.txt"
    echo PASS_completed_supported_counter_groups | tee "$run/passed.txt"
    exit
fi
[[ $# == 1 && "$1" =~ ^[a-z0-9][a-z0-9-]{0,31}$ && ! -e runs/$1 ]] || trellis_die 'Usage: NEW_RUN_LABEL'
sha256sum -c --quiet counters-v2-frozen.sha256
exec bash tools/run_gpu_turn.sh --seconds 180 --maintenance-check tools/check_maintenance.sh \
    --queue-arg strix-trellis-hardware-counters -- "$TRELLIS_SOURCE/tools/run_roofline_counters_v2.sh" --payload "$1"
