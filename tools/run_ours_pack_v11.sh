#!/usr/bin/env bash
# Qualify supplied K4/K5 dense weights through the frozen API and local example.
set -euo pipefail
source "$(dirname -- "$0")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/ours-pack-v11" ]]
root="$HOME/gpu-site/scratch-trellis"
export TMPDIR="$TRELLIS_SOURCE/.tmp" XDG_CACHE_HOME="$TRELLIS_SOURCE/.cache"
export AMD_COMGR_CACHE_DIR="$XDG_CACHE_HOME/comgr"
sha256sum -c --quiet frozen.sha256
if (( $# )); then
    [[ $# == 1 && "$1" == --payload && ! -e runs/qualification-v1 ]]
    run="$TRELLIS_SOURCE/runs/qualification-v1"
    mkdir -p "$run/outputs"
    { hostname; TZ=:/etc/localtime date -Is; uname -r; cat frozen.sha256; } > "$run/meta.txt"
    phase=correctness
    trap 'status=$?; if (( status )); then echo "FAILED phase=$phase status=$status" > "$run/failed.txt"; fi' EXIT
    "$root/runtime-v11/build-hip/strix_trellis_gpu_runtime" fixtures/*.stgpu > "$run/correctness.log" 2> "$run/correctness.stderr"
    echo PHASE_PASS=correctness
    phase=api
    "$root/runtime-v11/build-hip/strix_trellis_api_linear_bench" fixtures/*.stgpu > "$run/api.log" 2> "$run/api.stderr"
    echo PHASE_PASS=api
    phase=graphs
    "$root/runtime-v11/build-hip/strix_trellis_graph_linear_bench" fixtures/*.stgpu > "$run/graphs.log" 2> "$run/graphs.stderr"
    echo PHASE_PASS=graphs
    phase=engine
    exe="$TRELLIS_SOURCE/build-engine/engine/strix-trellis-dense-example"
    for bits in 4 5; do
        for row in {0..7}; do
            "$exe" "$TRELLIS_SOURCE/k$bits/wkv" "$TRELLIS_SOURCE/k$bits/input-r$row.f32" "$run/outputs/k$bits-r$row.f32" 0 "$bits" > "$run/engine-k$bits-r$row.log" 2> "$run/engine-k$bits-r$row.stderr"
        done
    done
    for mode in plain bias; do
        "$exe" "$root/integration-$mode/wkv" "$root/input.f32" "$run/outputs/k6-$mode.f32" 0 > "$run/engine-k6-$mode.log" 2> "$run/engine-k6-$mode.stderr"
    done
    python3 - "$run" "$TRELLIS_SOURCE" "$root" > "$run/engine-comparison.jsonl" <<'PY'
from pathlib import Path
import sys,struct,math,json,hashlib
run,source,root=map(Path,sys.argv[1:])
cases=[(f'k{bits}-r{row}',source/f'k{bits}'/f'expected-r{row}.f32') for bits in [4,5] for row in range(8)]
cases += [(f'k6-{mode}',root/f'expected-{mode}.f32') for mode in ['plain','bias']]
for name,reference in cases:
    raw=(run/'outputs'/f'{name}.f32').read_bytes();ref=reference.read_bytes()
    assert len(raw)==len(ref)==2048
    a=struct.unpack('<512f',raw);b=struct.unpack('<512f',ref)
    assert all(math.isfinite(x) for x in a+b)
    error=max(abs(x-y) for x,y in zip(a,b));scale=max(abs(x) for x in b)
    nrms=math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/max(sum(x*x for x in b),1e-30))
    assert error<=5e-4*max(1,scale) and nrms<=3e-5
    print(json.dumps(dict(case=name,values=512,maxabs=error,maxref=scale,nrms=nrms,sha256=hashlib.sha256(raw).hexdigest(),passed=True)))
PY
    echo PHASE_PASS=engine-all-eight-rows-and-k6-regression
    sha256sum -c --quiet frozen.sha256
    sha256sum "$run"/outputs/* > "$run/outputs.sha256"
    TZ=:/etc/localtime date -Is >> "$run/meta.txt"
    echo PASS_ours_pack_v11 | tee "$run/passed.txt"
    exit
fi
[[ ! -e runs/qualification-v1 ]]
exec bash tools/run_gpu_turn.sh --seconds 180 --maintenance-check tools/check_maintenance.sh \
    --queue-arg strix-trellis-ours-v11 -- "$TRELLIS_SOURCE/tools/run_ours_pack_v11.sh" --payload
