#!/usr/bin/env bash
# Frozen candidate qualification; GPU use is exclusively inside the shared turn.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/runtime-v12" ]] || trellis_die 'Expected own runtime-v12 subtree'
root="$HOME/gpu-site/scratch-trellis"
if [[ ${1:-} == --payload ]]; then
    [[ $# == 2 && "$2" =~ ^[a-z0-9][a-z0-9-]{0,31}$ ]] || trellis_die 'Invalid run label'
    run="$TRELLIS_SOURCE/runs/$2"
    trellis_require_owned "$run"
    [[ ! -e "$run" ]] || trellis_die 'Existing run is never overwritten'
    sha256sum -c --quiet frozen.sha256
    mkdir -p "$run"
    phase=metadata
    trap 'status=$?; if (( status )); then printf "FAILED phase=%s status=%d\n" "$phase" "$status" > "$run/failed.txt"; fi' EXIT
    { hostname; TZ=:/etc/localtime date -Is; uname -r; cat frozen.sha256; } > "$run/meta.txt"
    fixtures=("$root"/runtime-fixtures/*.stgpu "$root"/gpu-dense-prefill-tests/*.stgpu "$root"/gpu-numerical-tests/*.stgpu
              "$root"/prefill-long/fixtures/*.stgpu "$root"/k2-core/data/*.stgpu)
    [[ ${#fixtures[@]} == 84 ]] || trellis_die 'Expected exactly 84 fixtures'
    sha256sum "${fixtures[@]}" > "$run/fixtures.sha256"
    phase=correctness
    build-hip/strix_trellis_gpu_runtime "${fixtures[@]}" > "$run/correctness.log" 2> "$run/correctness.stderr"
    echo PHASE_PASS=correctness
    dense=("$root/gpu-dense-prefill-tests/layers.0.attn.wq_b.stgpu"
           "$root/gpu-dense-prefill-tests/layers.0.attn.wo_b.stgpu"
           "$root/gpu-dense-prefill-tests/layers.0.ffn.shared_experts.w1.stgpu"
           "$root/gpu-dense-prefill-tests/layers.0.ffn.shared_experts.w2.stgpu"
           "$root/gpu-dense-prefill-tests/layers.0.ffn.shared_experts.w3.stgpu"
           "$root/gpu-dense-prefill-tests/layers.0.attn.wq_a.stgpu"
           "$root/gpu-dense-prefill-tests/layers.0.attn.wo_a.slice.0.stgpu"
           "$root/runtime-fixtures/eaq-wkv-k6-mul1.stgpu" "$root/k2-core/data/wo_b.stgpu" "$root/k2-core/data/q_b.stgpu")
    phase=api-v12
    build-hip/strix_trellis_api_linear_bench "${dense[@]}" "$root"/prefill-long/fixtures/*.stgpu > "$run/api-v12.log" 2> "$run/api-v12.stderr"
    echo PHASE_PASS=api-v12
    phase=causal
    mkdir -p "$run/causal"
    for repetition in 0 1 2; do
        if (( repetition % 2 )); then order=(v12 v11); else order=(v11 v12); fi
        for index in "${!dense[@]}"; do
            for version in "${order[@]}"; do
                executable="$root/runtime-v11/build-hip/strix_trellis_api_linear_bench"
                if [[ "$version" == v12 ]]; then executable="$TRELLIS_SOURCE/build-hip/strix_trellis_api_linear_bench"; fi
                "$executable" "${dense[$index]}" > "$run/causal/r$repetition-f$index-$version.log" 2> "$run/causal/r$repetition-f$index-$version.stderr"
            done
        done
        for version in "${order[@]}"; do
            executable="$root/runtime-v11/build-hip/strix_trellis_head_linear_bench"
            if [[ "$version" == v12 ]]; then executable="$TRELLIS_SOURCE/build-hip/strix_trellis_head_linear_bench"; fi
            "$executable" "$root/gpu-head-eaq-ref-v2" > "$run/causal/r$repetition-head-$version.log" 2> "$run/causal/r$repetition-head-$version.stderr"
        done
    done
    echo PHASE_PASS=causal-three-pairs
    phase=graphs
    build-hip/strix_trellis_graph_linear_bench "${dense[@]}" > "$run/graphs.log" 2> "$run/graphs.stderr"
    echo PHASE_PASS=graphs
    phase=head
    build-hip/strix_trellis_head_linear_bench "$root/gpu-head-eaq-ref-v2" > "$run/head.log" 2> "$run/head.stderr"
    echo PHASE_PASS=head
    phase=head-full
    build-hip/head_crosscheck "$root/gpu-head-eaq-ref-v2" "$run/head-full" > "$run/head-full.log" 2> "$run/head-full.stderr"
    sha256sum "$run"/head-full/* > "$run/head-full.sha256"
    echo PHASE_PASS=head-full-dumps
    phase=head-full-v11
    "$root/runtime-v11/build-hip/head_crosscheck" "$root/gpu-head-eaq-ref-v2" "$run/head-full-v11" > "$run/head-full-v11.log" 2> "$run/head-full-v11.stderr"
    for rows in {1..8}; do
        for stage in a.f16 y.f32; do
            cmp "$run/head-full/r$rows-$stage" "$run/head-full-v11/r$rows-$stage"
        done
    done
    sha256sum "$run"/head-full-v11/* > "$run/head-full-v11.sha256"
    echo PHASE_PASS=head-full-all-rows-identical-v11
    phase=engine
    (cd "$root/engine-v12" && sha256sum -c --quiet frozen.sha256 && cat frozen.sha256) > "$run/engine.meta"
    "$root/engine-v12/build/engine/strix-trellis-dense-example" "$root/integration-plain/wkv" "$root/input.f32" "$run/engine-plain.f32" 0 > "$run/engine-plain.log" 2> "$run/engine-plain.stderr"
    "$root/engine-v12/build/engine/strix-trellis-dense-example" "$root/integration-bias/wkv" "$root/input.f32" "$run/engine-bias.f32" 0 > "$run/engine-bias.log" 2> "$run/engine-bias.stderr"
    python3 - "$run" "$root" > "$run/engine-comparison.jsonl" <<'PY'
import struct, math, hashlib, json, sys
from pathlib import Path
run, root=map(Path,sys.argv[1:])
for mode in ['plain','bias']:
    raw=(run/f'engine-{mode}.f32').read_bytes()
    reference=(root/f'expected-{mode}.f32').read_bytes()
    assert len(raw)==len(reference)==2048
    a=struct.unpack('<512f',raw);b=struct.unpack('<512f',reference)
    assert all(math.isfinite(v) for v in a+b)
    error=max(abs(x-y) for x,y in zip(a,b));scale=max(abs(v) for v in b)
    nrms=math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/max(sum(x*x for x in b),1e-30))
    assert error<=5e-4*max(1,scale) and nrms<=3e-5
    print(json.dumps(dict(mode=mode,maxabs=error,maxref=scale,nrms=nrms,sha256=hashlib.sha256(raw).hexdigest(),passed=True)))
PY
    echo PHASE_PASS=engine
    TZ=:/etc/localtime date -Is >> "$run/meta.txt"
    echo PASS_runtime_v12_candidate | tee "$run/passed.txt"
    exit
fi
[[ $# == 1 && "$1" =~ ^[a-z0-9][a-z0-9-]{0,31}$ ]] || trellis_die 'Usage: RUN_LABEL'
[[ ! -e "runs/$1" ]] || trellis_die 'Existing run is never overwritten'
sha256sum -c --quiet frozen.sha256
exec bash tools/run_gpu_turn.sh --seconds 300 --maintenance-check tools/check_maintenance.sh \
    --queue-arg strix-trellis-runtime-v12 -- "$TRELLIS_SOURCE/tools/run_runtime_v12.sh" --payload "$1"
