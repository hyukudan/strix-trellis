#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/gemv-combined" ]] || trellis_die 'Expected own gemv-combined subtree'
if [[ ${1:-} == --build ]]; then
 export TMPDIR="$TRELLIS_SOURCE/.tmp" XDG_CACHE_HOME="$TRELLIS_SOURCE/.cache" AMD_COMGR_CACHE_DIR="$TRELLIS_SOURCE/.cache/comgr" MAX_JOBS=1
 mkdir -p "$TMPDIR" "$AMD_COMGR_CACHE_DIR" bin build
 nice -n 10 /opt/rocm/core-10.1/bin/hipcc -O3 -ffp-contract=off -std=c++17 --offload-arch=gfx1151 -Isrc src/gemv_combined.hip bench/gemv_combined_bench.hip -o bin/gemv_combined_bench
 nice -n 10 /opt/rocm/core-10.1/bin/hipcc -O3 -ffp-contract=off -std=c++17 --offload-arch=gfx1151 -Isrc --cuda-device-only -S src/gemv_combined.hip -o build/combined.s
 sha256sum src/* include/* bench/* tools/* bin/* build/combined.s > build/frozen.sha256
 exit
fi
if [[ ${1:-} == --payload ]]; then
 [[ ${2:-} == search || ${2:-} == confirm ]] || trellis_die 'Expected search or confirm'
 run="$TRELLIS_SOURCE/runs/$2"
 [[ ! -e "$run" ]] || trellis_die 'Run already exists'
 mkdir -p "$run"
 { hostname; TZ=:/etc/localtime date -Is; uname -a; /opt/rocm/core-10.1/bin/hipcc --version; sha256sum -c build/frozen.sha256;
 for f in /sys/class/drm/card*/device/uevent /sys/class/drm/card*/device/pp_dpm_sclk /sys/class/drm/card*/device/pp_dpm_mclk; do if [[ -r "$f" ]]; then echo "$f";cat "$f";fi;done; } > "$run/meta.txt"
 root="$HOME/gpu-site/scratch-trellis"
 fixtures=("$root/runtime-fixtures/eaq-wkv-k6-mul1.stgpu" "$root/gpu-dense-prefill-tests/layers.0.attn.wq_b.stgpu" "$root/gpu-dense-prefill-tests/layers.0.attn.wo_b.stgpu" "$root/gpu-dense-prefill-tests/layers.0.ffn.shared_experts.w1.stgpu" "$root/gpu-dense-prefill-tests/layers.0.ffn.shared_experts.w2.stgpu")
 sha256sum "${fixtures[@]}" > "$run/fixtures.sha256"
 if [[ "$2" == search ]]; then ./bin/gemv_combined_bench --validate > "$run/validation.log" 2> "$run/validation.stderr"; fi
 ./bin/gemv_combined_bench --fixtures "${fixtures[@]}" > "$run/fixtures-before.log" 2> "$run/fixtures-before.stderr"
 ./bin/gemv_combined_bench --bench > "$run/benchmark.log" 2> "$run/benchmark.stderr"
 ./bin/gemv_combined_bench --fixtures "${fixtures[@]}" > "$run/fixtures-after.log" 2> "$run/fixtures-after.stderr"
 sha256sum -c build/frozen.sha256 > "$run/frozen-after.txt"
 TZ=:/etc/localtime date -Is >> "$run/meta.txt"
 echo PASS_combined_$2 | tee "$run/passed.txt"
 exit
fi
[[ ${1:-} == search || ${1:-} == confirm ]] || trellis_die 'Usage: --build|search|confirm'
sha256sum -c build/frozen.sha256 > /dev/null
exec bash tools/run_gpu_turn.sh --seconds 180 --maintenance-check tools/check_maintenance.sh --queue-arg strix-trellis-gemv-combined -- "$TRELLIS_SOURCE/tools/run_gemv_combined_bench.sh" --payload "$1"
