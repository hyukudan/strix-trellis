#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/gemv-split-operator" ]] || trellis_die 'Expected own subtree'
export TMPDIR="$TRELLIS_SOURCE/.tmp" XDG_CACHE_HOME="$TRELLIS_SOURCE/.cache" AMD_COMGR_CACHE_DIR="$TRELLIS_SOURCE/.cache/comgr" MAX_JOBS=1
mkdir -p "$TMPDIR" "$AMD_COMGR_CACHE_DIR" bin build
fixtures=("$HOME/gpu-site/scratch-trellis/gpu-dense-prefill-tests/"*.stgpu "$HOME/gpu-site/scratch-trellis/runtime-fixtures/eaq-wkv-k6-mul1.stgpu")
if [[ ${1:-} == --build ]]; then
 echo '938dbf966c634ef7f0d5322006a827d043b7e428d1c0b881a82108db3e65fda3  reference/libstrix_trellis_fused_v8.a' | sha256sum -c -
 nice -n 10 /opt/rocm/core-10.1/bin/hipcc -O3 -ffp-contract=off -std=c++17 --offload-arch=gfx1151 -Isrc -Iinclude src/gemv_combined.hip bench/gemv_split_operator_bench.hip -x none reference/libstrix_trellis_fused_v8.a -o bin/gemv_split_operator_bench
 sha256sum src/* include/* tests/gpu_runtime.hip bench/gemv_split_operator_bench.hip tools/* reference/* bin/* > build/frozen.sha256
 sha256sum "${fixtures[@]}" > build/fixtures.sha256
 exit
fi
if [[ ${1:-} == --payload ]]; then
 [[ ${2:-} == search || ${2:-} == confirm ]] || trellis_die 'Expected campaign'
 run="$TRELLIS_SOURCE/runs/$2"
 [[ ! -e "$run" ]] || trellis_die 'Fresh logs required'
 mkdir -p "$run"
 { hostname; TZ=:/etc/localtime date -Is; uname -a; /opt/rocm/core-10.1/bin/hipcc --version;
 for f in /sys/class/drm/card*/device/uevent /sys/class/drm/card*/device/pp_dpm_sclk /sys/class/drm/card*/device/pp_dpm_mclk; do if [[ -r "$f" ]]; then echo "$f";cat "$f";fi;done; } > "$run/meta.txt"
 sha256sum -c build/frozen.sha256 > "$run/frozen-before.txt"
 sha256sum -c build/fixtures.sha256 > "$run/fixtures-before.txt"
 /usr/bin/time -v ./bin/gemv_split_operator_bench "${fixtures[@]}" > "$run/benchmark.raw" 2> "$run/benchmark.stderr"
 sha256sum -c build/frozen.sha256 > "$run/frozen-after.txt"
 sha256sum -c build/fixtures.sha256 > "$run/fixtures-after.txt"
 TZ=:/etc/localtime date -Is >> "$run/meta.txt"
 echo "PASS_$2" | tee "$run/passed.txt"
 exit
fi
[[ ${1:-} == search || ${1:-} == confirm ]] || trellis_die 'Usage: --build|search|confirm'
sha256sum -c build/frozen.sha256 >/dev/null
sha256sum -c build/fixtures.sha256 >/dev/null
exec bash tools/run_gpu_turn.sh --seconds 180 --maintenance-check tools/check_maintenance.sh --queue-arg strix-trellis-gemv-split-operator -- "$TRELLIS_SOURCE/tools/run_gemv_split_operator_bench.sh" --payload "$1"
