#!/usr/bin/env bash
set -euo pipefail
own="$HOME/gpu-site/scratch-trellis/fused-output-core"
[[ $(pwd -P) == "$own" ]] || { echo 'Wrong private directory' >&2; exit 1; }
export TMPDIR="$own/.tmp" TMP="$own/.tmp" TEMP="$own/.tmp"
export XDG_CACHE_HOME="$own/.cache" AMD_COMGR_CACHE_DIR="$own/.cache/comgr"
export HIP_CACHE_DIR="$own/.cache/hip" POCL_CACHE_DIR="$own/.cache/pocl"
export CUDA_CACHE_PATH="$own/.cache/cuda" MIOPEN_USER_DB_PATH="$own/.cache/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="$own/.cache/miopen" MAX_JOBS=1
mkdir -p "$TMPDIR" "$AMD_COMGR_CACHE_DIR" "$HIP_CACHE_DIR" "$POCL_CACHE_DIR" "$CUDA_CACHE_PATH" "$MIOPEN_USER_DB_PATH" build bin
check() {
    local now hour minute
    now=$(TZ=:/etc/localtime date +%s)
    hour=$(TZ=:/etc/localtime date +%H); minute=$(TZ=:/etc/localtime date +%M)
    minute=$((10#$hour*60+10#$minute))
    (( minute < 1260 || minute >= 1410 )) || { echo 'User veto 21:00-23:30' >&2; return 1; }
    if (( minute < 1260 )); then
        (( $(TZ=:/etc/localtime date -d 'today 21:00' +%s)-now > 200 )) || return 1
    fi
    [[ ! -e "$HOME/gpu-site/.gpu-turn/maintenance" ]]
}
case ${1:-} in
 --build)
    lib="$HOME/gpu-site/scratch-trellis/build-hip/libstrix_trellis_fused_v9.a"
    echo "1e111789526692f0b9a3e713fcbf6723a1bec4d959a1a24cb9dde175001b0325  $lib" | sha256sum -c -
    nice -n 10 /opt/rocm/core-10.1/bin/hipcc -O3 -ffp-contract=off -std=c++17 --offload-arch=gfx1151 -Isrc src/gemv_fused_output.hip bench/fused_output_probe.hip -x none "$lib" -o bin/fused_output_probe
    # CPU-only device compilation for readable comparison with the linked ELF.
    nice -n 10 /opt/rocm/core-10.1/bin/hipcc -O3 -ffp-contract=off -std=c++17 --offload-arch=gfx1151 --cuda-device-only -S -Isrc src/gemv_fused_output.hip -o build/candidate.s
    /opt/rocm/core-10.1/llvm/bin/llvm-objdump --offloading bin/fused_output_probe > build/offloading.txt
    /opt/rocm/core-10.1/llvm/bin/llvm-readobj --notes bin/fused_output_probe.0.hip-amdgcn-amd-amdhsa--gfx1151 > build/linked-notes.txt
    /opt/rocm/core-10.1/llvm/bin/llvm-objdump -d --mcpu=gfx1151 bin/fused_output_probe.0.hip-amdgcn-amd-amdhsa--gfx1151 > build/linked-isa.txt
    /opt/rocm/core-10.1/bin/hipcc --version > build/toolchain.txt
    sha256sum /opt/rocm/core-10.1/llvm/bin/clang* /opt/rocm/core-10.1/llvm/bin/llvm-objdump > build/toolchain.sha256
    python3 reports/fused-output-core-data/prepare.py
    ;;
 --freeze)
    sha256sum src/* bench/* tools/* reports/fused-output-core-data/prepare.py bin/* build/candidate.s build/offloading.txt build/linked-notes.txt build/linked-isa.txt build/toolchain.txt build/toolchain.sha256 build/fixtures.sha256 "$HOME/gpu-site/scratch-trellis/build-hip/libstrix_trellis_fused_v9.a" > build/frozen.sha256
    ;;
 --payload)
    [[ ${2:-} == search || ${2:-} == confirm ]] || exit 1
    check
    [[ ! -e runs/$2 ]] || exit 1
    mkdir -p "runs/$2"
    if [[ ${FO_INTERNAL:-0} != 1 ]]; then
        rmdir "runs/$2"
        exec timeout --signal=TERM --kill-after=3s 145s env FO_INTERNAL=1 "$0" --payload "$2"
    fi
    { hostname; TZ=:/etc/localtime date -Is; uname -a; cat build/toolchain.txt; } > "runs/$2/meta.txt"
    sha256sum -c build/frozen.sha256 > "runs/$2/frozen-before.txt"
    sha256sum -c build/fixtures.sha256 > "runs/$2/fixtures-before.txt"
    /usr/bin/time -v ./bin/fused_output_probe fixtures > "runs/$2/probe.log" 2> "runs/$2/probe.stderr"
    sha256sum -c build/frozen.sha256 > "runs/$2/frozen-after.txt"
    sha256sum -c build/fixtures.sha256 > "runs/$2/fixtures-after.txt"
    TZ=:/etc/localtime date -Is >> "runs/$2/meta.txt"
    ;;
 search|confirm)
    check
    sha256sum -c build/frozen.sha256 >/dev/null
    sha256sum -c build/fixtures.sha256 >/dev/null
    # Exactly one synchronous pending request; the payload gets at most 180s.
    exec "$HOME/gpu-site/gpu-turn-any.sh" strix-trellis-fused-output-core \
        timeout --signal=TERM --kill-after=3s 180s "$own/tools/run_fused_output_core.sh" --payload "$1"
    ;;
 *) echo 'Usage: --build|--freeze|search|confirm' >&2; exit 1;;
esac
