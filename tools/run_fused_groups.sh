#!/usr/bin/env bash
set -euo pipefail
own="$HOME/gpu-site/scratch-trellis/fused-groups"
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
    lib="$HOME/gpu-site/scratch-trellis/runtime-v11/build-hip/libstrix_trellis.a"
    echo "201d596e469c3d81a91aaeffa4588801fad5ed2d3b8d8cc1279458b78f6b0597  $lib" | sha256sum -c -
    nice -n 10 /opt/rocm/core-10.1/bin/hipcc -O3 -ffp-contract=off -std=c++17 --offload-arch=gfx1151 -Isrc -Dfused_output=qualified_g4 -c src/gemv_fused_output.hip -o build/original.o
    nice -n 10 /opt/rocm/core-10.1/bin/hipcc -O3 -ffp-contract=off -std=c++17 --offload-arch=gfx1151 -Isrc -c src/gemv_fused_groups.hip -o build/groups.o
    nice -n 10 /opt/rocm/core-10.1/bin/hipcc -O3 -ffp-contract=off -std=c++17 --offload-arch=gfx1151 -Isrc bench/fused_groups_probe.hip -x none build/groups.o build/original.o "$lib" -o bin/fused_groups_probe
    nice -n 10 /opt/rocm/core-10.1/bin/hipcc -O3 -ffp-contract=off -std=c++17 --offload-arch=gfx1151 --cuda-device-only -S -Isrc src/gemv_fused_groups.hip -o build/candidate.s
    /opt/rocm/core-10.1/llvm/bin/llvm-objdump --offloading bin/fused_groups_probe > build/offloading.txt
    : > build/linked-notes.txt
    : > build/linked-isa.txt
    for image in bin/*.hip-amdgcn-amd-amdhsa--gfx1151; do
        /opt/rocm/core-10.1/llvm/bin/llvm-readobj --notes "$image" >> build/linked-notes.txt
        /opt/rocm/core-10.1/llvm/bin/llvm-objdump -d --mcpu=gfx1151 "$image" >> build/linked-isa.txt
    done
    /opt/rocm/core-10.1/bin/hipcc --version > build/toolchain.txt
    sha256sum /opt/rocm/core-10.1/llvm/bin/clang* /opt/rocm/core-10.1/llvm/bin/llvm-objdump > build/toolchain.sha256
    python3 reports/fused-groups-data/prepare.py
    python3 reports/fused-groups-data/proof.py --isa build > build/proof.txt
    ;;
 --freeze)
    sha256sum src/* bench/* tools/* reports/fused-groups-data/*.py reports/fused-groups-data/protocol.json bin/* build/candidate.s build/offloading.txt build/linked-notes.txt build/linked-isa.txt build/toolchain.txt build/toolchain.sha256 build/fixtures.sha256 build/reference-tensors.json reports/fused-groups-data/static-proof.json "$HOME/gpu-site/scratch-trellis/runtime-v11/build-hip/libstrix_trellis.a" > build/frozen.sha256
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
    /usr/bin/time -v ./bin/fused_groups_probe fixtures > "runs/$2/probe.log" 2> "runs/$2/probe.stderr"
    sha256sum -c build/frozen.sha256 > "runs/$2/frozen-after.txt"
    sha256sum -c build/fixtures.sha256 > "runs/$2/fixtures-after.txt"
    TZ=:/etc/localtime date -Is >> "runs/$2/meta.txt"
    ;;
 search|confirm)
    check
    sha256sum -c build/frozen.sha256 >/dev/null
    sha256sum -c build/fixtures.sha256 >/dev/null
    # Exactly one synchronous pending request; the payload gets at most 180s.
    exec "$own/tools/run_gpu_turn_fused_groups.sh" "$1"
    ;;
 *) echo 'Usage: --build|--freeze|search|confirm' >&2; exit 1;;
esac
