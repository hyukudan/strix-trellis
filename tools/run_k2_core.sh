#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/k2-core" ]] || trellis_die 'Expected private k2-core subtree'
export TMPDIR="$TRELLIS_SOURCE/.tmp" XDG_CACHE_HOME="$TRELLIS_SOURCE/.cache" MAX_JOBS=1
export AMD_COMGR_CACHE_DIR="$XDG_CACHE_HOME/comgr"
if [[ ${1:-} == --build ]]; then
    mkdir -p "$TMPDIR" "$AMD_COMGR_CACHE_DIR" build bin reference
    archive="$HOME/gpu-site/scratch-trellis/build-hip/libstrix_trellis_fused_v9.a"
    [[ $(sha256sum "$archive" | cut -d' ' -f1) == 1e111789526692f0b9a3e713fcbf6723a1bec4d959a1a24cb9dde175001b0325 ]]
    cp "$archive" reference/
    hipcc=/opt/rocm/core-10.1/bin/hipcc
    nice -n 10 "$hipcc" -O3 -ffp-contract=off -std=c++17 --offload-arch=gfx1151 -Isrc -c src/gemv_k2.hip -o build/gemv_k2.o
    nice -n 10 ar rcs build/libk2_private.a build/gemv_k2.o
    nice -n 10 "$hipcc" -O3 -ffp-contract=off -std=c++17 --offload-arch=gfx1151 -Isrc tests/k2_core_probe.hip -x none build/libk2_private.a reference/libstrix_trellis_fused_v9.a -o bin/k2_core_probe
    llvm=/opt/rocm/core-10.1/lib/llvm/bin
    nice -n 10 "$llvm/llvm-objcopy" --dump-section .hip_fatbin=build/k2.fatbin build/gemv_k2.o
    nice -n 10 "$llvm/clang-offload-bundler" --unbundle --type=o --input=build/k2.fatbin --targets=hip-amdgcn-amd-amdhsa--gfx1151 --output=build/k2.elf
    nice -n 10 "$llvm/llvm-objdump" -d --mcpu=gfx1151 build/k2.elf > build/k2-isa.txt
    nice -n 10 "$llvm/llvm-readelf" --notes build/k2.elf > build/k2-notes.txt
    nice -n 10 "$hipcc" --version > build/compiler.txt
    sha256sum src/* tests/* tools/* reference/* build/libk2_private.a build/gemv_k2.o build/*txt bin/* data/* > frozen.sha256
    exit
fi
if [[ ${1:-} == --payload ]]; then
    [[ $# == 2 && ( "$2" == search || "$2" == confirm ) ]]
    [[ ! -e "runs/$2.log" ]] || trellis_die 'Never overwrite a completed run'
    mkdir -p runs
    sha256sum -c --quiet frozen.sha256
    { hostname; TZ=:/etc/localtime date -Is; uname -a; cat build/compiler.txt; cat frozen.sha256; } > "runs/$2.meta"
    ./bin/k2_core_probe data/wo_b.stgpu data/q_b.stgpu > "runs/$2.log" 2> "runs/$2.stderr"
    sha256sum -c --quiet frozen.sha256
    cat "runs/$2.log"
    exit
fi
[[ $# == 1 && ( "$1" == --search || "$1" == --confirm ) ]]
exec bash tools/run_gpu_turn.sh --seconds 180 --maintenance-check tools/check_maintenance.sh --queue-arg strix-trellis-k2-core -- "$TRELLIS_SOURCE/tools/run_k2_core.sh" --payload "${1#--}"
