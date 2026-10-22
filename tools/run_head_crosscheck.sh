#!/usr/bin/env bash
# One queued numerical run against the immutable, qualified v7 library.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/head-crosscheck" ]] || trellis_die 'Expected head-crosscheck subtree'
root="$HOME/gpu-site/scratch-trellis"
library="$root/build-hip/libstrix_trellis_fused_v7.a"
data="$root/gpu-head-eaq-ref-v2"
expected_library=c6b49730b358ec04e4ce6cd08c9a4ff4a67ab31ee393c16cb602c4a689ca056e
export TMPDIR="$TRELLIS_SOURCE/.tmp" XDG_CACHE_HOME="$TRELLIS_SOURCE/.cache"
export AMD_COMGR_CACHE_DIR="$XDG_CACHE_HOME/comgr" MAX_JOBS=2
if [[ ${1:-} == --build ]]; then
    [[ $# == 1 && ! -e build/frozen.sha256 && ! -e head_crosscheck ]] || trellis_die 'Build requires a fresh, unfrozen destination'
    [[ $(sha256sum "$library" | cut -d' ' -f1) == "$expected_library" ]] || trellis_die 'Qualified library mismatch'
    mkdir -p "$TMPDIR" "$AMD_COMGR_CACHE_DIR" build
    sed '/^int main(/,$d' bench/head_linear_bench.hip > build/head_crosscheck_helpers.hpp
    nice -n 10 /opt/rocm/core-10.1/bin/hipcc -O3 -ffp-contract=off -std=c++17 \
      --offload-arch=gfx1151 -Iinclude -Ibuild bench/head_crosscheck.hip \
      -L"$root/build-hip" -l:libstrix_trellis_fused_v7.a -o head_crosscheck
    sha256sum head_crosscheck build/head_crosscheck_helpers.hpp bench/head_crosscheck.hip bench/head_linear_bench.hip \
      tests/gpu_runtime.hip include/strix_trellis.h src/*.hpp tools/*.sh "$library" > build/frozen.sha256
    /opt/rocm/core-10.1/bin/hipcc --version > build/toolchain.txt
    exit
fi
if [[ ${1:-} == --payload ]]; then
    [[ $# == 2 && $2 =~ ^run-[a-zA-Z0-9-]+$ && ! -e $2 ]] || trellis_die 'New run directory required'
    hostname; TZ=:/etc/localtime date -Is
    cat build/toolchain.txt
    sha256sum -c build/frozen.sha256
    (cd "$data" && sha256sum -c manifest.sha256)
    sha256sum "$data/"{head.blob,head.meta,input.f32,expected.f32,manifest.json,manifest.sha256}
    start=$(date +%s%N)
    "$TRELLIS_SOURCE/head_crosscheck" "$data" "$TRELLIS_SOURCE/$2"
    stop=$(date +%s%N)
    echo "PAYLOAD_EXEC_WALL_NS=$((stop-start))"
    sha256sum "$2"/*
    TZ=:/etc/localtime date -Is
    exit
fi
[[ $# == 1 && $1 =~ ^run-[a-zA-Z0-9-]+$ && ! -e $1 ]] || trellis_die 'Usage: run_head_crosscheck.sh NEW-run-name'
sha256sum -c build/frozen.sha256
exec bash tools/run_gpu_turn.sh --seconds 180 --maintenance-check tools/check_maintenance.sh \
  --queue-arg strix-trellis-head-crosscheck -- "$TRELLIS_SOURCE/tools/run_head_crosscheck.sh" --payload "$1"
