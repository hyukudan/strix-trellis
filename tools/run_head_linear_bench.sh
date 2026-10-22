#!/usr/bin/env bash
# CPU build or queued synthetic-head operator validation; never direct GPU entry.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/head-linear" ]] || trellis_die 'Expected head-linear subtree'
root="$HOME/gpu-site/scratch-trellis"
library="$root/build-hip/libstrix_trellis_fused_v7.a"
data="$root/gpu-head-ref-v1"
expected_library=c6b49730b358ec04e4ce6cd08c9a4ff4a67ab31ee393c16cb602c4a689ca056e
if [[ ${1:-} == --build ]]; then
    [[ $# == 1 ]] || trellis_die 'No extra build arguments'
    [[ $(sha256sum "$library" | cut -d' ' -f1) == "$expected_library" ]] || trellis_die 'Qualified v7 library mismatch'
    export TMPDIR="$TRELLIS_SOURCE/.tmp" XDG_CACHE_HOME="$TRELLIS_SOURCE/.cache"
    export AMD_COMGR_CACHE_DIR="$XDG_CACHE_HOME/comgr"
    trellis_require_owned "$TMPDIR"; trellis_require_owned "$AMD_COMGR_CACHE_DIR"
    mkdir -p "$TMPDIR" "$AMD_COMGR_CACHE_DIR" build
    nice -n 10 /opt/rocm/core-10.1/bin/hipcc -O3 -ffp-contract=off -std=c++17 \
      --offload-arch=gfx1151 -Iinclude bench/head_linear_bench.hip -L"$root/build-hip" -l:libstrix_trellis_fused_v7.a -o head_linear_bench_v1
    sha256sum head_linear_bench_v1 bench/head_linear_bench.hip tests/gpu_runtime.hip include/strix_trellis.h src/*.hpp "$library" > build/compiled.sha256
    exit
fi
if [[ ${1:-} == --payload ]]; then
    [[ $# == 1 ]] || trellis_die 'No extra payload arguments'
    hostname; TZ=:/etc/localtime date -Is
    /opt/rocm/core-10.1/bin/hipcc --version
    sha256sum -c build/compiled.sha256
    (cd "$data" && sha256sum -c manifest.sha256)
    sha256sum "$data/manifest.json" "$data/manifest.sha256"
    exec "$TRELLIS_SOURCE/head_linear_bench_v1" "$data"
fi
[[ $# == 0 ]] || trellis_die 'No arguments expected'
[[ -x head_linear_bench_v1 && -s build/compiled.sha256 && -s "$data/manifest.sha256" ]] || trellis_die 'Missing prepared binary or data manifest'
[[ $(stat -c%s "$data/head.blob") == 496704000 ]] || trellis_die 'Missing complete head blob'
exec bash tools/run_gpu_turn.sh --seconds 180 --maintenance-check tools/check_maintenance.sh \
  --queue-arg strix-trellis-head-linear -- "$TRELLIS_SOURCE/tools/run_head_linear_bench.sh" --payload
