#!/usr/bin/env bash
# Execute on the Linux build host; this script does not connect remotely.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scratch_paths.sh"
trellis_require_own_source
if (( $# != 0 )); then
    echo "Usage: $0 (no arguments; fixed source/build paths)" >&2
    exit 2
fi
rocm=/opt/rocm/core-10.1
[[ -x "$rocm/lib/llvm/bin/clang++" ]] || trellis_die "Missing $rocm/llvm/bin/clang++"
build_dir="$TRELLIS_SOURCE/build-hip"
trellis_require_owned "$build_dir"
export TMPDIR="$TRELLIS_SOURCE/.tmp"
export XDG_CACHE_HOME="$TRELLIS_SOURCE/.cache"
export AMD_COMGR_CACHE_DIR="$XDG_CACHE_HOME/comgr"
trellis_require_owned "$TMPDIR"
trellis_require_owned "$XDG_CACHE_HOME"
trellis_require_owned "$AMD_COMGR_CACHE_DIR"
mkdir -p -- "$TMPDIR" "$AMD_COMGR_CACHE_DIR"
# No install, cache deletion, GPU probing or program execution.
nice -n 10 cmake -S "$TRELLIS_SOURCE" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release -DSTRIX_TRELLIS_ENABLE_HIP=ON \
    -DSTRIX_TRELLIS_BUILD_TESTING=ON \
    -DCMAKE_HIP_COMPILER="$rocm/lib/llvm/bin/clang++" \
    -DCMAKE_HIP_COMPILER_ROCM_ROOT="$rocm" \
    -DCMAKE_PREFIX_PATH="$rocm" -DCMAKE_HIP_ARCHITECTURES=gfx1151
nice -n 10 cmake --build "$build_dir" --parallel 2
