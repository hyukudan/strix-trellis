#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ "$PWD" == "$HOME/gpu-site/scratch-trellis/lowbit-core" ]]
export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache"
export AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr"
mkdir -p "$TMPDIR" "$AMD_COMGR_CACHE_DIR" build
rocm=/opt/rocm/core-10.1
archive="$HOME/gpu-site/scratch-trellis/build-hip/libstrix_trellis_fused_v9.a"
echo '1e111789526692f0b9a3e713fcbf6723a1bec4d959a1a24cb9dde175001b0325  '"$archive" | sha256sum -c -
cp -n "$archive" build/v9.a
nice -n 10 "$rocm/lib/llvm/bin/clang++" -x hip -O3 -ffp-contract=off -std=c++17 \
  --offload-arch=gfx1151 --rocm-path="$rocm" -Isrc \
  src/gemv_lowbit.hip bench/lowbit_core_probe.hip -x none build/v9.a \
  -L"$rocm/lib" -Wl,-rpath,"$rocm/lib" -lamdhip64 -o build/lowbit_core_probe
{ "$rocm/lib/llvm/bin/clang++" --version; uname -a; sha256sum "$rocm/lib/llvm/bin/clang++"; } > build/toolchain.txt
"$rocm/lib/llvm/bin/llvm-objdump" --offloading build/lowbit_core_probe > build/offloading.txt
