#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/../.."
[[ "$PWD" == "$HOME/gpu-site/scratch-trellis/k2-rows-core" ]]
export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr" MAX_JOBS=1
mkdir -p "$TMPDIR" "$AMD_COMGR_CACHE_DIR" build bin reference
control="$HOME/gpu-site/scratch-trellis/runtime-v11/build-hip/libstrix_trellis.a"
[[ $(sha256sum "$control" | cut -d' ' -f1) == 201d596e469c3d81a91aaeffa4588801fad5ed2d3b8d8cc1279458b78f6b0597 ]]
cmp src/backend.hpp ../runtime-v11/src/backend.hpp
cp -n "$control" reference/libstrix_trellis_v11.a
hipcc=/opt/rocm/core-10.1/bin/hipcc
llvm=/opt/rocm/core-10.1/lib/llvm/bin
nice -n 10 "$hipcc" -O3 -ffp-contract=off -std=c++17 --offload-arch=gfx1151 -Isrc -c src/gemv_k2_rows.hip -o build/gemv_k2_rows.o
nice -n 10 ar rcs build/libk2_rows_private.a build/gemv_k2_rows.o
nice -n 10 "$hipcc" -O3 -ffp-contract=off -std=c++17 --offload-arch=gfx1151 -Isrc bench/k2_rows_probe.hip -x none build/libk2_rows_private.a reference/libstrix_trellis_v11.a -o bin/k2_rows_probe
nice -n 10 "$llvm/llvm-objcopy" --dump-section .hip_fatbin=build/rows.fatbin build/gemv_k2_rows.o
nice -n 10 "$llvm/clang-offload-bundler" --unbundle --type=o --input=build/rows.fatbin --targets=hip-amdgcn-amd-amdhsa--gfx1151 --output=build/rows.elf
nice -n 10 "$llvm/llvm-objdump" -d --mcpu=gfx1151 build/rows.elf > build/rows-isa.txt
nice -n 10 "$llvm/llvm-readelf" --notes build/rows.elf > build/rows-notes.txt
nice -n 10 "$llvm/llvm-objcopy" --dump-section .hip_fatbin=build/exe.fatbin bin/k2_rows_probe
# Archive extraction stays in our private build directory.
(cd build; ar x ../reference/libstrix_trellis_v11.a gemv_lds.hip.o)
nice -n 10 "$llvm/llvm-objcopy" --dump-section .hip_fatbin=build/control.fatbin build/gemv_lds.hip.o
nice -n 10 "$llvm/clang-offload-bundler" --unbundle --type=o --input=build/control.fatbin --targets=hip-amdgcn-amd-amdhsa--gfx1151 --output=build/control.elf
nice -n 10 "$llvm/llvm-objdump" -d --mcpu=gfx1151 build/control.elf > build/control-isa.txt
nice -n 10 "$llvm/llvm-readelf" --notes build/control.elf > build/control-notes.txt
nice -n 10 "$hipcc" --version > build/compiler.txt
python3 -B tools/k2-rows-core/audit.py > build/audit.json
