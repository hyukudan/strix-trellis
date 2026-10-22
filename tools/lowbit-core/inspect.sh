#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ "$PWD" == "$HOME/gpu-site/scratch-trellis/lowbit-core" ]]
rocm=/opt/rocm/core-10.1/lib/llvm/bin
# --offloading extracts the images embedded in the actual linked host executable.
"$rocm/llvm-objdump" --offloading build/lowbit_core_probe > build/offloading.txt
for f in build/*.hip-amdgcn-amd-amdhsa--gfx1151; do
    "$rocm/llvm-objdump" -d --mcpu=gfx1151 "$f" > "$f.isa"
    "$rocm/llvm-readobj" --notes "$f" > "$f.notes"
done
python3 tools/lowbit-core/audit.py build/lowbit_core_probe.0.hip-amdgcn-amd-amdhsa--gfx1151.isa \
    build/lowbit_core_probe.0.hip-amdgcn-amd-amdhsa--gfx1151.notes src/gemv_lowbit.hip > build/audit.json
