#!/usr/bin/env bash
# CPU tools only. This script NEVER executes the HIP-linked probe.
set -euxo pipefail
[[ "$PWD" == "$HOME/gpu-site/scratch-trellis/grouped-k3-abi" && ! -e frozen.sha256 ]]
export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" HIP_CACHE_DIR="$PWD/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr" MAX_JOBS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
hipcc=/opt/rocm/core-10.1/bin/hipcc
llvm=/opt/rocm/core-10.1/lib/llvm/bin
src=experiments/grouped-k3-abi
"$hipcc" --version > build/toolchain.txt
"$llvm/clang++" --version >> build/toolchain.txt
g++ --version >> build/toolchain.txt
uname -a >> build/toolchain.txt
sha256sum "$hipcc" "$llvm/clang++" "$llvm/llvm-objdump" "$llvm/llvm-readelf" > build/toolchain.sha256
# The actual HIP host frontend compiles all ABI asserts. No runtime dependency.
"$llvm/clang++" -x hip --cuda-host-only -nogpuinc -nogpulib -std=c++17 -O2 -Wall -Wextra -Werror -c "$src/cpu_test.hip" -o build/cpu_test.o
g++ build/cpu_test.o -o bin/cpu_selftest
ldd bin/cpu_selftest > build/cpu-dependencies.txt
! grep -Ei 'amdhip|hsa-runtime' build/cpu-dependencies.txt
/usr/bin/time -v bin/cpu_selftest > build/cpu-selftest.log 2> build/cpu-selftest.stderr
# Re-run the immutable HIGH C++ proof using its real source, not a replacement.
g++ -std=c++17 -O2 -Wall -Wextra -Werror reports/grouped_k3_guard_test.cpp -o bin/high_guard
bin/high_guard > build/high-guard.log
/usr/bin/time -v "$hipcc" -std=c++17 -O2 -ffp-contract=off --offload-arch=gfx1151 -Wall -Wextra -Werror "$src/pilot.hip" -o bin/abi_pilot 2> build/hip-compile.stderr
"$llvm/llvm-objcopy" --dump-section .hip_fatbin=build/linked.fatbin bin/abi_pilot
python3 -I -S -B "$src/inspect.py" --extract
"$llvm/llvm-readelf" --notes build/linked.elf > build/linked-notes.txt
"$llvm/llvm-objdump" -d --mcpu=gfx1151 build/linked.elf > build/linked-isa.txt
python3 -I -S -B "$src/inspect.py" > build/resources.json
