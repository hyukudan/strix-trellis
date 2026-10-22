#!/usr/bin/env bash
# CPU compilation/inspection only. NEVER execute any HIP-linked binary.
set -euxo pipefail
[[ "$PWD" == "$HOME/gpu-site/scratch-trellis/grouped-k3-math-r1" && ! -e frozen.sha256 ]]
[[ $(ps -o ni= -p $$) -ge 10 ]]
export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" HIP_CACHE_DIR="$PWD/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr"
export MAX_JOBS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
hipcc=/opt/rocm/core-10.1/bin/hipcc
llvm=/opt/rocm/core-10.1/lib/llvm/bin
src=experiments/grouped_k3_math
exec > >(tee build/build.log) 2>&1
hostname > build/host.txt
"$hipcc" --version > build/toolchain.txt
"$llvm/clang++" --version >> build/toolchain.txt
g++ --version >> build/toolchain.txt
uname -a >> build/toolchain.txt
sha256sum "$hipcc" "$llvm/clang++" "$llvm/llvm-objdump" "$llvm/llvm-readelf" > build/toolchain.sha256
echo '0b1453a3c9c25e3f7e4f12251d310a3a9f2c3b53c76d78c314e9624ba763e3b3  lib/libstrix_trellis.a' | sha256sum -c -
python3 -B tools/verify_grouped_k3_adapter.py --candidate "$src/kernels.hip" --helper-dir src > build/source-contract.log
python3 -B "$src/extract.py" --check >> build/source-contract.log
g++ -std=c++17 -O2 -Wall -Wextra -Werror reports/grouped_k3_guard_test.cpp -o bin/high_guard
bin/high_guard > build/high-guard.log
"$llvm/clang++" -x hip --cuda-host-only -nogpuinc -nogpulib -std=c++17 -O2 -ffp-contract=off -Wall -Wextra -Werror -c "$src/cpu_test.cpp" -o build/cpu_test.o
g++ build/cpu_test.o -o bin/math_cpu_test
ldd bin/math_cpu_test > build/cpu-dependencies.txt
! grep -Ei 'amdhip|hsa-runtime' build/cpu-dependencies.txt
bin/math_cpu_test > build/math-cpu.log
g++ -std=c++17 -O2 -Wall -Wextra -Werror "$src/attribute_test.cpp" -o bin/attribute_cpu_test
ldd bin/attribute_cpu_test > build/attribute-cpu-dependencies.txt
! grep -Ei 'amdhip|hsa-runtime' build/attribute-cpu-dependencies.txt
bin/attribute_cpu_test > build/attribute-cpu.log
# Effective dependency closure and preprocessed source use exactly the build flags.
flags=(-std=c++17 -O2 -ffp-contract=off --offload-arch=gfx1151 -Iinclude -Isrc -Wall -Wextra -Werror)
"$hipcc" "${flags[@]}" -M "$src/native_qa.hip" > build/effective-headers.d
"$hipcc" "${flags[@]}" -E "$src/native_qa.hip" > build/effective-source.ii
/usr/bin/time -v "$hipcc" "${flags[@]}" "$src/native_qa.hip" -Wl,--wrap=hipFuncGetAttributes,-Map=build/link.map,--cref,--emit-relocs -Llib -lstrix_trellis -o bin/math_native_qa 2> build/hip-compile.stderr
"$llvm/llvm-objcopy" --dump-section .hip_fatbin=build/linked.fatbin bin/math_native_qa
python3 -I -S -B "$src/inspect_linked.py" --extract
for image in build/linked-*.elf; do
    "$llvm/llvm-readelf" --notes "$image" > "${image%.elf}.notes"
    "$llvm/llvm-objdump" -d --mcpu=gfx1151 "$image" > "${image%.elf}.isa"
done
python3 -I -S -B "$src/inspect_linked.py" > build/resources.json
python3 -I -S -B "$src/audit_build.py" > build/audit.log
# Preserve actual archive caller relocations and final host link bindings.
"$llvm/llvm-objdump" -r lib/libstrix_trellis.a > build/archive-relocations.txt
"$llvm/llvm-objdump" -dr --demangle bin/math_native_qa > build/host-disassembly.txt
"$llvm/llvm-readelf" --symbols --relocations --dynamic bin/math_native_qa > build/host-symbols-relocations.txt
"$llvm/llvm-readelf" --dyn-syms /opt/rocm/core-10.1/lib/libamdhip64.so > build/real-HIP-exports.txt
sha256sum /opt/rocm/core-10.1/lib/libamdhip64.so > build/real-HIP-library.sha256
python3 -I -S -B "$src/audit_attribute_link.py" > build/attribute-link.json
