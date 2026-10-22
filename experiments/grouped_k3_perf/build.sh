#!/usr/bin/env bash
# Compilation and CPU execution only. NEVER run perf_native, including --help.
set -euo pipefail
[[ "$PWD" == "$HOME/gpu-site/scratch-trellis/grouped-k3-perf-r1/runs/host-r2" && ! -e frozen.sha256 ]]
[[ $(ps -o ni= -p $$) -ge 10 ]]
export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" HIP_CACHE_DIR="$PWD/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr"
export MAX_JOBS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec > >(tee build/build.log) 2>&1
set -x
hipcc=/opt/rocm/core-10.1/bin/hipcc
llvm=/opt/rocm/core-10.1/lib/llvm/bin
src=experiments/grouped_k3_perf
hostname > build/host.txt
"$hipcc" --version > build/toolchain.txt
"$llvm/clang++" --version >> build/toolchain.txt
g++ --version >> build/toolchain.txt
uname -a >> build/toolchain.txt
sha256sum "$hipcc" "$llvm/clang++" "$llvm/llvm-objdump" "$llvm/llvm-readelf" > build/toolchain.sha256
echo '0b1453a3c9c25e3f7e4f12251d310a3a9f2c3b53c76d78c314e9624ba763e3b3  lib/libstrix_trellis.a' | sha256sum -c -
python3 -B tools/verify_grouped_k3_adapter.py --candidate experiments/grouped_k3_math/kernels.hip --helper-dir src > build/source-contract.log
python3 -B experiments/grouped_k3_math/extract.py --check >> build/source-contract.log
heap_wraps=(-Wl,--wrap=_Znwm,--wrap=_Znam,--wrap=_ZdlPv,--wrap=_ZdaPv,--wrap=_ZdlPvm,--wrap=_ZdaPvm)
g++ -std=c++17 -O2 -Wall -Wextra -Werror "${heap_wraps[@]}" "$src/cpu_test.cpp" -o bin/perf_cpu_test
g++ -std=c++17 -O2 -Wall -Wextra -Werror experiments/grouped_k3_math/attribute_test.cpp -o bin/attribute_cpu_test
g++ -std=c++17 -O2 -Wall -Wextra -Werror reports/grouped_k3_guard_test.cpp -o bin/high_guard
g++ -std=c++17 -O2 -Wall -Wextra -Werror "$src/submit_hooks_test.cpp" -o bin/submit_hooks_test
for exe in perf_cpu_test attribute_cpu_test high_guard submit_hooks_test;do
    ldd "bin/$exe" > "build/$exe.dependencies.txt"
    ! grep -Ei 'amdhip|hsa-runtime' "build/$exe.dependencies.txt"
    "bin/$exe" > "build/$exe.log"
done
wraps=(-Wl,--wrap=hipMalloc,--wrap=hipFree,--wrap=hipMemcpy,--wrap=hipMemcpyAsync,--wrap=hipMemsetAsync,--wrap=hipStreamSynchronize,--wrap=hipPointerGetAttributes,--wrap=hipMemGetAddressRange,--wrap=hipStreamGetDevice,--wrap=hipStreamGetFlags,--wrap=hipGetDevice,--wrap=hipEventRecord,--wrap=hipEventElapsedTime,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=_Znwm,--wrap=_Znam,--wrap=_ZdlPv,--wrap=_ZdaPv,--wrap=_ZdlPvm,--wrap=_ZdaPvm)
flags=(-std=c++17 -O2 -ffp-contract=off --offload-arch=gfx1151 -Iinclude -Isrc -Wall -Wextra -Werror)
"$hipcc" "${flags[@]}" -M "$src/native.hip" > build/effective-headers.d
"$hipcc" "${flags[@]}" -E "$src/native.hip" > build/effective-source.ii
/usr/bin/time -v "$hipcc" "${flags[@]}" "$src/native.hip" "${wraps[@]}" -Wl,--wrap=hipFuncGetAttributes,-Map=build/link.map,--cref,--emit-relocs -Llib -lstrix_trellis -o bin/perf_native 2> build/hip-compile.stderr
"$llvm/llvm-objcopy" --dump-section .hip_fatbin=build/linked.fatbin bin/perf_native
python3 -I -S -B "$src/inspect_linked.py" --extract
for image in build/linked-*.elf;do
    "$llvm/llvm-readelf" --notes "$image" > "${image%.elf}.notes"
    "$llvm/llvm-objdump" -d --mcpu=gfx1151 "$image" > "${image%.elf}.isa"
done
python3 -I -S -B "$src/inspect_linked.py" > build/resources.json
python3 -I -S -B experiments/grouped_k3_math/audit_build.py > build/source-isa-audit.log
python3 -I -S -B "$src/identity.py" > build/identity.json
python3 -I -S -B "$src/cpu_audit.py" > build/cpu-audit.log
"$llvm/llvm-objdump" -r lib/libstrix_trellis.a > build/archive-relocations.txt
"$llvm/llvm-objdump" -dr --demangle bin/perf_native > build/host-disassembly.txt
"$llvm/llvm-readelf" --symbols --relocations --dynamic bin/perf_native > build/host-symbols-relocations.txt
"$llvm/llvm-readelf" --dyn-syms /opt/rocm/core-10.1/lib/libamdhip64.so > build/real-HIP-exports.txt
sha256sum /opt/rocm/core-10.1/lib/libamdhip64.so > build/real-HIP-library.sha256
python3 -I -S -B "$src/audit_attribute_link.py" > build/attribute-link.json
python3 -I -S -B "$src/audit_baseline_binding.py" > build/baseline-binding.json

python3 -I -S -B "$src/audit_heap_link.py" > build/heap-link.json
nice -n 10 bash "$src/heap_cpu/run.sh" build/heap-cpu
python3 -I -S -B "$src/provenance_test.py" > build/provenance-test.log
python3 -I -S -B "$src/raw_fixture_test.py" > build/raw-fixture-test.log
