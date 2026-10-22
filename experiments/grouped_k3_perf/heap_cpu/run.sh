#!/usr/bin/env bash
set -euo pipefail
out=$1
mkdir "$out"
src=experiments/grouped_k3_perf/heap_cpu
c++ -std=c++17 -O0 -Wall -Wextra -Werror -fPIC -shared "$src/heap_dso.cpp" -o "$out/libcpu_heap_probe.so"
c++ -std=c++17 -O0 -Wall -Wextra -Werror -c "$src/heap_static.cpp" -o "$out/static.o"
ar rcs "$out/libcpu_static.a" "$out/static.o"
for mode in global wrap; do
 flags=()
 if [[ $mode == wrap ]]; then
  flags=(-DREVIEW_LINK_WRAP -Wl,--wrap=_Znwm,--wrap=_Znam,--wrap=_ZdlPv,--wrap=_ZdaPv,--wrap=_ZdlPvm,--wrap=_ZdaPvm)
 fi
 c++ -std=c++17 -O0 -Wall -Wextra -Werror "$src/heap_probe.cpp" "${flags[@]}" -L"$out" -lcpu_static -lcpu_heap_probe -Wl,-rpath,'$ORIGIN' -o "$out/$mode"
 readelf -W --dyn-syms --version-info --dynamic "$out/$mode" > "$out/$mode.elf.txt"
 ! grep -Ei 'amdhip|hsa-runtime' "$out/$mode.elf.txt"
 "$out/$mode" | tee "$out/$mode.log"
done
readelf -W --dyn-syms --relocs --version-info "$out/libcpu_heap_probe.so" > "$out/dso.elf.txt"
c++ -std=c++17 -O2 -Wall -Wextra -Werror "$src/abi_pool_test.cpp" -o "$out/abi"
"$out/abi" > "$out/abi.log"
c++ -std=c++17 -O0 -Wall -Wextra -Werror "$src/forward.cpp" -Wl,--wrap=_Znwm,--wrap=_Znam,--wrap=_ZdlPv,--wrap=_ZdaPv,--wrap=_ZdlPvm,--wrap=_ZdaPvm -o "$out/forward"
ldd "$out/forward" > "$out/forward.dependencies.txt"
! grep -Ei 'amdhip|hsa-runtime' "$out/forward.dependencies.txt"
"$out/forward" > "$out/forward.log"
