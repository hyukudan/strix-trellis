#!/usr/bin/env bash
# Local WSL CPU-only builds. No network, HIP/HSA headers or remote writes.
set -euo pipefail
cd "$(dirname "$0")/../.."
out=${1:-build-grouped-k3-perf-review}
[[ "$out" =~ ^build-grouped-k3-perf-review(-[0-9]+)?$ ]]
mkdir "$out"
c++ --version > "$out/compiler.txt"
c++ -std=c++17 -O0 -Wall -Wextra -Werror -fPIC -shared tools/grouped_k3_perf_review/heap_dso.cpp -o "$out/libcpu_heap_probe.so"
c++ -std=c++17 -O0 -Wall -Wextra -Werror -c tools/grouped_k3_perf_review/heap_static.cpp -o "$out/heap_static.o"
ar rcs "$out/libcpu_static.a" "$out/heap_static.o"
c++ -std=c++17 -O0 -Wall -Wextra -Werror tools/grouped_k3_perf_review/heap_probe.cpp -L"$out" -lcpu_static -lcpu_heap_probe -Wl,-rpath,'$ORIGIN' -o "$out/probe_global"
c++ -std=c++17 -O0 -Wall -Wextra -Werror -DREVIEW_LINK_WRAP tools/grouped_k3_perf_review/heap_probe.cpp tools/grouped_k3_perf_review/heap_wrap.cpp -L"$out" -lcpu_static -lcpu_heap_probe -Wl,-rpath,'$ORIGIN',--wrap=_Znwm,--wrap=_Znam,--wrap=_ZdlPv,--wrap=_ZdaPv,--wrap=_ZdlPvm,--wrap=_ZdaPvm -o "$out/probe_wrap"
c++ -std=c++17 -O2 -Wall -Wextra -Werror tools/grouped_k3_perf_review/abi_pool_test.cpp -o "$out/abi_pool_test"
for name in libcpu_heap_probe.so probe_global probe_wrap abi_pool_test; do
    readelf -W --dynamic --dyn-syms --version-info "$out/$name" > "$out/$name.elf.txt"
    if readelf -W --dynamic "$out/$name" | grep -Ei 'amdhip|hsa-runtime'; then exit 4; fi
done
"$out/probe_global"
"$out/probe_wrap"
"$out/abi_pool_test"
