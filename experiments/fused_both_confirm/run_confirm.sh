#!/usr/bin/env bash
set -euo pipefail
own="$HOME/gpu-site/scratch-trellis/fused-both-confirm"
cd "$own"
export TMPDIR="$own/.tmp" TMP="$own/.tmp" TEMP="$own/.tmp"
export XDG_CACHE_HOME="$own/.cache" HIP_CACHE_DIR="$own/.cache/hip"
export LC_ALL=C
export AMD_COMGR_CACHE_DIR="$own/.cache/comgr" MAX_JOBS=1
mkdir -p "$TMPDIR" "$HIP_CACHE_DIR" "$AMD_COMGR_CACHE_DIR" build bin
check() {
    local hour minute now
    now=$(date +%s);hour=$(date +%H);minute=$(date +%M)
    minute=$((10#$hour*60+10#$minute))
    ((minute<1260 || minute>=1410)) || { echo 'VETO local 21:00-23:30';return 1; }
    if ((minute<1260));then (( $(date -d 'today 21:00' +%s)-now>200 ));fi
    [[ ! -e "$HOME/gpu-site/.gpu-turn/maintenance" ]]
}
case ${1:-} in
 --cpu)
    nice -n 10 python3 -I -S -B "$HOME/gpu-site/scratch-trellis/fused-both-core/reports/fused-both-review-proof.py" > build/cpu-normative-linux.txt 2>&1
    python3 -I -S -B experiments/fused_both/source_proof.py > build/source-proof.json
    ;;
 --build)
    lib="$HOME/gpu-site/scratch-trellis/runtime-v12/build-hip/libstrix_trellis.a"
    echo "97b4af73c40ffd86a7656ad864eac1c485b3004f60ce457c5019a8340d5224cd  $lib" | sha256sum -c -
    cmp src/backend.hpp "$HOME/gpu-site/scratch-trellis/runtime-v12/src/backend.hpp"
    nice -n 10 python3 experiments/fused_both_confirm/extract_core_confirm.py
    nice -n 10 /opt/rocm/core-10.1/bin/hipcc -O3 -ffp-contract=off -std=c++17 --offload-arch=gfx1151 --cuda-host-only -Xclang -fcuda-include-gpubinary -Xclang build/original-core.hipfb -Isrc -Iexperiments/fused_both -c experiments/fused_both/kernel.hip -o build/core_host_confirm.o
    nice -n 10 /opt/rocm/core-10.1/bin/hipcc -O3 -ffp-contract=off -std=c++17 --offload-arch=gfx1151 --cuda-host-only -Isrc -Iexperiments/fused_both -c experiments/fused_both_confirm/probe_confirm.hip -o build/probe_host_confirm.o
    nice -n 10 /opt/rocm/core-10.1/bin/hipcc --offload-arch=gfx1151 build/core_host_confirm.o build/probe_host_confirm.o "$lib" -o bin/probe_confirm
    /opt/rocm/core-10.1/llvm/bin/llvm-objdump --offloading bin/probe_confirm > build/offloading.txt
    /opt/rocm/core-10.1/llvm/bin/llvm-readobj --notes bin/probe_confirm.0.hip-amdgcn-amd-amdhsa--gfx1151 > build/linked-notes.txt
    /opt/rocm/core-10.1/llvm/bin/llvm-objdump -d --mcpu=gfx1151 bin/probe_confirm.0.hip-amdgcn-amd-amdhsa--gfx1151 > build/linked-isa.txt
    /opt/rocm/core-10.1/bin/hipcc --version > build/toolchain.txt
    python3 experiments/fused_both_confirm/identity_confirm.py > build/identity.json
    python3 experiments/fused_both/audit.py build > build/resources.json
    ;;
 --fixtures)
    [[ ! -e fixtures ]]
    cp -a "$HOME/gpu-site/scratch-trellis/fused-both-core/fixtures" fixtures
    cp references/r8-y-fp64-core.f32 fixtures/head/
    nice -n 10 python3 experiments/fused_both_confirm/cpu_references.py > build/cpu-references.jsonl
    find fixtures -type f -print0 | sort -z | xargs -0 sha256sum > build/fixtures.sha256
    ;;
 --freeze)
    sha256sum src/* experiments/fused_both/* experiments/fused_both_confirm/* bin/probe_confirm bin/probe_confirm.* build/linked* build/offloading.txt build/toolchain.txt build/source-proof.json build/identity.json build/resources.json build/host-harness.json build/host-guard.json build/cpu-validation.json build/incident-receipt.json build/cpu-references.jsonl build/cpu-normative.txt build/fixtures.sha256 build/original-core.hipfb references/* "$HOME/gpu-site/scratch-trellis/runtime-v12/build-hip/libstrix_trellis.a" > build/frozen.sha256
    ;;
 --payload)
    [[ ${2:-} == search || ${2:-} == confirm ]]
    check
    [[ ! -e runs/$2 ]]
    if [[ ${FB_INTERNAL:-0} != 1 ]];then
        exec timeout --signal=TERM --kill-after=3s 145s env FB_INTERNAL=1 bash "$0" --payload "$2"
    fi
    export TMPDIR="$own/.tmp" TMP="$own/.tmp" TEMP="$own/.tmp"
    export XDG_CACHE_HOME="$own/.cache" HIP_CACHE_DIR="$own/.cache/hip" AMD_COMGR_CACHE_DIR="$own/.cache/comgr"
    mkdir -p "runs/$2"
    printf '%s\n' "$$" > "runs/$2/payload.pid"
    date -Is > "runs/$2/physical.started"
    { hostname;date -Is;uname -a;cat build/toolchain.txt; } > "runs/$2/meta.txt"
    sha256sum -c build/frozen.sha256 > "runs/$2/frozen-before.txt"
    sha256sum -c build/fixtures.sha256 > "runs/$2/fixtures-before.txt"
    nice -n 10 /usr/bin/time -v ./bin/probe_confirm fixtures > "runs/$2/probe.log" 2> "runs/$2/probe.stderr"
    sha256sum -c build/frozen.sha256 > "runs/$2/frozen-after.txt"
    sha256sum -c build/fixtures.sha256 > "runs/$2/fixtures-after.txt"
    date -Is >> "runs/$2/meta.txt"
    ;;
 search|confirm)
    check
    sha256sum -c build/frozen.sha256 >/dev/null
    sha256sum -c build/fixtures.sha256 >/dev/null
    export GPU_TURN_MAX=180
    exec timeout --signal=TERM --kill-after=5s 1195s "$HOME/gpu-site/gpu-turn-any.sh" strix-trellis-fused-both-confirm-$1 timeout --signal=TERM --kill-after=3s 180s nice -n 10 env TMPDIR="$own/.tmp" TMP="$own/.tmp" TEMP="$own/.tmp" XDG_CACHE_HOME="$own/.cache" HIP_CACHE_DIR="$own/.cache/hip" AMD_COMGR_CACHE_DIR="$own/.cache/comgr" MAX_JOBS=1 bash "$own/experiments/fused_both_confirm/run_confirm.sh" --payload "$1"
    ;;
 *) exit 2;;
esac
