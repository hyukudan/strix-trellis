#!/usr/bin/env bash
set -euo pipefail
own="$HOME/gpu-site/scratch-trellis/fused-both-core"
cd "$own"
export TMPDIR="$own/.tmp" TMP="$own/.tmp" TEMP="$own/.tmp"
export XDG_CACHE_HOME="$own/.cache" HIP_CACHE_DIR="$own/.cache/hip"
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
    nice -n 10 python3 -I -S -B reports/fused-both-review-proof.py > build/cpu-normative-linux.txt 2>&1
    python3 -I -S -B experiments/fused_both/source_proof.py > build/source-proof.json
    ;;
 --build)
    lib="$HOME/gpu-site/scratch-trellis/runtime-v12/build-hip/libstrix_trellis.a"
    echo "97b4af73c40ffd86a7656ad864eac1c485b3004f60ce457c5019a8340d5224cd  $lib" | sha256sum -c -
    cmp src/backend.hpp "$HOME/gpu-site/scratch-trellis/runtime-v12/src/backend.hpp"
    nice -n 10 /opt/rocm/core-10.1/bin/hipcc -O3 -ffp-contract=off -std=c++17 --offload-arch=gfx1151 -Isrc -Iexperiments/fused_both experiments/fused_both/kernel.hip experiments/fused_both/probe.hip -x none "$lib" -o bin/probe
    /opt/rocm/core-10.1/llvm/bin/llvm-objdump --offloading bin/probe > build/offloading.txt
    /opt/rocm/core-10.1/llvm/bin/llvm-readobj --notes bin/probe.0.hip-amdgcn-amd-amdhsa--gfx1151 > build/linked-notes.txt
    /opt/rocm/core-10.1/llvm/bin/llvm-objdump -d --mcpu=gfx1151 bin/probe.0.hip-amdgcn-amd-amdhsa--gfx1151 > build/linked-isa.txt
    /opt/rocm/core-10.1/bin/hipcc --version > build/toolchain.txt
    ;;
 --fixtures) nice -n 10 python3 -I -S -B experiments/fused_both/prepare.py ;;
 --freeze)
    sha256sum src/* experiments/fused_both/* bin/probe bin/probe.* build/linked* build/offloading.txt build/toolchain.txt build/source-proof.json build/fixtures.sha256 "$HOME/gpu-site/scratch-trellis/runtime-v12/build-hip/libstrix_trellis.a" > build/frozen.sha256
    ;;
 --payload)
    [[ ${2:-} == search || ${2:-} == confirm ]]
    check
    [[ ! -e runs/$2 ]]
    if [[ ${FB_INTERNAL:-0} != 1 ]];then
        exec timeout --signal=TERM --kill-after=3s 145s env FB_INTERNAL=1 bash "$0" --payload "$2"
    fi
    mkdir -p "runs/$2"
    { hostname;date -Is;uname -a;cat build/toolchain.txt; } > "runs/$2/meta.txt"
    sha256sum -c build/frozen.sha256 > "runs/$2/frozen-before.txt"
    sha256sum -c build/fixtures.sha256 > "runs/$2/fixtures-before.txt"
    /usr/bin/time -v ./bin/probe fixtures > "runs/$2/probe.log" 2> "runs/$2/probe.stderr"
    sha256sum -c build/frozen.sha256 > "runs/$2/frozen-after.txt"
    sha256sum -c build/fixtures.sha256 > "runs/$2/fixtures-after.txt"
    date -Is >> "runs/$2/meta.txt"
    ;;
 search|confirm)
    check
    sha256sum -c build/frozen.sha256 >/dev/null
    sha256sum -c build/fixtures.sha256 >/dev/null
    exec timeout --signal=TERM --kill-after=5s 1195s "$HOME/gpu-site/gpu-turn-any.sh" strix-trellis-fused-both-core timeout --signal=TERM --kill-after=3s 180s bash "$own/experiments/fused_both/run.sh" --payload "$1"
    ;;
 *) exit 2;;
esac
