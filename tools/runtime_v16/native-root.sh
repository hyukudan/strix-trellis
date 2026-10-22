#!/usr/bin/env bash
# Canonical ROOT-only V16 aggregate; CPU preparation never queues this payload.
set -euo pipefail
[[ ${ROOT_GPU_LEASE:-0} == 1 && $# == 1 ]]
[[ "$PWD" == "$HOME/gpu-site/scratch-trellis/runtime-v16-qa" ]]
[[ $(dirname -- "$1") == "$PWD" && $(basename -- "$1") =~ ^root-v16-native-[a-zA-Z0-9_-]+$ && ! -e "$1" ]]
root="$HOME/gpu-site/scratch-trellis"
export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache"
export HIP_CACHE_DIR="$XDG_CACHE_HOME/hip" AMD_COMGR_CACHE_DIR="$XDG_CACHE_HOME/comgr"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
verify_all(){
    sha256sum -c frozen.sha256 >/dev/null
    (cd "$root/runtime-v16" && sha256sum -c frozen.sha256 >/dev/null)
    (cd "$root/engine-v16" && sha256sum -c frozen.sha256 >/dev/null)
    sha256sum -c preserved-inputs.sha256 >/dev/null
}
verify_all
mkdir -- "$1"
# Aggregate hard bound, not a measured duration; ROOT alone supplies the lease.
/usr/bin/time -v timeout --signal=TERM --kill-after=3s 600s bash -c '
set -euo pipefail
./v14/runtime_v16_v14b_qa "$HOME/gpu-site/scratch-trellis" > "$1/v14b-native.log" 2> "$1/v14b-native.stderr"
grep -Fx "PASS native_runtime_v14 attributes=5 K2_all_partials=1 WMMA_tuples=16 eligible=13 fallback=3" "$1/v14b-native.log"
cd direct
./build/v14b_control "$PWD/inputs" "$1/v14b" > "$1/control.log" 2> "$1/control.stderr"
./build/v14b_q1_control "$PWD/inputs" "$1/v14b" > "$1/control-q1.log" 2> "$1/control-q1.stderr"
./build/runtime_v16_complete "$PWD/inputs" "$1/v14b" > "$1/direct-native.log" 2> "$1/direct-native.stderr"
grep -Fx "PASS prepared V16 native contract" "$1/direct-native.log"
for mask in 0 1 2 3; do grep -q "PASS Q1 mask=$mask " "$1/direct-native.log"; done
' payload "$1" > "$1/payload.stdout" 2> "$1/payload.stderr"
verify_all > "$1/freezes-after.stdout" 2> "$1/freezes-after.stderr"
printf '%s\n' 'PASS V16 native aggregate raw EXIT0' > "$1/passed.txt"
