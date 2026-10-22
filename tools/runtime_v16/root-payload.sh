#!/usr/bin/env bash
# ROOT may queue this reviewed payload; this file never requests a lease.
set -euo pipefail
[[ ${ROOT_GPU_LEASE:-0} == 1 && $# == 1 && "$1" =~ ^[a-z0-9][a-z0-9-]{0,22}$ ]]
root="$HOME/gpu-site/scratch-trellis"
[[ "$PWD" == "$root/runtime-v16" ]]
/usr/bin/time -v timeout --signal=TERM --kill-after=3s 1180s bash -c '
set -euo pipefail
root="$HOME/gpu-site/scratch-trellis"
(cd "$root/runtime-v16-qa" && bash native-root.sh "$PWD/root-v16-native-$1")
cd "$root/runtime-v16"
timeout --signal=TERM --kill-after=3s 550s bash tools/run_runtime_v16.sh --payload "root-v16-$1"
' payload "$1"
