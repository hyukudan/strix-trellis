#!/usr/bin/env bash
set -euo pipefail
own="$HOME/gpu-site/scratch-trellis/fused-groups"
[[ $(pwd -P) == "$own" && ( ${1:-} == search || ${1:-} == confirm ) ]]
# The payload repeats maintenance/time checks after acquisition. One synchronous
# controller remains authoritative until this remote invocation has closed.
exec timeout --signal=TERM --kill-after=5s 1195s \
    "$HOME/gpu-site/gpu-turn-any.sh" strix-trellis-fused-groups \
    timeout --signal=TERM --kill-after=3s 180s \
    "$own/tools/run_fused_groups.sh" --payload "$1"
