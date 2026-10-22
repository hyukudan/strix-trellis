#!/usr/bin/env bash
set -euo pipefail
[[ "$PWD" == "$HOME/gpu-site/scratch-trellis/grouped-k3-perf-r1/runs/host-r2" && ! -e frozen.sha256 ]]
[[ $(ps -o ni= -p $$) -ge 10 ]]
export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" HIP_CACHE_DIR="$PWD/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr"
python3 -I -S -B experiments/grouped_k3_perf/verify_data.py
python3 -I -S -B experiments/grouped_k3_perf/identity.py > build/final-identity.json
sha256sum -c --quiet source.sha256
# Only runs is mutable. The manifest itself is the unavoidable self-hash exception.
find . -path ./runs -prune -o -type f ! -name frozen.sha256 -printf '%P\0' | sort -z | xargs -0 sha256sum > runs/freeze-manifest.tmp
mv runs/freeze-manifest.tmp frozen.sha256
sha256sum -c --quiet frozen.sha256
sha256sum frozen.sha256
