#!/usr/bin/env bash
# CPU ONLY. Freeze both independently built nodes after data/source review.
set -euxo pipefail
[[ "$PWD" == "$HOME/gpu-site/scratch-trellis/grouped-k3-math-r1" && ! -e frozen.sha256 ]]
export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" HIP_CACHE_DIR="$PWD/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr"
[[ $(ps -o ni= -p $$) -ge 10 ]]
python3 -I -S -B experiments/grouped_k3_math/verify_data.py
python3 -I -S -B experiments/grouped_k3_math/audit_build.py
sha256sum -c source.sha256
find experiments tools reports docs src include lib bin build data attempts -type f -print0 | sort -z | xargs -0 sha256sum > frozen.sha256
sha256sum AGENTS.md source.tar source.sha256 >> frozen.sha256
sha256sum -c --quiet frozen.sha256
sha256sum frozen.sha256
