#!/usr/bin/env bash
# CPU-only mirror and immutable manifest. Run once after all build gates pass.
set -euxo pipefail
[[ "$PWD" == "$HOME/gpu-site/scratch-trellis/grouped-k3-abi" && ! -e frozen.sha256 ]]
export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" HIP_CACHE_DIR="$PWD/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr" MAX_JOBS=1
grep -q '^PASS linked trace' build/resources.json
find experiments reports docs src bin build -type f -print0 | sort -z | xargs -0 sha256sum > frozen.sha256
sha256sum AGENTS.md source.tar >> frozen.sha256
tar -cf mirror.tar experiments reports docs src bin build AGENTS.md source.tar frozen.sha256
scp -q mirror.tar node2:gpu-site/scratch-trellis/grouped-k3-abi/mirror.tar
ssh -o BatchMode=yes node2 'cd ~/gpu-site/scratch-trellis/grouped-k3-abi && export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" HIP_CACHE_DIR="$PWD/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr"; nice -n 10 bash -c '\''tar -xf mirror.tar; sha256sum --check frozen.sha256; sha256sum frozen.sha256'\'''
sha256sum --check frozen.sha256
sha256sum frozen.sha256
