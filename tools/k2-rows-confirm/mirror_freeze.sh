#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ "$PWD" == "$HOME/gpu-site/scratch-trellis/k2-rows-confirm" && ! -e frozen.sha256 ]]
export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" HIP_CACHE_DIR="$PWD/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr"
tar -cf mirror.tar bench src tools reports docs original reference data build bin
ssh node2 'mkdir -p ~/gpu-site/scratch-trellis/k2-rows-confirm/{.tmp,.cache/hip,.cache/comgr,runs/1,runs/2,attempts/1,attempts/2}'
scp -q mirror.tar node2:gpu-site/scratch-trellis/k2-rows-confirm/mirror.tar
ssh node2 'cd ~/gpu-site/scratch-trellis/k2-rows-confirm && export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" HIP_CACHE_DIR="$PWD/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr" && tar -xf mirror.tar && (cd ../k2-rows-core; sha256sum --check --quiet frozen.sha256)'
nice -n 10 python3 -I -S -B tools/k2-rows-confirm/freeze.py
ssh node2 'cd ~/gpu-site/scratch-trellis/k2-rows-confirm && export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" HIP_CACHE_DIR="$PWD/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr" && nice -n 10 python3 -I -S -B tools/k2-rows-confirm/freeze.py'
