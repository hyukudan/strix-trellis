#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gpu-site/scratch-trellis"
./microbench-v2 --fixture layers.0.attn.wkv.trellis.bin ref-wkv.states.bin ref-wkv.values.bin
exec ./microbench-v2 256 11
