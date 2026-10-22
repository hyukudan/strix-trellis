#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gpu-site/scratch-trellis"
exec bash tools/run_gpu_turn.sh --seconds 180 --maintenance-check "$PWD/tools/check_maintenance.sh" --queue-arg strix-trellis-micro -- "$PWD/microbench" 256 11
