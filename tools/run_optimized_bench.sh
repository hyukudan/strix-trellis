#!/usr/bin/env bash
# Synchronize the perf source subtree, binaries and fixtures on both nodes first.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/perf" ]] || trellis_die 'Expected dedicated perf subtree'
if [[ ${1:-} == --payload ]]; then
    date --iso-8601=seconds
    hostname
    /opt/rocm/core-10.1/bin/hipcc --version
    sha256sum src/trellis_optimized.hpp src/optimized_microbench.hip src/trellis_device.hpp src/host_reference.hpp optimized_microbench_v3 *.bin
    "$TRELLIS_SOURCE/optimized_microbench_v3" --fixture layers.0.attn.wkv.trellis.bin ref-wkv.states.bin ref-wkv.values.bin
    exec "$TRELLIS_SOURCE/optimized_microbench_v3" 256 11 5 6
fi
[[ $# == 0 ]] || trellis_die 'No arguments expected'
exec bash "$TRELLIS_SOURCE/tools/run_gpu_turn.sh" --seconds 180 \
  --maintenance-check "$TRELLIS_SOURCE/tools/check_maintenance.sh" \
  --queue-arg strix-trellis-perf-bounded-v3 -- "$TRELLIS_SOURCE/tools/run_optimized_bench.sh" --payload
