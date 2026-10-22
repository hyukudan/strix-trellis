#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gpu-site/scratch-trellis"
{
 hostname; TZ=:/etc/localtime date -Is
 /opt/rocm/core-10.1/bin/hipcc --version
 sha256sum microbench.hip trellis_device.hpp host_reference.hpp design_probe.hip microbench-v3 design-probe
} > split-v3.meta
./design-probe --split-only
exec ./microbench-v3 256 11
