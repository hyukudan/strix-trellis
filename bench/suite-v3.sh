#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/gpu-site/scratch-trellis"
{
  hostname; TZ=:/etc/localtime date -Is
  /opt/rocm/core-10.1/bin/hipcc --version
  printf '%s\n' 'nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1151 microbench.hip -o microbench-v3' 'nice -n 10 hipcc -O3 -std=c++17 --offload-arch=gfx1151 design_probe.hip -o design-probe'
  sha256sum microbench.hip trellis_device.hpp host_reference.hpp design_probe.hip microbench-v3 design-probe
  for clock in /sys/class/drm/card*/device/pp_dpm_sclk; do [[ ! -f "$clock" ]] || { echo "$clock"; cat "$clock"; }; done
} > suite-v3.meta
./microbench-v3 --fixture layers.0.attn.wkv.trellis.bin ref-wkv.states.bin ref-wkv.values.bin
./design-probe
exec ./microbench-v3 256 11
