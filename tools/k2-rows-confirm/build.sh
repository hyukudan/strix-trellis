#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ "$PWD" == "$HOME/gpu-site/scratch-trellis/k2-rows-confirm" && ! -e frozen.sha256 ]]
export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.cache" HIP_CACHE_DIR="$PWD/.cache/hip" AMD_COMGR_CACHE_DIR="$PWD/.cache/comgr" MAX_JOBS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
hipcc=/opt/rocm/core-10.1/bin/hipcc
llvm=/opt/rocm/core-10.1/lib/llvm/bin
cmp src/backend.hpp ../k2-rows-core/src/backend.hpp
[[ $(sha256sum reference/libstrix_trellis_v11.a | cut -d' ' -f1) == 201d596e469c3d81a91aaeffa4588801fad5ed2d3b8d8cc1279458b78f6b0597 ]]
nice -n 10 g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ -I/opt/rocm/core-10.1/include -Isrc tools/k2-rows-confirm/factory_cpu.cpp -o bin/factory_confirm_cpu
nice -n 10 bin/factory_confirm_cpu > build/factory-cpu.log
nice -n 10 python3 -I -S -B reports/k2-rows-review-proof.py > build/design-proof.json
nice -n 10 python3 -I -S -B tools/k2-rows-confirm/proof.py > build/tail-proof.json
# Compile the new host harness while embedding the ORIGINAL diagnostic GPU bundle.
# There are no new device kernels; this avoids filename/CUID metadata churn.
nice -n 10 python3 -I -S -B tools/k2-rows-confirm/extract_diagnostics.py
nice -n 10 "$hipcc" -O3 -ffp-contract=off -std=c++17 --offload-arch=gfx1151 --cuda-host-only -Xclang -fcuda-include-gpubinary -Xclang build/original-diagnostics.hipfb -Isrc bench/k2_rows_confirm_probe.hip -x none original/libk2_rows_private.a reference/libstrix_trellis_v11.a -o bin/k2_rows_confirm_probe
nice -n 10 "$llvm/llvm-objcopy" --dump-section .hip_fatbin=build/exe.fatbin bin/k2_rows_confirm_probe
# The untouched original candidate and control are the actual linked archive objects.
cp -n original/{rows.elf,control.elf,rows-isa.txt,rows-notes.txt,control-isa.txt,control-notes.txt,compiler.txt} build/
nice -n 10 python3 -I -S -B tools/k2-rows-confirm/identity.py > build/identity.json
