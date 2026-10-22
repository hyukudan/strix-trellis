#!/usr/bin/env bash
set -euo pipefail
root="$HOME/gpu-site/scratch-trellis/k2-rows-confirm"
old="$HOME/gpu-site/scratch-trellis/k2-rows-core"
[[ -d "$root" && ! -e "$root/frozen.sha256" ]]
cd "$root"
export TMPDIR="$root/.tmp" XDG_CACHE_HOME="$root/.cache" HIP_CACHE_DIR="$root/.cache/hip" AMD_COMGR_CACHE_DIR="$root/.cache/comgr" MAX_JOBS=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HIP_CACHE_DIR" "$AMD_COMGR_CACHE_DIR" runs/1 runs/2 attempts/1 attempts/2 build bin original data/derived reference src reports tools/k2-rows-core
(cd "$old"; sha256sum --check --quiet frozen.sha256)
cp -rn "$old/src/." src/
for name in wo_b.stgpu q_b.stgpu; do cp -n "$old/data/$name" "data/$name"; done
cp -n "$old/data/derived/"{wo_b.fp64,q_b.fp64,provenance.json} data/derived/
cp -n "$old/reference/libstrix_trellis_v11.a" reference/
cp -n "$old/bin/k2_rows_probe" original/k2_rows_probe
cp -n "$old/build/"{libk2_rows_private.a,rows.elf,control.elf,exe.fatbin,rows-isa.txt,rows-notes.txt,control-isa.txt,control-notes.txt,compiler.txt} original/
cp -n "$old/frozen.sha256" original/frozen-core.sha256
cp -n "$old/tools/k2-rows-core/audit.py" tools/k2-rows-core/
cp -n "$old/reports/k2-rows-review-proof.py" reports/
cp -n "$old/tools/k2-rows-core/prep.py" tools/k2-rows-core/
mkdir -p docs
cp -n "$old/tools/make_k2_fixtures.py" tools/
cp -n "$old/docs/"{K2_MULTIROW.md,NUMERICS.md} docs/
cp -n "$old/bench/k2_rows_probe.hip" original/k2_rows_probe.hip
sha256sum original/k2_rows_probe reference/libstrix_trellis_v11.a
