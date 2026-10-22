#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "$0")/../.."
[[ "$PWD" == "$HOME/gpu-site/scratch-trellis/r8-wmma-core" ]]
export TMPDIR="$PWD/.tmp" XDG_CACHE_HOME="$PWD/.tmp/cache" HIP_CACHE_DIR="$PWD/.tmp/hip" AMD_COMGR_CACHE_DIR="$PWD/.tmp/comgr"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p build "$TMPDIR" "$XDG_CACHE_HOME" "$HIP_CACHE_DIR" "$AMD_COMGR_CACHE_DIR" control/src
cp "$HOME/gpu-site/scratch-trellis/runtime-v13/build-hip/libstrix_trellis.a" control/
cp "$HOME/gpu-site/scratch-trellis/runtime-v13/src/"* control/src/
cmp src/backend.hpp control/src/backend.hpp
echo '0b1453a3c9c25e3f7e4f12251d310a3a9f2c3b53c76d78c314e9624ba763e3b3  control/libstrix_trellis.a' | sha256sum -c -
# Exact opaque-object definitions are generated from the matching API source.
python3 -B - <<'PY'
from pathlib import Path
s=Path('control/src/api.hip').read_text()
a=s.index('struct strix_context_impl {');b=s.index('\nnamespace {',a)
Path('tools/r8_wmma_core/private_layout.hpp').write_text(s[a:b])
assert Path('src/api.hip').read_bytes()==Path('control/src/api.hip').read_bytes()
PY
hipcc=/opt/rocm/core-10.1/bin/hipcc
flags=(-O3 -std=c++17 --offload-arch=gfx1151 -ffp-contract=off)
[[ $(ps -o ni= -p $$ | tr -d ' ') == 10 ]]
"$hipcc" --version > build/toolchain.txt
"$hipcc" "${flags[@]}" -Rpass-analysis=kernel-resource-usage -c src/wmma_decoder.hip -o build/core.o > build/core.log 2>&1
"$hipcc" "${flags[@]}" -Rpass-analysis=kernel-resource-usage -c tools/r8_wmma_core/driver.hip -o build/driver.o > build/driver.log 2>&1
"$hipcc" "${flags[@]}" build/core.o build/driver.o control/libstrix_trellis.a -Wl,-Map,build/link.map -o build/r8-driver > build/link.log 2>&1
python3 -B tools/r8_wmma_core/prove.py > build/prove.log
python3 -B tools/r8_wmma_core/host_domain.py > build/host-domain.log
cp "$HOME/gpu-site/scratch-trellis/wmma-packed-probe/tools/audit_wmma_decoder.py" tools/r8_wmma_core/audit_inherited.py
python3 -B tools/r8_wmma_core/audit_inherited.py > build/isa-audit.log
