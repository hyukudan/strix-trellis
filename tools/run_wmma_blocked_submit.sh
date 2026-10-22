#!/usr/bin/env bash
# Two physical turns only: small validation + fixed grid, independent replay.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scratch_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/wmma-blocked-submit" ]] || trellis_die 'Wrong experimental subtree'
export TMPDIR="$TRELLIS_SOURCE/.tmp"
export XDG_CACHE_HOME="$TMPDIR/cache"
export AMD_COMGR_CACHE_DIR="$TMPDIR/comgr"
mkdir -p "$TMPDIR" build
oracle="$TRELLIS_SOURCE/build/oracles"
shared="$HOME/gpu-site/scratch-trellis"
dense=("$shared/gpu-dense-prefill-tests/layers.0.attn.wq_b.stgpu"
 "$shared/gpu-dense-prefill-tests/layers.0.attn.wo_b.stgpu"
 "$shared/gpu-dense-prefill-tests/layers.0.ffn.shared_experts.w1.stgpu")
small=("$oracle/blocked-k5.stgpu" "$oracle/blocked-k6.stgpu")
case ${1:-} in
 --build)
  hipcc=/opt/rocm/core-10.1/bin/hipcc
  flags=(-O3 -std=c++17 --offload-arch=gfx1151 -ffp-contract=off)
  nice -n 10 "$hipcc" "${flags[@]}" -Rpass-analysis=kernel-resource-usage -c src/wmma_blocked_submit.hip -o build/wmma_blocked_submit.o > build/resources.log 2>&1
  nice -n 10 "$hipcc" "${flags[@]}" --cuda-device-only -S src/wmma_blocked_submit.hip -o build/wmma_blocked_submit.s > build/isa-compile.log 2>&1
  nice -n 10 "$hipcc" "${flags[@]}" -DTRELLIS_GEMM_REUSE_BENCH build/wmma_blocked_submit.o src/gemm.hip src/gemv_lds.hip bench/wmma_blocked_submit_bench.hip -o build/wmma_blocked_submit_bench > build/compile.log 2>&1
  "$hipcc" --version > build/toolchain.txt
  # No GPU calls. Compare extracted copies only; never edit the old objects.
  nice -n 10 python3 tools/verify_wmma_submit.py
  sha256sum src/* bench/*.* tools/*.sh tools/verify_wmma_submit.py build/wmma_blocked_submit_bench build/wmma_blocked_submit.o build/wmma_blocked_submit.s build/resources.log build/toolchain.txt build/kernel-identity.json > build/compiled.sha256
  sha256sum "${small[@]}" "${dense[@]}" "$oracle"/*.oracle "$oracle"/*.bin > build/fixtures.sha256
  sha256sum build/compiled.sha256 build/fixtures.sha256 > build/campaign.sha256
  ;;
 --payload)
  [[ ${2:-} == first || ${2:-} == confirm ]] || trellis_die 'Unknown turn'
  # Exclusive create: a payload attempt consumes this turn even on failure.
  (set -o noclobber; date --iso-8601=seconds > "build/turn-$2.started") || trellis_die 'Turn already attempted'
  date --iso-8601=seconds; hostname
  sha256sum -c build/compiled.sha256
  sha256sum -c build/fixtures.sha256
  sha256sum -c build/campaign.sha256
  cat build/kernel-identity.json build/toolchain.txt
  if [[ $2 == first ]]; then
   nice -n 10 /usr/bin/time -v -o build/validate.memory build/wmma_blocked_submit_bench --validate "$oracle" "${small[@]}"
   cp build/campaign.sha256 build/validated.sha256
  else
   cmp build/validated.sha256 build/campaign.sha256
   sha256sum -c build/first-result.sha256
  fi
  nice -n 10 /usr/bin/time -v -o "build/$2.memory" build/wmma_blocked_submit_bench --search "$oracle" "${dense[@]}"
  date --iso-8601=seconds
  ;;
 first|confirm)
  [[ ! -e build/turn-$1.started ]] || trellis_die 'Turn already attempted'
  [[ $1 == first || -f build/first-result.sha256 ]] || trellis_die 'Missing first turn evidence'
  exec bash tools/run_gpu_turn.sh --seconds 180 --maintenance-check "$TRELLIS_SOURCE/tools/check_maintenance.sh" --queue-arg strix-trellis-wmma-blocked-submit -- "$TRELLIS_SOURCE/tools/run_wmma_blocked_submit.sh" --payload "$1"
  ;;
 *) trellis_die 'Usage: --build | first | confirm';;
esac
