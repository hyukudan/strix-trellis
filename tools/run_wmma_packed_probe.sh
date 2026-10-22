#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/wmma_decoder_paths.sh"
trellis_require_own_source
[[ "$TRELLIS_SOURCE" == "$HOME/gpu-site/scratch-trellis/wmma-packed-probe" ]] || trellis_die 'Wrong experimental subtree'
export TMPDIR="$TRELLIS_SOURCE/.tmp"
export XDG_CACHE_HOME="$TMPDIR/cache"
export AMD_COMGR_CACHE_DIR="$TMPDIR/comgr"
export HIP_CACHE_DIR="$TMPDIR/hip"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$AMD_COMGR_CACHE_DIR" "$HIP_CACHE_DIR" build
dense="$TRELLIS_SOURCE/fixtures"
fixtures=("$dense/layers.0.attn.wq_b.stgpu" "$dense/layers.0.attn.wo_b.stgpu"
 "$dense/layers.0.ffn.shared_experts.w1.stgpu" "$dense/layers.0.ffn.shared_experts.w2.stgpu"
 "$dense/layers.0.attn.wq_a.stgpu" "$dense/layers.0.attn.wo_a.slice.0.stgpu")
case ${1:-} in
 --build)
  hipcc=/opt/rocm/core-10.1/bin/hipcc
  flags=(-O3 -std=c++17 --offload-arch=gfx1151 -ffp-contract=off)
  nice -n 10 python3 -B tools/prepare_wmma_packed.py > build/prepare.log
  nice -n 10 "$hipcc" "${flags[@]}" -Rpass-analysis=kernel-resource-usage -c src/wmma_decoder.hip -o build/wmma_decoder.o > build/resources.log 2>&1
  nice -n 10 "$hipcc" "${flags[@]}" -c bench/wmma_decoder_probe.hip -o build/wmma_decoder_probe.o > build/compile.log 2>&1
  nice -n 10 "$hipcc" "${flags[@]}" build/wmma_decoder.o build/wmma_decoder_probe.o control/libstrix_trellis.a -o build/wmma_decoder_probe >> build/compile.log 2>&1
  "$hipcc" --version > build/toolchain.txt
  nice -n 10 python3 -B tools/verify_wmma_decoder.py > build/verify.log
  ;;
 --freeze)
  sha256sum source-final.tar build/large-oracle-proof.json build/cpu-review.json tools/*.py python/strix_trellis/*.py docs/* build/packed-proof.json build/probe-isa.txt tools/*wmma_packed* tools/run_wmma_packed* fixtures/* src/* bench/wmma_decoder_probe.hip tools/*wmma_decoder* tools/prepare_wmma_decoder.py control/src/* control/libstrix_trellis.a build/wmma_decoder_probe build/wmma_decoder.o build/linked-*.elf build/linked-*.isa build/linked-*.notes build/resources.log build/toolchain.txt build/source-linked-proof.json build/isa-audit.json > build/compiled.sha256
  sha256sum build/oracles/* "${fixtures[@]}" > build/fixtures.sha256
  sha256sum build/compiled.sha256 build/fixtures.sha256 > build/campaign.sha256
  ;;
 --payload)
  [[ ${2:-} == first || ${2:-} == confirm ]] || trellis_die 'Unknown campaign'
  (set -o noclobber; date --iso-8601=seconds > "build/turn-$2.started") || trellis_die 'Campaign already consumed'
  date --iso-8601=seconds; hostname
  sha256sum -c build/campaign.sha256
  sha256sum -c build/compiled.sha256
  sha256sum -c build/fixtures.sha256
  cat build/toolchain.txt build/source-linked-proof.json build/isa-audit.json
  nice -n 10 timeout --signal=TERM --kill-after=2s 145s /usr/bin/time -v -o "build/$2.memory" build/wmma_decoder_probe build/oracles "${fixtures[@]}" "$2"
  touch "build/$2.passed"
  date --iso-8601=seconds
  ;;
 first|confirm)
  [[ $1 != confirm || -e build/first.passed ]] || trellis_die "First campaign did not pass"
  [[ ! -e build/turn-$1.started ]] || trellis_die 'Campaign already consumed'
  exec bash tools/run_wmma_packed_turn.sh --seconds 180 --maintenance-check "$TRELLIS_SOURCE/tools/wmma_decoder_maintenance.sh" --queue-arg strix-trellis-wmma-packed-probe -- "$TRELLIS_SOURCE/tools/run_wmma_packed_probe.sh" --payload "$1"
  ;;
 *) trellis_die 'Usage: --build | --freeze | first | confirm';;
esac
