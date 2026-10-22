old="$HOME/gpu-site/scratch-trellis/fused-both-core"
(cd "$old" && sha256sum -c build/frozen.sha256) > .old-frozen-node2-before.txt
find "$old" -type f -print0 | sort -z | xargs -0 sha256sum > .old-baseline-node2.sha256
