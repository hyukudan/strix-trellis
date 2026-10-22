ls -l build/old* build/restoration/bin/probe || true
sha256sum "$HOME/gpu-site/scratch-trellis/fused-both-core/bin/probe"
tail -10 build/old-restoration-verify.txt || true
tail -10 build/host-rebuild.log || true
tail -10 build/host-harness.json || true
