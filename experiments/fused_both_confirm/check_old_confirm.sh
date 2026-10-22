old="$HOME/gpu-site/scratch-trellis/fused-both-core"
date -Is
sha256sum "$old/bin/probe" "$old/bin/probe.0.hip-amdgcn-amd-amdhsa--gfx1151"
stat -c '%s %y %n' "$old/bin/probe"
find "$HOME/gpu-site/scratch-trellis" -maxdepth 1 -name '*coord*' -print
