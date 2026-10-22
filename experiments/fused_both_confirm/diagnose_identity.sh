old="$HOME/gpu-site/scratch-trellis/fused-both-core"
image=bin/probe_confirm.0.hip-amdgcn-amd-amdhsa--gfx1151
cmp -l "$old/bin/probe.0.hip-amdgcn-amd-amdhsa--gfx1151" "$image" | head -25 || true
/opt/rocm/core-10.1/llvm/bin/llvm-readobj --sections --notes "$image" > build/new-sections.txt
/opt/rocm/core-10.1/llvm/bin/llvm-readobj --sections --notes "$old/bin/probe.0.hip-amdgcn-amd-amdhsa--gfx1151" > build/old-sections.txt
diff -u build/old-sections.txt build/new-sections.txt | head -100 || true
ls -l "$image" "$old/bin/probe.0.hip-amdgcn-amd-amdhsa--gfx1151"
