# One-time recovery from the documented objcopy in-place rewrite, not a build step.
old="$HOME/gpu-site/scratch-trellis/fused-both-core"
echo "6e6548b809f9cc569669454506b3b57a8d6ca1b97a49e57d3c012c57d642abf4  $old/bin/probe" | sha256sum -c -
cp -p "$old/bin/probe" build/old-probe-objcopy-rewritten.elf
ssh -n -o BatchMode=yes node2 "nice -n 10 env TMPDIR='$own/.tmp' TMP='$own/.tmp' TEMP='$own/.tmp' XDG_CACHE_HOME='$own/.cache' HIP_CACHE_DIR='$own/.cache/hip' AMD_COMGR_CACHE_DIR='$own/.cache/comgr' tar -C '$old' -cf - bin/probe" > build/old-restoration.tar
mkdir -p build/restoration
tar -xf build/old-restoration.tar -C build/restoration
echo 'e4323d12987936c7c6503d7a697b3fa92d88b13e03cbdbb5263de1cd065f8436  build/restoration/bin/probe' | sha256sum -c -
cp -p build/restoration/bin/probe "$old/bin/probe"
(cd "$old" && sha256sum -c build/frozen.sha256) > build/old-restoration-verify.txt
sha256sum "$old/bin/probe"
