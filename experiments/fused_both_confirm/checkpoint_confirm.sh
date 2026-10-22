date -Is
find "$HOME" -maxdepth 3 -name 'strix-trellis-coord.md' -print
cat "$HOME/gpu-site/.gpu-turn/maintenance" 2>/dev/null || true
cat "$HOME/gpu-site/.gpu-turn/holder" 2>/dev/null || true
cat build/host-harness.json
tail -4 build/cpu-normative.txt
tail -2 build/old-restoration-verify.txt 2>/dev/null || true
