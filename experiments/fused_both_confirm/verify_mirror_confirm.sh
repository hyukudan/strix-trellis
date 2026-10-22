date -Is
cmp src/backend.hpp "$HOME/gpu-site/scratch-trellis/runtime-v12/src/backend.hpp"
sha256sum -c build/frozen.sha256 > build/mirror-frozen.txt
sha256sum -c build/fixtures.sha256 > build/mirror-fixtures.txt
if [[ $(hostname) == node2 ]];then
    sha256sum -c .old-baseline-node2.sha256 > build/mirror-old-preserved.txt
else
    sha256sum -c build/old-preservation.sha256 > build/mirror-old-preserved.txt
fi
sha256sum build/frozen.sha256 build/fixtures.sha256 bin/probe_confirm
cat "$HOME/gpu-site/.gpu-turn/maintenance" 2>/dev/null || true
cat "$HOME/gpu-site/.gpu-turn/holder" 2>/dev/null || true
