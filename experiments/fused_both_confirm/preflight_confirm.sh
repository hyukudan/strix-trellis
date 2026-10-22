bash experiments/fused_both_confirm/run_confirm.sh --build > build/host-rebuild.log 2>&1
python3 experiments/fused_both_confirm/host_harness_check.py > build/host-harness.json
python3 - <<'PY'
from pathlib import Path
import hashlib,json,numpy as np
manifest=json.loads(Path('references/manifest.json').read_text())
p=Path('references/r8-y-fp64-core.f32')
assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['files'][p.name]['sha256']
a=np.fromfile('fixtures/head/cpu-y.f32','<f4');b=np.fromfile(p,'<f4',count=129280)
error=a.astype('d')-b.astype('d');m=float(abs(error).max());nrms=float(np.linalg.norm(error)/np.linalg.norm(b.astype('d')))
assert m<=5e-4*max(1,float(abs(b).max())) and nrms<=3e-5
print('HEAD_PREFIX_CPU',m,nrms,'full_columns',len(a),'not_standalone_CUDA_R1')
PY
old="$HOME/gpu-site/scratch-trellis/fused-both-core"
(cd "$old" && sha256sum -c build/frozen.sha256) > build/old-frozen-before.txt
find "$old" -type f -print0 | sort -z | xargs -0 sha256sum > build/old-preservation.sha256
sha256sum src/backend.hpp "$HOME/gpu-site/scratch-trellis/runtime-v12/src/backend.hpp"
