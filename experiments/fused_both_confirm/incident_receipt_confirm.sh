python3 - <<'PY' > build/incident-receipt.json
from pathlib import Path
import hashlib,json
old=Path.home()/'gpu-site/scratch-trellis/fused-both-core/bin/probe'
rewritten=Path('build/old-probe-objcopy-rewritten.elf')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(old)=='e4323d12987936c7c6503d7a697b3fa92d88b13e03cbdbb5263de1cd065f8436'
assert sha(rewritten)=='6e6548b809f9cc569669454506b3b57a8d6ca1b97a49e57d3c012c57d642abf4'
print(json.dumps(dict(incident='objcopy --dump-section without explicit output rewrote original input',
                     rewritten_sha256=sha(rewritten),restored_sha256=sha(old),
                     rewritten_bytes=rewritten.stat().st_size,original_bytes=old.stat().st_size,
                     recovery='Exact node2 copy, cp -p; original frozen manifest verified',
                     fix='Explicit build/original-read-copy.elf output; original input read only'),indent=2))
PY
