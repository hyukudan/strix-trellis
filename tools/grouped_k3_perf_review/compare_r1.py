"""Read-only byte comparison with the prior frozen R1 source/images, no ISA run."""
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import sys
import tarfile

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'tools'))
from audit_grouped_k3_math import sections

def sha(b):
    return hashlib.sha256(b).hexdigest()

def read(path):
    with tarfile.open(path) as t:
        return {m.name:t.extractfile(m).read() for m in t.getmembers() if m.isfile()}

spec = importlib.util.spec_from_file_location('review_identity', root / 'experiments/grouped_k3_perf/identity.py')
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)
base = root.parent
old_path = base / 'wt-grouped-k3-math-r1/reports/grouped-k3-math-r1-data/node2-evidence.tar.gz'
new_path = base / 'wt-grouped-k3-perf/reports/grouped-k3-perf-data/node2-evidence.tar.gz'
old, new = read(old_path), read(new_path)
pin = json.loads((root / 'reports/grouped-k3-math-r1-data/handoff.json').read_text())['nodes']['node2']
assert sha(old_path.read_bytes()) == pin['evidence_archive_sha256']
assert sha(old['bin/math_native_qa']) == pin['binary_sha256']
assert old['lib/libstrix_trellis.a'] == new['lib/libstrix_trellis.a']
images = []
for j in range(8):
    name = f'build/linked-{j:02d}.elf'
    prior, current = old[name], new[name]
    if j:
        assert prior == current
    else:
        assert prior != current
        assert identity.function_bytes(prior) == identity.function_bytes(current)
        for section in ('.text','.note'):
            assert sections(prior)[section] == sections(current)[section]
    images.append(dict(name=name, old_sha256=sha(prior), new_sha256=sha(current),
                       whole_identical=prior == current))
with tarfile.open(fileobj=io.BytesIO(new['source.tar'])) as t:
    source = {m.name:t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
with tarfile.open(fileobj=io.BytesIO(old['source.tar'])) as t:
    old_source = {m.name:t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
shared = [n for n in source if n.startswith('experiments/grouped_k3_math/')]
assert shared and all(source[n] == old_source[n] for n in shared)
print(json.dumps(dict(scope='READ_ONLY_FROZEN_R1_COMPARISON', old_archive_sha256=sha(old_path.read_bytes()),
                      new_archive_sha256=sha(new_path.read_bytes()), reused_math_source_files=len(shared),
                      all_math_FUNC_text_note_equal=True, images=images), indent=2))
