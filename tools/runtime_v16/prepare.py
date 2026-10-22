"""Restore historical compiled source byte spelling only in the new scope."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tarfile

own = Path.home()/'gpu-site/scratch-trellis/runtime-v16'
assert Path.cwd() == own and os.getpriority(os.PRIO_PROCESS, 0) == 10
assert not (own/'source-compiled.tar').exists()
root = own.parent
records = []
for directory in ('src', 'include'):
    for p in sorted((own/directory).glob('*')):
        if not p.is_file():
            continue
        local = p.read_bytes()
        selected = 'local LF source'
        for previous in ('runtime-v14b', 'runtime-v15'):
            old = root/previous/p.relative_to(own)
            if old.is_file() and old.read_bytes().replace(b'\r\n', b'\n') == local.replace(b'\r\n', b'\n'):
                p.write_bytes(old.read_bytes())
                selected = str(old)
                break
        assert p.read_bytes().replace(b'\r\n', b'\n') == local.replace(b'\r\n', b'\n')
        records.append(dict(path=p.relative_to(own).as_posix(), local_LF_sha256=hashlib.sha256(local).hexdigest(), compiled_sha256=hashlib.sha256(p.read_bytes()).hexdigest(), byte_source=selected))
(own/'source-byte-alignment.json').write_text(json.dumps(records, indent=2)+'\n', encoding='utf-8')
with tarfile.open(own/'source-compiled.tar', 'w') as archive:
    for directory in ('src', 'include'):
        archive.add(own/directory, arcname=directory)
shutil.copytree(root/'runtime-v14b/cpu-fixtures', own/'cpu-fixtures')
for directory in ('.tmp', '.cache/hip', '.cache/comgr', 'build', 'delivery'):
    (own/directory).mkdir(parents=True, exist_ok=True)
print('PREPARED_NEW_V16_SOURCE_ONLY')
