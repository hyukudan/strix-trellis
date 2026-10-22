"""Freeze immutable inputs after both mirrors exist, CPU only."""
from pathlib import Path
import hashlib
assert not Path('frozen.sha256').exists()
for name in ('.tmp','.cache/hip','.cache/comgr','runs/1','runs/2','attempts/1','attempts/2'):assert Path(name).is_dir(),name
files=sorted(p for directory in ('bench','src','tools','reports','docs','original','reference','data','build','bin') for p in Path(directory).rglob('*') if p.is_file())
Path('frozen.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.as_posix()+'\n' for p in files))
print(hashlib.sha256(Path('frozen.sha256').read_bytes()).hexdigest())
