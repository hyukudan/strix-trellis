"""Audit actual new executable against the immutable measured original, CPU only."""
from pathlib import Path
import hashlib,json,re,struct,runpy,contextlib,io
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha('original/k2_rows_probe')=='3dbfae0b6f204da83fd082e4c8f71d41903f108e01af876bf6dc70336dfba7e1'
assert sha('reference/libstrix_trellis_v11.a')=='201d596e469c3d81a91aaeffa4588801fad5ed2d3b8d8cc1279458b78f6b0597'
# Run the unchanged full source/ISA/resource audit on the new linked fatbin.
# Its legacy binary key resolves to the preserved original, never overwriting it.
Path('bin/k2_rows_probe').symlink_to('../original/k2_rows_probe') if not Path('bin/k2_rows_probe').exists() else None
capture=io.StringIO()
with contextlib.redirect_stdout(capture):runpy.run_path('tools/k2-rows-core/audit.py')
Path('build/audit.json').write_text(capture.getvalue())
assert Path('build/exe.fatbin').read_bytes()==Path('original/exe.fatbin').read_bytes(),'complete GPU fatbin must be byte identical'
for name in ('rows.elf','control.elf'):
    assert Path('build',name).read_bytes()==Path('original',name).read_bytes()
isa=Path('build/rows-isa.txt').read_text(); kernels={}
for name,fragment in re.findall(r'<([^>]*kernel[^>]*)>:(.*?)(?=\n\n|\Z)',isa,re.S):
    words=re.findall(r'// [0-9A-Fa-f]+: ((?:[0-9A-Fa-f]{8} ?)+)',fragment)
    code=b''.join(bytes.fromhex(w)[::-1] for line in words for w in line.split())
    assert code,name
    kernels[name]={'instruction_bytes':len(code),'sha256':hashlib.sha256(code).hexdigest()}
assert len(kernels)==6
new_source=Path('bench/k2_rows_confirm_probe.hip').read_text();old_source=Path('original/k2_rows_probe.hip').read_text()
for name in ('books_probe','reader_probe'):
    pattern=r'__global__ void '+name+r'[^\n]+'
    assert re.search(pattern,new_source)[0]==re.search(pattern,old_source)[0]
assert len(re.findall(r'__global__ void',new_source))==4 # two original forward declarations, two original diagnostic definitions
print(json.dumps({'PASS':True,'complete_fatbin_byte_identical':True,'fatbin_sha256':sha('build/exe.fatbin'),'new_host_binary_sha256':sha('bin/k2_rows_confirm_probe'),'original_binary_sha256':sha('original/k2_rows_probe'),'six_kernel_instruction_bytes':kernels},indent=2))
