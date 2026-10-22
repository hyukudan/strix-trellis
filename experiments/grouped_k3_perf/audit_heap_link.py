"""Read-only native ELF operator boundary audit; never executes HIP."""
from pathlib import Path
import subprocess,json,re,hashlib
ops=['_Znwm','_Znam','_ZdlPv','_ZdaPv','_ZdlPvm','_ZdaPvm']
def run(*args):return subprocess.check_output(args,text=True)
symbols=run('readelf','-Ws','bin/perf_native')
dynamic=run('readelf','-W','--dyn-syms','bin/perf_native')
relocs=run('objdump','-r','bin/perf_native')
archive=run('nm','-u','lib/libstrix_trellis.a')
source=Path('build/effective-source.ii').read_text()
records={}
for op in ops:
    imports=[x for x in dynamic.splitlines() if re.search(r'\b'+op+r'@',x)]
    assert imports and all(' UND ' in x for x in imports),(op,imports)
    assert not any(re.search(r'\b'+op+r'(?:@|$)',x) and ' UND ' not in x for x in symbols.splitlines())
    assert '__wrap_'+op in symbols,op
    records[op]=dict(versioned_imports=imports,linked_wrapper_relocations=[x.strip() for x in relocs.splitlines() if re.search(r'__wrap_'+op+r'(?:-|\s|$)',x)],static_archive_references=[x.strip() for x in archive.splitlines() if x.strip().endswith(op)])
assert '__real__Znwm' in source and '__wrap__Znwm' in source
# Effective overload inventory is evidence, not a claim that all heap is guarded.
operator_names=sorted(set(re.findall(r'\b_Z(?:nw|na|dl|da)[A-Za-z0-9_]*',symbols+'\n'+archive)))
driver=run('readelf','-W','--dyn-syms','--relocs','/opt/rocm/core-10.1/lib/libamdhip64.so')
assert '__wrap_' not in driver
for op in ops:assert re.search(r' UND '+op+r'@',driver),op
Path('build/HIP-heap-imports.txt').write_text('\n'.join(x for x in driver.splitlines() if any(re.search(r'\b'+op+r'@',x) for op in ops))+'\n')
inventory={op:dict(symbol_rows=[x.strip() for x in symbols.splitlines() if re.search(r'\b'+op+r'(?:@|$)',x)],static_archive_references=[x.strip() for x in archive.splitlines() if x.strip().endswith(op)]) for op in operator_names}
Path('build/heap-symbols.txt').write_text(symbols+'\n'+dynamic+'\n'+relocs)
print(json.dumps(dict(status='PASS_LINK_BOUNDARY_ONLY',ordinary=records,effective_operator_symbols=operator_names,overload_inventory=inventory,preprocessed_overload_declarations=sorted(set(re.findall(r'operator\s+(?:new|delete)\s*(?:\[\])?\s*\([^;{}]*\)',source))),nothrow_aligned=[x for x in operator_names if 'nothrow' in x or 'align_val' in x],scope='User/static unresolved link references only; driver DSO imports remain ordinary versioned imports; no HIP internal allocations observed',preprocessed_sha256=hashlib.sha256(source.encode()).hexdigest()),indent=2))
