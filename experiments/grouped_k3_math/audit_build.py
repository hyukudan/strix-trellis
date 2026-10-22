"""CPU-only checks of effective headers and finite scalar selection ISA patterns."""
from pathlib import Path
import hashlib
import json
import re


def sha(b):
    return hashlib.sha256(b).hexdigest()


def selection(stage, text):
    axis = 's4' if stage == 'grouped_baseline_core' else 's3'
    expected = [rf's_mul_i32 s\d+, {axis}, 0x48', rf's_mul_hi_i32 s\d+, {axis}, 0x48',
                r's_add_u32 s0, s0, s\d+', r's_addc_u32 s1, s1, s\d+']
    for pattern in expected:
        assert re.search(pattern,text), (stage,pattern)
    # Exact dynamic address derived from SGPR CTA y/z (72-byte Group).
    # Descriptor loads stay scalar. Vector loads later are payloads/reader data.
    loads = {'grouped_input_transform': [('b128','0x18'),('b64','0x40')],
             'grouped_baseline_core': [('b128','0x40'),('b64','0x10')],
             'grouped_output_transform': [('b128','0x28'),('b64','0x38'),('b64','0x50')]}
    for size, offset in loads[stage]:
        assert re.search(r's_load_'+size+r' s\[\d+:\d+\], s\[0:1\], '+offset+r'\s',text), (stage,size,offset)
    return dict(cta_index=axis,stride=72,scalar_group_address=True,loads=loads[stage])


def main():
    root=Path.cwd().resolve()
    pins=json.loads(Path('reports/grouped-k3-adapter-pins.json').read_text())
    dependency_text=Path('build/effective-headers.d').read_text().replace('\\\n',' ')
    dependencies=sorted(set(t for t in dependency_text.split() if not t.endswith(':')))
    records=[]
    actual=set()
    for name in dependencies:
        p=Path(name).resolve();assert p.is_file(),name
        records.append(dict(spelling=name,resolved=str(p),sha256=sha(p.read_bytes())))
        actual.add(p)
    for name in ('src/hadamard.hpp','src/trellis_optimized.hpp','src/trellis_device.hpp','src/packed_decoder.hpp',
                 'src/backend.hpp','src/api.hip','include/strix_trellis.h'):
        p=(root/name).resolve()
        # api.hip contributes literal private views, not a second API implementation.
        if name!='src/api.hip':assert p in actual,name
        assert sha(p.read_text(encoding='utf-8').encode())==pins['files_lf'][name],name
    Path('build/effective-header-hashes.json').write_text(json.dumps(records,indent=2)+'\n')
    result={}
    for stage in ('grouped_input_transform','grouped_baseline_core','grouped_output_transform'):
        text=Path('build/'+stage+'.isa').read_text()
        result[stage]=selection(stage,text)
        for before,after in [('s_mul_i32','v_mul_lo_u32'),('s_load_b128','global_load_b128')]:
            try:selection(stage,text.replace(before,after))
            except AssertionError:pass
            else:raise AssertionError('scalar selection mutation accepted')
    Path('build/scalar-selection.json').write_text(json.dumps(dict(selection=result,
        mutation_tests_rejected=6,scope='finite emitted ISA pattern review, not compiled math identity',
        effective_preprocessed_sha256=sha(Path('build/effective-source.ii').read_bytes())),indent=2)+'\n')
    print('PASS effective actual compile headers and SGPR selection; six ISA mutations rejected')


if __name__=='__main__':main()
