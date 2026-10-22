"""Hash every performance tensor; never imports or loads native code."""
from pathlib import Path
import hashlib,json,struct
def verify(directory):
    m=json.loads((directory/'manifest.json').read_text());assert len(m['cases'])==128 and m['alpha']==1 and m['beta']==0
    assert m['peak_host_bytes']<2*1024**3 and m['original_scales'] and m['original_packed']
    for c in m['cases']:
        raw=(directory/c['file']).read_bytes();assert len(raw)==c['bytes'] and hashlib.sha256(raw).hexdigest()==c['sha256']
        assert raw[:8]==b'GK3CPU01';i,n,s,g,bits,count=struct.unpack_from('<6I',raw,8)
        assert (i,n,s) in ((5120,2304,16),(2304,5120,4)) and g==c['group'] and bits==3 and count==12
        at=32;seen=set()
        for _ in range(count):
            name=raw[at:at+8].rstrip(b'\0').decode();size,=struct.unpack_from('<Q',raw,at+8);at+=16
            b=raw[at:at+size];at+=size;assert name not in seen;seen.add(name)
            assert len(b)==c['tensors'][name]['bytes'] and hashlib.sha256(b).hexdigest()==c['tensors'][name]['sha256']
            if name=='scalars':assert b==struct.pack('<2f',1.,0.)
            if name=='bias':assert not b
            if name=='packed':assert len(b)==4423680
        assert at==len(raw)
    print('PASS CPU full tensor hashes, 128 alpha1/beta0 original-weight cases; native QA OPEN')
if __name__=='__main__':verify(Path('data/cpu'))
