"""CPU guards/data/window/statistics counterexamples; no HIP imports or execution."""
from pathlib import Path
import importlib.util
import json
import math
import os
import struct
import tempfile
import numpy as np

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
root=Path(__file__).resolve().parents[2]
refs=load('tp_refs',root/'tools/tp_decode_refs.py')
admission=load('tp_admission',Path(__file__).with_name('admission.py'))
stats=load('tp_stats',Path(__file__).with_name('summarize.py'))
for minute,second,wanted in [(0,0,True),(1244,59,True),(1245,0,False),(1259,59,False),(1260,0,False),(1409,59,False),(1410,0,True)]:
    assert admission.allowed(minute,second)==wanted
table=refs.lut();assert table.shape==(65536,) and np.isfinite(table).all()
for i in (1280,4096):
    boundaries=[(i//16)*s//4 for s in range(5)]
    for r in range(1,9):
        visits=np.zeros((r,i//16),dtype=int)
        for b,e in zip(boundaries,boundaries[1:]):
            for h in range(b//8,(e+7)//8):
                for k in range(max(h*8,b),min(h*8+8,e)):visits[:,k]+=1
        assert np.all(visits==1)
        for b,e in zip(boundaries,boundaries[1:]):
            for cb in range(b,e,8 if r>=5 else 1024):
                ce=min(cb+(8 if r>=5 else 1024),e)
                addresses=[]
                for k in range(cb,ce,2):
                    addresses.append(k)
                    if k+1<ce:addresses.append(k+1)
                assert addresses==list(range(cb,ce)) # includes LAST without end+1
                events=[('load',cb)];k=cb
                while k+2<ce:
                    events.extend([('load',k+1),('dot',k),('load',k+2),('dot',k+1)]);k+=2
                if k+1<ce:events.extend([('load',k+1),('dot',k),('dot',k+1)])
                else:events.append(('dot',k))
                live=set()
                for op,tile in events:
                    assert cb<=tile<ce
                    if op=='load':assert tile not in live;live.add(tile);assert len(live)<=2
                    else:
                        if tile+1<ce:assert tile+1 in live
                        live.remove(tile)
                assert not live and [t for op,t in events if op=='dot']==list(range(cb,ce))
# Literal scalar states vs the full independent block oracle, including wrap.
words=np.arange(40,dtype='<u4').reshape(1,1,40)*np.uint32(0x1234567)
q=refs.full_q(words,table)
for lane in range(32):
    for p in range(8):
        start=((8*lane+p+1)*5-16)%1280;state=0
        for bit in range(16):
            pos=(start+bit)%1280;state=(state<<1)|((int(words[0,0,pos//32])>>(31-pos%32))&1)
        assert q[2*(lane%4)+p%2+8*((p//2)%2),lane//4+8*(p//4)].tobytes()==table[state].tobytes()
# RN32 before RN16 is distinguishable from direct F64 -> half on a tie.
half=np.float16(1);other=np.nextafter(half,np.float16(2));tie=(float(half)+float(other))/2
x=tie+2**-30
assert np.float16(np.float32(x)).tobytes()!=np.float16(x).tobytes()
# Cancellation and subnormal controls remain finite; grouped split sums differ.
p=np.array([2**25,1,-2**25,1],dtype='<f4');ordered=np.float32(0)
for value in p:ordered=np.float32(ordered+value)
assert ordered==1 and np.float32(np.float32(p[0]+p[1])+np.float32(p[2]+p[3]))==0
tiny=np.frombuffer(struct.pack('<H',1),dtype='<f2')[0];assert tiny>0 and float(tiny)*float(tiny)>0
with tempfile.TemporaryDirectory(prefix='tp-proof-',dir=os.environ.get('TMPDIR',root/'reports/tp-decode-private-data')) as directory:
    directory=Path(directory);path=directory/'bad.stgpu'
    for raw in (b'',struct.pack('<8sII',b'BADMAGIC',1,13),struct.pack('<8sII',b'STGPU001',1,13)+b'broken'):
        path.write_bytes(raw)
        try:refs.sections(path)
        except (AssertionError,struct.error):pass
        else:raise AssertionError('Malformed data accepted')
    # LAST byte mutation must be rejected by the exact frozen checker.
    (directory/'last').write_bytes(b'abc');(directory/'frozen.sha256').write_text(refs.sha(directory/'last')+'  last\n')
    admission.verify_freeze(directory);(directory/'last').write_bytes(b'abd')
    try:admission.verify_freeze(directory)
    except AssertionError:pass
    else:raise AssertionError('LAST freeze corruption accepted')
    # Synthetic timing records are test data, explicitly not benchmark evidence.
    lines=[]
    for site in ('q_b','o_b'):
        for rank in (0,1):
            for rows in range(1,9):
                candidates=[1,2,3,4]+([5] if rows==1 or(site=='o_b' and rows==6) else [])
                for candidate in candidates:
                    for sweep in range(11):
                        for variant in ([0,candidate] if sweep%2==0 else [candidate,0]):
                            for clone in range(21):lines.append(f'SAMPLE,L0.{site}.rank{rank},{rows},{candidate},{variant},{sweep},{clone},100,120')
    path=directory/'synthetic-test.log';path.write_text('\n'.join(lines))
    result=stats.summarize(path);assert len(result['results'])==268 and len(result['raw_chronology'])==61908
    assert all(x['event_speedup']==1 and x['wall_speedup']==1 for x in result['paired_all_losses_and_tails'])
print(json.dumps(dict(passed=True,cpu_only=True,window_cases=7,raw_masks='PASS',all_hash_states=65536,
                     last_corruption='rejected',timings='synthetic UNIT TEST ONLY',native_fdots='OPEN')))
