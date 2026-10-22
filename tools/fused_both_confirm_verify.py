"""CPU-only audit of immutable payload sources and both recorded physical campaigns."""
from pathlib import Path
import hashlib
import json
import math
import re

root=Path(__file__).resolve().parents[1]
data=root/'reports/fused-both-confirm-data'
expected='4b381b7b5d32b5e3a014df1b05f61996496fa40aced0ecd28f8bd55078bca0f3'
assert hashlib.sha256((data/'core-original-identical.elf').read_bytes()).hexdigest()=='a2616be9221bd9db77cc626b86d9437e499bfbb82f9138d36024de11744ef636'
geometries={'q_b':(1280,32768,5),'wo_b':(8192,5120,5),'inverse':(2304,5120,5),
            'shared':(5120,2304,5),'head':(5120,129280,6)}
rows=[]
for node,phase in [('node2','search'),('node1','confirm')]:
    folder=data/node
    frozen=folder/'build/frozen.sha256'
    assert hashlib.sha256(frozen.read_bytes()).hexdigest()==expected
    source_count=0
    for line in frozen.read_text().splitlines():
        digest,path=line.split('  ',1)
        if path.startswith(('src/','experiments/')):
            actual=(root/path).read_bytes().replace(b'\r\n',b'\n')
            assert hashlib.sha256(actual).hexdigest()==digest,path
            source_count+=1
    for relative in ['build/mirror-frozen.txt','build/mirror-fixtures.txt','build/mirror-old-preserved.txt',
                     'build/postflight-frozen.txt','build/postflight-fixtures.txt','build/postflight-old-preserved.txt',
                     f'runs/{phase}/frozen-before.txt',f'runs/{phase}/frozen-after.txt',
                     f'runs/{phase}/fixtures-before.txt',f'runs/{phase}/fixtures-after.txt']:
        lines=(folder/relative).read_text().splitlines()
        assert lines and all(line.endswith(': OK') for line in lines),relative
    baseline=folder/('old-baseline.sha256' if node=='node2' else 'build/old-preservation.sha256')
    original_paths=[line.split('  ',1)[1] for line in baseline.read_text().splitlines()]
    checked_paths=[line[:-4] for line in (folder/'build/postflight-old-preserved.txt').read_text().splitlines()]
    assert original_paths==checked_paths
    identity=json.loads((folder/'build/identity.json').read_text())
    assert identity['whole_core_identical'] and identity['instructions_identical']
    assert len(identity['six_functions'])==6 and identity['whole_core_bytes']==35936
    log=(folder/f'runs/{phase}/probe.log').read_text()
    assert 'SMALL_ALL PASS epilogues=1800' in log and 'PASS private_G4_confirm ' in log
    resources=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in log.splitlines() if line.startswith('RESOURCE ')][:6]
    assert len(resources)==6
    for r in resources:
        assert int(r['maxThreads'])>=1024 and int(r['local'])==0
        assert int(r['requested'])==16384<=int(r['dynamic_limit'])
        assert int(r['shared'])+int(r['requested'])<=65536
    samples=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in log.splitlines() if line.startswith('SAMPLE ')]
    summaries=[dict(re.findall(r'(\w+)=([^ ]+)',line)) for line in log.splitlines() if line.startswith('RESULT ')]
    assert len(samples)==220 and len(summaries)==20
    for name,(i,n,bits) in geometries.items():
        shape=[r for r in samples if r['name']==name]
        assert len(shape)==44
        assert log.count(f'VALID name={name} phase=before ')==1
        assert log.count(f'VALID name={name} phase=after ')==1
        assert log.count(f'LAST_ALL_MODES name={name} ')==1
        for sweep in range(11):
            sweep_rows=[r for r in shape if int(r['sweep'])==sweep]
            assert [r['path'] for r in sweep_rows]==(list('DCBA') if sweep%2==0 else list('ABCD'))
            assert all(r['all_clones_last_Y_exact']==r['full_cpu']=='1' for r in sweep_rows)
        packed=i*n*bits//8;clones=((256<<20)+packed-1)//packed
        for path in 'ABCD':
            result=next(r for r in summaries if r['name']==name and r['path']==path)
            assert int(result['clones'])==clones and int(result['packed_resident'])==packed*clones>=256<<20
            for clock in ('event','wall'):
                values=sorted(float(r[clock+'_us']) for r in shape if r['path']==path)
                assert len(values)==11
                for statistic,index in [('min',0),('median',5),('p90',9)]:
                    assert math.isclose(float(result[f'{clock}_{statistic}_us']),values[index],rel_tol=1e-9,abs_tol=1e-8)
    rows.append(dict(node=node,phase=phase,sources=source_count,samples=220,results=20,
                     exact_statistics=True,all_modes_and_clones=True,positive_physical_dynamic_limits=True,
                     immutable_payload_and_restored_old_bytes=True))
result=json.loads((data/'results.json').read_text())
assert result['private_direct_D_proposal_only']==['wo_b','inverse']
assert not result['activated']
print(json.dumps(dict(passed=True,campaigns=rows,physical_campaigns=2,gpu_executed_by_audit=False,
                     caveat='Original node1 host executable was rewritten during preparation and restored exactly; uninterrupted immutability is not claimed'),indent=2))
