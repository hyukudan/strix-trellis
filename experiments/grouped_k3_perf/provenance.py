"""Shared strict CPU provenance gate. No HIP loading; no lease acquisition."""
from pathlib import Path
import hashlib,json,re,math
from collections import Counter
PINS=json.loads(Path(__file__).with_name('provenance-pins.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def keys(value,names):assert set(value)==set(names.split()), ('schema keys',set(value))
def ref(value,base):
    keys(value,'path sha256');assert re.fullmatch('[0-9a-f]{64}',value['sha256'])
    p=Path(value['path']);p=p if p.is_absolute() else Path(base)/p
    assert sha(p)==value['sha256'],str(p)
    return p

def evidence(scope):
    for name,digest in PINS['qa'].items():assert sha(scope/'data/qa'/name)==digest,name
    for name,digest in PINS['review'].items():assert sha(scope/'data/review'/name)==digest,name
    q=read(scope/'data/qa/summary.json')
    assert q['passed'] and q['configs']==14 and q['negative_capture_cases']==46 and q['observed_attribute_calls']==7
    assert len(q['after_creation_attribute_deltas'])==6 and all(x.endswith('delta=0') for x in q['after_creation_attribute_deltas'])
    assert sha(scope/'data/cpu/manifest.json')==PINS['cpu']

def frozen(manifest):
    scope=manifest.resolve().parent
    rows={}
    for line in manifest.read_text().splitlines():
        digest,name=line.split('  ',1);assert name not in rows and not Path(name).is_absolute() and '..' not in Path(name).parts
        assert not name.startswith('runs/');assert sha(scope/name)==digest,name;rows[name]=digest
    actual={p.relative_to(scope).as_posix() for p in scope.rglob('*') if p.is_file() and 'runs' not in p.relative_to(scope).parts[:1]}
    assert actual==set(rows)|{'frozen.sha256'}, ('unlisted or missing frozen files',actual^(set(rows)|{'frozen.sha256'}))
    for name in ('bin/perf_native','lib/libstrix_trellis.a','source.tar','source.sha256','data/cpu/manifest.json','build/identity.json','build/heap-link.json'):
        assert name in rows,name
    assert rows['lib/libstrix_trellis.a']==PINS['library'];evidence(scope)
    proof=read(scope/'build/identity.json');assert proof['binary_sha256']==rows['bin/perf_native']
    assert proof['reference_binary']==PINS['qa_binary'] and len(proof['images'])==8
    assert all(x['whole_image_identical'] for x in proof['images'][1:])
    assert read(scope/'build/heap-link.json')['status']=='PASS_LINK_BOUNDARY_ONLY'
    return scope,rows

def admission(receipt,manifest):
    r=read(receipt);base=receipt.parent
    keys(r,'schema kind node physical_id campaign label inner_seconds outer_seconds previous_physical_id freeze root_review parent_meta')
    assert r['schema']==2 and r['kind']=='ROOT_ADMISSION_GROUPED_K3_R1'
    assert r['node'] in ('node1','node2') and r['campaign'] in (1,2)
    assert re.fullmatch('ROOT-grouped-k3-perf-r1-[A-Za-z0-9_-]+',r['label'])
    assert re.fullmatch('[A-Za-z0-9_-]{8,128}',r['physical_id'])
    assert r['inner_seconds']==270 and 0<r['outer_seconds']<=300
    assert (r['campaign']==1 and r['previous_physical_id'] is None) or (r['campaign']==2 and isinstance(r['previous_physical_id'],str) and r['previous_physical_id']!=r['physical_id'])
    assert ref(r['freeze'],base).resolve()==manifest.resolve()
    scope,rows=frozen(manifest)
    review=read(ref(r['root_review'],base))
    keys(review,'schema kind freeze_sha256 review_pins qa_pins decision')
    assert review==dict(schema=2,kind='ROOT_HOST_REVIEW',freeze_sha256=sha(manifest),review_pins=PINS['review'],qa_pins=PINS['qa'],decision='APPROVED_HOST_POOL_TIMING')
    meta=read(ref(r['parent_meta'],base))
    keys(meta,'schema node physical_id campaign label outer_seconds started_epoch')
    assert meta==dict(schema=2,**{k:r[k] for k in ('node','physical_id','campaign','label','outer_seconds')},started_epoch=meta['started_epoch'])
    assert isinstance(meta['started_epoch'],(int,float)) and meta['started_epoch']>0
    assert (scope/'build/host.txt').read_text().strip()==r['node']
    return r

def raw_pass(log):
    text=log.read_text();assert 'STOP' not in text and 'FAIL' not in text
    assert text.count('PASS physical performance campaign all fixed cells; contexts=2 calls=14')==1
    samples=[json.loads(line[7:]) for line in text.splitlines() if line.startswith('SAMPLE ')]
    expected={(i,g,mode,variant,sweep,clone) for i in (5120,2304) for g in (4,8,16,32) for mode in ('direct','graph') for variant in ('V13','new') for sweep in range(11) for clone in range(64//g)}
    observed=set()
    for row in samples:
        keys(row,'input G mode variant sweep clone event_ms wall_ms')
        key=tuple(row[k] for k in ('input','G','mode','variant','sweep','clone'))
        assert key in expected and key not in observed;observed.add(key)
        assert all(math.isfinite(row[k]) and row[k]>0 for k in ('event_ms','wall_ms'))
    assert observed==expected and len(samples)==2640
    assert text.count('PASS_VERIFY ')==216
    for i in (5120,2304):
        for g in (4,8,16,32):assert text.count(f'PASS_VERIFY input={i} G={g} coverage=64 ')==27
    assert text.count('attribute_event=')==14
    assert 'logical_creations=2 contexts_with_handle=2 contexts_admitted=2 calls=14 stored=14 overflow=0 failed=0 first_error=0' in text
    for phase in ('prepare','submit','capture','replay','reuse','cleanup'):
        assert re.search(r'attribute_zero_region phase='+phase+r' regions=\d+ delta=0(?:\n|$)',text)
    assert text.count('ledger explicit_gpu_bytes=')==2
    assert text.count('GRAPH clone_start=')==1716
    assert text.count('SUMABS id=')==216*64
    assert text.count('stage=Yordered ')==216*64
    assert text.count('stage=YoriginalFP64 ')==216*64
    assert text.count('stage=P64 ')==108*64*20
    previous=0;seen=[]
    pattern=r'PASS_VERIFY input=(5120|2304) G=(4|8|16|32) coverage=64 ownership=PASS guards=PASS immutable=PASS RSS=PASS ledger=PASS A_P_Y=PASS\n'
    for match in re.finditer(pattern,text):
        i,g=map(int,match.groups());seen.append((i,g));part=text[previous:match.start()];previous=match.end()
        checks=[line for line in part.splitlines() if line.startswith('CHECK ')]
        expected={(j,split,'P64') for j in range(64) for split in range(16 if i==5120 else 4)}
        expected|={(j,-1,stage) for j in range(64) for stage in ('Yordered','YoriginalFP64')}
        actual=set()
        for line in checks:
            fields=dict(item.split('=',1) for item in line.split()[1:])
            keys(fields,'id split stage maxabs nrms maxref worst enforce')
            key=(int(fields['id']),int(fields['split']),fields['stage']);assert key in expected and key not in actual;actual.add(key)
            maximum,nrms,maxref=[float(fields[k]) for k in ('maxabs','nrms','maxref')]
            assert all(math.isfinite(x) and x>=0 for x in (maximum,nrms,maxref))
            assert int(fields['enforce'])==(fields['stage']!='YoriginalFP64')
            if fields['enforce']=='1':assert maximum<=5e-4*max(1,maxref) and nrms<=3e-5
        assert actual==expected
        sums=[dict(x.split('=',1) for x in line.split()[1:]) for line in part.splitlines() if line.startswith('SUMABS ')]
        assert len(sums)==64 and {int(x['id']) for x in sums}==set(range(64))
        assert all(x['empirical_only']=='1' and math.isfinite(float(x['ratio'])) and float(x['ratio'])>=0 for x in sums)
    assert seen==[(i,g) for i in (5120,2304) for g in (4,8,16,32) for _ in range(27)]
    expected_graph=Counter((c*g,v,index) for i in (5120,2304) for g in (4,8,16,32) for v in (0,1) for c in range(64//g) for index in range(3 if v else 3*g))
    observed_graph=Counter()
    for line in text.splitlines():
        if not line.startswith('GRAPH '):continue
        f=dict(x.split('=',1) for x in line.split()[1:]);keys(f,'clone_start variant index func grid block shared')
        assert int(f['func'],16)>0 and all(int(x)>0 for x in f['grid'].split(',')) and f['block']=='128' and f['shared']=='0'
        observed_graph[tuple(int(f[k]) for k in ('clone_start','variant','index'))]+=1
    assert observed_graph==expected_graph
    events=[dict(x.split('=',1) for x in line.split()) for line in text.splitlines() if line.startswith('attribute_event=')]
    assert [int(e['attribute_event']) for e in events]==list(range(14))
    for creation in (1,2):
        ev=[e for e in events if e['creation']==str(creation)]
        assert len(ev)==7 and len({e['context'] for e in ev})==1
        assert len({e['function'] for e in ev})==7
        assert all(e['valid']=='1' and e['result']=='0' and int(e['context'],16)>0 and int(e['function'],16)>0 for e in ev)
        assert [e['phase'] for e in ev]==['creation']*4+['math']*3
    for line in text.splitlines():
        if line.startswith('ledger '):
            f=dict(x.split('=',1) for x in line.split()[1:])
            assert 0<int(f['explicit_gpu_bytes'])<1024**3 and int(f['packed_resident'])==283115520
    return text

def full(receipt,log):
    r=read(receipt);base=receipt.parent
    keys(r,'schema kind admission log stderr child_close parent_close parent_log child_pass')
    assert r['schema']==2 and r['kind']=='ROOT_PHYSICAL_FULL_PASS_GROUPED_K3_R1'
    ap=ref(r['admission'],base);a=read(ap);mp=ref(a['freeze'],ap.parent);admission(ap,mp)
    assert ref(r['log'],base).resolve()==log.resolve();raw_pass(log)
    stderr=ref(r['stderr'],base).read_text();assert 'STOP' not in stderr
    rss=re.search(r'Maximum resident set size \(kbytes\): (\d+)',stderr)
    assert rss and 0<int(rss[1])*1024<2*1024**3
    assert ref(r['child_close'],base).read_text()=='payload_exit=0 integrity_exit=0\n'
    assert ref(r['child_pass'],base).read_text()==a['physical_id']+' '+a['label']+' PASS\n'
    closure=read(ref(r['parent_close'],base));meta=read(ref(a['parent_meta'],ap.parent))
    keys(closure,'schema physical_id node campaign label started_epoch ended_epoch exit child_exit admission_sha256 child_log_sha256 parent_log_sha256')
    assert closure['schema']==2
    for k in ('physical_id','node','campaign','label'):assert closure[k]==a[k]
    assert closure['started_epoch']==meta['started_epoch']
    assert 0<=closure['ended_epoch']-closure['started_epoch']<=a['outer_seconds']<=300
    assert closure['exit']==0 and closure['child_exit']==0
    assert closure['admission_sha256']==sha(ap) and closure['child_log_sha256']==sha(log)
    parentlog=ref(r['parent_log'],base);assert sha(parentlog)==closure['parent_log_sha256']
    assert 'STOP' not in parentlog.read_text() and 'FAIL' not in parentlog.read_text()
    return a

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Verify full immutable scope, excluding runs only; CPU only')
    parser.add_argument('--manifest',type=Path,required=True)
    frozen(parser.parse_args().manifest)
    print('PASS immutable scope and pinned QA/review; new native gates remain OPEN')
