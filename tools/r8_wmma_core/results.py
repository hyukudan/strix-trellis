"""Preserve all raw measurements and freeze the first-pass FULL choice."""
from pathlib import Path
import argparse
import hashlib
import json
import math

p=argparse.ArgumentParser()
p.add_argument('root',type=Path)
p.add_argument('--select',action='store_true')
a=p.parse_args();root=a.root
records=[];campaigns={}
for name in ('first','confirm'):
    run=root/'runs'/name
    if not (run/'native.log').exists():continue
    lines=(run/'native.log').read_text().splitlines()
    errors=(run/'native.stderr').read_text().splitlines()
    complete=(run/'passed').exists() and any(s.startswith('PASS_R8_CAMPAIGN ') for s in lines)
    campaigns[name]=dict(passed=complete,failed=[s for s in errors+lines if s.startswith('FAIL ')])
    for s in lines:
        if not s.startswith('RESULT '):continue
        r=dict(x.split('=',1) for x in s.split()[1:]);r['campaign']=name
        for k in ('geometry','mode','clones','cold_bytes'):r[k]=int(r[k])
        for k in ('min_us','median_us','p90_us'):r[k]=float(r[k])
        r['raw']=[float(x) for x in r['raw'].split(',') if x]
        assert len(r['raw'])==11
        for k,j in [('min_us',0),('median_us',5),('p90_us',9)]:assert math.isclose(r[k],sorted(r['raw'])[j],rel_tol=1e-9)
        records.append(r)
    if complete:assert sum(r['campaign']==name for r in records)==56
def get(c,g,m,scope='full',clock='event'):
    found=[r for r in records if (r['campaign'],r['geometry'],r['mode'],r['scope'],r['clock'])==(c,g,m,scope,clock)]
    assert len(found)==1
    return found[0]
if a.select:
    assert campaigns['first']['passed'] and 'confirm' not in campaigns
    selection=dict(rule='Lowest first PASS FULL event median C/D versus actual V13 A; exact tie chooses C (AlignedLane Mode1). Fixed before confirm. Require <=0.98 versus A in BOTH new campaigns.',
                   first_log_sha256=hashlib.sha256((root/'runs/first/native.log').read_bytes()).hexdigest(),choices=[])
    for g in range(4):
        mode=min((2,3),key=lambda m:(get('first',g,m)['median_us'],m))
        selection['choices'].append(dict(geometry=g,mode=mode,decoder_mode=mode-1,
            first_ratio=get('first',g,mode)['median_us']/get('first',g,0)['median_us']))
    path=root/'selection.json'
    with path.open('x') as f:json.dump(selection,f,indent=2);f.write('\n')
    (root/'selection.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'  selection.json\n')
    print(json.dumps(selection,indent=2))
pairs=[]
for r in records:
    if r['scope']=='core' and r['mode']==1 or r['scope']=='full' and r['mode']==0:continue
    baseline=get(r['campaign'],r['geometry'],0 if r['scope']=='full' else 1,r['scope'],r['clock'])
    pairs.append(dict(campaign=r['campaign'],geometry=r['geometry'],mode=r['mode'],scope=r['scope'],clock=r['clock'],
        median_ratio=r['median_us']/baseline['median_us'],
        raw_ratios=[x/y for x,y in zip(r['raw'],baseline['raw'])],
        losses=sum(x>y for x,y in zip(r['raw'],baseline['raw']))))
winners=[]
if (root/'selection.json').exists():
    selection=json.loads((root/'selection.json').read_text())
    if all(campaigns.get(c,{}).get('passed',False) for c in ('first','confirm')):
        for choice in selection['choices']:
            g,m=choice['geometry'],choice['mode']
            ratios=[get(c,g,m)['median_us']/get(c,g,0)['median_us'] for c in ('first','confirm')]
            if max(ratios)<=.98:winners.append(dict(**choice,ratios=ratios))
result=dict(campaigns=campaigns,records=records,pairs=pairs,confirmed_proposals_only=winners,
            API_causal_claim=False,global_CAPI_required=True)
(root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(campaigns=campaigns,series=len(records),samples=sum(len(r['raw']) for r in records),confirmed_proposals=winners),indent=2))
