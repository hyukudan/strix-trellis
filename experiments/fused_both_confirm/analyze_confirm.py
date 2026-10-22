"""Read preserved campaigns, retain every sample/loss, and fail closed on gates."""
from pathlib import Path
import json
import re
import sys

root=Path(sys.argv[1])
records=[]
for path in sorted(root.glob('node*/runs/*/probe.log')):
    text=path.read_text()
    phase=path.parent.name
    samples=[]; results=[]
    for line in text.splitlines():
        if line.startswith(('SAMPLE ','RESULT ')):
            row=dict(re.findall(r'(\w+)=([^ ]+)',line))
            (samples if line.startswith('SAMPLE ') else results).append(row)
    passed='PASS private_G4_confirm ' in text
    if passed:
        assert len(samples)==220 and len(results)==20
        assert text.count('VALID name=')==10
        assert text.count('LAST_ALL_MODES name=')==5
        assert 'SMALL_ALL PASS epilogues=1800' in text
        assert all(r['all_clones_last_Y_exact']=='1' and r['full_cpu']=='1' for r in samples)
        memory=path.with_name('probe.stderr').read_text()
        rss=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',memory).group(1))
        assert rss<2*1024*1024
        for r in results:
            assert int(r['packed_resident'])>=256*1024*1024
    rows=[]
    for r in results:
        a=next(z for z in results if z['name']==r['name'] and z['path']=='A')
        rows.append(dict(r,event_gain_percent=100*(1-float(r['event_median_us'])/float(a['event_median_us'])),
                         wall_gain_percent=100*(1-float(r['wall_median_us'])/float(a['wall_median_us']))))
    records.append(dict(node=path.parts[-4],phase=phase,passed=passed,samples=samples,results=rows))
proposal=[]
if len(records)==2 and all(r['passed'] for r in records):
    for name in ('q_b','wo_b','inverse','shared','head'):
        rows=[next(z for z in r['results'] if z['name']==name and z['path']=='D') for r in records]
        if all(min(z['event_gain_percent'],z['wall_gain_percent'])>=2 for z in rows):
            proposal.append(name)
print(json.dumps(dict(campaigns=records,private_direct_D_proposal_only=proposal,
                     activated=False,limitation='No qualification/proposal on incomplete or failed campaign; retain all raw losses'),indent=2))
