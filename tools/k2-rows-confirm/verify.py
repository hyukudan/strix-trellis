"""Strict new-campaign gate and unchanged event statistics, fixed ROOT selection."""
import argparse,csv,hashlib,json,math,re,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs',nargs='+',type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
assert 1<=len(a.runs)<=2
allrows=[];groups=[];campaigns=[]
for campaign,run in enumerate(a.runs,1):
    text=(run/'benchmark.log').read_text();err=(run/'benchmark.stderr').read_text()
    assert 'payload_exit=0 integrity_exit=0 old_integrity_exit=0' in (run/'closed.txt').read_text()
    assert (run/'fresh-label.txt').read_text().strip()==f'strix-trellis-k2-rows-confirm-new-{campaign}'
    assert text.count('PASS k2_rows_probe')==1 and 'FAIL' not in text+err
    assert 'EXHAUSTIVE PASS hashes=524288 halves=524288 reader_states=196864' in text
    assert 'R1_IDENTITY PASS no timing' in text
    small=set(tuple(map(int,m)) for m in re.findall(r'SMALL PASS I=(\d+) N=(\d+) R=(\d+) S=1..32 layouts=3',text))
    assert small=={(I,N,R) for I in (16,80,128,176,272,112,240) for N in (16,32,48,80) for R in (2,4,8)}
    assert 'SMALL_ALL PASS cases=8064 modes=3' in text
    coverage=re.findall(r'COVERAGE PASS mode=(\d) R=(\d) chunk_mask=(\d+) tail_mask=(\d+) original_cases=5760 added_cases=2304',text)
    assert set(tuple(map(int,m)) for m in coverage)=={(m,R,510,15) for m in range(3) for R in (2,4,8)} and len(coverage)==9
    assert set(tuple(map(int,m)) for m in re.findall(r'LARGE_ROWS PASS I=(\d+) N=(\d+) R=(\d+) all_splits_columns=1 all_clones_before_after_last=1',text))=={(I,N,R) for I,N in ((8192,5120),(1280,32768)) for R in (2,4,8)}
    large=re.findall(r'LARGE PASS I=(\d+) N=(\d+) partial_ratio=(\S+) nrms=(\S+) maxabs_relative=(\S+) gpu_peak_bytes=(\d+)',text)
    assert len(large)==2
    for I,N,partial,nrms,relative,peak in large:
        assert float(partial)<=2e-6 and float(nrms)<=3e-5 and float(relative)<=5e-4 and int(peak)<1<<30
    rss=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',err)[1]);assert rss<2*1024*1024
    assert 'Exit status: 0' in err
    streams={}
    results={}
    for line in text.splitlines():
        if line.startswith(('SAMPLE ','HOST_SAMPLE ')):
            metric='host' if line.startswith('HOST_') else 'event'
            d=dict(part.split('=',1) for part in line.split()[1:]);d={k:float(v) if k=='us' else v if k=='protocol' else int(v) for k,v in d.items()}
            assert math.isfinite(d['us']) and d['us']>0 and d['S']==4
            key=(metric,d['I'],d['N'],d['R'],d['protocol'],d['mode'])
            streams.setdefault(key,[]).append(d)
            allrows.append(dict(campaign=campaign,metric=metric,chronological_index=len(allrows),**d))
        elif line.startswith('RESULT '):
            d=dict(part.split('=',1) for part in line.split()[1:]);key=(int(d['I']),int(d['N']),int(d['R']),d['protocol'],int(d['mode']))
            assert int(d['clones'])==26 and int(d['resident_bytes'])==272629760
            assert key not in results;results[key]=d
    assert len(streams)==72 and len(results)==36
    expected={(metric,I,N,R,protocol,m) for metric in ('host','event') for I,N in ((8192,5120),(1280,32768)) for R in (2,4,8) for protocol in ('core','operator') for m in range(3)}
    assert set(streams)==expected
    for key,rows in streams.items():
        assert [d['sweep'] for d in rows]==list(range(11))
        values=sorted(d['us'] for d in rows);mn=values[0];med=values[5];p90=values[9]
        metric,I,N,R,protocol,mode=key
        if metric=='event':
            d=results[I,N,R,protocol,mode]
            for name,val in [('min_us',mn),('median_us',med),('p90_us',p90)]:assert abs(float(d[name])-val)<1e-8
        groups.append(dict(campaign=campaign,metric=metric,I=I,N=N,R=R,S=4,protocol=protocol,mode=mode,count=11,min_us=mn,median_us=med,p90_us=p90))
    for metric in ('host','event'):
        chronological=[d for d in allrows if d['campaign']==campaign and d['metric']==metric]
        expected=[(I,N,R,protocol,sweep,m) for I,N in ((8192,5120),(1280,32768)) for R in (2,4,8) for protocol in ('core','operator') for sweep in range(11) for m in ((2,1,0) if sweep%2 else (0,1,2))]
        assert [(d['I'],d['N'],d['R'],d['protocol'],d['sweep'],d['mode']) for d in chronological]==expected
    campaigns.append({'campaign':campaign,'PASS':True,'node':(run/'meta.txt').read_text().splitlines()[0],'rss_kib':rss,'log_sha256':hashlib.sha256(text.encode()).hexdigest(),'large':large})
selection=[]
for I,N in ((8192,5120),(1280,32768)):
    for R in (2,4,8):
        gains=[]
        for campaign in range(1,len(a.runs)+1):
            lookup={d['mode']:d for d in groups if d['metric']=='event' and d['campaign']==campaign and (d['I'],d['N'],d['R'],d['protocol'])==(I,N,R,'operator')}
            gains.append(100*(1-lookup[2]['median_us']/lookup[0]['median_us']))
        selection.append(dict(I=I,N=N,R=R,S=4,preselected_mode=2,gains_percent=gains,qualified=len(a.runs)==2 and all(g>=2 for g in gains)))
a.out.mkdir(parents=True,exist_ok=True)
with (a.out/'samples.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,['campaign','metric','chronological_index','I','N','R','S','protocol','mode','sweep','us']);w.writeheader();w.writerows(allrows)
result=dict(PASS=True,campaigns=campaigns,groups=groups,selection=selection,prior_attempts='EXPLORATION ONLY; immutable empty selection unchanged')
(a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'campaigns':campaigns,'selection':selection},indent=2))
