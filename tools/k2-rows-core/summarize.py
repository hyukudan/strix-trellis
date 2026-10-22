"""Chronological RAW extraction, fixed nearest-rank p90 and campaign-1 selection."""
import argparse,csv,hashlib,json,math,re,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('logs',nargs='+',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
all_rows=[];groups=[]
for campaign,path in enumerate(a.logs,1):
    text=path.read_text();complete='PASS k2_rows_probe' in text
    samples={}
    for line in text.splitlines():
        if line.startswith('SAMPLE '):
            d=dict(part.split('=',1) for part in line.split()[1:]);d={k:(float(v) if k=='us' else v if k=='protocol' else int(v)) for k,v in d.items()};d['campaign']=campaign;d['chronological_index']=len(all_rows);all_rows.append(d)
            key=(d['I'],d['N'],d['R'],d['protocol'],d['mode']);samples.setdefault(key,[]).append(d['us'])
    if complete:assert len(samples)==36 and all(len(v)==11 for v in samples.values()) and len([d for d in all_rows if d['campaign']==campaign])==396
    for (I,N,R,protocol,mode),values in samples.items():
        ordered=sorted(values);groups.append(dict(campaign=campaign,I=I,N=N,R=R,S=4,protocol=protocol,mode=mode,count=len(values),min_us=min(values),median_us=statistics.median(values),p90_us=ordered[math.ceil(.9*len(values))-1],complete=complete))
fields=['campaign','chronological_index','I','N','R','S','protocol','mode','sweep','us']
with (a.out/'samples.csv').open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(all_rows)
selection=[]
for I,N in ((8192,5120),(1280,32768)):
    for R in (2,4,8):
        first=[d for d in groups if d['campaign']==1 and d['I']==I and d['N']==N and d['R']==R and d['protocol']=='operator']
        if len(first)!=3:continue
        chosen=min(first,key=lambda d:(d['median_us'],d['mode']))['mode'];entry=dict(I=I,N=N,R=R,S=4,chosen_mode_campaign1=chosen,gains_percent=[])
        qualified=chosen!=0 and len(a.logs)==2
        for campaign in range(1,len(a.logs)+1):
            subset=[d for d in groups if d['campaign']==campaign and d['I']==I and d['N']==N and d['R']==R and d['protocol']=='operator'];lookup={d['mode']:d for d in subset}
            if 0 not in lookup or chosen not in lookup:qualified=False;continue
            gain=100*(1-lookup[chosen]['median_us']/lookup[0]['median_us']);entry['gains_percent'].append(gain);qualified &= gain>=2 and lookup[chosen]['complete']
        entry['timing_threshold_pass']=qualified;selection.append(entry)
result=dict(groups=groups,selection=selection,log_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.logs},qualification='Threshold pass is necessary; source/ISA/integrity/CPU/physical correctness gates remain mandatory.')
(a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(selection,indent=2))
