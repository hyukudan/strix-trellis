"""Summarize raw per-operator events/walls, preserving every sweep, loss and tail."""
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

def summary(values):
    values=sorted(values);return dict(min=values[0],median=statistics.median(values),p90=values[int(.9*(len(values)-1))],samples=len(values))
def summarize(path):
    grouped=defaultdict(list);chronology=[]
    for line in path.read_text().splitlines():
        if not line.startswith('SAMPLE,'):continue
        _,fixture,rows,candidate,variant,sweep,clone,event,wall=line.split(',')
        record=dict(fixture=fixture,rows=int(rows),candidate=int(candidate),variant=int(variant),sweep=int(sweep),clone=int(clone),event_us=float(event),wall_us=float(wall))
        assert record['event_us']>0 and record['wall_us']>0
        chronology.append(record);grouped[(fixture,int(rows),int(candidate),int(variant))].append(record)
    assert len(grouped)==268 # 134 comparison points, old/new at every point
    results=[]
    for key,records in sorted(grouped.items()):
        assert {(r['sweep'],r['clone']) for r in records}=={(s,c) for s in range(11) for c in range(21)} and len(records)==231
        entry=dict(fixture=key[0],rows=key[1],candidate=key[2],variant=key[3],event=summary([r['event_us'] for r in records]),wall=summary([r['wall_us'] for r in records]),
                   sweeps=[dict(sweep=s,event=summary([r['event_us'] for r in records if r['sweep']==s]),wall=summary([r['wall_us'] for r in records if r['sweep']==s])) for s in range(11)])
        entry['unique_packed_GBs_event']=13107200/entry['event']['median']/1000
        results.append(entry)
    losses=[]
    for r in chronology:
        if r['variant']==0:continue
        baseline=next(b for b in grouped[(r['fixture'],r['rows'],r['candidate'],0)] if (b['sweep'],b['clone'])==(r['sweep'],r['clone']))
        losses.append(dict(fixture=r['fixture'],rows=r['rows'],candidate=r['candidate'],sweep=r['sweep'],clone=r['clone'],
            event_speedup=baseline['event_us']/r['event_us'],wall_speedup=baseline['wall_us']/r['wall_us']))
    return dict(raw_chronology=chronology,results=results,paired_all_losses_and_tails=losses,
                normalization='one logical operator with R real rows per sample; never divide pool time by 21',device_global_subtraction=False)
if __name__=='__main__':print(json.dumps(summarize(Path(sys.argv[1])),indent=2))
