"""CPU-only statistics and proposal gate. Preserves all cells and all losses."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import argparse,hashlib,json,math
from provenance import full,ref,read,keys as strict_keys
def summarize(path):
    text=path.read_text();assert 'PASS physical performance campaign all fixed cells;' in text and 'STOP ' not in text
    raw=[json.loads(line[7:]) for line in text.splitlines() if line.startswith('SAMPLE ')]
    cells={};keys=set()
    for row in raw:
        strict_keys(row,'input G mode variant sweep clone event_ms wall_ms')
        assert row['input'] in (5120,2304) and row['G'] in (4,8,16,32) and row['mode'] in ('direct','graph') and row['variant'] in ('V13','new')
        assert 0<=row['sweep']<11 and 0<=row['clone']<64//row['G']
        key=tuple(row[k] for k in ('input','G','mode','variant','sweep','clone'));assert key not in keys;keys.add(key)
        assert all(math.isfinite(row[k]) and row[k]>0 for k in ('event_ms','wall_ms'))
    for i in (5120,2304):
        for g in (4,8,16,32):
            for mode in ('direct','graph'):
                cell={}
                for timer in ('event_ms','wall_ms'):
                    samples={}
                    for v in ('V13','new'):
                        samples[v]=[]
                        for sweep in range(11):
                            rows=[r for r in raw if (r['input'],r['G'],r['mode'],r['variant'],r['sweep'])==(i,g,mode,v,sweep)]
                            assert len(rows)==64//g
                            # Mean across all cold clones = milliseconds per Batch G, never per pool64.
                            samples[v].append(math.fsum(r[timer] for r in rows)/len(rows))
                        values=sorted(samples[v]);cell[timer+'_'+v]=dict(min=values[0],median=values[5],p90=values[math.ceil(.9*len(values))-1],sweeps=samples[v])
                    cell[timer+'_gain']=1-sorted(samples['new'])[5]/sorted(samples['V13'])[5]
                    cell[timer+'_paired_sweep_gains']=[1-n/o for o,n in zip(samples['V13'],samples['new'])]
                cells[f'{i}/G{g}/{mode}']=cell
    return dict(log_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),raw=raw,cells=cells)
def proposal(campaigns):
    assert len(campaigns)==2 and campaigns[0]['log_sha256']!=campaigns[1]['log_sha256']
    direct=all(c[t+'_gain']>=.02 for run in campaigns for name,c in run['cells'].items() if name.endswith('/direct') for t in ('event_ms','wall_ms'))
    graph_negative=any(c[t+'_gain']<0 for run in campaigns for name,c in run['cells'].items() if name.endswith('/graph') for t in ('event_ms','wall_ms'))
    return dict(performance_eligible=direct,proposal_mode=('direct-only' if graph_negative else 'direct-primary_graph-secondary') if direct else 'NONE',
                graph_losses_preserved=True,selection='ALL fixed cells, no posthoc fast-mode selection')
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('logs',type=Path,nargs='+');p.add_argument('--root-full-pass-receipts',type=Path,nargs=2);a=p.parse_args()
    runs=[summarize(path) for path in a.logs];result=dict(campaigns=runs,proposal='BLOCKED_TWO_ROOT_PHYSICAL_FULL_PASS_REQUIRED')
    if a.root_full_pass_receipts:
        assert len(runs)==2
        receipts=[full(path,log) for path,log in zip(a.root_full_pass_receipts,a.logs)]
        first,second=receipts
        assert [first['campaign'],second['campaign']]==[1,2]
        assert first['physical_id']!=second['physical_id'] and first['label']!=second['label']
        assert second['previous_physical_id']==first['physical_id']
        freezes=[ref(a['freeze'],ref(read(r)['admission'],r.parent).parent) for a,r in zip(receipts,a.root_full_pass_receipts)]
        source_pins=[dict(line.split('  ',1)[::-1] for line in f.read_text().splitlines())['source.sha256'] for f in freezes]
        assert source_pins[0]==source_pins[1]
        result['proposal']=proposal(runs)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
