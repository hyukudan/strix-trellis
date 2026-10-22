"""Preserve every measured pair, including losses and incomplete campaigns."""
import json
import math
from pathlib import Path
import sys

directory=Path(sys.argv[1])
records=[]
completion={}
gates=[]
for turn in ('first','confirm'):
    stem='physical-first' if turn=='first' else 'physical-confirm-retry'
    lines=(directory/f'{stem}.log').read_text().splitlines()
    errors=(directory/f'{stem}.stderr').read_text().splitlines()
    completion[turn]=dict(complete=any(line.startswith('PASS campaign_complete=1') for line in lines),
                          outer=(directory/f'{stem}.outer').read_text().strip(),
                          failures=[line for line in lines+errors if line.startswith('FAIL')])
    for line in lines:
        if line.startswith(('HARDWARE ','GATE ','PASS ')):
            gates.append(dict(turn=turn,raw=line))
        if not line.startswith('RESULT '):
            continue
        row=dict(item.split('=',1) for item in line.split()[1:])
        row['turn']=turn
        for key in ('shape','rows','splits','pair','decoder','clones','cloned_bytes','gpu_bound'):
            row[key]=int(row[key])
        for key in ('min_us','median_us','p90_us'):
            row[key]=float(row[key])
        row['samples']=[float(x) for x in row['samples'].split(',') if x]
        assert len(row['samples'])==11
        ordered=sorted(row['samples'])
        for key,index in (('min_us',0),('median_us',5),('p90_us',9)):
            assert math.isclose(row[key],ordered[index],rel_tol=1e-8)
        assert row['parity']==row['CPU_ref']==row['canaries']=='PASS'
        records.append(row)
    if completion[turn]['complete']:
        assert sum(row['turn']==turn for row in records)==144
pairs=[]
for row in records:
    if not row['decoder']:
        continue
    baseline=[r for r in records if r['turn']==row['turn'] and r['shape']==row['shape'] and
              r['rows']==row['rows'] and r['splits']==row['splits'] and
              r['pair']==row['pair'] and r['mode']==row['mode'] and not r['decoder']]
    assert len(baseline)==1
    pairs.append(dict(turn=row['turn'],shape=row['shape'],rows=row['rows'],splits=row['splits'],
                      decoder=row['decoder'],mode=row['mode'],baseline=baseline[0],candidate=row,
                      ratio=row['median_us']/baseline[0]['median_us'],
                      raw_paired_ratios=[c/b for c,b in zip(row['samples'],baseline[0]['samples'])]))
shapes=('q_b','wo_b','shared w1','shared w2','q_a','wo_a')
table=['| Forma | M | S | Decoder | T1 core | T1 operador | T2 core | T2 operador | Cumple 2 % ambos |',
       '|---|---:|---:|---|---:|---:|---:|---:|---|']
winners=[]
for shape in range(6):
    for rows in (64,128,256):
        for decoder in (1,2):
            matched={(r['turn'],r['mode']):r for r in pairs if r['shape']==shape and r['rows']==rows and r['decoder']==decoder}
            if len(matched)!=4:
                continue
            ratios=[matched[key]['ratio'] for key in (('first','core'),('first','operator'),('confirm','core'),('confirm','operator'))]
            winner=all(completion[t]['complete'] and matched[t,'operator']['ratio']<=.98 for t in ('first','confirm'))
            if winner:winners.append(dict(shape=shapes[shape],rows=rows,decoder=decoder))
            table.append('| '+ ' | '.join([shapes[shape],str(rows),str(matched['first','core']['splits']),('AlignedLane' if decoder==1 else 'SADDR')]+[f'{(r-1)*100:+.2f} %' for r in ratios]+['sí' if winner else 'no'])+' |')
(directory/'results.json').write_text(json.dumps(dict(completion=completion,records=records,pairs=pairs,gates=gates,qualified_2percent_points=winners,
    median_losses=[{k:r[k] for k in ('turn','shape','rows','decoder','mode','ratio')} for r in pairs if r['ratio']>1],
    raw_paired_loss_count=sum(v>1 for r in pairs for v in r['raw_paired_ratios'])),indent=2)+'\n',encoding='utf-8',newline='\n')
(directory/'ratios.md').write_text('\n'.join(table)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(dict(completion=completion,records=len(records),pairs=len(pairs),raw_samples=sum(len(r['samples']) for r in records),qualified_2percent_points=winners,
    median_losses=[{k:r[k] for k in ('turn','shape','rows','decoder','mode','ratio')} for r in pairs if r['ratio']>1],
    raw_paired_loss_count=sum(v>1 for r in pairs for v in r['raw_paired_ratios'])),indent=2))
