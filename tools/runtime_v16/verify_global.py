"""Validate ROOT's completed global run records on CPU; no benchmark execution."""
import argparse
import json
import hashlib
import math
import re
from pathlib import Path
from collections import Counter

def verify_graph_records(rows):
    assert rows and rows[-1]=='PASS complete_operator_graphs'
    assert sum(line.startswith('SETUP ') for line in rows)==20
    results=[line for line in rows if line.startswith('RESULT ')]
    assert len(results)==60
    modes=Counter(line.split()[1] for line in results)
    assert modes=={'mode=direct':20,'mode=graph_individual':20,'mode=graph_rotation':20}

def verify_head_terminal(rows):
    assert rows and rows[-1]=='PASS sampled_head_operator (sampled numerical coverage)'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args();r=a.run
    def lines(name):return (r/name).read_text(encoding="utf-8").splitlines()
    def count(name,prefix):return sum(s.startswith(prefix) for s in lines(name))
    assert lines('passed.txt')==['PASS_runtime_v16_candidate']
    assert not (r/'failed.txt').exists()
    for directory in ['head-full','head-full-v14b']:
        hashes=lines(directory+'.sha256')
        assert len(hashes)==16
        for line in hashes:
            digest,name=line.split(maxsplit=1)
            assert hashlib.sha256((r/directory/Path(name).name).read_bytes()).hexdigest()==digest
    assert len(lines('fixtures.sha256'))==84
    assert lines('correctness.log')[-1].startswith('PASS route=all fixtures=84 ')
    assert count('api-v16.log','RESULT ')==156
    assert lines('api-v16.log')[-1]=='PASS complete_operator'
    verify_graph_records(lines('graphs.log'))
    causal_api=causal_head=0
    for rep in range(3):
        for version in ('v14b','v16'):
            for index in range(10):
                name=f'causal/r{rep}-f{index}-{version}.log'
                assert lines(name)[-1]=='PASS complete_operator'
                causal_api+=count(name,'RESULT ')
            name=f'causal/r{rep}-head-{version}.log'
            verify_head_terminal(lines(name))
            causal_head+=count(name,'HEAD_RESULT ')
    assert causal_api==732 and causal_head==48
    for rows in range(1,9):
        candidate=r/'head-full'/f'r{rows}-a.f16';old=r/'head-full-v14b'/candidate.name
        assert candidate.stat().st_size==rows*5120*2 and candidate.read_bytes()==old.read_bytes()
        candidate=r/'head-full'/f'r{rows}-y.f32';old=r/'head-full-v14b'/candidate.name
        assert candidate.stat().st_size==rows*129280*4
        assert candidate.read_bytes()==old.read_bytes()
    independent=json.loads((r/'head-independent.json').read_text(encoding="utf-8"))
    assert independent['full_CPU_gate']=='PASS'
    assert {x['source'] for x in independent['references']}=={'CPU','CUDA'}
    for x in independent['references']:
        assert x['aggregate']['values']==1034240 and x['aggregate']['passed']
        assert len(x['rows'])==8 and all(y['passed'] and y['values']==129280 for y in x['rows'])
    assert len(lines('engine-comparison.jsonl'))==2
    for line in lines('engine-comparison.jsonl'):
        engine=json.loads(line)
        assert engine['passed'] and engine['mode'] in ['plain','bias']
        raw=(r/('engine-'+engine['mode']+'.f32')).read_bytes()
        assert len(raw)==2048 and hashlib.sha256(raw).hexdigest()==engine['sha256']
        assert engine['nrms']<=3e-5 and engine['maxabs']<=5e-4*max(1,engine['maxref'])
    for name in ['correctness.log','api-v16.log','graphs.log']:
        for line in lines(name):
            fields=dict(re.findall(r'(\w+)=([^ ]+)',line))
            if 'maxabs' in fields and 'nrmse' in fields:
                values={key:float(fields[key]) for key in ['maxabs','nrmse','maxref']}
                assert all(math.isfinite(x) for x in values.values())
                assert values['nrmse']<=3e-5
                assert values['maxabs']<=5e-4*max(1,values['maxref'])
    print(json.dumps(dict(state='PHYSICAL_GLOBAL_RECORDS_PASS',fixtures=84,api_cases=156,graph_cases=20,graph_result_records=60,causal_pairs=3,
        causal_api_records=causal_api,causal_head_records=causal_head,head_A_R1_R8_identical_V14b=True,
        head_Y_R1_R8_identical_V14b=True,head_R8_full_CPU_CUDA_tolerance=True),indent=2))

if __name__=='__main__':main()
