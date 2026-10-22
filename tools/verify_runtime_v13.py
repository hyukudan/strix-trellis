"""Validate ROOT's completed global run records on CPU; no benchmark execution."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--causal-root',type=Path,help='Separate immutable raw logs during offline collection')
    p.add_argument('--head-root',type=Path,help='Detached full binary output archive')
    p.add_argument('--previous-head-root',type=Path,help='Detached previous-version full binary output archive')
    a=p.parse_args();r=a.run
    def lines(name):
        path=a.causal_root/name[len('causal/'):] if a.causal_root and name.startswith('causal/') else r/name
        return path.read_text().splitlines()
    def count(name,prefix):return sum(s.startswith(prefix) for s in lines(name))
    assert len(lines('fixtures.sha256'))==84
    assert lines('correctness.log')[-1].startswith('PASS route=all fixtures=84 ')
    assert count('api-v13.log','RESULT ')==156
    assert lines('api-v13.log')[-1]=='PASS complete_operator'
    assert count('graphs.log','SETUP ')==20
    assert count('graphs.log','RESULT ')==60 and lines('graphs.log')[-1]=='PASS complete_operator_graphs'
    for mode in ('direct','graph_individual','graph_rotation'):
        assert count('graphs.log','RESULT mode='+mode+' ')==20
    causal_api=causal_head=0
    for rep in range(3):
        for version in ('v12','v13'):
            for index in range(10):
                name=f'causal/r{rep}-f{index}-{version}.log'
                assert lines(name)[-1]=='PASS complete_operator'
                causal_api+=count(name,'RESULT ')
            name=f'causal/r{rep}-head-{version}.log'
            assert lines(name)[-1]=='PASS sampled_head_operator (sampled numerical coverage)'
            causal_head+=count(name,'HEAD_RESULT ')
    assert causal_api==732 and causal_head==48
    candidate_head=a.head_root if a.head_root else r/'head-full'
    previous_head=a.previous_head_root if a.previous_head_root else r/'head-full-v12'
    for rows in range(1,9):
        candidate=candidate_head/f'r{rows}-a.f16';old=previous_head/candidate.name
        assert candidate.stat().st_size==rows*5120*2 and candidate.read_bytes()==old.read_bytes()
        candidate=candidate_head/f'r{rows}-y.f32';old=previous_head/candidate.name
        assert candidate.stat().st_size==rows*129280*4
        if rows<8:assert candidate.read_bytes()==old.read_bytes()
    independent=json.loads((r/'head-independent.json').read_text())
    assert independent['full_CPU_gate']=='PASS'
    assert {x['source'] for x in independent['references']}=={'CPU','CUDA'}
    for x in independent['references']:
        assert x['aggregate']['values']==1034240 and x['aggregate']['passed']
        assert len(x['rows'])==8 and all(y['passed'] and y['values']==129280 for y in x['rows'])
    assert len(lines('engine-comparison.jsonl'))==2
    for line in lines('engine-comparison.jsonl'):assert json.loads(line)['passed']
    print(json.dumps(dict(state='PHYSICAL_GLOBAL_RECORDS_PASS',fixtures=84,api_cases=156,graph_cases=20,causal_pairs=3,
        causal_api_records=causal_api,causal_head_records=causal_head,head_A_R1_R8_identical_V12=True,
        head_Y_R1_R7_identical_V12=True,head_R8_full_CPU_CUDA_tolerance=True),indent=2))

if __name__=='__main__':main()
