"""SYNTHETIC CPU pair-policy unit test. Full receipt validation is mocked here only."""
from pathlib import Path
import sys,importlib.util,tempfile,json,gzip,io,contextlib
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/grouped_k3_perf'))
spec=importlib.util.spec_from_file_location('perf_stats',ROOT/'experiments/grouped_k3_perf/statistics.py')
stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)
from provenance import sha

def main():
    data=ROOT/'reports/grouped-k3-perf-r1-initial-data'
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-CPU-pair-') as tmp:
        d=Path(tmp);logs=[];receipts=[];admissions=[]
        def pin(p):return dict(path=str(p),sha256=sha(p))
        for n in (1,2):
            log=d/f'CPU{n}.log';log.write_bytes(gzip.decompress((ROOT.parent/'wt-grouped-k3-perf-review/reports/grouped-k3-perf-review-data'/f'synthetic-campaign{n}.log.gz').read_bytes()));logs.append(log)
            freeze=ROOT/'reports/grouped-k3-perf-r1-data'/f'node{n}-frozen.sha256'
            a=dict(campaign=n,label=f'SYNTHETIC-CPU-{n}',physical_id=f'SYNTHETIC-PHYSICAL-{n}',previous_physical_id=None if n==1 else 'SYNTHETIC-PHYSICAL-1',freeze=pin(freeze))
            ap=d/f'admission{n}.json';ap.write_text(json.dumps(a));admissions.append(a)
            r=d/f'receipt{n}.json';r.write_text(json.dumps(dict(admission=pin(ap))));receipts.append(r)
        args=['statistics.py',*map(str,logs),'--root-full-pass-receipts',*map(str,receipts)]
        def run():
            with patch.object(sys,'argv',args),patch.object(stats,'full',side_effect=admissions),contextlib.redirect_stdout(io.StringIO()) as output:
                stats.main();return json.loads(output.getvalue())
        assert run()['proposal']['proposal_mode']=='direct-only'
        for key,value in [('physical_id',admissions[0]['physical_id']),('campaign',1),('label',admissions[0]['label']),('previous_physical_id','wrong')]:
            saved=admissions[1][key];admissions[1][key]=value
            try:run()
            except AssertionError:pass
            else:raise AssertionError('invalid physical pair accepted')
            admissions[1][key]=saved
    print('PASS SYNTHETIC CPU pair policy: distinct physical IDs/campaigns/labels/chain; 4 negatives; positive uses mocked full validation, not a real performance proposal')
if __name__=='__main__':main()
