"""Adversarial CPU-only tests of admission, stats and timing-boundary source."""
from pathlib import Path
import importlib.util,json,sys,tempfile
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
def load(name):
    spec=importlib.util.spec_from_file_location('perf_'+name,HERE/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
stats=load('statistics');admission=load('admission')
def rejects(fn):
    try:fn()
    except (AssertionError,KeyError,FileNotFoundError):return True
    return False
def body(text,marker):
    at=text.index('{',text.index(marker));depth=1;end=at+1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return text[at+1:end-1]
def submit_source(text):
    code=body(text,'inline hipError_t submit(const Prepared& p)')
    for token in ('hipMalloc','hipMemcpy','hipStreamSynchronize','hipPointerGetAttributes','hipFuncGetAttributes','hipEvent','new ','std::vector','malloc','prepare('):assert token not in code,token
    assert 'dispatch(' in code and 'SubmitScope' in code
def main():
    pool=(HERE/'pool.hpp').read_text();native=(HERE/'native.hip').read_text();submit_source(pool)
    for call in ('hipMalloc(nullptr,1);','hipMemcpy(nullptr,nullptr,1,0);','hipStreamSynchronize(nullptr);','hipPointerGetAttributes(nullptr,nullptr);','hipFuncGetAttributes(nullptr,nullptr);','auto p=new int;'):
        mutant=pool.replace('    SubmitScope prohibition;','    SubmitScope prohibition;'+call)
        assert rejects(lambda:submit_source(mutant))
    submit=body(pool,'inline hipError_t submit(const Prepared& p)')
    for timing in ('hipEventRecord','Clock::now','hipStreamSynchronize','hipEventElapsedTime'):assert timing not in submit and timing in native
    assert native.index('hipEventRecord(events.a')<native.index('auto t=Clock::now()')<native.index('if(mode)graphs[v][c]->enqueue();else hip(submit(*plans[v][c]));',native.index('auto t=Clock::now()'))<native.index('hipEventRecord(events.b')
    # Independent hand-computable timing fixture. Never run a native module.
    with tempfile.TemporaryDirectory(prefix='grouped-k3-perf-cpu-') as tmp:
        tmp=Path(tmp);rows=[]
        for i in (5120,2304):
            for g in (4,8,16,32):
                for mode in ('direct','graph'):
                    for v in ('V13','new'):
                        for sweep in range(11):
                            for clone in range(64//g):
                                value=10+sweep
                                if v=='new':value*=.8 if mode=='direct' else 1.1
                                rows.append(dict(input=i,G=g,mode=mode,variant=v,sweep=sweep,clone=clone,event_ms=value,wall_ms=2*value))
        def write(rows):
            path=tmp/'samples.log';path.write_text('\n'.join('SAMPLE '+json.dumps(r) for r in rows)+'\nPASS physical performance campaign all fixed cells; contexts=2 calls=14\n');return path
        run=stats.summarize(write(rows));cell=run['cells']['5120/G4/direct']
        assert cell['event_ms_V13']['min']==10 and cell['event_ms_V13']['median']==15 and cell['event_ms_V13']['p90']==19
        assert abs(cell['event_ms_gain']-.2)<1e-12
        second=dict(run,log_sha256='different-physical-log')
        assert stats.proposal([run,second])['proposal_mode']=='direct-only'
        second=json.loads(json.dumps(second));second['cells']['2304/G32/direct']['wall_ms_gain']=.019
        assert not stats.proposal([run,second])['performance_eligible']
        assert rejects(lambda:stats.summarize(write(rows[:-1])))
        assert rejects(lambda:stats.summarize(write(rows+[rows[0]])))
        assert rejects(lambda:stats.summarize(write([dict(r,event_ms=float('nan')) if j==0 else r for j,r in enumerate(rows)])))
        assert rejects(lambda:stats.summarize(write([dict(r,unexpected=True) if j==0 else r for j,r in enumerate(rows)])))
        nonuniform=[]
        for r in rows:
            multiplier=1 if r['variant']=='V13' else .8 if r['mode']=='direct' else 1.1
            value=(10+r['sweep']+r['clone'])*multiplier
            nonuniform.append(dict(r,event_ms=value,wall_ms=2*value))
        varied=stats.summarize(write(nonuniform))
        for g in (4,8,16,32):
            cell=varied['cells'][f'5120/G{g}/direct'];offset=(64//g-1)/2
            assert cell['event_ms_V13']['min']==10+offset
            assert cell['event_ms_V13']['median']==15+offset
            assert cell['event_ms_V13']['p90']==19+offset
        manifest=tmp/'frozen.sha256';manifest.write_text('CPU fixture only')
        receipt=tmp/'receipt.json';receipt.write_text(json.dumps(dict(schema=1,actual_native_QA='COMPILED')))
        assert rejects(lambda:admission.check(receipt,manifest))
        receipt.write_text(json.dumps(dict(schema=1,actual_native_QA='PASS',QA_binary_sha256='wrong')))
        assert rejects(lambda:admission.check(receipt,manifest))
    print('PASS CPU adversarial timing-boundary mutations, per-Batch stats/tails/losses, full fixed selection and blocked receipts; native gates OPEN')
if __name__=='__main__':main()
