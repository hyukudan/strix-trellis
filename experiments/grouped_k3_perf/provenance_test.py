"""CPU-only adversarial receipt tests. Positive fixture is SYNTHETIC, never a proposal."""
from pathlib import Path
import sys,json,tempfile,copy,subprocess
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import provenance as v

def rejects(fn):
    try:fn()
    except (AssertionError,KeyError,ValueError,FileNotFoundError,TypeError):return
    raise AssertionError('invalid receipt accepted')

def main():
    count=0
    with tempfile.TemporaryDirectory(prefix='SYNTHETIC-CPU-NOT-PERFORMANCE-') as directory:
        root=Path(directory);(root/'build').mkdir();(root/'build/host.txt').write_text('node2\n')
        def write(name,obj):
            path=root/name;path.write_text(json.dumps(obj));return path
        def pin(path):return dict(path=str(path),sha256=v.sha(path))
        manifest=root/'frozen.sha256';manifest.write_text('SYNTHETIC CPU FIXTURE ONLY')
        review=write('review.json',dict(schema=2,kind='ROOT_HOST_REVIEW',freeze_sha256=v.sha(manifest),review_pins=v.PINS['review'],qa_pins=v.PINS['qa'],decision='APPROVED_HOST_POOL_TIMING'))
        meta=write('meta.json',dict(schema=2,node='node2',physical_id='SYNTHETIC-physical-1',campaign=1,label='ROOT-grouped-k3-perf-r1-SYNTHETIC-1',outer_seconds=300,started_epoch=1000))
        a=dict(schema=2,kind='ROOT_ADMISSION_GROUPED_K3_R1',node='node2',physical_id='SYNTHETIC-physical-1',campaign=1,label='ROOT-grouped-k3-perf-r1-SYNTHETIC-1',inner_seconds=270,outer_seconds=300,previous_physical_id=None,freeze=pin(manifest),root_review=pin(review),parent_meta=pin(meta))
        ap=write('admission.json',a)
        log=root/'native.log';log.write_text('SYNTHETIC CPU STRUCTURE FIXTURE, not a real performance proposal\n')
        stderr=root/'stderr';stderr.write_text('Maximum resident set size (kbytes): 1000\n')
        close=root/'closed';close.write_text('payload_exit=0 integrity_exit=0\n')
        passed=root/'passed';passed.write_text(a['physical_id']+' '+a['label']+' PASS\n')
        parentlog=root/'parent.log';parentlog.write_text('SYNTHETIC CPU ONLY\n')
        pc=dict(schema=2,physical_id=a['physical_id'],node=a['node'],campaign=1,label=a['label'],started_epoch=1000,ended_epoch=1001,exit=0,child_exit=0,admission_sha256=v.sha(ap),child_log_sha256=v.sha(log),parent_log_sha256=v.sha(parentlog))
        parentclose=write('parent-close.json',pc)
        f=dict(schema=2,kind='ROOT_PHYSICAL_FULL_PASS_GROUPED_K3_R1',admission=pin(ap),log=pin(log),stderr=pin(stderr),child_close=pin(close),parent_close=pin(parentclose),parent_log=pin(parentlog),child_pass=pin(passed))
        fp=write('full.json',f)
        # Only structure is mocked for this positive fixture. Real raw/full validators
        # are exercised as negatives below and are never bypassed by either CLI.
        with patch.object(v,'frozen',return_value=(root,{})),patch.object(v,'raw_pass',return_value='SYNTHETIC'):
            assert v.full(fp,log)['physical_id']==a['physical_id']
            for field,value in [('schema',1),('kind','BUILD_ONLY'),('extra',True)]:
                mutant=dict(f);mutant[field]=value;write('bad-full.json',mutant);rejects(lambda:v.full(root/'bad-full.json',log));count+=1
            for field,value in [('campaign',0),('inner_seconds',269),('outer_seconds',301),('physical_id','x'),('previous_physical_id','same'),('node','unknown'),('label','ROOT-grouped-k3-perf-old'),('extra',True)]:
                mutant=dict(a);mutant[field]=value;write('bad-admission.json',mutant);rejects(lambda:v.admission(root/'bad-admission.json',manifest));count+=1
            for field,value in [('exit',1),('child_exit',1),('physical_id','reused'),('ended_epoch',1301),('started_epoch',999),('child_log_sha256','0'*64),('admission_sha256','0'*64)]:
                mutant=dict(pc);mutant[field]=value;write('parent-close.json',mutant)
                f['parent_close']=pin(parentclose);write('full.json',f);rejects(lambda:v.full(fp,log));count+=1
            write('parent-close.json',pc);f['parent_close']=pin(parentclose);write('full.json',f)
            passed.write_text('PASS\n');rejects(lambda:v.full(fp,log));count+=1
        # Full raw and actual evidence are mandatory without the synthetic mocks.
        rejects(lambda:v.full(fp,log));rejects(lambda:v.frozen(manifest));count+=2
        for text in ('PASS physical performance campaign all fixed cells; contexts=2 calls=14\n','PASS native mathematical QA only; no performance claim\nfailed=0 overflow=0\n'):
            log.write_text(text);rejects(lambda:v.raw_pass(log));count+=1
        # Every required key, wrong hash and missing file fail closed.
        for field in a:
            mutant=dict(a);del mutant[field];write('bad-admission.json',mutant);rejects(lambda:v.admission(root/'bad-admission.json',manifest));count+=1
        ref=pin(log);log.write_text('tampered');rejects(lambda:v.ref(ref,root));count+=1
    print(f'PASS {count} CPU provenance negatives; positive STRUCTURE fixture SYNTHETIC, no real performance proposal; raw/native gates OPEN')
if __name__=='__main__':main()
