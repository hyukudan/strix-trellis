"""Collect actual fixture closures and freeze both CPU builds, without requesting GPU."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
repo=Path(__file__).resolve().parents[1]
data=repo/'reports/tp-decode-private-data'
external=Path('external-data/work/tp-decode-v1-build')
source=Path('external-data/work/tp-real-fixtures-cpu')
refs=Path('external-data/work/tp-decode-v1-cpu/refs')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def save(p,value):p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
p=argparse.ArgumentParser();p.add_argument('node',choices=('node1','node2'));p.add_argument('--attempt',required=True);args=p.parse_args()
remote='gpu-site/scratch-trellis/tp-decode-v1/host-r2';log=data/(args.node+'-freeze-'+args.attempt+'.log');assert not log.exists()
def call(cmd):
    result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    with log.open('ab') as f:f.write(('COMMAND '+json.dumps(cmd)+'\n').encode());f.write(result.stdout)
    print(result.stdout.decode(errors='replace'));result.check_returncode()
validation=json.loads((source/'validation.json').read_text(encoding='utf-8'))
assert validation['passed'] and len(validation['fixtures'])==14 and len(validation['corruption_negatives'])==7
assert all(x['rejected'] for x in validation['corruption_negatives'])
validator=Path('external-data/work/wt-tp-real-fixtures/tools/validate_tp_real_fixtures.py')
assert sha(validator)==validation['validator_sha256']
reference=json.loads((refs/'manifest.json').read_text(encoding='utf-8'));assert reference['passed'] and len(reference['fixtures'])==12
assert sha(repo/'tools/tp_decode_refs.py')==reference['reference_source_sha256']
originals={}
for item in reference['fixtures']:
    manifest=source/item['name']/'manifest.json';original=json.loads(manifest.read_text(encoding='utf-8'))
    assert sha(manifest)==item['source_manifest_sha256']
    for name,info in original['files'].items():assert sha(manifest.parent/name)==info['sha256']
    assert sha(refs/item['file'])==item['sha256'];originals[item['name']]=original
evidence=external/args.node/('evidence-'+args.attempt);evidence.mkdir(parents=True,exist_ok=False)
save(evidence/'cpu-validation.json',dict(passed=True,full_worker_validation=validation,full_validation_sha256=sha(source/'validation.json'),
    derivative_manifest_sha256=sha(refs/'manifest.json'),derivative=reference,original_manifests=originals,
    generator_fingerprints={str(q):sha(q) for q in [validator,validator.with_name('tp_real_common.py'),validator.with_name('tp_logical_fp64_oracle.py'),repo/'tools/tp_decode_refs.py']},
    independent_native_semantics='OPEN'))
sourcefiles=json.loads((data/(args.node+'-sources-'+args.attempt+'.json')).read_text())
save(evidence/'source-files.json',sourcefiles)
(evidence/'tp_decode_refs.py').write_bytes((repo/'tools/tp_decode_refs.py').read_bytes())
# Provenance evidence is small JSON/source, no packed/model payloads enter Git.
save(data/'cpu-validation.json',json.loads((evidence/'cpu-validation.json').read_text(encoding='utf-8')))
call(['scp','-r',str(evidence),args.node+':'+remote+'/evidence'])
call(['ssh','-o','BatchMode=yes',args.node,'cd ~/'+remote+' && nice -n 10 python3 -B experiments/tp_decode/freeze.py '+args.attempt])
destination=data/args.node/args.attempt;destination.mkdir(exist_ok=True)
for name in ('admission.json','handoff.json','frozen.sha256'):
    call(['scp',args.node+':'+remote+'/'+name,str(destination/name)])
