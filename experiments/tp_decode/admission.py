"""Small CPU admission checker; records no fabricated physical phase closure."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import socket

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def allowed(minute,second=0):
    if 1260<=minute<1410:return False
    return minute>=1410 or (1260*60-minute*60-second)>900
def verify_freeze(root):
    frozen=root/'frozen.sha256'
    for line in frozen.read_text().splitlines():
        digest,name=line.split('  ',1);p=root/name
        assert p.is_file() and sha(p)==digest, 'STOP frozen mutation '+name
    return sha(frozen)
def window(root,budget=True):
    now=dt.datetime.now().astimezone()
    minute=now.hour*60+now.minute
    assert (allowed(minute,now.second) if budget else not 1260<=minute<1410),'STOP cluster local veto/budget window'
    assert not (Path.home()/'gpu-site/.gpu-turn/maintenance').exists(),'STOP maintenance'
def verify(root,receipt,after=False):
    assert root==Path.home()/'gpu-site/scratch-trellis/tp-decode-v1/host-r2'
    window(root,not after);digest=verify_freeze(root)
    frozen=json.loads((root/'admission.json').read_text())
    assert sha(Path('/opt/rocm/core-10.1/lib/libamdhip64.so'))==frozen['HIP_runtime_library_sha256'],'STOP HIP runtime mutation'
    assert frozen['state']=='CPU_PREPARED_NATIVE_OPEN'
    assert not frozen['retired'] and frozen['isa_prefetch_gate']=='PASS_STATIC_ORDER'
    assert frozen['full_fixture_validation']=='PASS' and frozen['cpu_references']=='PASS'
    r=json.loads(receipt.read_text())
    assert r['campaign'] in (1,2) and r['label'].startswith('ROOT-tp-decode-v1-')
    assert r['node']==socket.gethostname() and r['frozen_sha256']==digest
    # ROOT supplies the actual OUTER lease receipt. This checker never creates it
    # and does not claim that an environment flag proves a scheduler lease.
    lease=Path(r['actual_outer_lease_receipt']);assert lease.is_file() and sha(lease)==r['actual_outer_lease_receipt_sha256']
    assert r['outer_seconds']==900 and r['root_confirms_active_lease'] is True
    assert all(r['root_review'][k]=='PASS' for k in ('source','resources','ownership','admission','budget'))
    assert not (root/'runs/STOP').exists()
    window(root,not after);return r
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path);p.add_argument('--prequeue',action='store_true');p.add_argument('--after',action='store_true');args=p.parse_args()
    if args.prequeue:
        window(Path.cwd());print(json.dumps(dict(state='PREQUEUE_CHECK_ONLY_NO_LEASE',frozen_sha256=verify_freeze(Path.cwd()),physical_closures='OPEN')))
    else:
        assert args.receipt is not None
        print(json.dumps(verify(Path.cwd(),args.receipt,args.after),indent=2))
