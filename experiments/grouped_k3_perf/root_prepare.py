"""ROOT only, inside a physical lease: bind existing review and actual parent metadata.
Does not obtain a lease, run HIP, or write a physical full PASS closure.
"""
from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from provenance import read,sha,admission,full,ref

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root-review',type=Path,required=True);p.add_argument('--actual-parent-meta',type=Path,required=True)
    p.add_argument('--previous-full',type=Path);a=p.parse_args()
    scope=Path.cwd();manifest=scope/'frozen.sha256';meta=read(a.actual_parent_meta)
    def pin(path):return dict(path=str(path.resolve()),sha256=sha(path))
    previous=None
    if a.previous_full:
        f=read(a.previous_full);log=ref(f['log'],a.previous_full.parent);prior=full(a.previous_full,log)
        assert prior['campaign']==1 and prior['label']!=meta['label'] and prior['physical_id']!=meta['physical_id'];previous=prior['physical_id']
    r=dict(schema=2,kind='ROOT_ADMISSION_GROUPED_K3_R1',**{k:meta[k] for k in ('node','physical_id','campaign','label','outer_seconds')},inner_seconds=270,previous_physical_id=previous,freeze=pin(manifest),root_review=pin(a.root_review),parent_meta=pin(a.actual_parent_meta))
    directory=scope/'runs'/('admission-'+r['label']);directory.mkdir()
    receipt=directory/'receipt.json';receipt.write_text(json.dumps(r,indent=2)+'\n')
    admission(receipt,manifest);print(receipt)
if __name__=='__main__':main()
