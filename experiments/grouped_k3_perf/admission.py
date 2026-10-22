"""ROOT admission and post-return closure; CPU only. No physical closure in payload."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import argparse,json
from provenance import admission as check,full,sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--receipt',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--finalize',type=Path,help='ROOT after outer return: candidate full receipt; validate before writing')
    p.add_argument('--output',type=Path)
    a=p.parse_args();r=check(a.receipt,a.manifest)
    if a.finalize:
        candidate=json.loads(a.finalize.read_text());assert candidate['admission']['sha256']==sha(a.receipt)
        log=Path(candidate['log']['path']);log=log if log.is_absolute() else a.finalize.parent/log
        full(a.finalize,log);assert a.output and not a.output.exists()
        # Relative references are anchored to the candidate's directory.
        assert a.output.parent.resolve()==a.finalize.parent.resolve()
        a.output.write_text(json.dumps(candidate,indent=2)+'\n')
    print(json.dumps(r,sort_keys=True))
if __name__=='__main__':main()
