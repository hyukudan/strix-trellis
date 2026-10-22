"""Generate 12 bounded real K5 TP CPU fixtures, L0 first. Never opens GPU APIs."""
from tp_real_common import *
import argparse


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=BASE)
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--layers',type=int,nargs='+',default=[0,21,39])
    args=p.parse_args()
    low_priority()
    args.out.mkdir(parents=True,exist_ok=True)
    headers,names=scan(args.base)
    snapshot=[dict(file=str(h.path),stat=h.before,header_sha256=h.header_sha256,
                   header_and_length_sha256=h.ranges[(0,h.data_offset)],header=h.header) for h in headers]
    dump(args.out/'source-headers.json',snapshot)
    table=independent_lut('mul1')
    records=[]
    for layer in args.layers:
        for site in ('q_b','o_b'):
            for rank in (0,1):
                records.append(generate_one(names,args.out,layer,site,rank,table))
                dump(args.out/'dense-manifest.json',dict(format='STGPU001',fixtures=records,
                    generator_sha256=sha_file(Path(__file__)),common_sha256=sha_file(Path(__file__).with_name('tp_real_common.py'))))


if __name__=='__main__':
    main()
