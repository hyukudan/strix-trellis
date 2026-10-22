"""New head-TP blobs/oracles, N<=256 blocks; canonical head data is untouched."""
from tp_real_common import *
import argparse


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=BASE)
    p.add_argument('--out',type=Path,default=OUT)
    args=p.parse_args()
    low_priority()
    headers,names=scan(args.base)
    table=independent_lut('mul1')
    records=[]
    for rank in (0,1):
        records.append(generate_one(names,args.out,0,'head',rank,table,True))
        dump(args.out/'head-manifest.json',dict(format='TP_HEAD_BLOB_CPU_1',fixtures=records,
             generator_sha256=sha_file(Path(__file__)),common_sha256=sha_file(Path(__file__).with_name('tp_real_common.py'))))


if __name__=='__main__':
    main()
