"""Deferred split oracle: FP64 P/sumabs, never a predicted GPU P checkpoint.

Use logical K16 boundaries floor(T*s/S), including non-H128 boundaries.
Inputs are already rounded A and exact half Q for a bounded N block <=256.
No native dot2 or GPU reduction semantics are claimed.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import numpy as np


def logical_fp64_oracle(a, q, splits):
    if (a.ndim != 2 or q.ndim != 2 or type(splits) is not int
            or not 1 <= splits <= a.shape[1]//16 or a.shape[1] != q.shape[0]
            or a.shape[1] % 16 or not 0 < q.shape[1] <= 256
            or a.dtype != np.dtype('<f2') or q.dtype != np.dtype('<f2')
            or not np.isfinite(a).all() or not np.isfinite(q).all()):
        raise ValueError('Expected finite half A/Q, K16 geometry, bounded N and valid S')
    tiles=a.shape[1]//16
    boundaries=[16*(tiles*s//splits) for s in range(splits+1)]
    p,mass=[],[]
    for begin,end in zip(boundaries,boundaries[1:]):
        left=a[:,begin:end].astype('f8')
        right=q[begin:end].astype('f8')
        p.append(left @ right)
        mass.append(np.abs(left) @ np.abs(right))
    return dict(K_boundaries=boundaries,P_fp64=np.stack(p),sumabs_fp64=np.stack(mass),
                contract='FP64 mathematical split oracle; not ordered GPU P')


if __name__=='__main__':
    # Coverage and ownership tests deliberately include q_b S32's uneven K16 cuts.
    for i in (1280,4096,5120):
        a=np.ones((8,i),dtype='<f2')
        q=np.ones((i,128),dtype='<f2')
        for s in (4,8,16,32):
            result=logical_fp64_oracle(a,q,s)
            assert result['K_boundaries'][0]==0 and result['K_boundaries'][-1]==i
            assert np.all(np.sum(result['P_fp64'],axis=0)==i)
            assert np.array_equal(result['P_fp64'],result['sumabs_fp64'])
    print('PASS: 12 logical FP64 split coverage checks; no GPU P generated')
