"""Independent standard-library coverage proof; no project numeric helpers."""
import json,os,resource
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS,(1<<30,1<<30))
cases=0; old=0; masks={};counts={}
for I in (16,80,128,176,272,112,240):
    for N in (16,32,48,80):
        for R in (2,4,8):
            lengths=set();tails=set()
            for S in range(1,33):
                # Distribute actual tile numbers independently using rational interval ownership.
                K=I//16
                bounds=[K*s//S for s in range(S+1)]
                assert bounds[0]==0 and bounds[-1]==K
                assert sum(bounds[s+1]-bounds[s] for s in range(S))==K
                for layout in range(3):
                    T=R if layout==0 else R+3;O=int(layout==2)
                    assert O+R<=T and 1+R>R and -1<0
                    writers=set()
                    for s in range(S):
                        tiles=list(range(bounds[s],bounds[s+1]));consumed=[]
                        for start in range(0,len(tiles),8):
                            chunk=tiles[start:start+8];L=len(chunk)
                            lengths.add(L);tails.add(L%4)
                            # Cooperative staging owns each cell exactly once, all consumed pairs exist.
                            staging={(r,k,pair):(O+r,chunk[k],pair) for r in range(R) for k in range(L) for pair in range(8)}
                            assert len(staging)==R*L*8
                            for k,tile in enumerate(chunk):
                                for r in range(R):
                                    for lane in range(4):
                                        for pair in (lane,lane+4):assert staging[r,k,pair]==(O+r,tile,pair)
                            consumed.extend(chunk)
                        assert consumed==tiles
                        for r in range(R):
                            for c in range(N):
                                address=(s*T+O+r)*N+c
                                assert address not in writers and 0<=address<S*T*N
                                writers.add(address)
                    assert len(writers)==S*R*N
                    cases+=1;old+=I not in (112,240)
            masks[f'{I}/{N}/{R}']={'chunks':sorted(lengths),'tails':sorted(tails)}
            if I==112:assert 7 in lengths
            if I==240:assert {7,8}<=lengths
            counts.setdefault(R,[set(),set()]);counts[R][0]|=lengths;counts[R][1]|=tails
assert cases==8064 and old==5760
for chunks,tails in counts.values():assert chunks==set(range(1,9)) and tails==set(range(4))
print(json.dumps({'PASS':True,'original_cases':old,'added_cases':cases-old,'total_cases':cases,'coverage':masks},indent=2))
