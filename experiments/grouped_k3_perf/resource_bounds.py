"""Static ledger from actual reference tensor sizes, not GPU measurements."""
from pathlib import Path
import json,sys
def bounds(manifest):
    shapes=[]
    for i in (5120,2304):
        cases=[c for c in manifest['cases'] if c['input']==i];assert len(cases)==64
        gpu=host=0
        for c in cases:
            t={k:v['bytes'] for k,v in c['tensors'].items()};weight=0
            for tag in ('packed','suh','svh','bias'):weight=(weight+255)//256*256+t[tag]
            stride=(i*2+255)//256*256+c['splits']*c['output']*4
            gpu+=weight+t['x']+512+2*(t['old']+512+stride+512)
            host+=sum(t.values())+t['x']
        # One transport blob, full copied buffer snapshot, two A/P/Y snapshots,
        # FP64 output/metrics, fixed counter log, descriptors/graphs, parse frames,
        # independent immutable vectors and conservative 128MiB runtime allowance.
        host+=4423680+2*max((c['input']*2+c['splits']*c['output']*4+c['output']*4) for c in cases)
        host+=128*1024**2+128*1024
        assert gpu<1024**3 and host<2*1024**3
        shapes.append(dict(input=i,explicit_gpu_planned=gpu,host_payload_and_reserve_planned=host,
            packed_resident=64*4423680,workspace_sum_two_variants=128*stride,stride=stride))
    return dict(shapes=shapes,gpu_limit=1024**3,host_limit=2*1024**3,CPU_generation_RSS_measured=manifest['peak_host_bytes'],
        native_allocation_ledger='OPEN; Pool records actual weights allocation_bytes and guarded buffers',
        native_peak_RSS='OPEN; per-check getrusage and payload /usr/bin/time -v',runtime_memory='native observation required')
if __name__=='__main__':print(json.dumps(bounds(json.loads(Path(sys.argv[1]).read_text())),indent=2))
