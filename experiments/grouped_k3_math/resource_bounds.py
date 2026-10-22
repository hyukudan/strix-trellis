"""Recompute planned explicit bytes from the actual new reference buffers."""
from pathlib import Path
import json
import sys

manifest=json.loads(Path(sys.argv[1]).read_text())
total=0
cpu_payload=0
initial_copies=0
shapes={}
weight_bytes=0
for case in manifest['cases']:
    sizes={k:v['bytes'] for k,v in case['tensors'].items()}
    i,n,s=case['input'],case['output'],case['splits']
    weights=0
    for field in ('packed','suh','svh','bias'):
        weights=(weights+255)//256*256
        weights+=sizes[field]
    weight_bytes+=weights
    a,p=sizes['a'],sizes['p64']//2
    assert a==i*2 and p==s*n*4
    assert sizes['x']==i*4 and sizes['old']==n*4
    total+=weights+sizes['x']+512+2*(sizes['old']+512+a+512+p+512)
    cpu_payload+=sum(sizes.values())
    initial_copies+=sizes['x']+2*sizes['old']
    shapes[i]=dict(input=i,output=n,splits=s,a=a,p=p,
        payload_stride=((a+255)//256*256+p+255)//256*256,
        workspace_g32=32*(((a+255)//256*256+p+255)//256*256),
        actual_guarded_A_P_per_request_G32=32*(a+p+1024))
transient=max(v['payload_stride'] for v in shapes.values())
# CpuCase raw vectors, retained X/Y copies, full/operator/stage/capture snapshots,
# source/reference parsing and metrics temporaries; driver/library RSS is a native gate.
snapshot_reserve=32*max(v['a']+v['p']+v['output']*4 for v in shapes.values())*10
attribute_log_reserve=128*1024 # includes all fixed events/contexts/counters, not driver memory
host_payload_bound=attribute_log_reserve+cpu_payload+initial_copies+snapshot_reserve+64*1024**2
assert total+transient<1024**3 and host_payload_bound<2*1024**3
assert sorted(v['workspace_g32'] for v in shapes.values())==[2768896,5046272]
print(json.dumps(dict(shapes=list(shapes.values()),actual_v13_weight_allocation_bytes_planned=weight_bytes,
    persistent_explicit_gpu_bytes_planned=total,transient_full_original_workspace_peak=transient,
    maximum_explicit_gpu_bytes_planned=total+transient,gpu_limit=1024**3,
    attribute_log_reserve_bytes=attribute_log_reserve,host_payload_bound_planned=host_payload_bound,host_limit=2*1024**3,
    cpu_generation_peak_host_bytes_measured=manifest['peak_host_bytes'],
    note='Arithmetic from actual CPU payload sizes and literal V13 allocation order; no GPU allocation was performed. Driver/runtime RSS and actual ledger remain native gates.'),indent=2))
