"""CPU-only TP fixture primitives; no engine implementation is imported."""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import hashlib
import json
from pathlib import Path
import struct
import time
import numpy as np
from audit_ours_packs import Header, independent_lut, low_priority, sha_file, ROOT
from make_ours_pack_fixtures import peak_working_set, strict_parse
from make_dense_gpu_fixtures import write_stream
from strix_trellis.container import SafetensorsFile, load_exl3
from strix_trellis.codec import unpack_states, decode_tile
from strix_trellis.linear import (input_stages_fp32, output_stages_fp32,
                                  core_dot_fp32, hadamard)

BASE = Path('external-data/ours-K5')
OUT = Path('external-data/work/tp-real-fixtures-cpu')
SEED = 202610216451
MEMORY_LIMIT = 1024**3


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def array_sha(a):
    h = hashlib.sha256()
    if not a.size:
        return h.hexdigest()
    for row in a.reshape(-1, a.shape[-1] if a.ndim else 1):
        h.update(np.ascontiguousarray(row).tobytes())
    return h.hexdigest()


def save(directory, name, a, files):
    p = directory/name
    with p.open('xb') as f:
        np.asarray(a).tofile(f)
    files[name] = dict(bytes=p.stat().st_size, sha256=sha_file(p),
                       dtype=np.asarray(a).dtype.str, shape=list(np.asarray(a).shape))


def rss():
    value = peak_working_set()
    if value >= MEMORY_LIMIT:
        raise ValueError('CPU process exceeds 1 GiB working set')
    return value


def scan(base):
    """Actual headers, never the incomplete index, determine tensor locations."""
    headers, names = [], {}
    for p in sorted(base.glob('*.safetensors')):
        h = Header(p)
        headers.append(h)
        for name in h.header:
            if name == '__metadata__':
                continue
            if name in names:
                raise ValueError('Duplicate actual tensor: '+name)
            names[name] = h
    return headers, names


def partition(i, n, site, rank):
    if rank not in (0, 1) or site not in ('q_b', 'o_b', 'head'):
        raise ValueError('Invalid TP site/rank')
    expected = {'q_b': (1280,32768), 'o_b': (8192,5120), 'head': (5120,129280)}[site]
    if (i,n) != expected:
        raise ValueError('Wrong whole tensor shape/TP axis')
    return (rank*i//2, (rank+1)*i//2, 0, n) if site == 'o_b' else (0,i,rank*n//2,(rank+1)*n//2)


def aligned(t, u, v, bounds):
    i0,i1,n0,n1 = bounds
    if (not 0 <= i0 < i1 <= len(u) or not 0 <= n0 < n1 <= len(v)
            or any(x % 128 for x in bounds)):
        raise ValueError('TP boundaries must preserve H128 and complete tiles')
    return t[i0//16:i1//16,n0//16:n1//16], u[i0:i1], v[n0:n1]


def selected(names, prefix, site, rank):
    h = names[prefix+'.trellis']
    group = load_exl3(SafetensorsFile(h.path), prefix)
    marker = SafetensorsFile(h.path).tensor(prefix+'.mul1')
    if (group.bits != 5 or group.codebook != 'mul1' or marker.dtype != np.dtype('<i4')
            or marker.size != 1 or (int(marker.item()) & 0xffffffff) != 0x83dcd12d):
        raise ValueError('Strict K5 mul1 marker required')
    bounds = partition(group.input_size,group.output_size,site,rank)
    t,u,v = aligned(group.trellis, group.suh, group.svh,bounds)
    bias = group.bias
    if bias is not None and site != 'o_b':
        bias = bias[bounds[2]:bounds[3]]
    for scale in (u,v):
        if not np.isfinite(scale).all() or not np.any(np.signbit(scale)) or not np.any(np.abs(scale) != 1):
            raise ValueError('Expected finite signed magnitudes, not sign-only scales')
    provenance = {}
    # Hash only bytes selected. Physical packed N cuts are row-strided gathers.
    for suffix,a in [('trellis',t),('suh',u),('svh',v)]:
        name = prefix+'.'+suffix
        info = h.header[name]
        start = h.data_offset+info['data_offsets'][0]
        ranges = []
        if suffix == 'trellis':
            for ti in range(bounds[0]//16,bounds[1]//16):
                offset = start+(ti*group.trellis.shape[1]+bounds[2]//16)*160
                length = (bounds[3]-bounds[2])//16*160
                ranges.append(dict(offset=offset,bytes=length,sha256=h.remember(offset,length)))
        else:
            lo,hi = bounds[:2] if suffix == 'suh' else bounds[2:]
            ranges.append(dict(offset=start+lo*2,bytes=(hi-lo)*2,sha256=h.remember(start+lo*2,(hi-lo)*2)))
        provenance[suffix] = dict(source_tensor=name, header=info,selected_sha256=array_sha(a),ranges=ranges)
    provenance['mul1'] = dict(header=h.header[prefix+'.mul1'],sha256=h.tensor_sha(prefix+'.mul1'))
    if bias is not None:
        name = prefix+'.bias'
        raw = SafetensorsFile(h.path).tensor(name)
        lo,hi = (0,len(raw)) if site == 'o_b' else bounds[2:]
        start = h.data_offset+h.header[name]['data_offsets'][0]+lo*raw.dtype.itemsize
        provenance['bias'] = dict(header=h.header[name],offset=start,bytes=(hi-lo)*raw.dtype.itemsize,
                                  sha256=h.remember(start,(hi-lo)*raw.dtype.itemsize))
    return h,t,u,v,bias,bounds,provenance


def independent_states(t):
    """Literal circular BIT reader, independent of codec's Python integer ring."""
    # Storage u32 pairs are little endian; the ring consumes each u32 MSB first.
    b = np.ascontiguousarray(t).view('u1').reshape(*t.shape[:2],40,4)[...,::-1]
    bits = np.unpackbits(b,axis=-1,bitorder='big').reshape(*t.shape[:2],1280)
    ends = (5*(np.arange(256)+1)-16) % 1280
    states = np.zeros((*t.shape[:2],256), dtype=np.uint16)
    for k in range(16):
        states = (states << 1) | bits[...,(ends+k)%1280]
    return states


def checked_q(t, table):
    states = independent_states(t)
    canonical = unpack_states(t,5)
    if not np.array_equal(states,canonical):
        raise ValueError('Full independent states disagree with codec')
    tiles = np.empty((*t.shape[:2],16,16), dtype='<f2')
    # Inverse lane/fragment mapping differs from codec coordinate->position.
    for lane in range(32):
        for frag in range(8):
            r=2*(lane%4)+frag%2+8*((frag//2)%2)
            c=lane//4+8*(frag//4)
            tiles[...,r,c] = table[states[...,8*lane+frag]]
    expected = decode_tile(t,5,'mul1')
    if not np.array_equal(tiles.view('u2'),expected.view('u2')):
        raise ValueError('Full independent Q disagrees with codec')
    return tiles.transpose(0,2,1,3).reshape(t.shape[0]*16,t.shape[1]*16)


def block_oracle(a, q, splits=None):
    """FP64 core and sumabs only; never predicts a candidate GPU reduction."""
    if splits is None:
        return a.astype('f8') @ q.astype('f8'), np.abs(a.astype('f8')) @ np.abs(q.astype('f8'))
    if type(splits) is not int or splits < 1 or a.shape[1] % splits:
        raise ValueError('S must divide local I')
    width = a.shape[1]//splits
    return [block_oracle(a[:,k*width:(k+1)*width],q[k*width:(k+1)*width]) for k in range(splits)]


def references(t,u,v,bias,x,table,keep_q):
    i,n = len(u),len(v)
    stages = input_stages_fp32(x,u)
    a = stages['core_input_fp16']
    outputs = {key:np.empty((8,n),dtype=dtype) for key,dtype in
               [('core','<f4'),('ref','<f4'),('core64','<f8'),('sumabs','<f8'),('ideal','<f8')]}
    qfull = np.empty((i,n),dtype='<f2') if keep_q else None
    ideal_a = hadamard(x.astype('f8')*u.astype('f8'))
    gamma = (2*i*2**-24)/(1-2*i*2**-24)
    max_ratio,blocks,values = 0.,0,0
    for begin in range(0,n,256):
        end = min(begin+256,n)
        packed = t[:,begin//16:end//16]
        q = checked_q(packed,table)
        core = core_dot_fp32(a,q)
        exact,mass = block_oracle(a,q)
        error = np.abs(core.astype('f8')-exact)
        if np.any(error > gamma*mass+1e-30):
            raise ValueError('Ordered F32 core outside gamma_2I sumabs bound')
        max_ratio = max(max_ratio,float(np.max(error/np.maximum(mass,1e-30))))
        vb=v[begin:end]
        bb=None if bias is None else bias[begin:end]
        y=output_stages_fp32(core,vb,bb)['output_fp32']
        for rows in range(1,9):
            prefix = input_stages_fp32(x[:rows],u)
            for key in prefix:
                if prefix[key].tobytes() != stages[key][:rows].tobytes():
                    raise ValueError('Input row-prefix invariance failed')
            test = core_dot_fp32(prefix['core_input_fp16'],q)
            yp = output_stages_fp32(test,vb,bb)['output_fp32']
            if test.tobytes()!=core[:rows].tobytes() or yp.tobytes()!=y[:rows].tobytes():
                raise ValueError('Full CPU core/Y prefix invariance failed')
        outputs['core'][:,begin:end]=core
        outputs['ref'][:,begin:end]=y
        outputs['core64'][:,begin:end]=exact
        outputs['sumabs'][:,begin:end]=mass
        ideal=hadamard(ideal_a @ q.astype('f8'))*vb.astype('f8')
        if bb is not None:
            ideal+=bb.astype('f8')
        outputs['ideal'][:,begin:end]=ideal
        if keep_q:
            qfull[:,begin:end]=q
        blocks+=1
        values+=q.size
        rss()
        if blocks%32==0 or end==n:
            print('CPU_BLOCK',end,n,flush=True)
    return stages,outputs,qfull,dict(all_states_and_Q_checked=values,all_lut_states=65536,
        max_core_error_over_sumabs=max_ratio,gamma_2I=gamma,absolute_floor=1e-30,
        row_prefix_R1_to_R8_full_A_core_Y_bytes_equal=True,n_block_max=256,blocks=blocks)


def generate_one(names,out,layer,site,rank,table,head=False):
    started,cpu = time.perf_counter(),time.process_time()
    prefix='head' if head else f'layers.{layer}.attn.'+{'q_b':'wq_b','o_b':'wo_b'}[site]
    name=f'head.rank{rank}' if head else f'L{layer}.{site}.rank{rank}'
    directory=out/name
    directory.mkdir(exist_ok=False)
    h,t,u,v,bias,bounds,provenance=selected(names,prefix,site,rank)
    seed=SEED+layer*100+{'q_b':0,'o_b':1,'head':2}[site]*10+rank
    rng=np.random.Generator(np.random.PCG64(seed))
    x=rng.normal(0,.125,(8,len(u))).astype('<f4')
    if len({row.tobytes() for row in x})!=8:
        raise ValueError('Original X rows must be distinct')
    stages,outputs,q,checks=references(t,u,v,bias,x,table,not head)
    files={}
    if head:
        path=directory/'head.blob'
        with path.open('xb') as f:
            for row in t:
                np.ascontiguousarray(row).tofile(f)
            u.tofile(f)
            v.tofile(f)
            if bias is not None:
                bias.tofile(f)
        files[path.name]=dict(bytes=path.stat().st_size,sha256=sha_file(path))
        for tag,a0 in [('x.f32',x),('scaled32.f32',stages['scaled_fp32']),('scaled.f16',stages['scaled_fp16']),
                       ('inh.f32',stages['input_hadamard_fp32']),('a.f16',stages['core_input_fp16']),
                       ('core.f32',outputs['core']),('y-ordered.f32',outputs['ref'])]:
            save(directory,tag,a0,files)
    else:
        sections=dict(meta=np.frombuffer(struct.pack('<8I',len(u),len(v),5,2,0x83dcd12d,8,int(bias is not None),seed & 0xffffffff),dtype='u1'),
            trellis=t,suh=u,svh=v,bias=np.empty(0,dtype='<f4') if bias is None else bias,
            x=x,old=np.zeros((8,len(v)),dtype='<f4'),scaled=stages['scaled_fp16'],inh=stages['input_hadamard_fp32'],
            a=stages['core_input_fp16'],q=q,core=outputs['core'],ref=outputs['ref'])
        path=directory/'fixture.stgpu'
        size,sha=write_stream(path,sections)
        strict_parse(path)
        files[path.name]=dict(bytes=size,sha256=sha,section_sha256={tag:array_sha(a0) for tag,a0 in sections.items()})
        save(directory,'scaled32.f32',stages['scaled_fp32'],files)
    for tag,key in [('core-fp64.f64','core64'),('core-sumabs.f64','sumabs'),('ideal-unrounded-y.f64','ideal')]:
        save(directory,tag,outputs[key],files)
    h.verify()
    record=dict(name=name,source=str(h.path),source_prefix=prefix,rank=rank,layer=layer,site=site,
        input=len(u),output=len(v),bits=5,rows=8,seed=seed,site_seed_id={'q_b':0,'o_b':1,'head':2}[site],
        stgpu_seed_field='low 32 bits; full PCG64 seed recorded here',alpha=1,beta=0,bias=bias is not None,
        local_bounds_I_N=bounds,source_tensors=provenance,files=files,checks=checks,
        scale_stats={key:dict(negative=int(np.signbit(a0).sum()),nonunit=int((np.abs(a0)!=1).sum()),
                             min=float(a0.min()),max=float(a0.max())) for key,a0 in [('suh',u),('svh',v)]},
        blob_layout=None if not head else dict(trellis_offset=0,trellis_bytes=t.size*2,suh_offset=t.size*2,
             svh_offset=t.size*2+len(u)*2,bias_offset=t.size*2+len(u)*2+len(v)*2),
        elapsed_wall_seconds=time.perf_counter()-started,process_cpu_seconds=time.process_time()-cpu,
        peak_working_set_bytes=rss(),gpu_validation='OPEN; CPU only')
    dump(directory/'manifest.json',record)
    print('FIXTURE_DONE',name,record['elapsed_wall_seconds'],flush=True)
    return record
