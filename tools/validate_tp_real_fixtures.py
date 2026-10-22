"""Validate frozen CPU TP bytes, real source slices and full bounded references."""
from tp_real_common import *
import argparse
import shutil


def mapped(path, dtype, shape, offset=0):
    return np.memmap(path, mode='r', dtype=dtype, shape=shape, offset=offset)


def equal(a,b,label):
    if a.shape!=b.shape or a.dtype!=b.dtype or a.tobytes()!=b.tobytes():
        raise ValueError('Bytes differ: '+label)


def scalar_half(a):
    """Independent IEEE binary16 rounding using Python struct, not NumPy cast."""
    return np.frombuffer(b''.join(struct.pack('<e',float(v)) for v in a.flat),dtype='<f2').reshape(a.shape)


def xor_h128(a):
    """Independent index-XOR butterflies with literal separate F32 operations."""
    data=np.asarray(a,dtype='<f4').copy().reshape(-1,128)
    index=np.arange(128)
    for bit in (1,2,4,8,16,32,64):
        prior=data.copy()
        partner=prior[:,index ^ bit]
        data[:,(index & bit)==0]=(prior+partner)[:,(index & bit)==0]
        data[:,(index & bit)!=0]=(partner-prior)[:,(index & bit)!=0]
    data=np.float32(data*np.float32(1/np.sqrt(128)))
    return data.reshape(a.shape)


def guarded(shape, gap, lead, tail, pattern):
    rows,width=shape
    stride=width+gap
    backing=(np.arange(lead+rows*stride+tail,dtype='<u4') ^ np.uint32(pattern))
    before=backing.copy()
    mask=np.ones(backing.size,dtype=bool)
    for r in range(rows):
        mask[lead+r*stride:lead+r*stride+width]=False
    logical=np.ndarray(shape,dtype='<f4',buffer=backing,offset=lead*4,strides=(stride*4,4))
    return backing,before,mask,logical


def validate(directory, record, names, table, hashes=True, full=True):
    i,n,rows=record['input'],record['output'],record['rows']
    h,t,u,v,bias,bounds,provenance=selected(names,record['source_prefix'],record['site'],record['rank'])
    if record['local_bounds_I_N']!=list(bounds) or (i,n,rows)!=(len(u),len(v),8):
        raise ValueError('TP axis/bounds mismatch')
    if record['source_tensors'] != provenance:
        raise ValueError('Source selected SHA/ranges mismatch')
    for name,info in record['files'].items():
        path=directory/name
        if path.stat().st_size!=info['bytes'] or (hashes and sha_file(path)!=info['sha256']):
            raise ValueError('File SHA/size mismatch: '+name)
    data={}
    if record['site']=='head':
        path=directory/'head.blob'
        data['trellis']=mapped(path,'<i2',t.shape)
        data['suh']=mapped(path,'<f2',(i,),t.nbytes)
        data['svh']=mapped(path,'<f2',(n,),t.nbytes+2*i)
        data['bias']=None if bias is None else mapped(path,'<f4',(n,),t.nbytes+2*i+2*n)
        for key,name,dtype,shape in [('x','x.f32','<f4',(8,i)),('scaled','scaled.f16','<f2',(8,i)),
             ('inh','inh.f32','<f4',(8,i)),('a','a.f16','<f2',(8,i)),
             ('core','core.f32','<f4',(8,n)),('ref','y-ordered.f32','<f4',(8,n))]:
            data[key]=mapped(directory/name,dtype,shape)
    else:
        path=directory/'fixture.stgpu'
        parts=strict_parse(path)
        meta=mapped(path,'<u4',(8,),parts['meta'][0])
        expected=np.array([i,n,5,2,0x83dcd12d,8,int(bias is not None),record['seed'] & 0xffffffff],dtype='<u4')
        equal(meta,expected,'metadata')
        for key,dtype,shape in [('trellis','<i2',t.shape),('suh','<f2',(i,)),('svh','<f2',(n,)),
            ('x','<f4',(8,i)),('old','<f4',(8,n)),('scaled','<f2',(8,i)),('inh','<f4',(8,i)),
            ('a','<f2',(8,i)),('q','<f2',(i,n)),('core','<f4',(8,n)),('ref','<f4',(8,n))]:
            data[key]=mapped(path,dtype,shape,parts[key][0])
        if bias is not None:
            equal(mapped(path,'<f4',(n,),parts['bias'][0]),bias,'bias exact')
        equal(data['old'],np.zeros((8,n),dtype='<f4'),'beta=0 old')
    equal(data['suh'],u,'real suh')
    equal(data['svh'],v,'real svh')
    # Read raw physical ranges independently through file seeks, without loader slicing.
    with Path(record['source']).open('rb') as source:
        for local,span in enumerate(record['source_tensors']['trellis']['ranges']):
            source.seek(span['offset'])
            raw=source.read(span['bytes'])
            if raw!=np.ascontiguousarray(data['trellis'][local]).tobytes():
                raise ValueError('Packed physical TP row mismatch')
    seed=SEED+record['layer']*100+{'q_b':0,'o_b':1,'head':2}[record['site']]*10+record['rank']
    equal(data['x'],np.random.Generator(np.random.PCG64(seed)).normal(0,.125,(8,i)).astype('<f4'),'new local original X')
    stages=input_stages_fp32(data['x'],u)
    # F32 x F16 is exact in F64 (<=35 significant bits); this checks RN32
    # before struct RN16 independently of the production NumPy multiply/cast.
    scaled32=(data['x'].astype('f8')*u.astype('f8')).astype('<f4')
    scaled16=scalar_half(scaled32)
    inh=xor_h128(scaled16)
    equal(stages['scaled_fp32'],scaled32,'independent RN32 scale')
    equal(stages['scaled_fp16'],scaled16,'independent struct RN16 scale')
    equal(stages['input_hadamard_fp32'],inh,'independent F32 XOR H128')
    equal(stages['core_input_fp16'],scalar_half(inh),'independent struct RN16 A')
    xb,xbefore,xmask,strided_x=guarded((8,i),19,11,23,0x4a11733d)
    yb,ybefore,ymask,strided_y=guarded((8,n),37,13,29,0xca55a17e)
    strided_x[:]=data['x']
    strided_stages=input_stages_fp32(strided_x,u)
    for key in stages:
        equal(strided_stages[key],stages[key],'CPU asymmetric X stride '+key)
    for key,name in [('scaled','scaled_fp16'),('inh','input_hadamard_fp32'),('a','core_input_fp16')]:
        equal(data[key],stages[name],name)
    equal(mapped(directory/'scaled32.f32','<f4',(8,i)),stages['scaled_fp32'],'RN32 before RN16')
    if not full:
        return
    exact_file=mapped(directory/'core-fp64.f64','<f8',(8,n))
    mass_file=mapped(directory/'core-sumabs.f64','<f8',(8,n))
    ideal_file=mapped(directory/'ideal-unrounded-y.f64','<f8',(8,n))
    ideal_a=hadamard(data['x'].astype('f8')*u.astype('f8'))
    for begin in range(0,n,256):
        end=min(begin+256,n)
        q=checked_q(t[:,begin//16:end//16],table)
        if 'q' in data:
            equal(data['q'][:,begin:end],q,'all Q')
        core=core_dot_fp32(stages['core_input_fp16'],q)
        equal(data['core'][:,begin:end],core,'ordered F32 core all rows')
        bb=None if bias is None else bias[begin:end]
        y=output_stages_fp32(core,v[begin:end],bb)['output_fp32']
        independent_y=xor_h128(core)*v[begin:end].astype('<f4')
        if bb is not None:
            independent_y=independent_y+bb
        equal(y,independent_y,'independent literal output checkpoint')
        equal(data['ref'][:,begin:end],y,'ordered F32 Y all rows including LAST')
        strided_y[:,begin:end]=y
        exact,mass=block_oracle(stages['core_input_fp16'],q)
        equal(exact_file[:,begin:end],exact,'FP64 rounded-operands core')
        equal(mass_file[:,begin:end],mass,'FP64 sumabs')
        gamma=(2*i*2**-24)/(1-2*i*2**-24)
        if np.any(np.abs(core.astype('f8')-exact)>gamma*mass+1e-30):
            raise ValueError('FP64 tolerance exceeded')
        ideal=hadamard(ideal_a @ q.astype('f8'))*v[begin:end].astype('f8')
        if bb is not None:
            ideal+=bb.astype('f8')
        equal(ideal_file[:,begin:end],ideal,'separate ideal unrounded oracle')
        rss()
    equal(strided_y,data['ref'],'CPU asymmetric Y stride all rows')
    if not np.array_equal(xb[xmask],xbefore[xmask]) or not np.array_equal(yb[ymask],ybefore[ymask]):
        raise ValueError('CPU asymmetric leading/gap/tail canary modified')
    h.verify()


def negatives(out,record,names,table):
    """Copies only, outside models. Keep every corruption and exact failure text."""
    root=out/'corruption-negatives'
    root.mkdir(exist_ok=False)
    source=out/record['name']
    results=[]
    for case in ('LAST-row-Y','packed-bit','scale','TP-axis','misalignment','marker','nonfinite-scale'):
        folder=root/case
        folder.mkdir()
        modified=json.loads(json.dumps(record))
        # Unmodified sidecars are linked read-only by convention, never written.
        for name in record['files']:
            if name!='fixture.stgpu':
                os.link(source/name,folder/name)
        shutil.copyfile(source/'fixture.stgpu',folder/'fixture.stgpu')
        path=folder/'fixture.stgpu'
        parts=strict_parse(path)
        if case in ('LAST-row-Y','packed-bit','scale','marker','nonfinite-scale'):
            tag={'LAST-row-Y':'ref','packed-bit':'trellis','scale':'suh','marker':'meta','nonfinite-scale':'svh'}[case]
            offset=parts[tag][0]+(parts[tag][1]-4 if case=='LAST-row-Y' else 16 if case=='marker' else 0)
            with path.open('r+b') as f:
                f.seek(offset)
                if case=='nonfinite-scale':
                    f.write(struct.pack('<e',float('inf')))
                elif case=='scale':
                    value=struct.unpack('<e',f.read(2))[0]
                    f.seek(offset)
                    f.write(struct.pack('<e',-value if value else 1.))
                else:
                    b=f.read(1)
                    f.seek(offset)
                    f.write(bytes([b[0]^1]))
        elif case=='TP-axis':
            modified['local_bounds_I_N']=[0,640,0,32768]
        else:
            modified['local_bounds_I_N'][2]+=16
        dump(folder/'manifest.json',modified)
        try:
            validate(folder,modified,names,table,hashes=False)
        except ValueError as e:
            results.append(dict(case=case,rejected=True,error=str(e),sha256=sha_file(path),
                                hash_gate_disabled_to_exercise_semantics=True))
        else:
            raise ValueError('Corruption escaped: '+case)
        dump(root/'results.json',results)
    # Direct alignment and wrong-axis selector tests cannot rely on a manifest gate.
    t=np.zeros((512,320,80),dtype='<i2')
    for bounds in [(16,4112,0,5120),(0,4096,16,5008)]:
        try:
            aligned(t,np.ones(8192,dtype='<f2'),np.ones(5120,dtype='<f2'),bounds)
        except ValueError:
            pass
        else:
            raise ValueError('Misaligned selector escaped')
    for shape,site in [((4096,10240),'o_b'),((2560,16384),'q_b')]:
        try:
            partition(*shape,site,0)
        except ValueError:
            pass
        else:
            raise ValueError('Wrong TP axis escaped')
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=BASE)
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--negatives',action='store_true')
    p.add_argument('--primary-only',action='store_true',help='Validate the four priority L0 attention fixtures')
    args=p.parse_args()
    low_priority()
    started,cpu=time.perf_counter(),time.process_time()
    headers,names=scan(args.base)
    frozen=json.loads((args.out/'source-headers.json').read_text(encoding='utf-8'))
    current={str(h.path):h for h in headers}
    if set(current)!={entry['file'] for entry in frozen}:
        raise ValueError('Actual source file inventory changed')
    for entry in frozen:
        h=current[entry['file']]
        if h.before!=entry['stat'] or h.header_sha256!=entry['header_sha256'] or h.header!=entry['header']:
            raise ValueError('Frozen source header/stat changed')
    table=independent_lut('mul1')
    records=[]
    for name in (('dense-manifest.json',) if args.primary_only else ('dense-manifest.json','head-manifest.json')):
        records.extend(json.loads((args.out/name).read_text(encoding='utf-8'))['fixtures'])
    if args.primary_only:
        records=[r for r in records if r['layer']==0]
    expected=4 if args.primary_only else 14
    if len(records)!=expected or len({r['name'] for r in records})!=expected:
        raise ValueError('Expected exactly 12 dense + 2 head fixtures')
    checks=[]
    for record in records:
        validate(args.out/record['name'],record,names,table)
        checks.append(dict(name=record['name'],passed=True))
        print('VALIDATED',record['name'],flush=True)
    bad=negatives(args.out,records[0],names,table) if args.negatives else []
    total=sum(p.stat().st_size for p in args.out.rglob('*') if p.is_file())
    if total>=4*1024**3:
        raise ValueError('Output disk budget exceeded')
    result=dict(passed=True,fixtures=checks,corruption_negatives=bad,total_disk_bytes=total,
                elapsed_wall_seconds=time.perf_counter()-started,process_cpu_seconds=time.process_time()-cpu,
                peak_working_set_bytes=rss(),cpu_only=True,gpu_byte_identity='OPEN',
                cpu_strides=dict(x_extra=19,y_extra=37,x_lead=11,y_lead=13,x_tail=23,y_tail=29,
                    canary_patterns=['0x4a11733d XOR absolute_index','0xca55a17e XOR absolute_index'],
                    native_HIP_certificate=False),source_header_stat_snapshot_unchanged=True,
                validator_sha256=sha_file(Path(__file__)),numpy=np.__version__)
    dump(args.out/('validation-primary.json' if args.primary_only else 'validation.json'),result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
