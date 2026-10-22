#!/usr/bin/env python3
"""Prepare a synthetic or shared-cache EXL3 head with H128-block references."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from strix_trellis.linear import matmul_fp32_stages
from strix_trellis.container import SafetensorsFile, load_exl3


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8<<20), b''):
            result.update(data)
    return result.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source-safetensors', type=Path,
                   help='Read the existing local EAQ head instead of generating weights')
    p.add_argument('--source-kind', choices=('shared-eaq-v1','ours-local-snapshot'), default='shared-eaq-v1')
    args = p.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        p.error('Use a fresh output directory; existing fixtures are never replaced')
    args.out.mkdir(parents=True, exist_ok=True)
    blob = args.out/'head.blob'
    started = time.perf_counter()
    inputs, outputs, bits, rows = 5120, 129280, 6, 8
    seed = 20261022
    rng = np.random.Generator(np.random.PCG64(seed))
    packed_bytes = inputs*outputs*bits//8
    source = None
    weights = None
    if args.source_kind == 'ours-local-snapshot' and args.source_safetensors is None:
        p.error('OUR head requires an explicit local safetensors source')
    if args.source_safetensors is not None:
        file = SafetensorsFile(args.source_safetensors)
        source_stat = file.path.stat()
        weights = load_exl3(file, 'head')
        accepted = (6,) if args.source_kind == 'shared-eaq-v1' else (4,5)
        if (weights.input_size != inputs or weights.output_size != outputs or
                weights.bits not in accepted or weights.codebook != 'mul1' or weights.bias is not None):
            p.error('Unexpected head geometry, codebook, bias or source-specific rate')
        bits = int(weights.bits)
        packed_bytes = inputs*outputs*bits//8
        with file.path.open('rb') as stream:
            header_sha = hashlib.sha256(stream.read(file.data_offset)).hexdigest()
        source = dict(kind='existing_shared_cache' if args.source_kind == 'shared-eaq-v1' else 'existing_ours_snapshot', file_name=file.path.name,
                      header_sha256=header_sha, prefix='head',
                      file_bytes=source_stat.st_size,
                      selected_tensors={name: file.header['head.'+name]
                                        for name in ('trellis', 'suh', 'svh', 'mul1')},
                      independent_payload_redownload=False)
        source['tensor_sha256'] = {
            name: hashlib.sha256(file.tensor('head.'+name).tobytes()
                                 if name == 'mul1' else getattr(weights,name)).hexdigest()
            for name in ('trellis', 'suh', 'svh', 'mul1')}
    # Stream one I16 row at a time: private generation buffers stay below 2 MiB.
    temporary = args.out/'head.blob.tmp'
    with temporary.open('xb') as f:
        for row in range(inputs//16):
            if weights is None:
                packed = rng.integers(0, 65536, (outputs//16,16*bits), dtype=np.uint16)
                packed.astype('<u2',copy=False).tofile(f)
            else:
                np.asarray(weights.trellis[row],dtype='<i2').tofile(f)
        if weights is None:
            suh = (rng.uniform(0.03125,1.5,inputs)*rng.choice([-1,1],inputs)).astype('<f2')
            svh = (rng.uniform(0.03125,1.25,outputs)*rng.choice([-1,1],outputs)).astype('<f2')
        else:
            suh = np.array(weights.suh,dtype='<f2',copy=True)
            svh = np.array(weights.svh,dtype='<f2',copy=True)
        suh.tofile(f); svh.tofile(f)
    if source is not None:
        final_stat = file.path.stat()
        if ((final_stat.st_size,final_stat.st_mtime_ns) !=
                (source_stat.st_size,source_stat.st_mtime_ns)):
            raise RuntimeError('Shared source changed while reading; fixture was not published')
    temporary.replace(blob)
    trellis = np.memmap(blob, mode='r',dtype='<u2',shape=(inputs//16,outputs//16,16*bits))
    x = rng.normal(0,0.25,(rows,inputs)).astype('<f4')
    x[0]=0; x[0,[0,1,31,32,127,128,5119]]=[1,-0.5,0.25,-1,0.5,-0.25,1]
    x.tofile(args.out/'input.f32')
    blocks = [0,(outputs//2//128)*128,outputs-128]
    expected = []
    for begin in blocks:
        selected = np.array(trellis[:,begin//16:begin//16+8,:],copy=True)
        stages = matmul_fp32_stages(x,selected,suh,svh[begin:begin+128],bits,'mul1')
        expected.append(stages['output_fp32'])
    golden = np.stack(expected).astype('<f4')
    golden.tofile(args.out/'expected.f32')
    blob_bytes = packed_bytes+2*(inputs+outputs)
    version = 1 if source is None else (2 if args.source_kind == 'shared-eaq-v1' else 3)
    metadata = struct.pack('<8s8I6Q', b'STHEAD01',version,rows,inputs,outputs,bits,2,0x83DCD12D,len(blocks),
                           blob_bytes,packed_bytes,packed_bytes,packed_bytes+2*inputs,x.nbytes,golden.nbytes)
    metadata += struct.pack('<'+'I'*len(blocks),*blocks)
    if source is not None:
        metadata += struct.pack('<I',1 if version==2 else 2)
    (args.out/'head.meta').write_bytes(metadata)
    files = ['head.blob','head.meta','input.f32','expected.f32']
    records = {name:dict(bytes=(args.out/name).stat().st_size,sha256=sha(args.out/name)) for name in files}
    manifest = dict(format='STHEAD01',version=version,
                    source_weights='synthetic' if source is None else args.source_kind,
                    actual_model_values=source is not None,
                    geometry_source='tests/fixtures/eaq_head_metadata.json',
                    geometry_sha256=sha(ROOT/'tests/fixtures/eaq_head_metadata.json'),
                    input=inputs,output=outputs,bits=bits,codebook='mul1',rows=rows,seed=seed,
                    sampled_H128_blocks=blocks,total_H128_blocks=outputs//128,
                    reference='strix_trellis.linear.matmul_fp32_stages',numpy=np.__version__,files=records)
    manifest['generator_sha256'] = sha(Path(__file__))
    manifest['reference_source_sha256'] = {
        name: sha(ROOT/'python/strix_trellis'/name) for name in ('codec.py','linear.py')}
    if source is not None:
        manifest['source'] = source
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    (args.out/'manifest.sha256').write_text(''.join(f"{v['sha256']}  {k}\n" for k,v in records.items()),encoding='utf-8',newline='\n')
    print(f"Prepared {manifest['source_weights']} head {blob_bytes} bytes, {len(blocks)}/{outputs//128} H128 blocks, {time.perf_counter()-started:.3f}s")


if __name__ == '__main__':
    main()
