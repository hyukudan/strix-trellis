#!/usr/bin/env python3
"""CPU proof of a register-only EXL3-to-WMMA B-fragment mapping, bits5/6."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

MASK = (1<<32)-1


def scalar_state(words, state, bits):
    size = len(words)*32
    start = ((state+1)*bits-16) % size
    value = 0
    for j in range(16):
        position = (start+j) % size
        value = (value<<1) | ((words[position//32]>>(31-position%32))&1)
    return value


def register_pairs(data, lane, bits):
    base = (lane&7)*bits-1
    # Only the first word can wrap before this tile; no successor tile is read.
    words = [data[base if base>=0 else len(data)-1]]
    words.extend(data[base+j] for j in range(1,bits+1))
    cls = 2*(lane>>4)+4*((lane>>3)&1)
    result = []
    for pair in range(4):
        constant = 16+(1+8*pair)*bits
        first = constant//32
        start = constant+cls*bits
        upper = int(start>=32*(first+1))
        assert start//32 == first+upper
        low = words[first+upper]
        high_index = first+upper+1
        high = words[high_index] if high_index<=bits else 0
        shift = start&31
        assert 0<shift<32
        aligned = ((low<<shift)&MASK) | (high>>(32-shift))
        result.extend([aligned>>16,(aligned>>(16-bits))&65535])
    return result


def check_tile(data, bits):
    logical = [[None]*16 for _ in range(16)]
    for lane in range(32):
        for j in range(8):
            r = 2*(lane%4)+(j%2)+8*((j//2)%2)
            c = lane//4+8*(j//4)
            assert logical[r][c] is None
            logical[r][c] = scalar_state(data,8*lane+j,bits)
    owned = [register_pairs(data,lane,bits) for lane in range(32)]
    fragments = []
    for lane in range(32):
        peer = lane^16
        low = owned[lane] if lane<16 else owned[peer]
        high = owned[peer] if lane<16 else owned[lane]
        fragment = low+high
        expected = [logical[k][lane%16] for k in range(16)]
        if fragment != expected:
            raise AssertionError(dict(bits=bits,lane=lane,actual=fragment,expected=expected))
        fragments.append(fragment)
    assert fragments[:16] == fragments[16:]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--json',type=Path)
    args = p.parse_args()
    started = time.perf_counter();rng = random.Random(20261023)
    counts = []
    for bits in (5,6):
        tiles = 0
        for position in range(256*bits):
            data = [0]*(bits*8)
            data[position//32] = 1<<(31-position%32)
            check_tile(data,bits);tiles+=1
        for _ in range(8):
            check_tile([rng.getrandbits(32) for _ in range(bits*8)],bits);tiles+=1
        counts.append(dict(bits=bits,basis_tiles=256*bits,random_tiles=8,tiles=tiles,
                           decoded_states=tiles*256,fragment_values=tiles*32*16))
    # The accumulator mapping is unchanged; it stores each 16x16 cell once.
    outputs = {(2*j+lane//16,lane%16) for lane in range(32) for j in range(8)}
    assert outputs == {(r,c) for r in range(16) for c in range(16)}
    result = dict(status='passed_cpu_only',seed=20261023,results=counts,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  elapsed_seconds=time.perf_counter()-started,gpu_executed=False)
    if args.json:
        args.json.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
