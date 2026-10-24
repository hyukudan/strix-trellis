#!/usr/bin/env python3
"""Independent precision/order reference, NOT a bit-exact GPU dot emulator.

The 4096-case gfx1151 capture rejects both one-rounding and two-rounding IEEE
models. This exact-rational arithmetic reference emulates the lane/S schedule
under an explicit ideal IEEE hypothesis and tests precision only. Actual byte
identity MUST use GPU-vs-GPU legacy P/Y with the same S and operation order.
"""
import struct
from fractions import Fraction
import numpy as np


def round32(value):
    """Round an exact rational to IEEE binary32, ties to even, without double rounding."""
    value = Fraction(value)
    sign = int(value < 0)
    value = abs(value)
    if not value:
        return np.float32(0)
    n, d = value.numerator, value.denominator
    exponent = n.bit_length() - d.bit_length()
    if (n < (d << exponent)) if exponent >= 0 else ((n << -exponent) < d):
        exponent -= 1
    quantum = max(exponent - 23, -149)
    if quantum >= 0:
        d <<= quantum
    else:
        n <<= -quantum
    q, rem = divmod(n, d)
    if 2 * rem > d or (2 * rem == d and q & 1):
        q += 1
    # q*2**quantum is exactly representable in binary64 for finite binary32.
    result = np.float32(float(q) * (2.0 ** quantum))
    return -result if sign else result


def fdot2(w0, x0, w1, x1, acc):
    args = [Fraction(float(v)) for v in (np.float16(w0),np.float16(x0),
                                        np.float16(w1),np.float16(x1),np.float32(acc))]
    total = args[0]*args[1] + args[2]*args[3] + args[4]
    if not total and not args[0]*args[1] and not args[2]*args[3] and not args[4]:
        negative = (np.signbit(w0)!=np.signbit(x0)) and (np.signbit(w1)!=np.signbit(x1)) and np.signbit(acc)
        return np.float32(-0.) if negative else np.float32(0.)
    return round32(total)


def partials(q, activation, splits):
    """q is [K,N] decoded F16, activation [R,K] transformed F16.

    Emulates the lane map, the two fdot2 updates per K16/output, and the
    shuffle tree (lane0+lane2)+(lane1+lane3), without using the GPU scheduler.
    """
    q = np.asarray(q, dtype=np.float16)
    activation = np.asarray(activation, dtype=np.float16)
    rows, inputs = activation.shape
    assert q.shape[0] == inputs and inputs % 16 == 0
    out = np.zeros((splits, rows, q.shape[1]), dtype=np.float32)
    for s in range(splits):
        for r in range(rows):
            for n in range(q.shape[1]):
                lanes = np.zeros(4, dtype=np.float32)
                for k in range((inputs // 16) * s // splits, (inputs // 16) * (s + 1) // splits):
                    for lane in range(4):
                        for half in (0, 8):
                            i = k * 16 + lane * 2 + half
                            lanes[lane] = fdot2(q[i,n],activation[r,i],q[i+1,n],activation[r,i+1],lanes[lane])
                out[s,r,n] = np.float32(np.float32(lanes[0] + lanes[2]) + np.float32(lanes[1] + lanes[3]))
    return out


def fold(p, arrival=None):
    # Arrival order affects publication, never the ascending split reduction.
    published = np.empty_like(p)
    for s in range(len(p)) if arrival is None else arrival:
        published[s] = p[s]
    total = np.zeros(p.shape[1:], dtype=np.float32)
    for plane in published:
        total = np.add(total, plane, dtype=np.float32)
    return total


def write_native_cases(path):
    rng = np.random.default_rng(20261021)
    with open(path, "wb") as f:
        for i in range(4096):
            values = (rng.normal(size=4) * 2.0 ** (i % 20 - 10)).astype(np.float16)
            acc = np.float32(rng.normal() * 2.0 ** (i % 30 - 15))
            if i == 0:
                values = np.array([1,1,2**-12,2**-12],dtype=np.float16)
                acc = np.float32(2**-24)
            elif i == 1:
                values = np.array([2**-12,2**-12,1,1],dtype=np.float16)
                acc = np.float32(2**-24)
            expected = fdot2(*values, acc)
            # Store w0,x0,w1,x1 as raw halves, then acc and expected F32.
            f.write(values.astype('<f2').tobytes() + struct.pack('<ff',acc,expected))


def write_core_cases(path):
    from make_rank_fixtures import decode_columns
    rng = np.random.default_rng(20261022)
    with open(path,'wb') as f:
        f.write(struct.pack('<4I',512,16,2,2))
        for bits in (4,5):
            packed = rng.integers(0,65536,size=(32,1,bits*16),dtype=np.uint16)
            a = rng.normal(size=(2,512)).astype('<f2')
            q = decode_columns(packed,bits)
            f.write(struct.pack('<I',bits)+packed.astype('<u2').tobytes()+a.tobytes())
            for s in (4,16,32):
                f.write(partials(q,a,s).astype('<f4').tobytes())


if __name__ == '__main__':
    import sys
    write_native_cases(sys.argv[1])
