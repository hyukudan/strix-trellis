"""Bit-level reference expressed as circular integers, independent of GPU kernels."""
from functools import lru_cache
import math
import struct

import numpy as np

CODEBOOKS = ("3inst", "mcg", "mul1")
MUL1_MULTIPLIER = 0x83DCD12D


def step_widths(k):
    """Return the 256 transition widths for supported EXL3 bitrates."""
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, float, np.integer, np.floating)):
        raise ValueError("k must be integer 1..8 or one of 1.5, 2.5, 3.5")
    rate = float(k)
    if not math.isfinite(rate) or rate not in (*range(1, 9), 1.5, 2.5, 3.5):
        raise ValueError("Unsupported EXL3 bitrate")
    base = int(rate)
    return tuple(base + (i % 2 if rate != base else 0) for i in range(256))


def _half(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


@lru_cache(maxsize=3)
def _codebook(name):
    if name not in CODEBOOKS:
        raise ValueError(f"Unsupported codebook: {name}")
    table = np.empty(65536, dtype=np.float16)
    for state in range(65536):
        if name == "mul1":
            product = (state * MUL1_MULTIPLIER) % (1 << 32)
            byte_sum = sum(product.to_bytes(4, "little"))
            # Exact binary64 arithmetic followed by ONE half rounding models half FMA.
            value = (1024 + byte_sum) * _half(0x1EEE) + _half(0xC931)
        else:
            multiplier, offset = ((89226354, 64248484) if name == "3inst"
                                  else (0xCBAC1FED, 0))
            mixed = ((state * multiplier + offset) % (1 << 32))
            halves = (mixed & 0x8FFF8FFF) ^ 0x3B603B60
            value = _half(halves % 65536) + _half(halves // 65536)
        table[state] = value
    table.flags.writeable = False
    return table


def decode_states(states, codebook="mul1"):
    """Map integer states modulo 2**16 to binary16 codebook values."""
    indices = np.asarray(states)
    if indices.dtype.kind not in "iu":
        raise ValueError("States must be integers")
    # Array casts retain low bits for every integer width, including signed int16.
    return _codebook(codebook)[indices.astype(np.uint16)]


def unpack_states(packed, k):
    """Decode (..., 16*k) signed/unsigned 16-bit words to (..., 256) states.

    Each little-endian uint32 contains MSB-first ring bits. The first state ends
    after the first transition, so its high bits come from the end of the tile.
    """
    widths = step_widths(k)
    words = np.asarray(packed)
    size = sum(widths)
    if words.ndim < 1 or words.shape[-1] != size // 16:
        raise ValueError(f"Expected {size // 16} words per tile")
    if words.dtype.kind not in "iu" or words.dtype.itemsize != 2:
        raise ValueError("Packed storage must contain int16 or uint16 words")
    flat = words.reshape(-1, size // 16)
    result = np.empty((len(flat), 256), dtype=np.uint16)
    for tile_index, tile in enumerate(flat):
        # Reinterpret signed words numerically; construct a big-endian bit ring.
        ring = 0
        for pair in range(0, len(tile), 2):
            lo = int(tile[pair]) & 65535
            hi = int(tile[pair + 1]) & 65535
            ring = (ring << 32) | lo | (hi << 16)
        repeated = (ring << size) | ring
        end = 0
        for position, width in enumerate(widths):
            end += width
            result[tile_index, position] = (repeated >> (size - end)) & 65535
    return result.reshape(words.shape[:-1] + (256,))


def decode_tile(packed, k, codebook="mul1"):
    """Return (..., 16, 16) tiles in logical row/column order (binary16)."""
    step_widths(k)
    if float(k) % 1 and codebook != "mul1":
        raise ValueError("Half-integer bitrates require mul1")
    values = decode_states(unpack_states(packed, k), codebook)
    tiles = np.empty(values.shape[:-1] + (16, 16), dtype=np.float16)
    # Derive stream position from each logical coordinate's binary digits.
    for row in range(16):
        for col in range(16):
            position = ((col & 7) << 5) | ((row & 6) << 2)
            position |= (row & 1) | ((row & 8) >> 2) | ((col & 8) >> 1)
            tiles[..., row, col] = values[..., position]
    return tiles


@lru_cache(maxsize=16)
def _window_indices(k):
    """Bit indices [256, 16] (MSB first) of every state window in the tile ring."""
    widths = np.array(step_widths(k), dtype=np.int64)
    size = int(widths.sum())
    ends = np.cumsum(widths)
    indices = (ends[:, None] - 16 + np.arange(16)[None, :]) % size
    indices.flags.writeable = False
    return indices


_TILE_POSITION = np.empty((16, 16), dtype=np.int64)
for _row in range(16):
    for _col in range(16):
        _TILE_POSITION[_row, _col] = (((_col & 7) << 5) | ((_row & 6) << 2) | (_row & 1) |
                                      ((_row & 8) >> 2) | ((_col & 8) >> 1))
_TILE_POSITION.flags.writeable = False


def unpack_states_vectorized(packed, k, chunk_tiles=8192):
    """Array form of `unpack_states`: same ring, windows and output, bounded memory.

    Each tile's 16-bit word pairs form little-endian uint32 values whose bits are
    consumed MSB first; the window of position i is the 16 ring bits ending at
    the cumulative width S(i), modulo the ring length.
    """
    widths = step_widths(k)
    words = np.asarray(packed)
    size = sum(widths)
    if words.ndim < 1 or words.shape[-1] != size // 16:
        raise ValueError(f"Expected {size // 16} words per tile")
    if words.dtype.kind not in "iu" or words.dtype.itemsize != 2:
        raise ValueError("Packed storage must contain int16 or uint16 words")
    flat = words.reshape(-1, size // 16)
    windows = _window_indices(k)
    result = np.empty((len(flat), 256), dtype=np.uint16)
    for begin in range(0, len(flat), chunk_tiles):
        tiles = np.ascontiguousarray(flat[begin:begin + chunk_tiles]).astype("<u2", copy=False)
        pairs = tiles.reshape(len(tiles), -1, 2).astype(np.uint32)
        ring_words = (pairs[..., 0] | (pairs[..., 1] << 16)).astype(">u4")
        bits = np.unpackbits(ring_words.view(np.uint8), axis=-1)
        packed_windows = np.packbits(bits[:, windows], axis=-1)  # MSB-first 16 bits
        result[begin:begin + len(tiles)] = packed_windows.view(">u2")[..., 0]
    return result.reshape(words.shape[:-1] + (256,))


def dequantize_rotated_vectorized(trellis, k=None, codebook="mul1", chunk_tiles=8192):
    """Array form of `dequantize_rotated`; identical binary16 output."""
    data = np.asarray(trellis)
    if data.ndim != 3 or not data.shape[0] or not data.shape[1]:
        raise ValueError("Trellis must have three nonempty dimensions")
    if k is None:
        k = data.shape[-1] / 16
    step_widths(k)
    if float(k) % 1 and codebook != "mul1":
        raise ValueError("Half-integer bitrates require mul1")
    values = decode_states(unpack_states_vectorized(data, k, chunk_tiles), codebook)
    tiles = values[..., _TILE_POSITION]
    return tiles.transpose(0, 2, 1, 3).reshape(data.shape[0] * 16, data.shape[1] * 16)


def dequantize_rotated(trellis, k=None, codebook="mul1"):
    """Convert [input/16, output/16, words] tiles to [input, output] weights."""
    data = np.asarray(trellis)
    if data.ndim != 3 or not data.shape[0] or not data.shape[1]:
        raise ValueError("Trellis must have three nonempty dimensions")
    if k is None:
        k = data.shape[-1] / 16
    tiles = decode_tile(data, k, codebook)
    return tiles.transpose(0, 2, 1, 3).reshape(data.shape[0] * 16, data.shape[1] * 16)
