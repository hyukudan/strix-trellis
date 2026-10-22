# EXL3 CPU contract

> **Type:** user documentation.

Independent reference in `python/strix_trellis`, using the standard library and
NumPy. The contract was first published as provisional to support parallel
implementation work. It defines the implemented interpretation. It **does not
certify parity with an exllamav3 execution or full model compatibility**.

## Official sources, revisions and licenses

Source code was read to investigate the format, without copying implementations.
The reference uses Python integers for ring arithmetic, a scalar codebook table and NumPy
transforms. It does not import or execute upstream code.

Primary EXL3 source: revision `a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a`,
[MIT](https://github.com/turboderp-org/exllamav3/blob/a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a/LICENSE) license.
Files consulted, pinned to that revision:

- [codebook.cuh](https://github.com/turboderp-org/exllamav3/blob/a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a/exllamav3/exllamav3_ext/quant/codebook.cuh): codebook constants and arithmetic.
- [pack.cu](https://github.com/turboderp-org/exllamav3/blob/a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a/exllamav3/exllamav3_ext/quant/pack.cu): word packing.
- [exl3_dq.cuh](https://github.com/turboderp-org/exllamav3/blob/a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a/exllamav3/exllamav3_ext/quant/exl3_dq.cuh): 16-bit windows.
- [reconstruct.cu](https://github.com/turboderp-org/exllamav3/blob/a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a/exllamav3/exllamav3_ext/quant/reconstruct.cu): fragment order and bilateral transform.
- [bits_k.cuh](https://github.com/turboderp-org/exllamav3/blob/a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a/exllamav3/exllamav3_ext/quant/bits_k.cuh) and [frac.cuh](https://github.com/turboderp-org/exllamav3/blob/a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a/exllamav3/exllamav3_ext/quant/frac.cuh): supported rates.
- [exl3.py](https://github.com/turboderp-org/exllamav3/blob/a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a/exllamav3/modules/quant/exl3.py): names, orientation and scales.
- [tests/testlib/trellis.py](https://github.com/turboderp-org/exllamav3/blob/a3bfbfd0d3c0cea67815146a19fcd2bb3f84709a/tests/testlib/trellis.py): documentary corroboration; not used as an executable oracle.

The model declares conversion with v1.6.1. The following were also consulted:
[codebook.cuh](https://github.com/turboderp-org/exllamav3/blob/f45b84512ba65bca77f7d394399d4b68821fc71b/exllamav3/exllamav3_ext/quant/codebook.cuh)
and [exl3_dq.cuh](https://github.com/turboderp-org/exllamav3/blob/f45b84512ba65bca77f7d394399d4b68821fc71b/exllamav3/exllamav3_ext/quant/exl3_dq.cuh)
from tag v1.6.1, revision `f45b84512ba65bca77f7d394399d4b68821fc71b`, MIT.
The constants and integer windows agree with the implemented interpretation.

Cornell QTIP: revision `e90c6688c8dfae326a3a81b5eb032db7c6680ec0`,
[GNU GPL v3](https://github.com/Cornell-RelaxML/qtip/blob/e90c6688c8dfae326a3a81b5eb032db7c6680ec0/LICENSE) license.
Sources consulted: [bitshift.py](https://github.com/Cornell-RelaxML/qtip/blob/e90c6688c8dfae326a3a81b5eb032db7c6680ec0/lib/codebook/bitshift.py)
and [quantized_linear.py](https://github.com/Cornell-RelaxML/qtip/blob/e90c6688c8dfae326a3a81b5eb032db7c6680ec0/lib/linear/quantized_linear.py).
QTIP provides the concepts of bitshift trellises, procedural codebooks and
incoherence. Its storage and variants are not assumed interchangeable with
EXL3. Its `1mad`/`2mad` codebooks are not EXL3 `mul1`.

## Storage and states

The rotated matrix has orientation `[input, output]`. Its trellis has shape
`[input/16, output/16, 16*K]`, with int16 or uint16 words. Each logical tile is
16×16. K is the number of bits per weight for that tensor, not the global model
average. Integer rates 1–8 are supported, along with 1.5, 2.5 and 3.5 exclusively
with `mul1`.

Each word pair forms `uint32 = low16 | (high16 << 16)`. On-disk bytes are
little-endian, and the stream consumes bit 31 of each uint32 first. The ring
contains `R = 256*K` bits, with no extra initial state. Each position consumes K
bits for integer rates, or `floor(K) + (position % 2)` for half-integer rates
(repeated mask `0xAAAA`).

Let `S(i)` be the sum of widths from position 0 through i, inclusive. The state
at i consists of the 16 bits `[S(i)-16, S(i))`, MSB first, with indices modulo R.
The first window incorporates bits from the end of the same tile. All bit
patterns of length R are valid.

The state at position `8*lane + fragment`, lane 0–31 and fragment 0–7, maps to:

```text
row = 2*(lane mod 4) + (fragment mod 2) + 8*(floor(fragment/2) mod 2)
col = floor(lane/4) + 8*floor(fragment/4)
```

Tiles are stored first by input block, then by output block. No additional
transpose is applied when using `x @ W`.

## Codebooks and rounding

States are reduced modulo 65536 and integer products modulo 2^32. The result
is binary16, rounded to nearest with ties to even.

| Name | Procedural product | Conversion |
|---|---|---|
| `3inst` | `p = state*89226354 + 64248484` | `q = (p & 0x8FFF8FFF) xor 0x3B603B60`; sum its binary16 halves with one final rounding |
| `mcg` | `p = state*0xCBAC1FED` | Same conversion of two halves |
| `mul1` | `p = state*0x83DCD12D` | Sum b of four unsigned bytes; `half((1024+b)*(1774/262144) - 1329/128)` with one rounding |

For `mul1`, the binary16 constants have bit patterns `0x1EEE` and `0xC931`.
Rounding the product to half before adding does NOT reproduce the FMA.
The reference evaluates the exact expression in float64 before converting to
half. The loader checks the int32 `mul1` marker as uint32 `0x83DCD12D`. It does
not accept other multipliers or implement additional QTIP codebooks.

## Hadamard, signs and scales

H is block diagonal, with 128×128 Sylvester blocks divided by sqrt(128), in
natural order. Both layer dimensions must be multiples of 128.
`suh[input]` and `svh[output]` already contain signs and magnitudes:

```text
W = diag(suh) H_input W_rotated H_output diag(svh)
y = ((x * suh) H_input W_rotated) H_output * svh + bias
```

No sign is removed and no additional global factor is introduced. Transforms
and matmul use float64. Intermediate rounding of activations, half/int8 kernels,
GPU accumulators and MoE expert selection are not reproduced.

## API and reproduction

Install with `python -m pip install --no-build-isolation --no-deps -e ./python`
(requires NumPy to be installed). Alternatively, add `python/` to `PYTHONPATH`.

| API | Input / output |
|---|---|
| `step_widths(k)` | Tuple of 256 widths |
| `unpack_states(packed, k)` | `[..., 16*k]` 16-bit words → `[..., 256]` uint16 |
| `decode_states(states, codebook="mul1")` | Integer states → float16 |
| `decode_tile(packed, k, codebook="mul1")` | Words → logical `[..., 16, 16]` |
| `dequantize_rotated(trellis, k=None, codebook="mul1")` | 3D trellis → float16 matrix; K inferred from the last dimension |
| `hadamard(values, block_size=128, axis=-1)` | Normalized float64 transform |
| `reconstruct_weight(trellis, suh, svh, k=None, codebook="mul1")` | Original float64 matrix `[input, output]` |
| `matmul(x, trellis, suh, svh, k=None, codebook="mul1", bias=None)` | `x[..., input]` → float64 `y[..., output]` |
| `load_fixture(manifest, directory=None)` | Checks SHA-256 and marker; returns `(trellis, suh, svh, k)` |

```powershell
python tools/fetch_fixture.py
$env:PYTHONPATH = "$PWD/python"
python -m unittest discover -s tests -v
```

The tests do not use the network. The three tests using real data are explicitly
skipped if the external directory is missing. To claim sample validation, they
must pass without skips.

## Real-data evidence and limits

Model: [Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw/tree/45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b),
revision `45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b`.
The [model card](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw/blob/45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b/README.md)
declares EAQ v1 with standard EXL3 format, `mul1` and output scales always
enabled. This is the publisher's statement, not an independent certification.
Model license: [MIT](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw/blob/45ee5d0fd106c94a3d77f721e09cbb2f50e5a16b/LICENSE).

Sample: complete group `layers.0.attn.wkv` from shard 1 of 39; I16 trellis
`[320,32,96]` (K=6), F16 `suh` `[5120]`, F16 `svh` `[512]`, scalar I32 `mul1`.
It is stored in a local fixture directory (set `STRIX_TRELLIS_FIXTURES`). The
manifest `tests/fixtures/eaq_v1.json` records the pinned URL, shapes, dtypes,
absolute ranges `[start, exclusive end)`, sizes and SHA-256 hashes.

**2,470,950 bytes** were downloaded through six HTTP 206 ranges: an 8-byte
prefix, a 493,594-byte header and four tensors (1,977,348 bytes). The downloader
requires an exact Content-Range before reading the body and limits reads and
the budget to 90,000,000 bytes. It never requests the entire shard. It selects
the smallest complete `trellis/suh/svh/mul1` group from the first shard.

The sample passes integrity checks, window checks against a separate byte
reader, finite reconstruction and algebraic equivalence between two CPU matmul
paths. Both paths share codebook and layout rules. No golden output produced
by upstream was available at this milestone. Rates other than K=6 and codebooks
other than `mul1` are tested with synthetic data. Layer/logit validation, a
general safetensors loader, packed `su/sv` signs, other multipliers and parity
with actual kernels were still pending at this milestone.

## Subsequent validation status

The preceding section records the scope of the first CPU milestone. The external
original CUDA oracle was run later: 2,621,440 values from the real wkv and
65,536 states per codebook match bit for bit. Seven real
dense K5/K6 groups, the generic modern mmap loader and the historical su/sv
helper were added. The complete HIP API passed 56 fixtures up to M256. Full-model logits, external out_scales and old
containers without a scale contract remain outside that qualification.
