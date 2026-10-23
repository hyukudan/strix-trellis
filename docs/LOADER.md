# Reading EXL3 tensors

> **Type:** user documentation.

`SafetensorsFile` opens a local file and maps only the selected payloads.
It does not load the entire shard or depend on PyTorch. It accepts modern groups
with trellis I16, suh/svh F16 and optional F16/F32 bias. It validates the header,
duplicate JSON keys, shapes, offsets/sizes, overlaps among selected tensors,
markers and finite scales. It does not certify tensors outside the selected
group or their other formats.

```python
from strix_trellis import load_exl3

weights = load_exl3("model-00001-of-00039.safetensors", "layers.0.attn.wkv")
print(weights.input_size, weights.output_size, weights.bits, weights.codebook)
y = weights.matmul(x, production=True)  # CPU reference; this is not the HIP path.
blob, metadata_fields = weights.transport_blob()
```

Unquantized BF16 tensors (embeddings, norms) have no NumPy dtype: `tensor(name)`
rejects them unless `raw_bf16=True`, which maps the raw little-endian bit patterns as
uint16. `tensor_float32(name)` copies a BF16, F16 or F32 tensor to float32 (an exact
widening). EXL3 groups never contain BF16 regions.

## Preparation without an intermediate blob

To pass the four regions to the native preparation function without building a
complete copy of the group in Python:

```python
regions, metadata_fields = weights.transport_regions()
# regions: trellis/suh/svh/bias -> byte memoryview, or None if there is no bias.
# The caller builds the C descriptors for strix_weights_prepare_parts.
```

Each `memoryview` is contiguous, little-endian and read-only. If its array already
has that format, it retains the mapped memory directly. It does not create
`bytes`, concatenate the group or expand Q. If a region needs an endian change
or stride packing, only that region is materialized. The views retain their
arrays and mappings even if `weights` or `SafetensorsFile` is released. They must
remain alive until native preparation finishes. A read-only view is not an
immutable snapshot: the caller must keep the sources immutable.

The metadata fields are exactly those of `transport_blob()`, including the
offsets in the **equivalent blob**. Each region already starts at its payload;
do not apply that offset to its pointer. Absent bias is represented by `None`,
`bias_offset=0` and `bias_bytes=0`. F32 bias remains mapped. F16 bias is converted
once to F32 during loading and becomes read-only.

This removes host copies and allocations from packing. The HIP preparation
function still copies the weights to its own storage. These views do not promise
direct GPU access to the mmap or an unmeasured bandwidth improvement.

## Reusable contiguous buffer and compatibility

If the consumer requires a single buffer, `transport_buffer()` allocates one
`bytearray` of the final size. It copies each array directly to a NumPy view of
the destination with its transport dtype, without temporary `.tobytes()` objects.
It can also use a buffer previously allocated by the caller:

```python
buffer, metadata_fields = weights.transport_buffer()
# buffer is an owned bytearray; it can also be used for blob preparation.

storage = bytearray(len(buffer) + 256)
used, metadata_fields = weights.transport_buffer(storage)
# used is a writable memoryview of the used prefix; the tail is unchanged.
```

The destination must support the buffer protocol, be writable and C-contiguous,
and have sufficient capacity. A contiguous NumPy array is also accepted.
Before writing, any possible overlap with the sources is rejected, using the
memory spans of the entire supplied buffer conservatively. Matching capacity
does not permit in-place packing. The caller must not supply another writable
view of the same mapped file or mutate the sources during copying. Alignment
gaps between payloads are zeroed. Bytes after the used prefix are preserved.

`transport_blob()` retains its previous contract: it returns immutable `bytes`
and the same metadata. It uses `transport_buffer()` and makes the final copy
needed for immutability. All variants preserve little-endian bytes and logical
four-byte alignment for each payload. The fields correspond to
`strix_exl3_metadata_v1`; the C struct is not serialized in the file. The caller
builds the native descriptor and uses `strix_weights_prepare` for a blob or
`strix_weights_prepare_parts` for regions. The backend receives activations in
the original basis and performs the transforms. The caller must not apply H128
or out_scales again.

## Partitions and source lifetime

`weights.slice_outputs(begin,end)` supports output partitions aligned to 128,
for example to split the vocabulary across nodes. It preserves complete H128
blocks in each partition. It is a view of the tiles. `transport_regions()`
repacks the strided trellis and retains contiguous views of scales/bias. With
`transport_buffer()`, the strided trellis is copied directly to the final
destination, without a second intermediate packed copy. Neither path expands Q
to FP16.

The container's NumPy views are read-only. Keep the file immutable while views
exist and release references before replacing or deleting it, especially on
Windows. C preparation copies the weights: the lifetime of the host blob,
buffer and views may end after prepare returns.

The CPU reference supports integer rates 1…8 and mul1 at 1.5/2.5/3.5. GPU
transport v1 rejects half-integer rates. It accepts known numeric markers and
historical empty markers. It rejects simultaneous markers, unknown multipliers,
external out_scales and legacy groups without complete suh/svh. It does not infer
missing scales. The legacy sign helpers are not a complete importer for arbitrary
old checkpoints.

Custom weights with this trellis/codebook contract are accepted in the same way.
There is no dependency on a model or vendor name. Different custom codebooks or
layouts require a new contract/transport version. They must not silently
reinterpret EXL3 bytes.

## Views by TP rank

`rank_partition(tensor, site, rank, bias_rank=0)` prepares the two
mix-B/expert-TP ranks without expanding weights. It slices the output of q_b
and gate/up, slices the input of o_b/down and replicates q_a/kv. It validates
dimensions, rate and mul1. For input partitions, it retains bias on a single
rank so that bias is added only once when reducing outputs across nodes.

`tensor.slice_inputs(begin, end, include_bias=...)` also supports a general
H128-aligned slice; `include_bias` is a required boolean. The K tiles and suh
are views, while svh retains all outputs. Output partitioning may require
compacting trellis during export. Site details, APIs that accept strided
activation views and measurement commands are in [TP_RANK.md](TP_RANK.md).
