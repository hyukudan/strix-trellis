"""Selected EXL3 tensor groups from local safetensors, without loading a shard."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct

import numpy as np

from .codec import step_widths
from .linear import matmul, matmul_fp32

_DTYPES = {"I16": "<i2", "I32": "<i4", "F16": "<f2", "F32": "<f4"}
_MARKERS = {"3inst": 0, "mcg": 0xCBAC1FED, "mul1": 0x83DCD12D}
_TRANSPORT_PARTS = (("trellis", np.dtype("<i2")), ("suh", np.dtype("<f2")),
                    ("svh", np.dtype("<f2")), ("bias", np.dtype("<f4")))


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


class SafetensorsFile:
    """Read-only selected-tensor view; unrelated tensor dtypes need no decoder.

    Validates the header and every selected byte range, dtype and shape. This is
    not a validator for all unrelated tensors in a multi-format model shard.
    Keep the underlying file immutable while returned memory maps are in use.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.file_bytes = self.path.stat().st_size
        with self.path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("Truncated safetensors header length")
            header_bytes = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_bytes <= min(8_000_000, self.file_bytes - 8):
                raise ValueError("Invalid or oversized safetensors header")
            self.header = json.loads(stream.read(header_bytes), object_pairs_hook=_unique_object)
        if not isinstance(self.header, dict):
            raise ValueError("Safetensors header must be an object")
        self.data_offset = 8 + header_bytes

    def tensor(self, name):
        info = self.header.get(name)
        if not isinstance(info, dict) or info.get("dtype") not in _DTYPES:
            raise ValueError(f"Missing tensor or unsupported dtype: {name}")
        shape, offsets = info.get("shape"), info.get("data_offsets")
        if (not isinstance(shape, list) or len(shape) > 8 or
                any(type(x) is not int or x < 0 for x in shape) or
                not isinstance(offsets, list) or len(offsets) != 2 or
                any(type(x) is not int for x in offsets)):
            raise ValueError(f"Invalid tensor shape or offsets: {name}")
        begin, end = offsets
        dtype = np.dtype(_DTYPES[info["dtype"]])
        if not 0 <= begin <= end <= self.file_bytes - self.data_offset:
            raise ValueError(f"Tensor range exceeds file: {name}")
        if math.prod(shape) * dtype.itemsize != end - begin:
            raise ValueError(f"Tensor size does not match shape/dtype: {name}")
        if begin == end:
            result = np.empty(shape, dtype=dtype)
            result.flags.writeable = False
            return result
        return np.memmap(self.path, mode="r", dtype=dtype,
                         offset=self.data_offset + begin, shape=tuple(shape))

    def exl3(self, prefix):
        return load_exl3(self, prefix)


@dataclass(frozen=True)
class Exl3Tensor:
    """Modern, folded-scale EXL3 group; logical Q has [input, output] shape."""
    trellis: np.ndarray
    suh: np.ndarray
    svh: np.ndarray
    bits: float
    codebook: str
    bias: np.ndarray | None
    source: str
    prefix: str

    @property
    def input_size(self):
        return self.trellis.shape[0] * 16

    @property
    def output_size(self):
        return self.trellis.shape[1] * 16

    def matmul(self, x, *, production=False):
        operation = matmul_fp32 if production else matmul
        return operation(x, self.trellis, self.suh, self.svh, self.bits,
                         self.codebook, bias=self.bias)

    def slice_outputs(self, begin, end):
        """View a H128-aligned output partition; export repacks its tile stride.

        H128 is block diagonal, so each partition retains its complete output
        transform. Useful for a vocabulary partition without expanding weights.
        """
        if (type(begin) is not int or type(end) is not int or
                not 0 <= begin < end <= self.output_size or begin % 128 or end % 128):
            raise ValueError("Output partitions must be nonempty, in range and H128 aligned")
        return Exl3Tensor(self.trellis[:, begin//16:end//16, :], self.suh,
                          self.svh[begin:end], self.bits, self.codebook,
                          None if self.bias is None else self.bias[begin:end],
                          self.source, self.prefix + f"[{begin}:{end}]")

    def slice_inputs(self, begin, end, *, include_bias):
        """View an H128-aligned input partition without repacking its K tiles.

        Input H128 blocks are independent, so each rank consumes its original
        input slice and the matching suh slice. Output scales remain complete.
        Summing rank outputs reconstructs the projection, subject to floating
        point reduction error. Exactly one rank must include the output bias;
        requiring this explicit boolean prevents an accidental duplicate bias.
        """
        if (type(begin) is not int or type(end) is not int or
                not 0 <= begin < end <= self.input_size or begin % 128 or end % 128):
            raise ValueError("Input partitions must be nonempty, in range and H128 aligned")
        if type(include_bias) is not bool:
            raise ValueError("include_bias must be an explicit bool")
        return Exl3Tensor(self.trellis[begin//16:end//16, :, :], self.suh[begin:end],
                          self.svh, self.bits, self.codebook,
                          self.bias if include_bias else None,
                          self.source, self.prefix + f"[K{begin}:{end}]")

    def _transport_layout(self):
        """Compute the canonical layout without converting or copying arrays."""
        if not float(self.bits).is_integer():
            raise ValueError("GPU transport v1 supports integer bitrates only")
        metadata = dict(rows=self.output_size, columns=self.input_size,
                        bits_x2=int(self.bits) * 2,
                        codebook=("3inst", "mcg", "mul1").index(self.codebook),
                        marker=_MARKERS[self.codebook], transport_version=1)
        arrays, size = [], 0
        for name, dtype in _TRANSPORT_PARTS:
            values = getattr(self, name)
            if values is None:
                metadata[name + "_offset"] = metadata[name + "_bytes"] = 0
                continue
            values = np.asarray(values)
            size += -size % 4
            metadata[name + "_offset"] = size
            metadata[name + "_bytes"] = values.size * dtype.itemsize
            size += metadata[name + "_bytes"]
            arrays.append((name, values, dtype))
        return arrays, metadata, size

    def transport_regions(self):
        """Return (read-only byte views, canonical transport metadata).

        Contiguous little-endian arrays are borrowed without a payload copy.
        Other arrays are packed independently; an output slice normally repacks
        only trellis. Views retain their backing arrays and memory maps, even
        after this group is released. They are not snapshots: callers must keep
        the source arrays and mapped files immutable until native prepare ends.
        Bias is None when absent. Metadata offsets describe the equivalent blob;
        each returned view already starts at its own payload, not that offset.
        """
        arrays, metadata, _ = self._transport_layout()
        regions = dict.fromkeys(name for name, _ in _TRANSPORT_PARTS)
        for name, values, dtype in arrays:
            contiguous = np.ascontiguousarray(values, dtype=dtype)
            regions[name] = memoryview(contiguous).cast("B").toreadonly()
        return regions, metadata

    def transport_buffer(self, destination=None):
        """Return (mutable buffer, canonical transport metadata) without bytes copies.

        With no destination, allocate one bytearray. Otherwise accept a writable
        C-contiguous buffer and return a byte memoryview of its populated prefix;
        bytes beyond that prefix are unchanged. Sources are copied directly into
        typed destination views, including strided slices and endian conversion.
        Reject any possible overlap with the supplied destination before writing.
        Only inter-payload padding is zeroed; no expanded Q is materialized.
        """
        arrays, metadata, size = self._transport_layout()
        owned = destination is None
        if owned:
            destination = bytearray(size)
        try:
            view = memoryview(destination)
        except TypeError as error:
            raise ValueError("Destination must expose a writable contiguous buffer") from error
        if view.readonly or not view.c_contiguous:
            raise ValueError("Destination must be writable and C-contiguous")
        view = view.cast("B")
        if view.nbytes < size:
            raise ValueError("Destination is too small for the transport")
        if not owned:
            target = np.frombuffer(view, dtype=np.uint8)
            if any(np.may_share_memory(target, values) for _, values, _ in arrays):
                raise ValueError("Destination must not overlap transport source arrays")
        end = 0
        for name, values, dtype in arrays:
            offset, length = metadata[name + "_offset"], metadata[name + "_bytes"]
            if offset > end:
                view[end:offset] = b"\0" * (offset - end)
            target = np.ndarray(values.shape, dtype=dtype, buffer=view, offset=offset)
            np.copyto(target, values, casting="unsafe")
            end = offset + length
        return (destination if owned else view[:size]), metadata

    def transport_blob(self):
        """Return (owned immutable bytes, fields for strix_exl3_metadata_v1).

        Native C descriptors are deliberately not serialized here. CPU supports
        half rates; the GPU transport only accepts integer rates. Prefer regions
        or a caller-owned buffer when the final immutable copy is unnecessary.
        """
        data, metadata = self.transport_buffer()
        return bytes(data), metadata


def load_exl3(file, prefix):
    """Map a selected modern EXL3 group from a path or SafetensorsFile.

    Accepts validated numerical markers and historical empty codebook markers.
    Rejects ambiguous markers, unknown multipliers, external out_scales and
    legacy sign-only groups whose missing scale contract cannot be inferred.
    """
    file = file if isinstance(file, SafetensorsFile) else SafetensorsFile(file)
    if not isinstance(prefix, str) or not prefix:
        raise ValueError("A nonempty tensor prefix is required")
    if prefix + ".out_scales" in file.header:
        raise ValueError("External out_scales requires an explicit scale contract")
    names = [prefix + suffix for suffix in (".trellis", ".suh", ".svh")]
    selected = {name: file.tensor(name) for name in names}
    trellis, suh, svh = (selected[name] for name in names)
    if trellis.dtype != np.dtype("<i2") or trellis.ndim != 3 or not all(trellis.shape):
        raise ValueError("EXL3 trellis must be nonempty I16 [I/16,O/16,words]")
    bits = trellis.shape[-1] / 16
    step_widths(bits)
    inputs, outputs = trellis.shape[0] * 16, trellis.shape[1] * 16
    if inputs % 128 or outputs % 128:
        raise ValueError("Complete EXL3 linear requires I/O multiples of 128")
    for name, values, size in (("suh", suh, inputs), ("svh", svh, outputs)):
        if values.dtype != np.dtype("<f2") or values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid folded F16 scale vector: {name}")
    markers = [name for name in ("mul1", "mcg") if prefix + "." + name in file.header]
    if len(markers) > 1:
        raise ValueError("Simultaneous mul1/mcg markers")
    codebook = markers[0] if markers else "3inst"
    if markers:
        name = prefix + "." + codebook
        marker = file.tensor(name)
        selected[name] = marker
        if marker.size and (marker.dtype != np.dtype("<i4") or marker.size != 1 or
                            int(marker.reshape(-1)[0]) & 0xFFFFFFFF != _MARKERS[codebook]):
            raise ValueError("Unsupported EXL3 codebook marker")
    if bits % 1 and codebook != "mul1":
        raise ValueError("Half-integer EXL3 rates require mul1")
    bias = None
    if prefix + ".bias" in file.header:
        name = prefix + ".bias"
        raw = file.tensor(name)
        selected[name] = raw
        if raw.dtype.kind != "f" or raw.shape != (outputs,) or not np.isfinite(raw).all():
            raise ValueError("Invalid bias vector")
        bias = raw.astype(np.float32, copy=False)
        bias.flags.writeable = False
    ranges = sorted(file.header[name]["data_offsets"] for name, value in selected.items() if value.nbytes)
    if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
        raise ValueError("Overlapping selected EXL3 tensor ranges")
    return Exl3Tensor(trellis, suh, svh, bits, codebook, bias, str(file.path), prefix)
