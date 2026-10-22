"""Load-time, two-rank EXL3 partitions for the explicit fused TP interfaces.

The helpers retain compressed weights. Output partitions may require a single
trellis repack when exported; input partitions retain their contiguous K view.
There is no execution-time tensor adapter or expanded-weight representation.
"""
import numpy as np

from .codec import step_widths
from .container import Exl3Tensor


# Full logical [input, output] shapes. The packed layout is [I/16,N/16,16*bits].
_SITES = {
    "q_b": (1280, 32768, "output", (4, 5)),
    "o_b": (8192, 5120, "input", (4, 5)),
    "shared_gate": (5120, 2304, "output", (4, 5)),
    "shared_up": (5120, 2304, "output", (4, 5)),
    "shared_down": (2304, 5120, "input", (4, 5)),
    "q_a": (5120, 1280, "replicate", (4, 5)),
    "kv": (5120, 512, "replicate", (4, 5)),
    "expert_gate": (5120, 2304, "output", (2, 3)),
    "expert_up": (5120, 2304, "output", (2, 3)),
    "expert_down": (2304, 5120, "input", (2, 3)),
}


def _validate_tensor(tensor, site):
    if not isinstance(tensor, Exl3Tensor):
        raise ValueError("rank_partition requires an Exl3Tensor")
    inputs, outputs, axis, rates = _SITES[site]
    step_widths(tensor.bits)
    if tensor.bits not in rates or tensor.codebook != "mul1":
        raise ValueError(f"{site} requires mul1 at bitrates {rates}")
    packed = tensor.trellis
    if (not isinstance(packed, np.ndarray) or packed.dtype.kind != "i" or
            packed.dtype.itemsize != 2 or
            packed.shape != (inputs // 16, outputs // 16, int(tensor.bits) * 16)):
        raise ValueError(f"{site} requires a complete I16 trellis for I={inputs}, N={outputs}")
    for name, size in (("suh", inputs), ("svh", outputs)):
        value = getattr(tensor, name)
        if (not isinstance(value, np.ndarray) or value.shape != (size,) or
                value.dtype.kind != "f" or value.dtype.itemsize != 2 or
                not np.isfinite(value).all()):
            raise ValueError(f"Invalid folded F16 {name} for {site}")
    if tensor.bias is not None:
        bias = tensor.bias
        if (not isinstance(bias, np.ndarray) or bias.shape != (outputs,) or
                bias.dtype.kind != "f" or bias.dtype.itemsize != 4 or
                not np.isfinite(bias).all()):
            raise ValueError(f"Invalid F32 bias for {site}")
    return inputs, outputs, axis


def rank_partition(tensor, site, rank, *, bias_rank=0):
    """Return rank 0 or 1 for a complete, validated mix-B/expert TP site.

    ``q_b``, ``shared_gate/up`` and ``expert_gate/up`` split output columns.
    ``o_b``, ``shared_down`` and ``expert_down`` split the input K dimension;
    sum their rank outputs, adding bias only on ``bias_rank`` (default zero).
    ``q_a`` and ``kv`` are replicated and return the original tensor itself.
    Replicated outputs must not be summed. Output partitions are concatenated
    in rank order; their disjoint bias slices are both retained.

    Ranks and bias_rank are exact Python integers 0/1. Dense sites require
    K4/K5 mul1; experts require K2/K3 mul1. A local rank tensor is deliberately
    rejected as input, preventing accidental double partitioning.
    """
    if type(site) is not str or site not in _SITES:
        raise ValueError("Unknown TP site")
    if type(rank) is not int or rank not in (0, 1):
        raise ValueError("rank must be the integer 0 or 1")
    if type(bias_rank) is not int or bias_rank not in (0, 1):
        raise ValueError("bias_rank must be the integer 0 or 1")
    inputs, outputs, axis = _validate_tensor(tensor, site)
    if axis == "replicate":
        return tensor
    if axis == "output":
        width = outputs // 2
        return tensor.slice_outputs(rank * width, (rank + 1) * width)
    width = inputs // 2
    return tensor.slice_inputs(rank * width, (rank + 1) * width,
                               include_bias=rank == bias_rank)
