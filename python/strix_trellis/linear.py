"""Independent mathematical FP64 and explicitly staged production FP32 references."""
import numpy as np

from .codec import dequantize_rotated


def hadamard(values, block_size=128, axis=-1):
    """Apply normalized Sylvester Hadamard independently to contiguous blocks."""
    if isinstance(block_size, bool) or not isinstance(block_size, int):
        raise ValueError("block_size must be a positive power of two")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    data = np.asarray(values, dtype=np.float64)
    if data.ndim == 0 or not -data.ndim <= axis < data.ndim:
        raise ValueError("Invalid transform axis")
    data = np.moveaxis(data, axis, -1).copy()
    if data.shape[-1] % block_size:
        raise ValueError("Transform dimension must be divisible by block_size")
    blocks = data.reshape(-1, block_size)
    stride = 1
    while stride < block_size:
        groups = blocks.reshape(-1, 2, stride)
        left = groups[:, 0].copy()
        right = groups[:, 1].copy()
        groups[:, 0] = left + right
        groups[:, 1] = left - right
        stride *= 2
    data /= np.sqrt(block_size)
    return np.moveaxis(data, -1, axis)


def _scales(suh, svh, shape):
    left, right = np.asarray(suh, dtype=np.float64), np.asarray(svh, dtype=np.float64)
    if left.shape != (shape[0],) or right.shape != (shape[1],):
        raise ValueError("Scale vectors must match input/output dimensions")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Scale vectors must be finite")
    return left, right


def reconstruct_weight(trellis, suh, svh, k=None, codebook="mul1"):
    """W = diag(suh) H128 W_hat H128 diag(svh), returned in float64."""
    rotated = dequantize_rotated(trellis, k, codebook)
    left, right = _scales(suh, svh, rotated.shape)
    weight = hadamard(hadamard(rotated, axis=0), axis=1)
    return left[:, None] * weight * right[None, :]


def matmul(x, trellis, suh, svh, k=None, codebook="mul1", bias=None):
    """Apply signed scales and both Hadamards without forming original-basis W."""
    rotated = dequantize_rotated(trellis, k, codebook).astype(np.float64)
    left, right = _scales(suh, svh, rotated.shape)
    inputs = np.asarray(x, dtype=np.float64)
    if inputs.ndim < 1 or inputs.shape[-1] != rotated.shape[0]:
        raise ValueError("Input last dimension must match trellis input dimension")
    output = hadamard(hadamard(inputs * left) @ rotated) * right
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float64)
        if bias.shape != right.shape:
            raise ValueError("Bias must match output dimension")
        output += bias
    return output


def _finite_cast(values, dtype, name):
    raw = np.asarray(values)
    if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError(f"{name} must contain finite real numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        result = raw.astype(dtype)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} overflows {np.dtype(dtype).name}")
    return result


def _finite_stage(values, name):
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite result at {name}")
    return values


def hadamard_fp32(values, block_size=128, axis=-1):
    """Ascending-stride Sylvester butterflies, FP32 operations and final scale.

    The scale is RN32(1/sqrt(block_size)), computed from binary64. No FP16
    rounding occurs here. Nonfinite inputs/results are outside this contract.
    """
    if (isinstance(block_size, bool) or not isinstance(block_size, int)
            or block_size <= 0 or block_size & (block_size - 1)):
        raise ValueError("block_size must be a positive power of two")
    data = _finite_cast(values, np.float32, "Hadamard input")
    if data.ndim == 0 or not -data.ndim <= axis < data.ndim:
        raise ValueError("Invalid transform axis")
    data = np.moveaxis(data, axis, -1).copy()
    if not data.shape[-1] or data.shape[-1] % block_size:
        raise ValueError("Transform dimension must be positive and divisible by block_size")
    blocks = data.reshape(-1, block_size)
    stride = 1
    with np.errstate(over="ignore", invalid="ignore"):
        while stride < block_size:
            groups = blocks.reshape(-1, 2, stride)
            left, right = groups[:, 0].copy(), groups[:, 1].copy()
            groups[:, 0] = left + right
            groups[:, 1] = left - right
            _finite_stage(groups, "Hadamard butterfly")
            stride *= 2
        data *= np.float32(1.0 / np.sqrt(block_size))
    return np.moveaxis(_finite_stage(data, "Hadamard normalization"), -1, axis)


def input_stages_fp32(x, suh):
    """Return the input checkpoints of the production contract (see NUMERICS.md)."""
    inputs = _finite_cast(x, np.float32, "Input")
    left = _finite_cast(suh, np.float16, "suh")
    if inputs.ndim < 1 or left.shape != (inputs.shape[-1],):
        raise ValueError("suh must match input last dimension")
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = _finite_stage(inputs * left.astype(np.float32), "input scale")
    scaled_half = _finite_cast(scaled, np.float16, "Scaled input")
    transformed = hadamard_fp32(scaled_half)
    core_input = _finite_cast(transformed, np.float16, "Input Hadamard")
    return {"x_fp32": inputs, "scaled_fp32": scaled,
            "scaled_fp16": scaled_half, "input_hadamard_fp32": transformed,
            "core_input_fp16": core_input}


def core_dot_fp32(core_input, rotated):
    """Dot with FP16 operands, FP32 separate multiply/add, increasing input index.

    Deliberately avoids BLAS-dependent reduction order and implicit FMA. A GPU
    reduction/FMA may differ and must report its error against these checkpoints.
    """
    a = _finite_cast(core_input, np.float16, "Core input").astype(np.float32)
    q = _finite_cast(rotated, np.float16, "Rotated weights").astype(np.float32)
    if a.ndim < 1 or q.ndim != 2 or not q.shape[0] or not q.shape[1] or a.shape[-1] != q.shape[0]:
        raise ValueError("Core shapes must be [..., input] and [input, output]")
    result = np.zeros(a.shape[:-1] + (q.shape[1],), dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        for index in range(q.shape[0]):
            product = a[..., index, None] * q[index]
            result += product
            _finite_stage(result, "core accumulation")
    return result


def output_stages_fp32(core, svh, bias=None):
    """FP32 output Hadamard, signed FP16 scale in FP32, then optional FP32 bias."""
    values = _finite_cast(core, np.float32, "Core output")
    right = _finite_cast(svh, np.float16, "svh")
    if values.ndim < 1 or right.shape != (values.shape[-1],):
        raise ValueError("svh must match output last dimension")
    transformed = hadamard_fp32(values)
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = _finite_stage(transformed * right.astype(np.float32), "output scale")
    output = scaled.copy()
    if bias is not None:
        offset = _finite_cast(bias, np.float32, "Bias")
        if offset.shape != right.shape:
            raise ValueError("Bias must match output dimension")
        with np.errstate(over="ignore", invalid="ignore"):
            output = _finite_stage(output + offset, "bias addition")
    return {"output_hadamard_fp32": transformed, "scaled_output_fp32": scaled,
            "output_fp32": output}


def matmul_fp32_stages(x, trellis, suh, svh, k=None, codebook="mul1", bias=None):
    """Return every checkpoint of OUR production contract, not upstream parity."""
    rotated = dequantize_rotated(trellis, k, codebook)
    stages = input_stages_fp32(x, suh)
    if stages["core_input_fp16"].shape[-1] != rotated.shape[0]:
        raise ValueError("Input last dimension must match trellis input dimension")
    stages["rotated_fp16"] = rotated
    stages["core_fp32"] = core_dot_fp32(stages["core_input_fp16"], rotated)
    stages.update(output_stages_fp32(stages["core_fp32"], svh, bias))
    return stages


def matmul_fp32(x, trellis, suh, svh, k=None, codebook="mul1", bias=None):
    """Evaluate the staged FP32 production reference; preserve FP64 matmul separately."""
    return matmul_fp32_stages(x, trellis, suh, svh, k, codebook, bias)["output_fp32"]


def decode_legacy_signs(packed):
    """Decode legacy 1-D 16-bit su/sv: bit 16*j+b maps to sign 1-2*bit_b."""
    words = np.asarray(packed)
    if words.ndim != 1 or words.dtype.kind not in "iu" or words.dtype.itemsize != 2:
        raise ValueError("Legacy signs require a one-dimensional int16/uint16 array")
    signs = np.empty(words.size * 16, dtype=np.float16)
    for index, word in enumerate(words):
        bits = int(word) & 65535
        for bit in range(16):
            signs[16 * index + bit] = -1 if bits & (1 << bit) else 1
    return signs
