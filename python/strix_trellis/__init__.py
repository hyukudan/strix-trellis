"""Independent EXL3 CPU reference. See docs/FORMAT.md for evidence limits."""
from .codec import decode_states, decode_tile, dequantize_rotated, step_widths, unpack_states
from .linear import (hadamard, matmul, reconstruct_weight, hadamard_fp32,
                     input_stages_fp32, core_dot_fp32, output_stages_fp32,
                     matmul_fp32, matmul_fp32_stages, decode_legacy_signs)
from .fixture import load_fixture
from .container import Exl3Tensor, SafetensorsFile, load_exl3
from .tp import rank_partition

__all__ = ["decode_states", "decode_tile", "dequantize_rotated", "step_widths",
           "unpack_states", "hadamard", "matmul", "reconstruct_weight", "load_fixture",
           "hadamard_fp32", "input_stages_fp32", "core_dot_fp32", "output_stages_fp32",
           "matmul_fp32", "matmul_fp32_stages", "decode_legacy_signs",
           "Exl3Tensor", "SafetensorsFile", "load_exl3", "rank_partition"]
