"""Analytical forward work on real tokens, independent of a compiled program.

A multiply-add is two FLOPs. Projections count every real token; attention
counts each row's own square, not the square of the padded batch length.
The trainer multiplies these forward counts by three for forward/backward
MFU. Rematerialization belongs only to the separate HLO hardware count.
"""

import math

import numpy as np
from flax import linen as nn

from .protocols import Counted


class MissingFlopsCount(ValueError):
    """The requested model work needs a capability or geometry not supplied."""


def real_lengths(lengths: np.ndarray) -> np.ndarray:
    """Nonnegative integer row lengths, widened before sums and squares."""
    lengths = np.asarray(lengths)
    if lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer) or np.any(lengths < 0):
        raise ValueError("model FLOPs need nonnegative integer lengths of shape [rows]")
    return lengths.astype(np.float64)


def counted_flops(module: nn.Module | Counted, lengths: np.ndarray) -> float:
    """Read a module's declaration, or refuse by name; never read its HLO."""
    if not isinstance(module, Counted):
        raise MissingFlopsCount(f"{type(module).__name__} has no model_flops(lengths); "
                                "cannot report train/mfu")
    count = float(module.model_flops(lengths))
    if not math.isfinite(count) or count < 0:
        raise ValueError(f"{type(module).__name__}.model_flops returned {count}, "
                         "not finite nonnegative FLOPs")
    return count


def projection_flops(lengths: np.ndarray, inputs: int, outputs: int) -> float:
    """One dense matrix: two operations per input/output pair per real token."""
    return float(2 * inputs * outputs * np.sum(real_lengths(lengths)))


def attention_flops(lengths: np.ndarray, heads: int, key: int, value: int, *, causal: bool) -> float:
    """QK and AV, two FLOPs per multiply-add, causal attention halved.

    The conventional model count uses L squared (not L*(L+1)) for causal
    triangles: 2*h*d*L^2 for equal key/value widths, 4*h*d*L^2 for full
    attention. Masking, softmaxes and fused-kernel padding are not matmuls.
    """
    pairs = np.sum(np.square(real_lengths(lengths)))
    return float((1 if causal else 2) * heads * (key + value) * pairs)


def delta_flops(lengths: np.ndarray, heads: int, key: int, value: int, chunk: int) -> float:
    """The chunked delta rule, excluding learned projections and convolution.

    Reading and writing the key-by-value state each cost 2*H*Dk*Dv per
    token. In-chunk key products and value application each cost 2*H*C*D
    per token after dividing their C-by-C products by the C-token chunk:
    4*H*Dk*Dv + 2*H*C*(Dk+Dv). C is the configured algorithmic chunk size;
    padding and rematerialized repeats do not change this model count.
    """
    return float(np.sum(real_lengths(lengths)) * (4 * heads * key * value
                                                 + 2 * heads * chunk * (key + value)))
