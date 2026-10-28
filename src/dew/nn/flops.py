"""Analytical forward work on real tokens, independent of a compiled program.

A multiply-add is two FLOPs. Projections count every real token; attention
counts each row's own square, not the square of the padded batch length.
The trainer multiplies these forward counts by three for forward/backward
MFU. Rematerialization belongs only to the separate HLO hardware count.
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np
from flax import linen as nn

from .protocols import Counted, CountedParameters


class MissingFlopsCount(ValueError):
    """The requested model work needs a capability or geometry not supplied."""


def real_lengths(lengths: np.ndarray) -> np.ndarray:
    """Nonnegative integer row lengths, widened before sums and squares."""
    lengths = np.asarray(lengths)
    if lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer) or np.any(lengths < 0):
        raise ValueError("model FLOPs need nonnegative integer lengths of shape [rows]")
    return lengths.astype(np.float64)


def matching_lengths(*lengths: np.ndarray) -> tuple[np.ndarray, ...]:
    """Independent real sequence lengths belonging to the same batch rows."""
    widened = tuple(real_lengths(column) for column in lengths)
    if not widened or any(column.shape != widened[0].shape for column in widened[1:]):
        raise ValueError("model FLOPs need matching [rows] sequence geometries")
    return widened


def patch_lengths(sample_shape: Sequence[int], patch: Sequence[int], rows: int) -> np.ndarray:
    """Token counts of a known image/video shape after its patch convolution."""
    if (len(sample_shape) != len(patch) + 1 or rows < 0 or any(step <= 0 for step in patch)
            or any(size <= 0 or size % step for size, step in zip(sample_shape[:-1], patch, strict=True))):
        raise MissingFlopsCount(f"sample shape {tuple(sample_shape)} is not whole {tuple(patch)} patches")
    count = math.prod(size // step for size, step in zip(sample_shape[:-1], patch, strict=True))
    return np.full((rows,), count, np.int64)


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


def parameter_flops(module: nn.Module | CountedParameters, lengths: np.ndarray) -> float:
    """The declared learned-matrix contribution; no inference from parameter names."""
    if not isinstance(module, CountedParameters):
        raise MissingFlopsCount(f"{type(module).__name__} has no parameter_flops for PEFT training work")
    return float(module.parameter_flops(lengths))


def projection_flops(lengths: np.ndarray, inputs: int, outputs: int) -> float:
    """One dense matrix: two operations per input/output pair per real token."""
    return float(2 * inputs * outputs * np.sum(real_lengths(lengths)))


def conv_flops(lengths: np.ndarray, inputs: int, outputs: int, taps: int, *, groups: int = 1) -> float:
    """Convolution MACs per real output position, with grouped input width."""
    return projection_flops(lengths, taps * inputs // groups, outputs)


def frozen_training_flops(forward: float, frozen: float, adapter: float = 0.0) -> float:
    """The forward and backward work of layers whose matrices doing `frozen` of their `forward`
    work take no weight gradient, with trainable adapter factors doing `adapter` forward work: three
    times the forward work (forward, input gradient, weight gradient), less the frozen matrices'
    weight gradients, plus the adapters' three passes."""
    return 3 * forward - frozen + 3 * adapter


def dense_training_flops(lengths: np.ndarray, inputs: int, outputs: int, *,
                          trainable: bool = True, adapter_rank: int = 0) -> float:
    """Forward/backward work of a dense layer, separate from forward FLOPs.

    A frozen base matrix has forward and input-gradient products, 4*in*out
    per token. A trained matrix also has its weight-gradient product,
    6*in*out. Trainable rank-r adapter factors add 6*r*(in+out), so a frozen
    dense layer with LoRA costs exactly 4*in*out + 6*r*(in+out). Remat is
    excluded. The caller supplies actual frozen/trainable geometry; this
    helper never infers it from a parameter name or folds it into a forward
    model count that the trainer would incorrectly multiply by three.
    """
    if inputs < 0 or outputs < 0 or adapter_rank < 0:
        raise ValueError("dense training FLOPs need nonnegative matrix widths and adapter rank")
    forward = 2 * inputs * outputs
    adapter = 2 * adapter_rank * (inputs + outputs)
    return per_real_token(lengths, frozen_training_flops(forward, 0 if trainable else forward, adapter))


def attention_flops(lengths: np.ndarray, heads: int, key: int, value: int, *, causal: bool,
                    window: int | None = None, chunk: int | None = None) -> float:
    """QK and AV, two FLOPs per multiply-add, causal attention halved.

    The conventional model count uses L squared (not L*(L+1)) for causal
    triangles: 2*h*d*L^2 for equal key/value widths, 4*h*d*L^2 for full
    attention. Masking, softmaxes and fused-kernel padding are not matmuls.
    """
    def pairs(length):
        if window is None:
            return np.square(length) * (0.5 if causal else 1)
        if window < 1:
            raise ValueError("an attention FLOP window must be positive")
        visible = np.minimum(length, window)
        return (length * visible - 0.5 * visible * visible) * (1 if causal else 2)

    length = real_lengths(lengths)
    if chunk is not None:
        if chunk < 1:
            raise ValueError("an attention FLOP chunk must be positive")
        work = np.floor(length / chunk) * pairs(chunk) + pairs(np.remainder(length, chunk))
    else:
        work = pairs(length)
    return _attention_products(work, heads, key, value)


def self_attention_flops(lengths: np.ndarray, projections: Mapping[str, tuple[int, int]], heads: int,
                         head_dim: int) -> float:
    """A bidirectional self-attention block over real rows: its `projections` (name to input and
    output width) and the full QK and AV products of `heads` heads `head_dim` wide."""
    return (sum(projection_flops(lengths, *shape) for shape in projections.values())
            + attention_flops(lengths, heads, head_dim, head_dim, causal=False))


def cross_attention_flops(lengths: np.ndarray, context_lengths: np.ndarray,
                           heads: int, key: int, value: int) -> float:
    """QK and AV over independent query/key rows, rather than a joined square."""
    query, context = matching_lengths(lengths, context_lengths)
    return _attention_products(query * context, heads, key, value)


def _attention_products(pairs: np.ndarray, heads: int, key: int, value: int) -> float:
    return float(2 * heads * (key + value) * np.sum(pairs))


def per_real_token(lengths: np.ndarray, flops: float) -> float:
    """`flops` for each real token of `lengths`' rows."""
    return float(np.sum(real_lengths(lengths)) * flops)


def delta_flops(lengths: np.ndarray, heads: int, key: int, value: int) -> float:
    """The delta rule's recurrent function, not one implementation's work.

    Per token, w*S predicts the stored value, k^T*delta writes its correction
    and q*S reads the updated state. Each is 2*H*Dk*Dv FLOPs under the
    multiply-add convention: 6*H*Dk*Dv total. Chunked prep/inversion, padding
    and remat are hardware work, not model work, just as FlashAttention's
    tiling changes no model attention count. GDN and KDA use this same rule.
    """
    return per_real_token(lengths, 6 * heads * key * value)


def ssd_flops(lengths: np.ndarray, heads: int, state: int, channels: int) -> float:
    """SSD's recurrent state write and read, 4*H*N*P per token, without chunk work.

    Unlike the delta rule, SSD has no value-correction read. Its scalar
    decay/scaling is pointwise work, excluded like GDN's decay and gates.
    """
    return per_real_token(lengths, 4 * heads * state * channels)
