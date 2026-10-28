"""Rotary position embeddings: the angles of one axis or several, Qwen's
interleaved M-RoPE, the one rotation every model applies (half-split or
adjacent pairs, leading or trailing), Llama 3.1's frequency ramp, and
YaRN's frequency ramp and attention factor. Beside them the absolute
sinusoidal tables share their frequencies (`sinusoid_frequencies`) and their
sines and cosines (`sinusoids`, `sincos_grid`): a timestep's, a patch grid's
and audio's relative positions.

YaRN's ramp is transformers 5.16.1's `_compute_yarn_parameters` and its
scale `yarn_apply_mscale` (models/deepseek_v3), with `dim` at the rope
width. The inverse-frequency tables are static, built on the host in NumPy
as transformers builds them on the CPU, in the dtype the angles are computed
in (`at_least_fp32` of the rotated activations'). The one transcendental,
`theta ** x`, rounds once from float64 (`_base_powers`), within one ulp of
transformers' float32 `pow`.
"""

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike

from dew.registry import from_record

from .precision import at_least_fp32


def sinusoid_frequencies(count: int, divisor: float, *, dtype: DTypeLike, exponential: bool = False
                         ) -> np.ndarray:
    """`10000 ** (-i / divisor)` for i below `count`, an absolute sinusoidal table's frequencies, on the
    host in `dtype`: rounded once from float64 (`_base_powers`), or, `exponential`, as the published
    timestep embeddings form them, the whole exponent `i * -ln(10000)` in `dtype` before it is divided
    (the other order moves a sine by 1e-5 near timestep 1000) and its exp taken in float64 and rounded
    once."""
    if not exponential:
        return (1.0 / _base_powers(10000.0, np.arange(count, dtype=np.float64) / divisor)).astype(dtype)
    width = np.dtype(dtype).type
    exponent = np.arange(count, dtype=width) * width(-math.log(10000.0)) / width(divisor)
    return np.exp(exponent.astype(np.float64)).astype(width)


def sinusoids(angles, *, cosine_first: bool = False, xp=jnp):
    """Each angle's sine and cosine side by side, `[..., n]` to `[..., 2n]`, the sines first or, where
    `cosine_first`, the cosines. `xp` is the array module: `numpy` for a host table built while a model
    traces, which JAX would otherwise stage into the trace."""
    first, second = (xp.cos(angles), xp.sin(angles)) if cosine_first else (xp.sin(angles), xp.cos(angles))
    return xp.concatenate([first, second], axis=-1)


def sincos_grid(frequencies: np.ndarray, first: np.ndarray, second: np.ndarray, *, dtype: DTypeLike
                ) -> np.ndarray:
    """A 2D sinusoidal table on the host, `[positions, 4 * len(frequencies)]`: the `first` axis's
    coordinate of each position, then the `second`'s, each as its sines then cosines at `frequencies`,
    computed in the frequencies' dtype and rounded once to `dtype` (the caller's axis order: a patch
    grid's rows first, SD3's columns). Each axis's distinct coordinates are computed once and gathered
    into the one output, so a wide grid holds no full-width table in the wider dtype."""
    width = 2 * len(frequencies)
    table = np.empty((first.size, 2 * width), dtype)
    for half, coordinates in enumerate((first, second)):
        values, where = np.unique(coordinates.reshape(-1), return_inverse=True)
        angles = values.astype(frequencies.dtype)[:, None] * frequencies[None, :]
        table[:, half * width:(half + 1) * width] = sinusoids(angles, xp=np).astype(dtype)[where]
    return table


def inverse_frequencies(theta: float, dim: int, pairs: int | None = None, *,
                        dtype: DTypeLike) -> np.ndarray:
    """`1 / theta ** (2i / dim)` for i below `pairs` (all of `dim // 2` by
    default), transformers' `compute_default_rope_parameters` table, built on
    the host in `dtype` (`_base_powers`)."""
    count = dim // 2 if pairs is None else pairs
    return 1.0 / _base_powers(theta, np.arange(0, 2 * count, 2, dtype=dtype) / dim)


def _base_powers(theta: float, exponents: np.ndarray) -> np.ndarray:
    """`theta ** exponents` correctly rounded to the exponents' dtype,
    whatever the backend: computed in float64 and rounded once."""
    return np.power(np.float64(theta), exponents.astype(np.float64)).astype(exponents.dtype)


@dataclasses.dataclass(frozen=True)
class RopeScaling:
    """Scales rotary frequencies by Llama 3.1's ramp, under the reference's names.

    `_compute_llama3_parameters` (modeling_rope_utils.py:580) divides a
    frequency by `factor` when its wavelength exceeds
    original_max_position_embeddings / low_freq_factor, keeps it below
    original_max_position_embeddings / high_freq_factor, and interpolates
    linearly on (original_max_position_embeddings / wavelength -
    low_freq_factor) / (high_freq_factor - low_freq_factor) between. Only
    `rope_type` 'llama3' is this ramp.
    """

    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int
    rope_type: str = 'llama3'

    def __post_init__(self):
        if self.rope_type != 'llama3':
            raise ValueError(
                f"rope_scaling applies the llama3 ramp, got rope_type {self.rope_type!r}")
        if self.factor < 1.0:
            raise ValueError(f"rope_scaling factor is at least 1, got {self.factor}")
        if self.high_freq_factor <= self.low_freq_factor:
            raise ValueError(
                f"rope_scaling needs high_freq_factor above low_freq_factor, got "
                f"{self.high_freq_factor} and {self.low_freq_factor}")
        if self.original_max_position_embeddings < 1:
            raise ValueError(
                "rope_scaling original_max_position_embeddings is the pretraining "
                f"context, got {self.original_max_position_embeddings}")

    def apply(self, inv_freq: np.ndarray) -> np.ndarray:
        """Return the scaled inverse frequencies, the reference's arithmetic
        on the host in the table's own dtype."""
        old_context_len = float(self.original_max_position_embeddings)
        wavelen = 2 * math.pi / inv_freq
        divided = np.where(wavelen > old_context_len / self.low_freq_factor,
                           inv_freq / self.factor, inv_freq)
        smooth = ((old_context_len / wavelen - self.low_freq_factor)
                  / (self.high_freq_factor - self.low_freq_factor))
        smoothed = (1 - smooth) * divided / self.factor + smooth * divided
        medium = np.logical_and(wavelen >= old_context_len / self.high_freq_factor,
                                wavelen <= old_context_len / self.low_freq_factor)
        return np.where(medium, smoothed, divided).astype(inv_freq.dtype)


@dataclasses.dataclass(frozen=True)
class LongRopeScaling:
    """Phi-3's short/long inverse-frequency factors and cos/sin amplitude.

    Each row selects its table at its own largest position. This matches
    transformers at batch one and on uniform batches, while independent
    served requests cannot change one another's frequencies. Cached keys
    retain their table until the shared decode operation rebuilds the row.
    """

    short_factor: tuple[float, ...]
    long_factor: tuple[float, ...]
    original_max_position_embeddings: int
    factor: float = 1.0
    attention_factor: float | None = None
    rope_type: str = 'longrope'

    def __post_init__(self):
        for name in ('short_factor', 'long_factor'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.rope_type != 'longrope' or self.original_max_position_embeddings < 2:
            raise ValueError('longrope requires an original context of at least two positions')
        if not self.short_factor or len(self.short_factor) != len(self.long_factor):
            raise ValueError('longrope short_factor and long_factor must have the same positive length')
        if any(value <= 0 for value in (*self.short_factor, *self.long_factor)) or self.factor <= 0:
            raise ValueError('longrope frequency factors must be positive')
        if self.attention_factor is not None and self.attention_factor <= 0:
            raise ValueError('longrope attention_factor must be positive')

    @property
    def amplitude(self) -> float:
        """The reference's explicit amplitude, or its context-ratio formula."""
        if self.attention_factor is not None:
            return self.attention_factor
        return (1.0 if self.factor <= 1 else
                math.sqrt(1 + math.log(self.factor) / math.log(self.original_max_position_embeddings)))


def rope_scaling_from_record(record: Mapping[str, object]) -> "Scaling":
    """Build the scaling a record holds: a class record (`{"class", "fields"}`, as the registry writes a
    union field) by its class, else the one its rope_type names, LongRoPE's, YaRN's or Llama 3.1's."""
    if 'class' in record:
        built = from_record(Scaling, record)
        assert isinstance(built, RopeScaling | LongRopeScaling | YarnScaling)
        return built
    kinds = {'longrope': LongRopeScaling, 'yarn': YarnScaling}
    return from_record(kinds.get(str(record.get('rope_type')), RopeScaling), record)


@dataclasses.dataclass(frozen=True)
class YarnScaling:
    """YaRN rope scaling, the reference's `rope_parameters` fields.

    Both released DeepSeek configs carry this spelling (rope_type yarn,
    factor 40 off 4096 base positions), so the record keeps the reference's
    names and a translation renames nothing. The ramp transforms the base of
    the layer it rides on (`MixerContext.rope_theta`), which it does not
    repeat; the mscale is applied in the attention as a query pre-scale.
    """

    rope_type: str = 'yarn'
    factor: float = 40.0
    original_max_position_embeddings: int = 4096
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float | None = None
    mscale_all_dim: float | None = None
    truncate: bool = True
    # An explicit cos/sin amplitude, which the reference applies instead of
    # deriving one; None derives it from factor and the mscales above.
    attention_factor: float | None = None


type Scaling = RopeScaling | LongRopeScaling | YarnScaling
"""A rotary's one frequency scaling: Llama 3.1's ramp, LongRoPE's factors or YaRN's (`rotary_freqs`)."""


def rotary_freqs(positions, head_dim: int, theta: float, rot_dim: int | None = None,
                 partial_rotary_type: str = 'proportional',
                 rope_scaling: Scaling | None = None, *,
                 dtype: DTypeLike, reach=None, section: Sequence[int] | None = None):
    """Return cos and sin of the rotary angles at absolute `positions`: [P, pairs].

    `positions` is [P], or [B, P] for a packed batch whose documents restart at
    0. The angles are computed in `dtype` (at least fp32), so a token rotates
    the same in prefill and decode.

    With `section`, Qwen's interleaved M-RoPE, `positions` is [B, S, 3]: each
    token's temporal, height and width ids. Pair i turns by the height id
    when `i % 3 == 1` and `i < 3 * section[1]`, by the width id when
    `i % 3 == 2` and `i < 3 * section[2]`, and by the temporal id otherwise
    (Qwen3-VL's `apply_interleaved_mrope`), each at the frequency and
    amplitude the rope and its ramp give that pair (Llama 3.1's or YaRN's,
    the same at every position), as Qwen3_5TextRotaryEmbedding builds them
    through `ROPE_INIT_FUNCTIONS` (modeling_qwen3_5.py:143-213).

    LongRoPE picks its short or long factors by how far a sequence reaches: a
    row's furthest position, or `reach` `[B, P]`, each token's document's
    furthest place (`dew.nn.inputs.document_reach`), so a packed document
    rotates as it does alone.

    `rot_dim` narrows the rotation to the first rot_dim dimensions, under one
    of two published conventions that rotate different angles:

    - 'proportional' (Gemma 4, `_compute_proportional_rope_parameters`):
      `theta ** (2i / head_dim)` for the rot_dim // 2 rotated pairs, the rest at
      frequency zero; head_dim // 2 wide.
    - 'default' (Qwen3.5, modeling_qwen3_5.py:117-124): a rot_dim-wide rope,
      `theta ** (2i / rot_dim)`, rot_dim // 2 wide; `rotate` passes the
      rest through, the reference's `q_rot, q_pass` split.

    `rope_scaling` (Llama 3.1's ramp) applies before a proportional rope pads
    its zero-frequency tail, as the reference's `dim = head_dim *
    partial_rotary_factor` does. YaRN (`YarnScaling`) replaces the inverse
    frequencies of the rotated width with its ramp (`yarn_inv_freq`) and
    scales cos and sin by its attention factor, as the reference's rotary
    embedding does: over the whole head, or over `rot_dim` under 'default',
    the reference's `dim = head_dim * partial_rotary_factor`
    (`_compute_yarn_parameters`). A proportional rope has no YaRN form, so
    YaRN beside a proportional `rot_dim` is refused.
    """
    if partial_rotary_type not in ('proportional', 'default'):
        raise ValueError(
            "partial_rotary_type names the convention of a partial rotary, "
            f"'proportional' or 'default', got {partial_rotary_type!r}")
    pairs = head_dim // 2 if rot_dim is None else rot_dim // 2
    divisor = head_dim if rot_dim is None or partial_rotary_type == 'proportional' else rot_dim
    inv_freq = inverse_frequencies(theta, divisor, pairs, dtype=dtype)
    if isinstance(rope_scaling, LongRopeScaling):
        if len(rope_scaling.short_factor) != pairs:
            raise ValueError(f'longrope factors must have {pairs} entries for the rotated width')
        powers = _base_powers(theta, np.arange(0, 2 * pairs, 2, dtype=dtype) / divisor)
        short = 1.0 / (np.asarray(rope_scaling.short_factor, dtype=dtype) * powers)
        long = 1.0 / (np.asarray(rope_scaling.long_factor, dtype=dtype) * powers)
        rows = jnp.asarray(positions)
        furthest = jnp.max(rows, axis=-1, keepdims=rows.ndim > 1) if reach is None else jnp.asarray(reach)
        expanded = furthest + 1 > rope_scaling.original_max_position_embeddings
        inv_freq = jnp.where(expanded[..., None], long, short)
    elif isinstance(rope_scaling, YarnScaling):
        if rot_dim is not None and partial_rotary_type == 'proportional':
            raise ValueError(f"YaRN ramps the rotated width at its own frequencies and has no "
                             f"proportional form, so it takes no proportional rot_dim, got {rot_dim}")
        inv_freq = yarn_inv_freq(divisor, theta, rope_scaling, dtype=dtype)
    elif rope_scaling is not None:
        inv_freq = rope_scaling.apply(inv_freq)
    if rot_dim is not None and partial_rotary_type == 'proportional':
        padding = head_dim // 2 - pairs
        if isinstance(rope_scaling, LongRopeScaling):
            inv_freq = jnp.pad(inv_freq, (*((0, 0),) * (inv_freq.ndim - 1), (0, padding)))
        else:
            inv_freq = np.concatenate([inv_freq, np.zeros((padding,), inv_freq.dtype)])
    if section is None:
        angles = jnp.asarray(positions, inv_freq.dtype)[..., None] * inv_freq
    else:
        if positions.shape[-1] != 3:
            raise ValueError(f"M-RoPE reads three coordinates per position, got {positions.shape[-1]}")
        indices = jnp.arange(inv_freq.shape[-1])
        axes = jnp.zeros(indices.shape, jnp.int32)
        for axis in (1, 2):
            axes = jnp.where((indices % 3 == axis) & (indices < section[axis] * 3), axis, axes)
        selected = jnp.take_along_axis(positions, axes[None, None, :], axis=-1)
        angles = selected.astype(inv_freq.dtype) * inv_freq
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    if isinstance(rope_scaling, LongRopeScaling):
        cos, sin = cos * rope_scaling.amplitude, sin * rope_scaling.amplitude
    elif isinstance(rope_scaling, YarnScaling):
        factor = yarn_attention_factor(rope_scaling)
        cos, sin = cos * factor, sin * factor
    return cos, sin


def axis_tables[P: (np.ndarray, jax.Array)](positions: P, axes: Sequence[int], theta: float, *,
                                            dtype: DTypeLike, stored: DTypeLike | None = None
                                            ) -> tuple[P, P]:
    """cos and sin of the rotary angles of ids on several axes: `[..., sum(axes) // 2]`.

    `positions` is `[..., len(axes)]`. Axis i turns its id at
    `inverse_frequencies(theta, axes[i])`, `axes[i] // 2` pairs, and the axes'
    angles concatenate in order: Flux's `EmbedND`, Wan's three-axis rope and
    the vision towers' patch grids. The angles are computed in `dtype`. NumPy
    ids compute on the host, where `dtype` may be float64, as
    `FluxPosEmbed` computes its static grid; JAX ids compute on the device.

    `stored` is the dtype a source keeps its angles and tables in: the
    angles round to it before their cosine and sine, and those round to it
    after, as Z-Image's `precompute_freqs_cis` stores the float32 angles of
    float64 products and takes their float32 cosine and sine.
    """
    angles = [positions[..., axis, None].astype(dtype) * inverse_frequencies(theta, dim, dtype=dtype)
              for axis, dim in enumerate(axes)]
    if stored is not None:
        angles = [angle.astype(stored).astype(dtype) for angle in angles]
    table = dtype if stored is None else stored
    if isinstance(positions, np.ndarray):
        joined = np.concatenate(angles, axis=-1)
        return np.cos(joined).astype(table), np.sin(joined).astype(table)
    joined = jnp.concatenate(angles, axis=-1)
    return jnp.cos(joined).astype(table), jnp.sin(joined).astype(table)


def rotate(x, cos, sin, *, pairs: Literal['half', 'adjacent'] = 'half',
           span: Literal['leading', 'trailing'] = 'leading', scale: float | None = None):
    """Rotate the channel pairs of `[B, S, H, D]` heads or `[B, S, D]` rows.

    `cos` and `sin` hold one angle a pair, `[S, P]` for one sequence or
    `[B, S, P]` when each row has its own positions (a packed batch, an
    image's grid). The `2 * P` rotated channels lead the last axis or, with
    `span='trailing'`, end it (DeepSeek V4's nope-then-rope heads), and the
    rest pass through: the sliced partial rotary of
    `rotary_freqs(partial_rotary_type='default')`. 'half' pairs channel i
    with i + P, the rotate-half convention of the HF decoders. 'adjacent'
    pairs 2i with 2i + 1, the complex numbers of Flux, Llama 4's vision and
    DeepSeek V4, and each pair lands back in place, so a rotation by `-sin`
    undoes it. The arithmetic runs in at least fp32 and rounds once to x's
    dtype; `scale` multiplies the whole of x inside it, so a query's
    attention scale narrows once, with the product.

    The halves rotate as `x1 cos - x2 sin` and `x2 cos + x1 sin`, which is
    `x cos + rotate_half(x) sin` without the rotated copy. On a TPU v6e the
    copy's slice-and-negate ran as separate passes: without it Qwen3-0.6B's
    training step at 8 x 1024 took 140.6 against 145.6 ms, and Qwen3-1.7B's
    at 4 x 1024 150.0 against 153.8 (docs/performance.md).
    """
    if x.ndim == 4:
        cos, sin = cos[..., None, :], sin[..., None, :]
    wide = x.astype(at_least_fp32(x.dtype))
    count = cos.shape[-1]
    start = 0 if span == 'leading' else wide.shape[-1] - 2 * count
    lead, turned, tail = wide[..., :start], wide[..., start:start + 2 * count], wide[..., start + 2 * count:]
    if pairs == 'half':
        first, second = turned[..., :count], turned[..., count:]
        rotated = [first * cos - second * sin, second * cos + first * sin]
    else:
        split = turned.reshape(*turned.shape[:-1], count, 2)
        first, second = split[..., 0], split[..., 1]
        rotated = [jnp.stack([first * cos - second * sin, second * cos + first * sin],
                             axis=-1).reshape(turned.shape)]
    out = jnp.concatenate([part for part in (lead, *rotated, tail) if part.shape[-1]], axis=-1)
    return (out if scale is None else out * scale).astype(x.dtype)


def yarn_inv_freq(head_dim: int, theta: float, yarn: YarnScaling, *,
                  dtype: DTypeLike) -> np.ndarray:
    """YaRN inverse frequencies over the rope width, `[head_dim // 2]`, in `dtype`.

    Mirrors `modeling_rope_utils._compute_yarn_parameters` with `dim` at
    `head_dim`, the rotated width: a whole head's, as DeepSeek's configs point
    `head_dim` at the rope slice, or a partial rotary's `head_dim *
    partial_rotary_factor`. Low dims interpolate towards `1 / (factor * pos_freqs)`,
    high dims keep extrapolating, and the linear ramp between the correction
    bounds blends them.
    """
    dim = head_dim
    pairs = dim // 2
    pos_freqs = _base_powers(theta, np.arange(0, dim, 2, dtype=dtype) / dim)
    inv_extrapolation = 1.0 / pos_freqs
    inv_interpolation = 1.0 / (yarn.factor * pos_freqs)

    def correction_dim(rotations: float) -> float:
        """The dimension seeing `rotations` turns over the original context."""
        return (dim * math.log(yarn.original_max_position_embeddings
                               / (rotations * 2 * math.pi))
                / (2 * math.log(theta)))

    low = correction_dim(yarn.beta_fast)
    high = correction_dim(yarn.beta_slow)
    if yarn.truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    span = high - low
    if span == 0:
        # The reference nudges a degenerate bound to keep the division finite.
        span = 0.001
    ramp = np.clip((np.arange(pairs, dtype=dtype) - low) / span, 0, 1)
    return (inv_interpolation * ramp + inv_extrapolation * (1 - ramp)).astype(dtype)

def yarn_attention_factor(yarn: YarnScaling) -> float:
    """The cos/sin multiplier of `_compute_yarn_parameters`.

    Both released configs set mscale and mscale_all_dim to 1.0, so this is
    1.0 for them; a config that sets them apart rotates at a different
    amplitude.
    """
    if yarn.attention_factor is not None:
        return float(yarn.attention_factor)
    def mscale(scale: float, weight: float) -> float:
        return 1.0 if scale <= 1 else 0.1 * weight * math.log(scale) + 1.0

    if yarn.mscale and yarn.mscale_all_dim:
        return float(mscale(yarn.factor, yarn.mscale)
                     / mscale(yarn.factor, yarn.mscale_all_dim))
    return float(mscale(yarn.factor, 1.0))


def yarn_query_scale(yarn: YarnScaling) -> float:
    """The attention-scale multiplier of `yarn_apply_mscale`, squared.

    The reference folds this into the softmax scale, while dew's kernels
    scale by `1 / sqrt(head_dim)` themselves, so the query carries the
    ratio, the same way `attention_scale` does on the standard mixer. For
    the released configs this is `(0.1 * ln(40) + 1) ** 2`.
    """
    if not yarn.mscale_all_dim or yarn.factor <= 1:
        return 1.0
    mscale = 0.1 * yarn.mscale_all_dim * math.log(yarn.factor) + 1.0
    return mscale * mscale


def axis_halves(x, axes: Sequence[int]):
    """`x` with each axis's channels split in halves and gathered: every
    axis's first half, then every axis's second half. Gemma 4's vision rope
    turns each axis's channels in halves of their own
    (`apply_multidimensional_rope`); moved so, a query and key turn with
    `rotate`'s 'half' pairs over angles laid out by `axis_tables`, and their
    dot product is the reference's. Channels past the axes stay last."""
    starts = np.cumsum((0, *axes))
    spans = list(zip(starts[:-1], axes, strict=True))
    firsts = [x[..., start:start + width // 2] for start, width in spans]
    seconds = [x[..., start + width // 2:start + width] for start, width in spans]
    return jnp.concatenate([*firsts, *seconds, x[..., starts[-1]:]], axis=-1)


def deinterleaved(x):
    """`x`'s even channels, then its odd ones: the half layout
    `modeling_deepseek_v3.apply_rotary_pos_emb_interleave` moves adjacent
    pairs into before it rotates them, and leaves them in. A query and key
    in that layout rotate with `rotate`'s 'half' pairs, and their dot product
    keeps the complex structure."""
    return jnp.concatenate([x[..., 0::2], x[..., 1::2]], axis=-1)
