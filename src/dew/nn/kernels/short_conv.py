"""The short causal conv of the recurrent mixers on the recurrence template's Mosaic GPU kernels.

`linear.causal_conv1d` is the depthwise causal conv over a `[B, T, C]` projection with `[C, K]` taps
(K a few tokens), its bias and silu, which the gated delta net, Kimi's delta attention and Mamba run
before their scans; the delta rules then l2-normalise each query and key head. As XLA computes it, the
backward reads and writes the `[B, T, C]` activation once for each of the norm's, silu's and the conv's
transposes, several of them in fp32. Here a piece's forward reads the projection once and writes its output
once, and its backward reads the projection and its cotangent once and writes its input's gradient once,
everything between in fp32 registers (`templates.recurrence.conv`: a thread a channel walking its tokens,
the backward carrying the K-1 later tokens' pre-activation gradients, as causal-conv1d's channel-last backward
does, Dao-AILab/causal-conv1d csrc/causal_conv1d_bwd.cu). Neither saves anything but the projection.

The conv selector (`CONV`) runs them where their requirements hold and XLA's conv elsewhere, reported as
the call's route ('short_conv'). The arithmetic is `causal_conv1d`'s in fp32, each output rounded once to its
dtype: the conv's products summed tap by tap from the oldest (`preactivation`), the sigmoid's quotient and
the norm's root and inverse each rounded once. Every output and gradient is within the float64 gate of XLA's
conv (tests/test_short_conv.py).
"""

import dataclasses
import functools
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.activations import silu
from dew.nn.derivatives import KernelGlue, Protected, contracted, with_gradients_formed
from dew.nn.precision import at_least_fp32, compensated_sum, l2norm

from . import generation
from .generation import INHERENT, Choice
from .selection import Kernel, Op, Requirement

WIDEST_TAPS = 8
"""The longest conv the kernels unroll: each tap is a register tile the walk carries."""

DTYPES = (jnp.dtype(jnp.float32), jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16))


@dataclasses.dataclass(frozen=True)
class ConvCase:
    """A conv as the conv selector reads it (`CONV`): whether the kernels run here (`placed`), whether no
    mesh axis splits its operand (`whole`), the operand's `shape`
    and `dtype`, the dtype it is written in (`out`), its `taps` and its pieces (`parts`, `(width, head)`
    each)."""

    placed: bool
    whole: bool
    shape: tuple[int, ...]
    dtype: np.dtype
    out: np.dtype
    taps: int
    parts: tuple[tuple[int, int | None], ...]


SUPPORTS: tuple[Requirement[ConvCase], ...] = (
    Requirement('capability', lambda case: case.placed,
                "the kernels need a GPU a Mosaic GPU bundle claims (`mainloop.family_of`)", row=INHERENT),
    Requirement('placement', lambda case: case.whole,
                "a mesh axis splits the conv's operands, and the kernels see whole arrays",
                row='ledger 1306'),
    Requirement('shape', lambda case: len(case.shape) == 3 and case.shape[1] > 0,
                lambda case: f"a conv over {case.shape}; the kernels take [B, T, C]", row=INHERENT),
    Requirement('dtype', lambda case: case.dtype in DTYPES and case.out in DTYPES,
                lambda case: f"{case.dtype} inputs into {case.out}; the kernels read and write float32, "
                             "bfloat16 or float16, summing in float32", row=INHERENT),
    Requirement('taps', lambda case: 2 <= case.taps <= WIDEST_TAPS,
                lambda case: f"{case.taps} taps; the kernels unroll 2 to {WIDEST_TAPS}", row=INHERENT),
    Requirement('pieces', lambda case: sum(width for width, _ in case.parts) == case.shape[-1],
                lambda case: f"pieces {case.parts} of {case.shape[-1]} channels", row=INHERENT),
    Requirement('blocks', lambda case: all(_fit(width, head) for width, head in case.parts),
                lambda case: f"pieces {case.parts}: each whole warpgroups of channels, a normalised piece's "
                             "head one", row=INHERENT),
)
"""What the kernels need of a conv: a GPU they compile for, reverse mode, an operand no mesh axis splits,
and the shapes and dtypes they take."""

def _fit(width: int, head: int | None) -> bool:
    """Whether the kernels take a piece (`templates.recurrence.conv.fits`)."""
    from .templates.recurrence import conv

    return conv.fits(width, head)


CONV: Op[ConvCase] = Op(Choice.SHORT_CONV, (Kernel('mosaic_gpu', SUPPORTS), Kernel('xla', (), generic=True)),
                        lambda case: 'mosaic_gpu' if generation.backend() == 'gpu' else 'xla')
"""The conv selector: the template's kernels on a GPU where their requirements hold, else XLA's conv
(`xla_conv1d`)."""


def preactivation(window, taps, bias):
    """The conv's sum, tap by tap from the oldest token, then the bias: `window` and `taps` each tap's
    inputs `[..., C]` and weights `[C]` (a kernel's a thread's channels of each), oldest first, in the sum's
    dtype (fp32 in the kernels; the portable conv's and the decode reference's at least fp32): the one order
    and the one statement of it every conv path adds in, the template's conv and decode step included."""
    total = window[0] * taps[0]
    for inputs, tap in zip(window[1:], taps[1:], strict=True):
        total = total + inputs * tap
    return total if bias is None else total + bias


def conv_bias(bias: jax.Array, leading: tuple[int, ...]) -> jax.Array:
    """A conv's bias `[C]` added at every row and token of `leading` `[B, S]` (`_spread_bias`): its gradient
    the wide pre-activation cotangent summed over them in one compensated reduction, rounded once, the one
    rule of a conv's bias, XLA's conv's (`xla_conv1d`), a decode step's (`recurrence`) and
    the kernels' (their partials, the split conv's shards)."""
    return _spread_bias((*leading, bias.shape[0]))(bias)


# Bounded: a server's calls vary their rows, each shape a wrapper of its own.
@functools.lru_cache(maxsize=256)
def _spread_bias(shape: tuple[int, ...]) -> Callable:
    """A conv's bias spread over the rows and tokens of `shape` `[B, S, C]`, differentiated at every order
    (`generation.linear_tangent`): reverse mode sums its tokens' cotangents in one compensated reduction
    rounded once (`compensated_sum`), forward mode takes the broadcast's own tangent. XLA's float32 sum of a
    conv's bias gradient over its tokens left it 1.6 to 2.4 times as far from float64 as the sum rounded once
    (armada's CPU and an A100 and a GH200, row 1710), past what a split conv's shards
    (`linear._shards_summed`) and the kernels' partials add to it."""
    def spread(values: jax.Array) -> jax.Array:
        return jnp.broadcast_to(values, shape)

    return contracted("test_a_convs_bias_holds_every_order")(
        generation.linear_tangent(lambda values: (spread(values), None),
                                  lambda saved, operands, cotangent: (
                                      compensated_sum(cotangent, tuple(range(len(shape) - 1))),),
                                  spread, kind=Protected()))


def _launched(interpret: bool):
    """The family the kernels issue by, and whether they interpret (`mainloop.issued`)."""
    from .templates.recurrence import mainloop

    family, interpreted = mainloop.issued(jnp.float32)
    return {"family": family, "interpret": interpret or interpreted}


def _forward(x, taps, bias, valid, documents, parts, activation: bool, eps: float, dtype, interpret: bool):
    """Each piece's output, every piece's in one launch (`templates.recurrence.conv.forward`)."""
    from .templates.recurrence import conv

    return conv.forward(x, taps, bias, valid, documents, parts=tuple(parts), activation=activation, eps=eps,
                        dtype=dtype, **_launched(interpret))


def _backward(x, taps, bias, valid, documents, cotangents, parts, activation: bool, eps: float,
              interpret: bool):
    """Every piece's gradients in one launch (`templates.recurrence.conv.backward`) of one piece, the
    pieces' cotangents joined along the channels: a normalised piece's taken through its norm in fp32 by a
    launch joining them (`templates.recurrence.conv.through`), else by XLA's concatenation. A launch of
    several pieces emitted its walk once a piece, Qwen3.5-4B's three pieces' backward 0.354 ms on a GH200
    against one piece's 0.266 (M-JS-253b)."""
    from .templates.recurrence import conv

    if any(head is not None for _, head in parts):
        cotangent = conv.through(x, taps, bias, valid, documents, tuple(cotangents), parts=tuple(parts),
                                 activation=activation, eps=eps, **_launched(interpret))
    else:
        cotangent = jnp.concatenate(cotangents, axis=-1)
    dx, dw, db = conv.backward(x, taps, bias, valid, documents, cotangent, activation=activation,
                               **_launched(interpret))
    # The taps' and bias's gradients formed in their dtypes before the input's cotangent leaves (row 6027).
    dx, (d_taps, d_bias) = with_gradients_formed(
        dx, (dw.T.astype(taps.dtype), None if bias is None else db.astype(bias.dtype)))
    return dx, d_taps, d_bias


def conv(x, taps, bias, valid, documents, parts, *, activation: bool, eps: float, dtype, interpret: bool):
    """`linear.causal_conv1d` on the kernels: `x` `[B, T, C]` convolved causally by `taps` `[C, K]` behind
    K-1 zero tokens, plus `bias` `[C]` or None, silu'd where `activation`, as the `parts` pieces
    `((width, head), ...)` of its channels in order, each `[B, T, width]` in `dtype`, a piece with a `head`
    l2-normalised (`eps`) over each `head` channels. `valid` `[B, T]` int8 or None zeroes padding tokens'
    outputs; `documents` `[B, T]` int32 or None numbers packed documents, a tap reading a token only in
    its output token's. Differentiated at every order through XLA's conv (`xla_conv1d`, the same
    arithmetic; `generation.linear_tangent`): reverse mode runs the backward kernels, and forward mode takes
    XLA's tangent however a jit, a scan or a remat staged the call (row 2395)."""

    def forward(x, taps, bias, valid, documents):
        return _forward(x, taps, bias, valid, documents, parts, activation, eps, dtype, interpret), ()

    def backward(saved, operands, cotangents):
        del saved
        x, taps, bias, valid, documents = operands
        dx, dw, db = _backward(x, taps, bias, valid, documents, cotangents, parts, activation, eps, interpret)
        # The masks have no tangent space: float0's zero, as JAX takes it of them.
        masks = (None if mask is None else jax.custom_derivatives.zero_from_primal(mask)
                 for mask in (valid, documents))
        return dx, dw, db, *masks

    def portable(x, taps, bias, valid, documents):
        return xla_conv1d(x, taps, activation, bias, valid=valid, segments=documents, parts=parts,
                          dtype=dtype)

    return contracted("test_the_short_conv_holds_every_order")(
        generation.linear_tangent(forward, backward, portable,
                                  kind=KernelGlue()))(x, taps, bias, valid, documents)


def xla_conv1d(x, kernel, activation: bool = True, bias=None, *, valid=None, segments=None,
               parts: Sequence[tuple[int, int | None]] | None = None,
               dtype=None) -> jax.Array | tuple[jax.Array, ...]:
    """`causal_conv1d`'s arithmetic in XLA: a CPU's and a TPU's, and the oracle of the GPU's kernels."""
    dtype = x.dtype if dtype is None else jnp.dtype(dtype)
    _, taps_width = kernel.shape
    length = x.shape[1]
    wide = jnp.promote_types(at_least_fp32(x.dtype), dtype)
    padded = jnp.pad(x, ((0, 0), (taps_width - 1, 0), (0, 0))).astype(wide)
    taps = kernel.astype(wide)

    def read(tap):
        """The tokens `taps_width - 1 - tap` before each, those of its own document."""
        window = padded[:, tap:tap + length]
        if segments is None:
            return window
        before = jnp.pad(segments, ((0, 0), (taps_width - 1, 0)))[:, tap:tap + length]
        return jnp.where((before == segments)[..., None], window, 0)

    spread_bias = None if bias is None else conv_bias(bias.astype(wide), x.shape[:2])
    windows = preactivation([read(tap) for tap in range(taps_width)],
                            [taps[:, tap] for tap in range(taps_width)], spread_bias)
    if activation:
        windows = silu(windows)
    if valid is not None:
        windows = jnp.where(valid[:, :, None], windows, 0)
    return conv_pieces(windows, parts, dtype)


def conv_pieces(convolved, parts, dtype):
    """`convolved` `[B, S, D]` as `causal_conv1d` returns it for `parts` (the whole, where None), in `dtype`:
    each piece split off in order and the `head`-wide groups of a normalised one l2-normalised."""
    if parts is None:
        return convolved.astype(dtype)
    pieces, first = [], 0
    for width, head in parts:
        piece = convolved[..., first:first + width]
        first += width
        if head is not None:
            piece = l2norm(piece.reshape(*piece.shape[:-1], -1, head)).reshape(piece.shape)
        pieces.append(piece.astype(dtype))
    return tuple(pieces)
