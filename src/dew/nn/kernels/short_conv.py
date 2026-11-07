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

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.derivatives import KernelGlue, contracted

from . import generation
from .generation import INHERENT
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


CONV: Op[ConvCase] = Op('short_conv', (Kernel('mosaic_gpu', SUPPORTS), Kernel('xla', (), generic=True)),
                        lambda case: 'mosaic_gpu' if generation.backend() == 'gpu' else 'xla')
"""The conv selector: the template's kernels on a GPU where their requirements hold, else XLA's conv
(`dew.nn.linear.xla_conv1d`)."""


def preactivation(window, taps, bias):
    """The conv's sum, tap by tap from the oldest token, then the bias: `window` and `taps` each tap's
    inputs `[..., C]` and weights `[C]`, oldest first, in the sum's dtype (fp32 in the kernels; the portable
    conv's and the decode reference's at least fp32), the one order every conv path adds in."""
    total = window[0] * taps[0][None, :]
    for inputs, tap in zip(window[1:], taps[1:], strict=True):
        total = total + inputs * tap[None, :]
    return total if bias is None else total + bias[None, :]


def _launched(interpret: bool):
    """The family the kernels issue by, and whether they interpret (off a GPU, the interpreted bundle's)."""
    from .templates import bundles
    from .templates.recurrence import mainloop

    family = mainloop.family_of(jnp.float32)
    return {"family": family or bundles.INTERPRETED, "interpret": interpret or family is None}


def _forward(x, taps, bias, valid, documents, parts, activation: bool, eps: float, dtype, interpret: bool):
    """Each piece's output, a launch each (`templates.recurrence.conv.forward`)."""
    from .templates.recurrence import conv

    return tuple(conv.forward(x, taps, bias, valid, documents, first=first, width=width, head=head,
                              activation=activation, eps=eps, dtype=dtype, **_launched(interpret))
                 for first, width, head in conv.pieces(parts))


def _backward(x, taps, bias, valid, documents, cotangents, parts, activation: bool, eps: float,
              interpret: bool):
    """Every piece's gradients in one launch (`templates.recurrence.conv.backward`)."""
    from .templates.recurrence import conv

    dx, dw, db = conv.backward(x, taps, bias, valid, documents, tuple(cotangents), parts=tuple(parts),
                               activation=activation, eps=eps, **_launched(interpret))
    return dx, dw.T.astype(taps.dtype), None if bias is None else db.astype(bias.dtype)


def conv(x, taps, bias, valid, documents, parts, *, activation: bool, eps: float, dtype, interpret: bool):
    """`linear.causal_conv1d` on the kernels: `x` `[B, T, C]` convolved causally by `taps` `[C, K]` behind
    K-1 zero tokens, plus `bias` `[C]` or None, silu'd where `activation`, as the `parts` pieces
    `((width, head), ...)` of its channels in order, each `[B, T, width]` in `dtype`, a piece with a `head`
    l2-normalised (`eps`) over each `head` channels. `valid` `[B, T]` int8 or None zeroes padding tokens'
    outputs; `documents` `[B, T]` int32 or None numbers packed documents, a tap reading a token only in
    its output token's. Differentiated at every order through XLA's conv (`linear.xla_conv1d`, the same
    arithmetic; `generation.linear_tangent`): reverse mode runs the backward kernels, and forward mode takes
    XLA's tangent however a jit, a scan or a remat staged the call (row 2395)."""
    from dew.nn.linear import xla_conv1d

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

