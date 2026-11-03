"""Row 3180: a Mosaic launch that adds into two buffers in place is on par in the route's own program.

The Mosaic backward's one launch (`BackwardTiles.fused`: the queries-first launch, and the shared-keys one at
width 256) adds each query head's dK and dV into its key head's own, refs it hands the custom call as
operands aliased to its outputs (`output_to_operand_aliasing`). Two zeroed refs lowered to one `broadcast`,
and where the route's program gives the call an operand that is a slice (a training forward's stacked rows,
sliced to the maxima and denominators), XLA's dynamic-slice fusion rewriter wraps the call in a fusion whose
operands it deduplicates: one parameter for both aliased operands, which copy insertion gives one buffer for
both outputs, so dK and dV both held dK + dV (an H100's gradients 1.2 and 0.8 times their RMS from XLA's,
ET-3180-PLAIN-h100). A backward alone in its own program is not wrapped, so the Mosaic backward's own cells
(called with the residuals as arguments) never saw it. The dK/dV launch's hook gradients
(`attention_mosaic_dkv._dkv_launched`) are such refs too: two key parameters, or two tile parameters, of one
shape were one buffer alike, which the kernel matrix's bias cells hold (tests/attention_matrix.py: two key
or two tile parameters, the route's training step). The route's training steps through the one launch on an
H100, on par with the reference path's (check 1), hold the rest.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attention_matrix import operands, terms_of
from attention_oracle import kept
from reference_error import assert_as_exact
from test_attention_mosaic import ON_HOPPER

from dew.nn.attention import AttentionImpl, scaled_dot_product_attention
from dew.nn.kernels import tuning
from dew.nn.kernels.templates import attention as template, attention_mosaic, attention_route

TILES = attention_mosaic.BackwardTiles
CALLS = {
    "fused-64": (64, 8, 2, TILES(128, 64, 2, 2, fused=True, family="wgmma.fed")),
    "fused-128": (128, 8, 2, TILES(64, 64, 2, 2, fused=True, family="wgmma.fed")),
    "shared-256": (256, 8, 4, TILES(64, 64, 2, 2, fused=True, family="wgmma.fed", shared_keys=True)),
}
"""The one launch's forms on an H100, by head width: the queries-first launch at widths 64 and 128 and the
shared-keys launch at 256 (`MosaicBackward.derived`'s picks where it takes one launch), 8 query heads over 2
or 4 key heads, each pinned so the check holds wherever the derived default goes."""
LENGTH = 1024
HEADS = 8
SEEDS = 8
def _route(causal: bool, implementation: AttentionImpl = "template"):
    """The route's training step: the gradients of `sum(attention * dout)` through `implementation`."""
    def step(query, key, value, dout):
        def loss(query, key, value):
            out = scaled_dot_product_attention(query, key, value, implementation=implementation,
                                               causal=causal)
            return (out.astype(jnp.float32) * dout.astype(jnp.float32)).sum()

        return jax.grad(loss, argnums=(0, 1, 2))(query, key, value)

    return step


def _terms(keep, heads: int = HEADS):
    """The gradients' float64 terms (`terms_of`, without the output's) over the keys each query keeps: a
    causal query that keeps few keys has a dQ that cancels toward zero in float64, held to its terms'
    rounding, not to its own size."""
    terms = terms_of(keep, heads)

    def gradients(*drawn):
        return terms(*drawn)[1:]

    return gradients


@ON_HOPPER
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "full"])
@pytest.mark.parametrize("name", CALLS)
def a_routes_one_launch_backward_is_on_par_on_an_h100(name: str, causal: bool) -> None:
    """The route's training step through the template on an H100, its backward pinned to the one launch
    (`CALLS`), on par with the reference path's (check 1): the step's program holds the forward and the
    backward, whose launch the dynamic-slice fusion wrapped (row 3180)."""
    width, heads, kv_heads, config = CALLS[name]

    def ours(*operands):
        with tuning.pinned({attention_mosaic.MosaicBackward: config}):
            return _route(causal)(*operands)

    assert_as_exact(jax.jit(ours), jax.jit(_route(causal, "reference")),
                    operands(width, heads, kv_heads, jnp.bfloat16),
                    f"one launch {name} {'causal' if causal else 'full'}", seeds=SEEDS,
                    terms=_terms(kept(LENGTH, LENGTH, causal=causal)))


WINDOW = 256
SPANNED = {"fused-64": (64, TILES(128, 64, 2, 2, fused=True, family="wgmma.fed")),
           "fused-128": (128, TILES(64, 64, 2, 2, fused=True, family="wgmma.fed"))}
"""Row 3171's call (ET-3171-BAND-h100), the one launch pinned at the tiles a plain call derives: a packed
windowed causal call, three documents and padding over 1024 keys (ids from 1: 0 is padding), a window of
256, 8 query heads over 2, at width 64 and at 128."""


def _spanned(width: int, implementation: str):
    """Row 3171's training step through the route (`attention_route.attend`) or the template's reference path:
    the output, and the gradients of `sum(out * dout)` of the query, key and value, and the keys each query
    keeps."""
    lengths = (300, 400, 200)
    padding = np.zeros(LENGTH - sum(lengths), int)
    segments = np.concatenate([np.repeat(np.arange(1, 4), lengths), padding])[None]
    spans = template.spans(LENGTH, LENGTH, 1, causal=True, sliding_window=WINDOW,
                           segment_ids=jnp.asarray(segments, jnp.int32))

    def step(query, key, value, dout):
        def attended(query, key, value):
            if implementation == "reference":
                return jnp.asarray(template.reference(query, key, value, spans))
            return jnp.asarray(attention_route.attend(query, key, value, spans))

        out, pull = jax.vjp(attended, query, key, value)
        return (out, *pull(dout))

    return step, kept(LENGTH, LENGTH, causal=True, sliding_window=WINDOW, segment_ids=segments)


@ON_HOPPER
@pytest.mark.parametrize("name", SPANNED)
def a_span_calls_one_launch_is_on_par_on_an_h100(name: str) -> None:
    """Row 3171's packed windowed call through the route on an H100, its backward pinned to the one launch,
    on par with the reference path's (check 1): its dK and dV, wrong and unrepeatable there before row 3180's
    fix, on par with it (ET-SPAN-h100); span calls derive the one launch as plain ones do
    (`MosaicBackward.derived`)."""
    width, config = SPANNED[name]
    step, keep = _spanned(width, "template")

    def ours(*operands):
        with tuning.pinned({attention_mosaic.MosaicBackward: config}):
            return step(*operands)

    assert_as_exact(jax.jit(ours), jax.jit(_spanned(width, "reference")[0]),
                    operands(width, HEADS, 2, jnp.bfloat16),
                    f"span call one launch {name}", seeds=SEEDS, terms=terms_of(keep, HEADS))
