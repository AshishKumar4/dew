"""An output hook fused into the Mosaic forward's epilogue (row 1956, M-NA-47): a pointwise gate over a
forward 256 wide finishes a chunk of columns at a time in one launch of every value column, the gate read at
its own bytes; a hook laid at the whole width keeps the whole-width finish; each on par with the reference
(check 1)."""

import jax
import jax.numpy as jnp
import pytest
from generation_support import INTERPRETS
from reference_error import assert_as_exact

from dew.nn.kernels import generation, tuning
from dew.nn.kernels.templates import attention as template, attention_mosaic
from dew.nn.kernels.templates.hooks import Hook, Side

pytestmark = pytest.mark.kernel

SEEDS = 8


def _gate(out, gate):
    """An output gate, `o * sigmoid(g)` (Qwen3-Next's and Qwen3.5's gated attention)."""
    return out * jax.nn.sigmoid(gate)


def _gated(gate) -> template.Hooks:
    """`_gate` over `gate` `[B, Q, H, Dv]` (`Kind.OUTPUT`)."""
    return template.Hooks(output=Hook(_gate, (Side(gate, template.Kind.OUTPUT),)))


UNSPLIT = {"sm90": attention_mosaic.Tiles(64, 64, 2, 2, family="wgmma")}
"""A launch of every value column (`Tiles.columns` None) on an H100's family, pinned, so the epilogue its
width takes is the one judged whatever tile a derivation would pick (an A100's registers hold no such tile
at width 256, its derivation splitting the columns)."""


def _wide(queries: int, width: int):
    """A draw of a causal call's query, key and value `width` wide and a bf16 gate."""
    def draw(seed):
        streams = jax.random.split(jax.random.key(seed), 4)
        shapes = ((1, queries, 2, width), (1, queries, 1, width), (1, queries, 1, width),
                  (1, queries, 2, width))
        return tuple(jax.random.normal(stream, shape, jnp.float32).astype(jnp.bfloat16)
                     for stream, shape in zip(streams, shapes, strict=True))

    return draw


@INTERPRETS
@pytest.mark.parametrize("hook", ["pointwise", "whole-width", "width-reading"])
@pytest.mark.parametrize("device", ["sm90", "sm80"])
def test_a_wide_gated_launch_is_on_par_with_its_16_bit_gate_interpreted(device, hook):
    """A bf16 gate over a forward 256 wide under the interpreter: a pointwise gate in one launch of every
    value column (pinned on an H100, `UNSPLIT`), finished a chunk of columns at a time with the gate's own
    columns (`attention_mosaic_program.pointwise_tiles`); a gate its hook lays at the whole width (a reshape
    to `[-1, 256]`, no chunk's) at the tiles its call derives, the whole width's; one reading its width
    (`out.shape[-1]`), shape-preserving, in the pinned launch finished whole; the gate read at its own bytes
    and widened as the hook takes it; on par with the reference (check 1)."""
    queries, width = 128, 256
    spans = template.spans(queries, queries, 1, causal=True)

    def hooked(gate):
        if hook == "pointwise":
            return _gated(gate)
        if hook == "width-reading":
            # Shape-preserving at every width, its constant its width's: a chunk's would scale it wrong.
            def scaled(out, g):
                return out * jax.nn.sigmoid(g) * (out.shape[-1] / width)

            return template.Hooks(output=Hook(scaled, (Side(gate, template.Kind.OUTPUT),)))
        return template.Hooks(output=Hook(lambda out, g: out * jax.nn.sigmoid(g.reshape((-1, width))),
                                          (Side(gate, template.Kind.OUTPUT),)))

    pins = ({attention_mosaic.MosaicForward: UNSPLIT[device]}
            if hook != "whole-width" and device in UNSPLIT else {})

    def ours(query, key, value, gate):
        with generation.assumed(device, interpreted=True), tuning.pinned(pins):
            return attention_mosaic.forward(query, key, value, spans, hooked(gate), interpret=True)

    def reference(query, key, value, gate):
        return template.reference(query, key, value, spans, hooked(gate))

    assert_as_exact(jax.jit(ours), jax.jit(reference), _wide(queries, width),
                    f"mosaic forward, a {hook} bf16 gate 256 wide", seeds=SEEDS)
