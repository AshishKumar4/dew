"""An output gate fused into the Mosaic forward's epilogue (row 1956, M-NA-47): a gate over a forward 256
wide finishes a chunk of columns at a time in the launch its ungated call derives, the gate read at its own
bytes; on par with the reference (check 1)."""

import jax
import jax.numpy as jnp
import pytest
from generation_support import INTERPRETS
from reference_error import assert_as_exact

from dew.nn.kernels import generation
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


@INTERPRETS
@pytest.mark.parametrize("device", ["sm90", "sm80"])
def test_a_wide_gated_forward_is_on_par_with_its_16_bit_gate_interpreted(device):
    """A bf16 gate over a forward 256 wide, under the interpreter at each family's derived tiles: its launch
    finishes its sums a chunk of columns at a time, gating each with the gate's own columns, the gate read at
    its own bytes and widened as the hook takes it; on par with the reference (check 1)."""
    queries, width = 128, 256
    spans = template.spans(queries, queries, 1, causal=True)

    def draw(seed):
        streams = jax.random.split(jax.random.key(seed), 4)
        shapes = ((1, queries, 2, width), (1, queries, 1, width), (1, queries, 1, width),
                  (1, queries, 2, width))
        return tuple(jax.random.normal(stream, shape, jnp.float32).astype(jnp.bfloat16)
                     for stream, shape in zip(streams, shapes, strict=True))

    def ours(query, key, value, gate):
        with generation.assumed(device, interpreted=True):
            return attention_mosaic.forward(query, key, value, spans, _gated(gate), interpret=True)

    def reference(query, key, value, gate):
        return template.reference(query, key, value, spans, _gated(gate))

    assert_as_exact(jax.jit(ours), jax.jit(reference), draw, "mosaic forward, a bf16 gate 256 wide",
                    seeds=SEEDS)


@pytest.mark.parametrize("generation_name", ["sm90", "sm80"])
def test_a_wide_gated_forward_walks_the_value_columns_its_ungated_forward_does(generation_name):
    """Qwen3.5-4B's gated attention (16 heads over 4, width 256, a 12,288-token causal row) derives the
    launches of value columns its ungated call does: its gate finishes a chunk of columns at a time, where a
    whole-width gated finish split an H100's one launch into two of 128 columns, each walking every key (1.51x
    the ungated forward and XLA's gate on a GH200, M-NA-47)."""
    rows, tokens, heads, kv_heads, width = 1, 12288, 16, 4, 256
    spans = template.spans(tokens, tokens, rows, causal=True)
    query = jax.ShapeDtypeStruct((rows, tokens, heads, width), jnp.bfloat16)
    key = jax.ShapeDtypeStruct((rows, tokens, kv_heads, width), jnp.bfloat16)
    gate = jax.ShapeDtypeStruct((rows, tokens, heads, width), jnp.bfloat16)
    derived = {}
    for gated in (False, True):
        hook = {"output_mod": _gate, "output_kinds": (template.Kind.OUTPUT,),
                "output_dtypes": (template.kept_dtype(gate.dtype),)} if gated else {}
        op = attention_mosaic.MosaicForward(form="causal", keys=tokens, scale=width ** -0.5, **hook)
        ba = op.bind(query, key, key, spans.start, spans.end, None, None, (), (gate,) if gated else (),
                     None, residuals=template.Kept.NONE, interpret=False)
        with generation.assumed(generation_name):
            derived[gated] = op.derived(ba, generation.gpu_target())
    assert derived[True] is not None and derived[True].columns == derived[False].columns
