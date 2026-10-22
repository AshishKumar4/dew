"""The attention template's TPU mainloop (`attention_mosaic_tpu`), interpreted on the CPU: at the plain-JAX
mainloop's tiles it computes the same bits (the algebra/mainloop boundary, test_attention_algebra), and it is
as exact as the reference over full and causal calls in both of the algebra's forms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact
from test_attention_template import ROWS
from test_attention_triton import drawing

from dew.nn.kernels.templates import attention as template, attention_mosaic_tpu, attention_portable

DETERMINISTIC = {"xla_cpu_enable_fast_math": False,
                 "xla_disable_hlo_passes": "cpu-instruction-fusion,fusion,simplify-fp-conversions"}
"""XLA's CPU options under which both mainloops' bits are their arithmetic's alone: Pallas TPU's interpreter
runs a kernel inside its grid loop, where XLA's CPU fusion rounds some sums and products differently from
the same algebra in the plain mainloop (one ulp in up to 67% of an output's entries); unfused, none."""

TILES = attention_mosaic_tpu.Tiles(128, 128)
"""The tiles both mainloops walk: a product's rounding depends on its shape."""
QUERIES, KEYS = 200, 300
"""A call of two query blocks and three key blocks, the last of each cut by padding."""


@pytest.mark.parametrize("natural", [False, True], ids=["default", "natural"])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["float32", "bfloat16"])
def test_the_tpu_mainloop_computes_the_plain_mainloops_bits(dtype, causal, natural):
    """Both mainloops call the algebra on the same tiles in the same order, so the output agrees bit for
    bit: a mainloop that masked, scaled or normalized on its own would part from the other here."""
    query, key, value = drawing(QUERIES, KEYS, dtype, width=128, value_width=128)(0)
    spans = template.spans(QUERIES, KEYS, ROWS, causal=causal)
    def tpu(query, key, value):
        return attention_mosaic_tpu.forward(query, key, value, spans, natural=natural, tiles=TILES,
                                            interpret=True)

    def portable(query, key, value):
        return attention_portable.attend(query, key, value, spans, block_q=TILES.block_q,
                                         block_k=TILES.block_k, pad=max(TILES.block_q, TILES.block_k),
                                         natural=natural)

    tpu, portable = (jax.jit(fn).lower(query, key, value).compile(compiler_options=DETERMINISTIC)(
        query, key, value) for fn in (tpu, portable))
    np.testing.assert_array_equal(np.asarray(tpu, np.float32), np.asarray(portable, np.float32))


@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16], ids=["float32", "bfloat16"])
def test_the_tpu_mainloop_is_as_exact_as_the_reference(dtype, causal):
    """Check 1 against the reference's own XLA arithmetic, float64 the truth."""
    spans = template.spans(QUERIES, KEYS, ROWS, causal=causal)

    def ours(query, key, value):
        return attention_mosaic_tpu.forward(query, key, value, spans, tiles=TILES, interpret=True)

    def reference(query, key, value):
        return template.reference(query, key, value, spans)

    draw = drawing(QUERIES, KEYS, dtype, width=128, value_width=128)
    assert_as_exact(jax.jit(ours), jax.jit(reference), draw,
                    f"tpu mainloop {'causal' if causal else 'full'} {jnp.dtype(dtype).name}", seeds=8)
