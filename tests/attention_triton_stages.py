"""The template's Pallas-Triton dK/dV launch on an H100 is right on a prescaled query.

Triton's pipeliner (LowerLoops.cpp's `loadRequiresAdditionalBuffer`) gave a load of one use alone a buffer
of its own a stage. The dK/dV launch reads its query tile twice, raw and by the call's power of two
(`attention.prescale`: bf16's and fp32's), so at two stages the next stage's copy overwrote the tile under
dK's product: an H100's dK wrong, and different a repeat (FS-TRITON-RACE1-h100, ET-P0RACE-h100). The root
fix is the pipeliner's, in the jaxlib the locks pin. Each cell is a route's training step that reaches that
launch on an H100, on par with the reference path's (check 1, with its float64 terms) and the same bit for
bit a repeat (the launches it runs add nothing atomically).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from attention_oracle import attention_terms, kept
from reference_error import assert_as_exact
from test_attention_mosaic import ON_HOPPER

from dew.nn.attention import AttentionImpl, scaled_dot_product_attention
from dew.nn.kernels import generation

LENGTH = 1024
SEEDS = 8
CALLS: dict[str, tuple] = {
    "gemma-softcap-256": (256, 8, 4, jnp.bfloat16, 50.0, "auto", None),
    "triton-bf16-128": (128, 8, 2, jnp.bfloat16, None, "template", "triton"),
    "template-fp32-128": (128, 8, 2, jnp.float32, None, "template", None),
    "template-fp32-256": (256, 8, 4, jnp.float32, None, "template", None),
}
"""Causal calls that reach the Pallas-Triton backward on an H100, by width, query and key heads, dtype,
softcap, implementation and the mainloop named (`generation.chosen`): Gemma's softcap at its default route
(the Mosaic backward takes no hooked call 256 wide), a bf16 call on the Triton mainloop named, and fp32 calls,
which the Mosaic mainloop takes none of, 256 wide among them."""


def _step(name: str, implementation: AttentionImpl):
    """`name`'s training step: its output and the gradients of `sum(out * dout)` of its query, key and
    value."""
    softcap, mainloop = CALLS[name][4], CALLS[name][6]

    def step(query, key, value, dout):
        def attended(query, key, value):
            return scaled_dot_product_attention(query, key, value, implementation=implementation, causal=True,
                                                softcap=softcap)

        out, pull = jax.vjp(attended, query, key, value)
        return (out, *pull(dout))

    if mainloop is None or implementation == "reference":
        return step

    def chosen(*operands):
        with generation.chosen(attention_mainloop=mainloop):
            return step(*operands)

    return chosen


def _drawn(name: str):
    """A draw of `name`'s operands by seed: unit Gaussians in its dtype."""
    width, heads, kv_heads, dtype, *_ = CALLS[name]

    def draw(seed: int):
        streams = jax.random.split(jax.random.key(seed), 4)
        query, dout = (jax.random.normal(stream, (1, LENGTH, heads, width), dtype) for stream in streams[:2])
        key, value = (jax.random.normal(stream, (1, LENGTH, kv_heads, width), dtype)
                      for stream in streams[2:])
        return query, key, value, dout

    return draw


@ON_HOPPER
@pytest.mark.parametrize("name", CALLS)
def a_prescaled_tritons_backward_is_on_par_and_repeats_on_an_h100(name: str) -> None:
    """`name`'s training step on an H100 on par with the reference path's (check 1) and the same bit for bit
    run again on the same draw (the Triton pipeliner's race, the module's docstring)."""
    softcap, implementation = CALLS[name][4], CALLS[name][5]
    keep = kept(LENGTH, LENGTH, causal=True)

    def terms(query, key, value, dout):
        return tuple(attention_terms(query, key, value, dout, keep, softcap=softcap)[:4])

    ours = jax.jit(_step(name, implementation))
    assert_as_exact(ours, jax.jit(_step(name, "reference")), _drawn(name), name, seeds=SEEDS, terms=terms)
    operands = _drawn(name)(0)
    first, again = ours(*operands), ours(*operands)
    for index, (a, b) in enumerate(zip(first, again, strict=True)):
        assert np.array_equal(np.asarray(a), np.asarray(b)), f"{name}: leaf {index} differs run again"
