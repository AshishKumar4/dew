"""The template's Pallas-Triton dK/dV launch on an H100 is right on a prescaled query.

Triton's pipeliner (LowerLoops.cpp's `loadRequiresAdditionalBuffer`) gave a load of one use alone a buffer
of its own a stage. The dK/dV launch reads its query tile twice, raw and by the call's power of two
(`attention.prescale`: bf16's and fp32's), so at two stages the next stage's copy overwrote the tile under
dK's product: an H100's dK wrong, and different a repeat (FS-TRITON-RACE1-h100, ET-P0RACE-h100). The root
fix is the pipeliner's, in the PJRT plugin the locks pin (jax-cuda12 and jax-cuda13 pjrt post45). Each
cell is a route's training step that reaches that launch on an H100, on par with the reference path's
(check 1, with its float64 terms) and the same bit for bit a repeat (the launches it runs add nothing
atomically).
"""

import jax
import jax.numpy as jnp
import pytest
from attention_oracle import attention_terms, kept
from gpu_flow import repeats_bit_for_bit
from reference_error import assert_as_exact
from test_attention_mosaic import ON_HOPPER

from dew.nn.attention import AttentionImpl, scaled_dot_product_attention
from dew.nn.kernels import generation

LENGTH = 1024
SEEDS = 8
REPEATS = 5
CALLS: dict[str, tuple] = {
    "gemma-softcap-256": (256, 16, 8, jnp.bfloat16, 50.0, "auto", None),
    "triton-bf16-128": (128, 32, 8, jnp.bfloat16, None, "template", "triton"),
    "template-fp32-128": (128, 32, 8, jnp.float32, None, "template", None),
}
"""Causal calls that reach the Pallas-Triton backward on an H100, by width, query and key heads, dtype,
softcap, implementation and the mainloop named (`generation.chosen`): Gemma 2 9B's heads under its softcap at
its default route (the Mosaic backward takes no hooked call 256 wide), Qwen3-8B's heads on the Triton mainloop
named (FS-TRITON-RACE1-h100's call), and in fp32, which the Mosaic mainloop takes none of. At main 6bb411dcbd
the Triton mainloop's bf16 call failed (its dK 1.79 times the reference's error, a row 0.56 off,
ET-P0RACE2-h100), and Gemma's and the fp32 call passed: they reach the launch, but their draws showed no
race there."""


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
        # Keys of 0.3, as FS-TRITON-RACE1-h100's: logits of an ordinary spread.
        return query, (0.3 * key).astype(dtype), value, dout

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
    repeats_bit_for_bit(ours, *_drawn(name)(0), runs=REPEATS,
                        names=[f"{name}'s {leaf}" for leaf in ("output", "dQ", "dK", "dV")])
