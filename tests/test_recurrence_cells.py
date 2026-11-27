"""A receipt's cell routes the device generation it timed alone (`recurrence_chunk.PORT_CELLS`,
`SCAN_CELLS`): a sibling generation of the same instruction family, whose speed against the same reference a
receipt of its own would state, takes the template (docs/design/kernel-templates.md: a declared cell is a
device generation's, never an instruction family's)."""

import jax
import jax.numpy as jnp
import pytest
from test_recurrence_chunk import FORMED, _measured_routes

from dew.nn.kernels import generation, recurrence_chunk
from dew.nn.kernels.recurrence_chunk import Chunk, chunk


@pytest.mark.parametrize(('timed', 'sibling'), [('sm100', 'sm103')])
def test_a_b200s_port_cells_route_no_other_tcgen05_generation(timed, sibling):
    """Qwen3.5-27B's gated delta net forward at its B200 receipt's shapes: the port by its cell on a B200,
    the template on a B300 (sm103), the same family's, which no receipt timed."""
    assert generation.family(timed) == generation.family(sibling)
    shapes = (*[jax.ShapeDtypeStruct((1, 4096, heads, 128), jnp.bfloat16) for heads in (16, 16, 48)],
              jax.ShapeDtypeStruct((1, 4096, 48), jnp.float32),
              jax.ShapeDtypeStruct((1, 4096, 48), jnp.bfloat16))

    def forward(query, key, value, g, beta):
        return recurrence_chunk.chunk_gated_delta_rule(query, key, value, g, beta)[0]

    routes = {name: set(_measured_routes(recurrence_chunk.GATED_DELTA.choice, name, forward, *shapes))
              for name in (timed, sibling)}
    assert routes == {timed: {('fla', True)}, sibling: {('mosaic_gpu', False)}}, routes


def _ssd_training_routes(generation_name: str, heads: int, width: int, chunk_size: int) -> set:
    """The routes (`_measured_routes`) of an SSD's bf16 training on `generation_name` at a Nemotron-H
    receipt's shapes: one row of 4096 tokens, `heads` heads `width` wide, eight groups of 128-wide states,
    chunks of `chunk_size`, a float32 output."""
    def loss(x, B, C, dt, A, D):
        operands = Chunk(C, B, x, beta=dt, rate=A, skip=D)
        out, _ = chunk(FORMED[4], operands, chunk_size=chunk_size, dtype=jnp.float32)
        return out.astype(jnp.float32).sum()

    shapes = (jax.ShapeDtypeStruct((1, 4096, heads, width), jnp.bfloat16),
              *[jax.ShapeDtypeStruct((1, 4096, 8, 128), jnp.bfloat16)] * 2,
              jax.ShapeDtypeStruct((1, 4096, heads), jnp.float32),
              *[jax.ShapeDtypeStruct((heads,), jnp.float32)] * 2)
    return set(_measured_routes(recurrence_chunk.SCAN.choice, generation_name,
                                jax.grad(loss, argnums=(0, 1, 2, 3)), *shapes))


@pytest.mark.parametrize('sibling', ['sm86', 'sm89', 'sm120'])
def test_an_a100s_ssd_port_cells_route_no_other_mma_sync_generation(sibling):
    """Nemotron-H 4B's SSD training at its A100 receipt's shapes: the port by its cell on an A100, the
    template on another mma.sync generation (an A10's, an L4's, an RTX PRO 6000's), which no receipt
    timed."""
    assert generation.family('sm80') == generation.family(sibling)
    routes = {name: _ssd_training_routes(name, 96, 80, 256) for name in ('sm80', sibling)}
    assert ('pallas', True) in routes['sm80'] and routes[sibling] == {('mosaic_gpu', False)}, routes


def test_an_a100s_nemotron_h_30b_ssd_training_is_the_templates():
    """Nemotron-H 30B's SSD training at its A100 receipt's shapes takes the template on an A100, its port cell
    retired (M-JS-271b-a100: 1.898 ms against the port's 2.052): its every route the template's, none a
    receipt's override, no fallback to the port or the portable math."""
    routes = _ssd_training_routes('sm80', 64, 64, 128)
    assert routes == {('mosaic_gpu', False)}, routes
