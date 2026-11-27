"""The intra walk's plan (`intra._sub_chunk_walk`): the tiles its read-out takes past the walk
(`intra._read_out_staged`) in a scope of their own where its family scopes its walks apart, so an A100 holds
two of Kimi Linear's forward intra programs a multiprocessor (row 13507)."""

import jax.numpy as jnp
from test_recurrence_mosaic import resource_operands

from dew.nn.kda import KimiDeltaAttention
from dew.nn.kernels import generation
from dew.nn.kernels.recurrence_chunk import template_bind
from dew.nn.kernels.templates.recurrence import template


def test_an_a100_holds_two_of_kimi_linears_forward_intra_programs():
    """Kimi Linear's bf16 forward over 4,096 tokens on an A100 (sm80, mma.sync, its walks' rings scoped
    apart): its intra walk's read-out tiles, 30 KB, take the rings' 32 KB past the walk, so a multiprocessor
    holds two of its programs, where beside the walk's staging (108,544 bytes) it held one (row 13507)."""
    facts = generation.gpu_target("sm80")
    shapes = resource_operands("kimi-linear-kda", jnp.bfloat16, 1, 4096)
    with generation.assumed(facts):
        call, _ = template_bind(*(jnp.zeros(shape.shape, shape.dtype) for shape in shapes), rounded=True,
                                decay_bound=KimiDeltaAttention.lower_bound)
        _, bound = call.bound["mosaic_gpu"]
        tiles = template.MOSAIC.derived(bound, facts)
        assert tiles is not None
        [intra] = [launch for launch in template.MOSAIC.launches(bound, tiles.forward or tiles, facts,
                                                                 backward=False)
                   if launch.name == "recurrence_alone_intra"]
    assert intra.resident >= 2, intra
