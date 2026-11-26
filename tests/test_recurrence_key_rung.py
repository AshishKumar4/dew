"""The recurrence template's key-block rung (`template.Mosaic._narrowed`, `mainloop.Keys`): a launch walking a
key block of its own, spilled with no producer or residency rung left, walks half of it."""

import dataclasses

import jax
import jax.numpy as jnp

from dew.nn.kernels import generation
from dew.nn.kernels.templates import report
from dew.nn.kernels.templates.recurrence import mainloop, template

KEYS = "recurrence_keys_back"


def _keys_back(bound, tiles, facts) -> mainloop.Launch:
    [launch] = [launch for launch in template.MOSAIC.launches(bound, tiles, facts, backward=True)
                if launch.name == KEYS]
    return launch


def test_a_spilled_keys_backward_walks_half_its_key_block_and_nothing_else_moves():
    """The gated delta net's 16-bit training at Qwen3.5-27B's widths on an sm120, its keys' backward refused
    for a spill (as its packed training's spilled 8 bytes at 255 registers, row 13064): the call's tiles
    derive again with that launch alone at half its key block (`Tiles.keys`), its programs twice as many,
    every other launch and the forward alone's tiles as they were (head review of 78586ea1f); refused at
    every narrower block, its training derives none."""
    facts = generation.gpu_target("sm120")
    q = k = jax.ShapeDtypeStruct((1, 256, 16, 128), jnp.bfloat16)
    v = jax.ShapeDtypeStruct((1, 256, 48, 128), jnp.bfloat16)
    g = jax.ShapeDtypeStruct((1, 256, 48), jnp.float32)
    beta = jax.ShapeDtypeStruct((1, 256, 48), jnp.bfloat16)
    bound = template.MOSAIC.bind(q, k, v, g, beta, None, None, bound=1.0)
    block = mainloop.block_of(template.bundle(facts, jnp.bfloat16))
    with report.refusals(), generation.assumed(facts):
        tiles = template.MOSAIC.derived(bound, facts)
        assert tiles is not None and tiles.keys == mainloop.Keys()
        report.refuse_launch(template.MOSAIC, bound, tiles, {KEYS})
        narrowed = template.MOSAIC.derived(bound, facts)
        assert narrowed is not None and narrowed.keys.of(KEYS) == block // 2
        assert narrowed.forward == tiles.forward
        assert dataclasses.replace(narrowed, keys=mainloop.Keys()) == tiles
        assert _keys_back(bound, narrowed, facts).programs == 2 * _keys_back(bound, tiles, facts).programs
        rung = narrowed
        while rung is not None and rung.keys.of(KEYS):
            report.refuse_launch(template.MOSAIC, bound, rung, {KEYS})
            rung = template.MOSAIC.derived(bound, facts)
        assert rung is None
