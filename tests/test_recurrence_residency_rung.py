"""The recurrence template's residency rung (`template.Mosaic._fewer`, `resources.Residents`): a spilled launch
a program fewer a multiprocessor, each on its own, where it moves no registers as where its producer's ladder
ran out."""

import dataclasses

import jax
import jax.numpy as jnp

from dew.nn.kda import KimiDeltaAttention
from dew.nn.kernels import generation
from dew.nn.kernels.templates import report, resources
from dew.nn.kernels.templates.recurrence import template
from tools import benchmark_recurrence

SCORES, MEMORY = "recurrence_key_scores_back", "recurrence_key_memory_back"


def _kimi_packed_training(length: int = 256):
    """Kimi delta attention's 16-bit training over packed documents at its scoreboard geometry
    (`benchmark_recurrence.SHAPES`), its decays bounded as the layer declares them, bound for the template."""
    shape = benchmark_recurrence.SHAPES["kimi-linear-kda"]
    heads, width = (1, length, shape.key_heads, shape.key_width), (1, length, shape.value_heads)
    q = k = jax.ShapeDtypeStruct(heads, jnp.bfloat16)
    v = jax.ShapeDtypeStruct((*width, shape.value_width), jnp.bfloat16)
    g = jax.ShapeDtypeStruct((*width, shape.key_width), jnp.float32)
    beta = jax.ShapeDtypeStruct(width, jnp.bfloat16)
    h0 = jax.ShapeDtypeStruct((1, shape.value_heads, shape.key_width, shape.value_width), jnp.float32)
    starts = jax.ShapeDtypeStruct((1, length), jnp.bool_)
    return template.MOSAIC.bind(q, k, v, g, beta, h0, starts, max_segments=8, bound=1.0, rounded=True,
                                decay_bound=KimiDeltaAttention.lower_bound)


def test_a_spilled_launch_moving_no_registers_steps_to_a_program_fewer_on_its_own():
    """Kimi's packed 16-bit training on an sm90, its key-scores backward refused for a spill (a launch with
    no producer to raise, three `[C, C]` accumulators within 128 registers at two programs: JAX 0.12.1's
    compile spilled 176 bytes there, row 14169): its derivation keeps the tiles with that launch a program a
    multiprocessor (`Mosaic._fewer`), every launch held (`template.holds`); read exhausted at its first
    refusal, its tiles were dropped for the next ring depth, which plans the launch alike, down to no tiles.
    Refused again beside a launch that moves registers at two programs, that one alone steps a program
    fewer, its producer at the least, where the launch at one program held every rung back (head review of
    row 14169)."""
    facts = generation.gpu_target("sm90")
    bound = _kimi_packed_training()
    with report.refusals(), generation.assumed(facts):
        derived = template.MOSAIC.derived(bound, facts)
        report.refuse_launch(template.MOSAIC, bound, derived, {SCORES})
        stepped = template.MOSAIC.derived(bound, facts)
        assert stepped == dataclasses.replace(derived, residents=resources.Residents(((SCORES, 1),)))
        launches = {launch.name: launch for launch in template.MOSAIC.launches(bound, stepped, facts,
                                                                                backward=True)}
        assert launches[SCORES].resident == 1
        family, shared = template.bundle(facts), generation.shared_budget(facts)
        assert template.holds(tuple(launches.values()), family, shared)
        assert launches[MEMORY].resident == 2
        both = template.MOSAIC._fewer(bound, stepped, frozenset({SCORES, MEMORY}), facts)
    assert both == dataclasses.replace(stepped, residents=resources.Residents(((MEMORY, 1), (SCORES, 1))))
