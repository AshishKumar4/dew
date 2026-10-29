"""The recurrence template's Mosaic GPU mainloop's gradient (`mainloop.gradients`, its fast backward)
through the chunked op's explicit route, as test_recurrence_mosaic holds its forward: check 1 of the rule's
gradient against its reference's, both from float64, over a GPU's cases and, interpreted off one, the cases
that cover each launch (`INTERPRETED_GRADIENT`); a gradient the fast backward does not take, refused. A file
of its own: interpreted, the forward's and the gradient's checks together took 37 minutes in one process, past
a CI task's deadline."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, with_gradients
from test_recurrence_chunk import chunk_on, token_by_token
from test_recurrence_mosaic import (
    DTYPES,
    NATIVE,
    PACKED_DOCUMENTS,
    SEEDS,
    SHAPES,
    draw,
    geometry,
    interpreted,
    on_the_mainloop,
    template_call,
)

from dew.nn.backbones.decoder_block import RematPolicy
from dew.nn.kernels import recurrence
from dew.nn.kernels.recurrence_chunk import Chunk, mosaic_rule, recomputing
from dew.nn.kernels.templates import bundles
from dew.nn.kernels.templates.recurrence import mainloop, template
from dew.nn.mixer_base import DELTA_RULE_RESIDUALS

pytestmark = pytest.mark.kernel

INTERPRETED_GRADIENT = {('a ragged length', 'bf16', 'kept'), ('a ragged length', 'bf16', 'recomputed'),
                        ('a ragged length', 'fp32', 'kept'), ('one sequence', 'bf16', 'kept'),
                        ('packed documents', 'bf16', 'kept')}
"""The gradient's cases the CPU interprets: the fast backward's launches over a ragged chunk, two key blocks
and two key heads' groups, packed documents, its forward's memories kept and recomputed, and an fp32
call's gradient refused. Interpreted, the whole grid took 5.4 times these (7828 against 1452
seconds)."""

GRADIENT_CASES = [case for case, shape in SHAPES.items() if not shape.get('per_key')]
"""The cases whose gradient the mainloop takes: a decay a head's (the gated delta rule's)."""

REMATS = {'kept': ((), None), 'recomputed': (('delta_memory',), None), 'saved': ((), DELTA_RULE_RESIDUALS)}
"""The remat rungs a gradient case runs under, each the names its layers recompute (`recomputing`) and those a
checkpoint saves (`decoder_block.RematPolicy.save`, None for no checkpoint): the forward's memories kept,
recomputed in the backward, or kept by their names under a checkpoint saving the delta rule's (row 2023)."""


def drawn(dtype, seed: int, **shape) -> tuple[Chunk, tuple[jax.Array, jax.Array]]:
    """A draw's operands (`draw`), its documents' starts taken out (`restarted` puts them back: they are not
    differentiable), and cotangents of its output, in `dtype`, and its final memory, fp32, from the seed's
    own key."""
    (operands,) = draw(dtype, seed, **shape)
    keys = jax.random.split(jax.random.fold_in(jax.random.key(seed), 101))
    return dataclasses.replace(operands, starts=None), (
        jax.random.normal(keys[0], operands.value.shape).astype(dtype),
        jax.random.normal(keys[1], operands.state.shape))


def restarted(fn, case: str):
    """`fn` on operands whose documents start as the case's draws' do (`draw`'s, the same each seed)."""
    (operands,) = draw(jnp.float32, 0, **geometry(case))
    return lambda drawn: fn(dataclasses.replace(drawn, starts=operands.starts))


WHOLE_SUITE_INTERPRETED = {('packed documents', 'bf16', 'kept'): 225, ('one sequence', 'bf16', 'kept'): 122}
"""The interpreted gradient's cases past a group's two minutes (seconds, train 270's CI on f58abb8a3, ledger
2127), which the whole suite runs."""


@pytest.mark.parametrize(('case', 'dtype', 'remat'), [
    pytest.param(*case.values, id=case.id, marks=(*case.marks, pytest.mark.whole_suite))
    if case.id in {'-'.join(each) for each in WHOLE_SUITE_INTERPRETED} else case
    for case in interpreted([(case, dtype, remat) for remat in REMATS for dtype in DTYPES
                             for case in GRADIENT_CASES], INTERPRETED_GRADIENT)])
def test_the_mosaic_gradient_is_as_exact_as_its_reference(case, dtype, remat):
    """The gradient of the mainloop's rule on par from float64 with its reference's over the case's draws
    (check 1, truth the recurrence a token at a time's in float64): in a 16-bit call the mainloop's fast
    backward (`mainloop.gradients`, fla's DAG) against the fla port's, whose numbers it runs, from the
    forward's memories kept, under a remat rung that recomputes them (`recomputing`) recomputed, or under a
    checkpoint saving the delta rule's names kept by them. A call
    the template refuses is checked refused and its gradient on the mainloop's rule (`on_the_mainloop`). A
    gradient the fast backward does not take, an fp32 call's or one whose launches the device does not hold
    (`Mosaic.trains`), the rule refuses as it is traced: the delta rule's selector gives it the port or the
    portable math, whose check 1 test_recurrence_chunk holds (row 1926: no kernel falls back inside
    itself)."""
    hooks = recurrence.Hooks(decay='scalar', scale=SHAPES[case]['key_width'] ** -0.5)

    # Packed documents bounded as a layer bounds them (`PACKED_DOCUMENTS`).
    bound = {'max_segments': PACKED_DOCUMENTS} if SHAPES[case].get('packed') else {}

    # Plain functions: the gate compiles both arms alike (`reference_error._arms`).
    def compiled(fn):
        recomputed, saved = REMATS[remat]

        def traced(*arguments):
            run = restarted(functools.partial(fn, **bound), case)
            if saved is not None:
                run = jax.checkpoint(run, policy=RematPolicy(save=saved).checkpoint_policy())
            with recomputing(recomputed):
                return with_gradients(run)(*arguments)

        return traced

    truth = with_gradients(restarted(functools.partial(token_by_token, hooks), case))
    on = on_the_mainloop(hooks, case, dtype)
    if on is None:
        return
    draws = functools.partial(drawn, dtype, **geometry(case))
    (sample,) = draw(dtype, 0, **geometry(case))
    if not trained(sample, hooks):
        with pytest.raises(NotImplementedError):
            jax.eval_shape(compiled(on), *draws(0))
        return
    assert_as_exact(compiled(on), compiled(functools.partial(chunk_on, 'fla', hooks)), draws,
                    f'the mosaic gradient, {case} in {jnp.dtype(dtype).name}', seeds=SEEDS, truth_fn=truth)


@NATIVE
def test_the_fast_backward_is_bitwise_the_same_each_run():
    """The template's fast backward gives the same bits each run over packed documents (row 1298): each
    queries' and keys' gradient tile has one program, which stores its key head's first value head's and
    adds the rest's in order, a step's adds done before the next (`_gradient_stored`); no add's order varies
    between runs."""
    case = 'packed documents'
    hooks = recurrence.Hooks(decay='scalar', scale=SHAPES[case]['key_width'] ** -0.5)
    fn = on_the_mainloop(hooks, case, jnp.bfloat16)
    run = jax.jit(with_gradients(restarted(fn, case)))
    operands, cotangent = drawn(jnp.bfloat16, 0, **geometry(case))
    first, second = (jax.tree.leaves(run(operands, cotangent)) for _ in range(2))
    for ours, again in zip(first, second, strict=True):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(again))


WALKS = interpreted([('one sequence', 'forward'), ('one sequence', 'training')],
                    {('one sequence', 'forward')})
"""A two-role walk's products: the forward interpreted off a GPU, its fast backward the GPU tier's."""


@pytest.mark.parametrize(('case', 'product'), WALKS)
def test_a_walk_of_two_roles_is_as_exact_as_its_reference(case, product):
    """The states and reverse launches with two value blocks of a sequence and head a program, one a
    consumer role (`Tiles.walkers`, warpgroup MMA's bundle, which the derivation gives a call whose walks
    fill more than a wave), on par from float64 with the fla port's forward, or forward and gradient, over
    the case's draws (check 1, truth the recurrence a token at a time's): each role's columns, staging and
    stores its own. Off a GPU interpreted; on one whose family has no two consumer roles (an A100's
    mma.sync), whose device lowers no Hopper copy, refused rather than covered."""
    family = mainloop.family_of()
    if family is not None and family.consumers < 2:
        pytest.skip(f"the {family.name} bundle takes one consumer role a program, no two-role walk")
    hooks = recurrence.Hooks(decay='scalar', scale=SHAPES[case]['key_width'] ** -0.5)
    shape = {**geometry(case), 'value_width': 64}
    tiles = mainloop.Tiles(32, 1, walkers=2)

    def walked(given: Chunk):
        return mosaic_rule(given.query, given.key, given.value, given.log_decay, given.beta, given.state,
                           tiles=tiles, scale=hooks.scale, family=bundles.of('sm90', jnp.bfloat16))

    reference = functools.partial(chunk_on, 'fla', hooks)
    truth = functools.partial(token_by_token, hooks)
    if product == 'forward':
        assert_as_exact(jax.jit(walked), jax.jit(reference), functools.partial(draw, jnp.bfloat16, **shape),
                        f'a two-role walk, {case}', seeds=SEEDS, truth_fn=jax.jit(truth))
        return
    draws = functools.partial(drawn, jnp.bfloat16, **shape)
    assert_as_exact(with_gradients(walked), with_gradients(reference), draws,
                    f'a two-role walk\'s gradient, {case}', seeds=SEEDS, truth_fn=with_gradients(truth))


def trained(given: Chunk, hooks: recurrence.Hooks) -> bool:
    """Whether the template's gradient runs a call over `given`'s operands under `hooks` on the device its
    launches run for (`Mosaic.trains`: the fast backward takes it, its launches held at the call's tiles)."""
    return template.MOSAIC.trains(template_call(given, hooks), template.Mosaic.launched())
