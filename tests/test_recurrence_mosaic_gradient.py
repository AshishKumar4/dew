"""The recurrence template's Mosaic GPU mainloop's gradient (`mainloop_backward.gradients`, its fast backward)
through the chunked op's explicit route, as test_recurrence_mosaic holds its forward: check 1 of the rule's
gradient against its reference's, both from float64, over a GPU's cases and, interpreted off one, the cases
that cover each launch (`INTERPRETED_GRADIENT`); a gradient the fast backward does not take, refused. A file
of its own: interpreted, the forward's and the gradient's checks together took 37 minutes in one process, past
a CI task's deadline."""

import dataclasses
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fla_sites import the_template_is_as_exact_as_fla
from generation_support import runs
from reference_error import with_gradients
from test_recurrence_chunk import chunk_on
from test_recurrence_mosaic import (
    DTYPES,
    PACKED_DOCUMENTS,
    SHAPES,
    case_hooks,
    draw,
    geometry,
    interpreted,
    on_the_mainloop,
    recorded_site,
    refuses_its_gradient,
    template_call,
)

from dew.nn.backbones.decoder_block import RematPolicy
from dew.nn.kernels import generation, recurrence
from dew.nn.kernels.recurrence_chunk import Chunk, mosaic_rule, recomputing
from dew.nn.kernels.templates import bundles
from dew.nn.kernels.templates.recurrence import mainloop, template
from dew.nn.mixer_base import DELTA_RULE_RESIDUALS

pytestmark = pytest.mark.kernel

INTERPRETED_GRADIENT: set[tuple] = set()
"""The gradient's cases the CPU interprets, each a check-1 site of its own (`interpreted_gradient`): the fast
backward's launches over a ragged chunk, two key blocks and two key heads' groups, packed documents, its
forward's memories kept and recomputed, under a decay a key dimension (`key_backward`'s), and an fp32 call's
under the run's fast forms. Interpreted, the whole grid took 5.4 times these (7828 against 1452 seconds)."""

REMATS = {'kept': ((), None), 'recomputed': (('delta_memory',), None), 'saved': ((), DELTA_RULE_RESIDUALS)}
"""The remat rungs a gradient case runs under, each the names its layers recompute (`recomputing`) and those a
checkpoint saves (`decoder_block.RematPolicy.save`, None for no checkpoint): the forward's memories kept,
recomputed in the backward, or kept by their names under a checkpoint saving the delta rule's (row 2023)."""

GRADIENT_GRID = [(case, dtype, remat) for remat in REMATS for dtype in DTYPES for case in SHAPES]
"""The gradient's cases, the CPU's (`INTERPRETED_GRADIENT`) and the GPU flow's."""


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


def interpreted_gradient(case: str, dtype: str, remat: str) -> Callable:
    """An interpreted gradient case's test, its one declaration: the case among `INTERPRETED_GRADIENT` and
    its check 1 (`the_mosaic_gradient_is_as_exact_as_its_reference`) run by the test it decorates, a check-1
    site of its own (its `check1_shards` mark). Whole, these cases took 122 to 225 seconds (train 270's CI on
    66a222273, ledger 2127), 483 to 653 (train 479's on e82eab19a) and 474 (the recomputed one, train 521's
    on e65403fc9); a map task of 80 seeds of the three 256-seed cases took 525 s, one of 15 of the two
    128-seed cases 159 s (other-shards at 77a98366c), so each case is a site apart."""
    INTERPRETED_GRADIENT.add((case, dtype, remat))

    def check():
        the_mosaic_gradient_is_as_exact_as_its_reference(case, DTYPES[dtype], remat)

    return lambda test: functools.update_wrapper(check, test)


@pytest.mark.check1_shards(seeds=256, per_task=20)
@interpreted_gradient('packed documents', 'bf16', 'kept')
def test_a_packed_bf16_gradient_is_as_exact_as_its_reference():
    """`the_mosaic_gradient_is_as_exact_as_its_reference` at packed documents in bf16, from the forward's
    memories kept: a task of 32 seeds took 122 s (grad-shards at 61f6e8d39)."""


@pytest.mark.check1_shards(seeds=256, per_task=32)
@interpreted_gradient('a ragged length', 'bf16', 'kept')
def test_a_ragged_bf16_gradient_is_as_exact_as_its_reference():
    """`the_mosaic_gradient_is_as_exact_as_its_reference` at a ragged length in bf16, from the forward's
    memories kept."""


@pytest.mark.check1_shards(seeds=256, per_task=32)
@interpreted_gradient('one sequence', 'bf16', 'kept')
def test_a_bf16_gradient_over_one_sequence_is_as_exact_as_its_reference():
    """`the_mosaic_gradient_is_as_exact_as_its_reference` at one sequence in bf16, from the forward's
    memories kept."""


@pytest.mark.check1_shards(seeds=256, per_task=24)
@interpreted_gradient('a ragged length', 'bf16', 'recomputed')
def test_a_recomputed_ragged_bf16_gradient_is_as_exact_as_its_reference():
    """`the_mosaic_gradient_is_as_exact_as_its_reference` at a ragged length in bf16, its forward's memories
    recomputed under a remat rung (`recomputing`)."""


@pytest.mark.check1_shards(seeds=128, per_task=16)
@interpreted_gradient('a decay a key dimension', 'bf16', 'kept')
def test_a_bf16_gradient_under_a_decay_a_key_dimension_is_as_exact_as_its_reference():
    """`the_mosaic_gradient_is_as_exact_as_its_reference` under a decay a key dimension in bf16, from the
    forward's memories kept."""


@pytest.mark.check1_shards(seeds=128, per_task=12)
@interpreted_gradient('a ragged length', 'fp32', 'kept')
def test_a_ragged_fp32_gradient_is_as_exact_as_its_reference():
    """`the_mosaic_gradient_is_as_exact_as_its_reference` at a ragged length in fp32, from the forward's
    memories kept: 5.1 s a seed."""


def the_mosaic_gradient_is_as_exact_as_its_reference(case, dtype, remat):
    """The gradient of the mainloop's rule on par from float64 with its reference's over the case's draws
    (check 1, truth the recurrence a token at a time's in float64): in a 16-bit call the mainloop's fast
    backward (`mainloop_backward.gradients`, fla's DAG; under a decay a key dimension fla_kda's,
    `key_backward`) against recorded fla-core's (`fla_sites`: chunk_gated_delta_rule's, chunk_kda's), from
    the forward's memories kept, under a remat rung that recomputes them (`recomputing`) recomputed, or under
    a checkpoint saving the delta rule's names kept by them. A call the template refuses is checked refused
    and its gradient on the mainloop's rule (`on_the_mainloop`). A gradient the fast backward does not take,
    an fp32 call's outside the run's fast forms (`mainloop_backward.fast_backward_form`) or one whose launches
    the device does not hold (`Mosaic.trains`), the rule refuses as it is traced: the delta rule's selector
    gives it the port or the portable math, whose check 1
    test_recurrence_chunk holds (row 1926: no kernel falls back inside itself)."""
    hooks = case_hooks(case)

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

    on = on_the_mainloop(hooks, case, dtype)
    if on is None:
        return
    draws = functools.partial(drawn, dtype, **geometry(case))
    (sample,) = draw(dtype, 0, **geometry(case))
    if not trained(sample, hooks):
        refuses_its_gradient(compiled(on), *draws(0))
        return
    recomputed, saved = REMATS[remat]

    def rule(given: Chunk):
        run = functools.partial(on, **bound)
        if saved is not None:
            run = jax.checkpoint(run, policy=RematPolicy(save=saved).checkpoint_policy())
        return run(given)

    def under(run):
        def traced(*arguments):
            with recomputing(recomputed):
                return run(*arguments)

        return traced

    the_template_is_as_exact_as_fla(rule, recorded_site(case, dtype),
                                    label=f'the mosaic gradient, {case} in {jnp.dtype(dtype).name}',
                                    under=under)


def test_a_recomputed_decay_a_key_dimension_trains_as_its_kept_forward():
    """Kimi delta attention's gradient through the template (interpreted off a GPU) under the remat rung that
    recomputes the forward's memories (`recomputing(('delta_memory',))`), its recompute reading the inverse
    the forward kept (row 3591), over one draw of its recorded site: bitwise the gradient its kept memories
    give, whose check 1 against recorded fla holds both
    (`the_mosaic_gradient_is_as_exact_as_its_reference`)."""
    from fla_sites import LEAVES, ruled, sites

    case = 'a decay a key dimension'
    on = on_the_mainloop(case_hooks(case), case, jnp.bfloat16)
    arrays = sites()[recorded_site(case, jnp.bfloat16)].arrays(0)

    def under(names):
        def run(given):
            with recomputing(names):
                return ruled(on, given, LEAVES)
        return run

    found = (jax.tree.leaves(jax.jit(under(names))(arrays)) for names in ((), ('delta_memory',)))
    for leaf, a, b in zip(LEAVES, *found, strict=True):
        assert np.array_equal(np.asarray(a), np.asarray(b)), leaf


WALK_GRID = [('one sequence', 'forward'), ('one sequence', 'training')]
INTERPRETED_WALKS = {('one sequence', 'forward')}
"""A two-role walk's products: the forward interpreted off a GPU, its fast backward the GPU flow's."""
WALKS = interpreted(WALK_GRID, INTERPRETED_WALKS)


@runs('sm90')
@pytest.mark.parametrize(('case', 'product'), WALKS)
@pytest.mark.check1_shards(seeds=256, per_task=128)
def test_a_walk_of_two_roles_is_as_exact_as_its_reference(case, product):
    """The states and reverse launches with two value blocks of a sequence and head a program, one a
    consumer role (`Tiles.walkers`, warpgroup MMA's bundle, which the derivation gives a call whose walks
    fill more than a wave), on par from float64 with recorded fla-core's forward, or forward and gradient
    (`fla_sites`), over the case's draws (check 1, truth the recurrence a token at a time's): each role's
    columns, staging and stores its own. Off a GPU interpreted; a GPU that runs no Hopper kernels (an A100)
    skips it. Its interpreted forward's 256 recorded draws took 108 s whole with the shared cache warm
    (jsf-slow-37e5) and 135 to 148 s in main's CI (ea1d57903, 614382ac3): a site, two map tasks."""
    hooks = recurrence.Hooks(decay='scalar', scale=SHAPES[case]['key_width'] ** -0.5, write_bound=1.0)
    tiles = mainloop.Tiles(32, 1, walkers=2)

    def walked(given: Chunk):
        return mosaic_rule(given.query, given.key, given.value, given.log_decay, given.beta, given.state,
                           tiles=tiles, scale=hooks.scale, bound=hooks.inverse_bound,
                           family=bundles.of('sm90', jnp.bfloat16))

    site = f"walk/{'cpu' if mainloop.family_of() is None else 'gpu'}"
    if product == 'forward':
        the_template_is_as_exact_as_fla(walked, site, ('out', 'final'), f'a two-role walk, {case}')
        return
    the_template_is_as_exact_as_fla(walked, site, label=f'a two-role walk\'s gradient, {case}')


def test_the_fast_backward_trains_a_bf16_rated_scan_alone():
    """A rated scan's (Mamba2's SSD) 16-bit gradient under the run's fast forms, whose numbers are
    mamba_ssm's: the template's fast backward takes a bf16 call, the case recorded mamba_ssm holds it to, and
    refuses an fp16 one, which no recording holds, so the port trains it
    (`mainloop_backward.fast_backward_form`)."""
    hooks = recurrence.Hooks(decay='scalar', rated=True, delta=False, skip=True)
    shape = geometry('one sequence')
    with generation.run_precision(dataclasses.replace(generation.numerics(), forms='fast')):
        assert trained(drawn(jnp.bfloat16, 0, **shape)[0], hooks)
        assert not trained(drawn(jnp.float16, 0, **shape)[0], hooks)


ROUNDED_EXACTLY = {
    "a write": ({"x": 129 / 128, "B": 1.0, "C": 0.0, "dt": 129 / 256, "dy": 0.0}, "final", 16641 / 32768),
    "a write gate's gradient": ({"x": 1.0, "B": 1.0, "C": 129 / 128, "dt": 0.5, "dy": 129 / 128}, "ddt",
                                16641 / 16384),
}
"""One live token's rated scan whose value mamba_ssm's 16-bit arithmetic holds exactly, each case's operands
at the token (zero elsewhere, `A = D = 0`, no entering memory), the value it checks and that value: its
write, `(B dt)^T x` with `B dt` and `x` each a bf16 number (the gate on the keys' side, `_chunk_state_fwd`),
and its write gate's gradient, `dy (C B) x` summed in fp32 (`_chunk_scan_chunk_state_bwd_dx`)."""


@pytest.mark.parametrize('case', ROUNDED_EXACTLY)
def test_the_rated_scan_rounds_where_mamba_ssm_does(case):
    """A rated scan's (Mamba2's SSD) 16-bit call on the template under the run's fast forms, interpreted off
    a GPU, at one live token whose value mamba_ssm's arithmetic holds exactly (`ROUNDED_EXACTLY`): the
    template's is that value, its gate on the keys' side and its values' gradient in fp32 rounding nothing
    mamba_ssm's does not."""
    at, checked, value = ROUNDED_EXACTLY[case]
    hooks = recurrence.Hooks(decay='scalar', rated=True, delta=False, skip=True)
    rows, length, heads, states, width = 1, 64, 1, 64, 64

    def one(name, shape, dtype):
        return jnp.zeros(shape, jnp.float32).at[(0,) * len(shape)].set(at[name]).astype(dtype)

    x, B, C = (one(name, (rows, length, heads, size), jnp.bfloat16)
               for name, size in (("x", width), ("B", states), ("C", states)))
    dt = one("dt", (rows, length, heads), jnp.float32)
    dy = one("dy", (rows, length, heads, width), jnp.float32)
    zero = jnp.zeros((heads,), jnp.float32)

    def scanned(dt):
        return chunk_on('mosaic_gpu', hooks, Chunk(C, B, x, beta=dt, rate=zero, skip=zero), chunk_size=64)

    with generation.run_precision(dataclasses.replace(generation.numerics(), forms='fast')):
        if checked == "final":
            found = jnp.max(jnp.abs(scanned(dt)[1]))
        else:
            found = jax.grad(lambda dt: jnp.sum(scanned(dt)[0].astype(jnp.float32) * dy))(dt)[0, 0, 0]
    assert float(found) == value, f"{case}: {float(found)!r} where mamba_ssm's is {value!r}"


def test_the_rated_scans_skip_reaches_the_output_in_the_dtype_its_caller_asks():
    """A rated scan's (Mamba2's SSD) 16-bit call on the template under the run's fast forms, interpreted off
    a GPU, its output asked in float32 (Mamba-2's mixer asks it so, unrounded for the earlier shards' state it
    adds, `mamba2.chunk_ssd`): with no state term (`B = C = 0`) each output is its skip's product `x D` in
    float32, at `x = D = 129/128` 16641/16384, which a bfloat16 output rounds to 130/128."""
    hooks = recurrence.Hooks(decay='scalar', rated=True, delta=False, skip=True)
    rows, length, heads, states, width = 1, 64, 1, 64, 64
    x = jnp.full((rows, length, heads, width), 129 / 128, jnp.bfloat16)
    B = C = jnp.zeros((rows, length, heads, states), jnp.bfloat16)
    dt = jnp.full((rows, length, heads), 0.5, jnp.float32)
    skip = jnp.full((heads,), 129 / 128, jnp.float32)
    with generation.run_precision(dataclasses.replace(generation.numerics(), forms='fast')):
        out, _ = chunk_on('mosaic_gpu', hooks, Chunk(C, B, x, beta=dt, rate=jnp.zeros((heads,), jnp.float32),
                                                     skip=skip), chunk_size=64, dtype=jnp.float32)
    assert out.dtype == jnp.float32
    assert bool(jnp.all(out == jnp.float32(129 / 128) * jnp.float32(129 / 128)))


def trained(given: Chunk, hooks: recurrence.Hooks) -> bool:
    """Whether the template's gradient runs a call over `given`'s operands under `hooks` on the device its
    launches run for (`Mosaic.trains`: the fast backward takes it, its launches held at the call's tiles)."""
    return template.MOSAIC.trains(template_call(given, hooks), template.Mosaic.launched())
