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
from generation_support import COMPILES_ON_CPU, runs
from reference_error import assert_as_exact, with_gradients
from test_recurrence_chunk import ROUTES, chunk_on, cotangents, formed, on_route, routed, token_by_token
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
from dew.nn.kernels.recurrence_chunk import Chunk, recomputing
from dew.nn.kernels.templates import bundles, mosaic_sm90, mosaic_sm100
from dew.nn.kernels.templates.recurrence import chunk_forward, mainloop, mainloop_backward, template
from dew.nn.kernels.templates.recurrence.template import mosaic_rule
from dew.nn.residuals import DELTA_RULE_RESIDUALS

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
    own stream (`cotangents`)."""
    (operands,) = draw(dtype, seed, **shape)
    return dataclasses.replace(operands, starts=None), cotangents(operands, seed)


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
    128-seed cases 159 s (other-shards at 77a98366c), so each case is a site apart, on each route
    (`routed`)."""
    INTERPRETED_GRADIENT.add((case, dtype, remat))
    return routed(lambda: the_mosaic_gradient_is_as_exact_as_its_reference(case, DTYPES[dtype], remat))


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


@pytest.mark.check1_shards(seeds=32, per_task=8)
@interpreted_gradient('a decay a key dimension', 'fp32', 'kept')
def test_an_fp32_gradient_under_a_decay_a_key_dimension_is_as_exact_as_its_reference():
    """`the_mosaic_gradient_is_as_exact_as_its_reference` under a decay a key dimension in fp32 under the
    run's fast forms (`key_backward` in `mainloop.FAST_PARTS` bf16 parts, refused at exact products,
    `mainloop.per_key_refusal`), from the forward's memories kept, held to recorded fla-core's fp32."""


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
    # Judged under the forms the call runs (`on_the_mainloop`'s fast ones): an fp32 call's rounding reads the
    # product precision they set (`recurrence_chunk.fast_rounding`).
    if not formed('fast', trained)(sample, hooks):
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


@pytest.mark.check1_shards(seeds=8, per_task=4)
@pytest.mark.parametrize('case', ['one sequence', 'packed documents'])
def test_tcgen05s_chunks_train_as_wgmma_does_interpreted(case):
    """The template's training step at a B200's derived tiles on its bundle (tcgen05's 64-row products into
    tensor memory read back in the WGMMA layout, `mosaic_sm100.CHUNKS`, interpreted: Mosaic GPU's interpreter
    models tensor memory and tcgen05's products) on par from float64 with the same tiles on wgmma's bundle
    (check 1, the truth the recurrence a token at a time in float64): its output, final memory and every
    operand's gradient in bf16 (row 4747). Wgmma's arm is held to recorded fla-core
    (`the_mosaic_gradient_is_as_exact_as_its_reference`)."""
    hooks, dtype = case_hooks(case), jnp.bfloat16
    (sample, _) = drawn(dtype, 0, **geometry(case))
    tiles = template.MOSAIC.derived(template_call(sample, hooks), generation.gpu_target('sm100'))
    assert tiles is not None

    def on(family):
        def rule(given):
            return template.mosaic_rule(given.query, given.key, given.value, given.log_decay, given.beta,
                                        given.state, tiles=tiles, family=family, delta=hooks.delta,
                                        scale=given.key.shape[-1] ** -0.5, bound=hooks.inverse_bound,
                                        rounded=True)
        return jax.jit(with_gradients(rule))

    assert_as_exact(on(mosaic_sm100.CHUNKS), on(mosaic_sm90.WGMMA), functools.partial(drawn, dtype,
                    **geometry(case)), f'tcgen05 chunks training, {case}', seeds=8,
                    truth_fn=jax.jit(with_gradients(functools.partial(token_by_token, hooks))))


@pytest.mark.check1_shards(seeds=8, per_task=4)
@pytest.mark.parametrize('family', ['wgmma', 'mma.sync'])
@pytest.mark.parametrize('case', ['one sequence', 'a decay a key dimension'])
def test_a_128_token_chunk_trains_as_64_does_interpreted(case, family):
    """The template's forward and fast backward (`chunk_forward.kept_forward`,
    `mainloop_backward.kept_backward`) at 128 tokens a chunk (`Tiles.chunk`), interpreted on wgmma's and
    mma.sync's bundles, on par from float64
    with the same launches at 64 (check 1, the truth the recurrence a token at a time in float64): the output,
    final memory and every operand's gradient in bf16 (row 4770). No rule derives 128 at the scoreboard's
    geometries, whose launches at 128 pass a GPU's resources (`template.chunk_of`'s step down,
    jsf-chunkchoice), so the launches are called as the rule calls them."""
    hooks, dtype = case_hooks(case), jnp.bfloat16
    bundle = bundles.of({'wgmma': 'sm90', 'mma.sync': 'sm80'}[family], jnp.bfloat16)
    (sample, _) = drawn(dtype, 0, **geometry(case))
    tiles = template.MOSAIC.derived(template_call(sample, hooks), generation.gpu_target('sm90'))
    assert tiles is not None

    def at(chunk):
        common = {'tiles': dataclasses.replace(tiles, chunk=chunk), 'family': bundle, 'interpret': True,
                  'delta': hooks.delta, 'rounded': True}

        def trained(given, cotangent):
            operands = (given.query, given.key, given.value, given.log_decay, given.beta, given.state)
            out, final, (inverse, kept) = chunk_forward.kept_forward(*operands, **common)
            grads, _ = mainloop_backward.kept_backward(*operands, *cotangent, inverse=inverse, kept=kept,
                                                       **common)
            # Its gradients of the operands, the skip's last beside them none.
            names = ('query', 'key', 'value', 'log_decay', 'beta', 'state')
            return (out, final), dataclasses.replace(given, **{
                name: grad.astype(operand.dtype)
                for name, grad, operand in zip(names, grads[:len(names)], operands, strict=True)})
        return jax.jit(trained)

    assert_as_exact(at(128), at(64), functools.partial(drawn, dtype, **geometry(case)),
                    f'128-token chunks training on {family}, {case}', seeds=8,
                    truth_fn=jax.jit(with_gradients(functools.partial(token_by_token, hooks))))


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
    tiles = mainloop.Tiles(32, 1, walkers=2, chunk=64)

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


@pytest.mark.parametrize("route", ROUTES)
def test_a_decay_a_key_dimensions_first_token_takes_no_gradient_where_its_sequence_starts_from_none(route):
    """Packed documents under a decay a key dimension: each document after its row's first starts from no
    memory, so its first token's decay scales nothing and its gradient is exactly zero in float64, as fla's
    port takes it (`fla_kda.first_decay_grad`). The template's from its chunk's end, every term of the chunk,
    held 1.5e-7 there against fla's 3.3e-9 on an A100 (row 5194); taken from the memory it scales, zero."""
    case = 'a decay a key dimension, packed'
    hooks = case_hooks(case)
    with on_route(route):
        # The route's geometry: its rows a GPU's, its documents' starts the draw's.
        given, cotangent = drawn(jnp.bfloat16, 0, **geometry(case))
        rule = restarted(functools.partial(chunk_on, 'mosaic_gpu', hooks), case)
        grads = jax.device_get(with_gradients(rule)(given, cotangent))
        (operands,) = draw(jnp.float32, 0, **geometry(case))
    starts = np.asarray(operands.starts)
    later = starts.copy()
    later[:, 0] = False
    dg = np.asarray(grads[1].log_decay)
    assert later.any() and np.all(dg[later] == 0.0)


@COMPILES_ON_CPU
def test_a_saved_fp32_gradient_under_the_highest_precision_runs_the_numbers_its_route_took():
    """An fp32 call's gradient on the template under its fast forms, in a remat saving the delta rule's
    names (`DELTA_RULE_RESIDUALS`) within a caller's 'highest' product precision, lowered for an H100 on
    this CPU (`kernel_compile.lowered_for`): its fast backward runs at the numbers the selector gave the
    call (`recurrence_chunk.chunk`'s `rounded`), not the accurate forms' the remat's pullback read later
    under 'highest', whose solve no sm90 thread holds (`base.Unfitted` natively, M-JS-63)."""
    from fla_sites import ruled
    from kernel_compile import lowered_for

    from dew.nn.kernels.lowered import kernel_calls

    hooks = case_hooks("a ragged length")
    given, (dout, dfinal) = drawn(jnp.float32, 0, **SHAPES["a ragged length"])
    arrays = {"q": given.query, "k": given.key, "v": given.value, "g": given.log_decay, "beta": given.beta,
              "h0": given.state, "dy": dout, "dfinal": dfinal}
    run = formed("fast", functools.partial(chunk_on, "mosaic_gpu", hooks))
    saved = jax.checkpoint(run, policy=RematPolicy(save=DELTA_RULE_RESIDUALS).checkpoint_policy())
    abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), arrays)
    with generation.assumed("sm90"), jax.default_matmul_precision("highest"):
        lowered = lowered_for(lambda arrays: ruled(saved, arrays), (abstract,), "sm90")
    assert any(call.name.startswith("recurrence_values_back") for call in kernel_calls(lowered, unrun=True))
