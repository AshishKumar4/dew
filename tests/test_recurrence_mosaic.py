"""The recurrence template's Mosaic GPU mainloop (`dew.nn.kernels.templates.recurrence.mainloop`) through the
chunked op with the template named to its selectors (`chunk_on('mosaic_gpu', ...)`), compiled on a GPU of an
instruction family one of its bundles claims (mma.sync, wgmma): check 1 (`reference_error.assert_as_exact`) of
its output and final memory against its reference's (the recurrence a token at a time,
`test_recurrence_chunk.token_by_token`, in an fp32 call, the fla port in a 16-bit one), both from float64,
over one sequence a row and packed documents, value heads over fewer key heads, keys 64 and 128 wide, values
off a block's width, bf16 and fp32 operands and lengths off the chunks, and every hook combination the
mainloop takes (a decay a head or none, with and without the delta rule, Mamba2's rated decay, the skip and
the norm) on the chunked op's own draws (`test_recurrence_chunk`). Off such a GPU the cases run in Mosaic
GPU's interpreter on the wgmma bundle (`bundles.INTERPRETED`) at one row of a shorter length
(`INTERPRETED_LENGTH`), whose products are the dots they compute: coverage of the mainloop's algebra and its
roles' streams, not proof of its kernels, which the GPU gate is;
the stability suite holds the mainloop under its regimes there (tests/test_recurrence_stability.py)."""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fla_sites import the_template_is_as_exact_as_fla
from generation_support import lowers
from reference_error import assert_as_exact
from test_recurrence_chunk import (
    FORMED,
    HOOKS,
    SEEDS as HOOK_SEEDS,
    chunk_on,
    draw as hooked_draw,
    formed,
    token_by_token,
)

from dew.nn.kernels import generation, recurrence
from dew.nn.kernels.recurrence_chunk import Chunk, chunk_rounding, fast_rounding, recomputed_memory
from dew.nn.kernels.selection import KernelRefused
from dew.nn.kernels.templates import bundles, mosaic, product, report, resources
from dew.nn.kernels.templates.base import Unfitted
from dew.nn.kernels.templates.recurrence import mainloop, mainloop_backward, template
from dew.nn.kernels.templates.recurrence.template import mosaic_rule

pytestmark = pytest.mark.kernel

SEEDS = 8
"""Draws a case takes, check 1's fewest (`reference_error.FEWEST_SEEDS`)."""

FLOOR = -5.0
"""A decay a key dimension's declared floor a token, Kimi delta attention's `lower_bound`: the draws' decays
lie above it (`lower_bound * sigmoid`), and the mainloop's intra launch factors its pairs under it."""

SHAPES = {
    'one sequence': {'rows': 2, 'length': 192, 'key_heads': 2, 'value_heads': 4, 'key_width': 128,
                     'value_width': 128},
    'a ragged length': {'rows': 2, 'length': 150, 'key_heads': 1, 'value_heads': 2, 'key_width': 64,
                        'value_width': 96},
    'packed documents': {'rows': 2, 'length': 200, 'key_heads': 1, 'value_heads': 3, 'key_width': 64,
                         'value_width': 64, 'packed': True},
    'packed, wide keys': {'rows': 1, 'length': 300, 'key_heads': 1, 'value_heads': 2, 'key_width': 128,
                          'value_width': 128, 'packed': True},
    'a decay a key dimension': {'rows': 2, 'length': 192, 'key_heads': 2, 'value_heads': 2, 'key_width': 64,
                                'value_width': 64, 'per_key': True},
    'a decay a key dimension, packed': {'rows': 2, 'length': 150, 'key_heads': 1, 'value_heads': 1,
                                        'key_width': 64, 'value_width': 64, 'packed': True, 'per_key': True},
}
"""Each case's operands: rows of `length` tokens over `value_heads` value heads a `key_heads` key heads serve,
and packed documents where `packed` (starting on and off the chunks' grid, one shorter than a chunk)."""

INTERPRETED_LENGTH = 140
"""Off a GPU of a family a bundle claims, each case's draws are one row of at most this many tokens (three
chunks, the last ragged, the packed documents' starts still on and off the chunks' grid): the same structure
in Mosaic GPU's interpreter within a test's ten minutes, which the full geometry is not."""


DTYPES = {'bf16': jnp.bfloat16, 'fp32': jnp.float32}

INTERPRETED_FORWARD = {('one sequence', 'fp32'), ('a ragged length', 'bf16'), ('packed documents', 'bf16'),
                       ('a decay a key dimension', 'bf16'), ('a decay a key dimension', 'fp32'),
                       ('a decay a key dimension, packed', 'bf16'),
                       ('a decay a key dimension, packed', 'fp32')}
"""The forward's cases the CPU interprets: each launch in each arithmetic (an fp32 call's three parts, a
16-bit call's), over key blocks and key heads' groups, a ragged chunk, packed documents, a decay a key
dimension's launches packed and not, and an fp32 call's refusal under one. Interpreted, the whole grid took
3.5 times these (2838 against 805 seconds)."""

def interpreted(grid: list[tuple], chosen: set[tuple]) -> list:
    """`grid`'s parameters the CPU interprets, those in `chosen`, each id its values joined by '-'; the rest
    are the GPU flow's alone (`native`, tests/test_gpu_recurrence.py); a test's dtype taken by its id."""
    return [pytest.param(*(DTYPES.get(value, value) for value in values), id='-'.join(values))
            for values in grid if values in chosen]


def native(grid: list[tuple], chosen: set[tuple]) -> list[tuple[str, tuple]]:
    """`grid`'s parameters the GPU flow checks alone, those not in `chosen` (`interpreted`): each its id, its
    values joined by '-', and its values, a dtype taken by its id."""
    return [('-'.join(values), tuple(DTYPES.get(value, value) for value in values))
            for values in grid if values not in chosen]


FORWARD_GRID = [(case, dtype) for dtype in DTYPES for case in SHAPES]
"""The forward's cases, the CPU's (`INTERPRETED_FORWARD`) and the GPU flow's."""


INTERPRETED_HEADS = {'one sequence': {'key_heads': 1, 'value_heads': 2, 'value_width': 64},
                     'packed documents': {'value_heads': 2}}
"""Off a GPU, the cases' heads and values narrowed to what each still checks (one sequence two key blocks and
a key head's group of value heads, packed documents their starts on and off the chunks' grid): the
interpreted gradient of one sequence took 300 to 390 s and of packed documents 535 to 688, past the suite's
240 a test (row 1850); narrowed, 131 s and 336 s: one value head, or two chunks, left the packed log decay
unresolved within 40 seeds, its extra seeds taking 784 and 934 s."""


def case_hooks(case: str) -> recurrence.Hooks:
    """Case `case`'s rule: the gated delta rule with a decay a head, or a key dimension under `FLOOR` (Kimi
    delta attention's), its queries scaled by `Dk^-1/2`."""
    per_key = SHAPES[case].get('per_key')
    return recurrence.Hooks(decay='vector' if per_key else 'scalar', scale=SHAPES[case]['key_width'] ** -0.5,
                            floor=FLOOR if per_key else None, write_bound=1.0)


def geometry(case: str) -> dict:
    """Case `case`'s operands as `SHAPES` gives them, shortened and narrowed off a GPU of a family a bundle
    claims (`geometry_on`)."""
    return geometry_on(case, interpreted=mainloop.family_of() is None)


def recorded_site(case: str, dtype) -> str:
    """Case `case`'s site of recorded fla-core in `dtype` at this machine's geometry (`fla_sites`)."""
    label = {jnp.dtype(value): name for name, value in DTYPES.items()}[jnp.dtype(dtype)]
    return f"mosaic/{case}/{label}/{'cpu' if mainloop.family_of() is None else 'gpu'}"


def geometry_on(case: str, interpreted: bool) -> dict:
    """Case `case`'s operands as `SHAPES` gives them, shortened and narrowed where `interpreted`
    (`INTERPRETED_LENGTH`, `INTERPRETED_HEADS`)."""
    shape = SHAPES[case]
    if not interpreted:
        return shape
    return {**shape, 'rows': 1, 'length': min(shape['length'], INTERPRETED_LENGTH),
            **INTERPRETED_HEADS.get(case, {})}


PACKED_DOCUMENTS = 5
"""The most documents a row of a packed draw holds (`draw`'s starts: its first row's five), the bound a
packed call takes as a layer passes its batch's (`max_segments`): the template lays its launches for that
many sequences a row, where unbounded it lays a row's every token (row 1850: the interpreted gradient's
template arm 60 s a seed unbounded, 35 bounded)."""


def draw(dtype, seed: int, *, rows: int, length: int, key_heads: int, value_heads: int, key_width: int,
         value_width: int, packed: bool = False, per_key: bool = False) -> tuple[Chunk]:
    """A draw's operands from `seed`, its keys' and values' dimensions each permuted by an order the seed
    draws (an exact symmetry of every head's recurrence): unit queries and keys, normal values, a log decay a
    head in (-inf, 0) or a key dimension's in (`FLOOR`, 0) (Kimi delta attention's bounded gate), a write
    gate in (0, 1), a memory entering; the queries, keys, values and write gate in `dtype`."""
    keys = jax.random.split(jax.random.key(seed), 6)
    rng = np.random.default_rng(seed)
    key_order, value_order = rng.permutation(key_width), rng.permutation(value_width)

    def unit(key):
        x = jax.random.normal(key, (rows, length, key_heads, key_width))
        return jnp.take(x / jnp.linalg.norm(x, axis=-1, keepdims=True), key_order, axis=-1)

    query, key = unit(keys[0]), unit(keys[1])
    value = jax.random.normal(keys[2], (rows, length, value_heads, value_width))
    decays = (rows, length, value_heads) + ((key_width,) if per_key else ())
    log_decay = -jax.nn.softplus(jax.random.normal(keys[3], decays) - 1.0)
    if per_key:
        log_decay = FLOOR * jax.nn.sigmoid(jax.random.normal(keys[3], decays) - 1.0)
        log_decay = jnp.take(log_decay, key_order, axis=-1)
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (rows, length, value_heads)))
    state = 0.3 * jax.random.normal(keys[5], (rows, value_heads, key_width, value_width))
    state = jnp.take(jnp.take(state, key_order, axis=2), value_order, axis=3)
    starts = None
    if packed:
        starts = jnp.zeros((rows, length), bool).at[0, jnp.array([0, 23, 40, 128, length - 10])].set(True)
        if rows > 1:
            starts = starts.at[1, jnp.array([9, 64])].set(True)
    return (Chunk(query.astype(dtype), key.astype(dtype), jnp.take(value, value_order, axis=-1).astype(dtype),
                  log_decay=log_decay, beta=beta.astype(dtype), state=state, starts=starts),)


def template_call(given: Chunk, hooks: recurrence.Hooks):
    """The template's call over `given`'s operands under `hooks`, as the chunked op hands them over
    (`recurrence_chunk.chunk`): no decay a zero log decay a head, no write gate ones, the rule's form its
    hooks', its numbers the run's fast forms' where they are its forms (`rounded`, as its route resolves
    it: a rated scan's mamba_ssm's, `recurrence_chunk.fast_rounding`, else the delta rule's and an fp32
    call's two parts, `recurrence_chunk.chunk_rounding`)."""
    heads = given.value.shape[:3]
    g = jnp.zeros(heads, jnp.float32) if given.log_decay is None else given.log_decay
    beta = jnp.ones(heads, given.value.dtype) if given.beta is None else given.beta
    # The l2 norm takes the queries and keys to fp32 at least before the rule reads them.
    query, key = ((x.astype(jnp.promote_types(x.dtype, jnp.float32)) for x in (given.query, given.key))
                  if hooks.l2norm else (given.query, given.key))
    return template.MOSAIC.bind(query, key, given.value, g, beta, given.state, given.starts,
                                delta=hooks.delta, wide=hooks.skip or hooks.output is not None,
                                floor=hooks.floor, bound=hooks.inverse_bound,
                                rounded=fast_rounding() if hooks.rated else chunk_rounding(
                                    hooks.delta, jnp.result_type(query, key, given.value)))


def refuses_its_gradient(fn, *drawn) -> None:
    """`fn`'s gradient, traced over `drawn` where the template's fast backward does not take it (an fp32 call,
    a decay a key dimension's, launches the device does not hold): refused, by the template named to the
    selectors as its route's 'backward' requirement (`recurrence_chunk.MOSAIC_SUPPORTS`, `KernelRefused`), by
    the mainloop called directly (`mosaic_rule`) as not implemented."""
    with pytest.raises((KernelRefused, NotImplementedError)) as refused:
        jax.eval_shape(fn, *drawn)
    if isinstance(refused.value, KernelRefused):
        assert 'backward' in {refusal.code for refusal in refused.value.refusals}


def taken(given: Chunk, hooks: recurrence.Hooks) -> bool:
    """Whether the template takes a call over `given`'s operands under `hooks` on the device its launches run
    for (`Mosaic.eligible`: every launch's shared memory and registers within a block's and a thread's, a
    decay a key dimension's declared floor)."""
    return template.MOSAIC.eligible(template_call(given, hooks), template.Mosaic.launched())


REFUSED = (Unfitted, NotImplementedError, KernelRefused)
"""The template's refusals: no config whose launches the device holds, a call its mainloop does not take
(a decay a key dimension without a floor its sub-chunks factor under), or a requirement of its route's the
named template fails (`selection.KernelRefused`)."""


def on_the_mainloop(hooks: recurrence.Hooks, case: str, dtype, forms: str = 'fast'):
    """The chunked op on the mainloop for `case`'s calls in `dtype` under the run's `forms` (`formed`):
    through the template's route where it takes them (`chunk_on('mosaic_gpu', ...)`, the template named to
    the selectors, at its derived tiles);
    else, the route refusing them (a launch's registers past a thread's), the mainloop's rule at its narrowest
    value block, one chunk in flight (`mosaic_rule`), whose arithmetic is the same at any tiles. None where
    the mainloop's rule refuses the call too (checked): an fp32 call's decay a key dimension
    (`mainloop.per_key_refusal`); and on a GPU where its launches at those tiles pass the device's shared
    memory (an fp32 call's solve on an A100's block), whose arithmetic the CPU interprets."""
    (operands,) = draw(dtype, 0, **geometry(case))
    if formed(forms, taken)(operands, hooks):
        return formed(forms, functools.partial(chunk_on, 'mosaic_gpu', hooks))
    with pytest.raises(REFUSED):
        jax.eval_shape(formed(forms, functools.partial(chunk_on, 'mosaic_gpu', hooks)), operands)
    family = mainloop.family_of() or bundles.INTERPRETED
    if SHAPES[case].get('per_key') and mainloop.per_key_refusal(dtype):
        # The portable math's arm is test_recurrence_chunk's.
        with pytest.raises(NotImplementedError):
            jax.eval_shape(functools.partial(mosaic_rule, tiles=mainloop.Tiles(32, 1), floor=hooks.floor),
                           operands.query, operands.key, operands.value, operands.log_decay, operands.beta)
        return None
    narrowest = mainloop.Tiles(family.narrowest_columns(jnp.bfloat16), 1)
    if mainloop.family_of() is not None:
        shapes = [None if x is None else jax.ShapeDtypeStruct(x.shape, x.dtype)
                  for x in (operands.query, operands.key, operands.value, operands.log_decay, operands.beta,
                            operands.state, operands.starts)]
        *given, starts = shapes
        launches = template.planned(*given, tiles=narrowest, family=family, starts=starts,
                                             floor=hooks.floor)
        if max(launch.shared for launch in launches) > generation.shared_budget(template.Mosaic.launched()):
            return None

    def rule(given: Chunk, max_segments: int | None = None):
        return mosaic_rule(given.query, given.key, given.value, given.log_decay, given.beta, given.state,
                           tiles=narrowest, starts=given.starts, max_segments=max_segments, scale=hooks.scale,
                           floor=hooks.floor, bound=hooks.inverse_bound, rounded=forms == 'fast',
                           recomputed=recomputed_memory())

    return rule


DRAWN_FORWARD: set[tuple] = set()
"""The interpreted forward's drawn cases, each a test of its own (`drawn_forward`); the rest of
`INTERPRETED_FORWARD` the template refuses (the refusal test below)."""


def drawn_forward(case: str, dtype: str) -> Callable:
    """A drawn forward case's test: the case taken out of the refused ones (`DRAWN_FORWARD`) and
    `the_mosaic_forward_is_as_exact_as_its_reference` run by the test it decorates, a check-1 site of its own
    (its `check1_shards` mark). A map task of 128 seeds of the three 256-seed cases took 1045 to 1071 s on
    main's CI (Main, 2026-11-02); the 128-seed cases under a decay a key dimension 250 s and past 600 s whole
    (int-g1-each at 7683420535), a task of 8 seeds of both 117 to 125 s, half of it their compiles
    (keyed-shards at 7e3a5fb48)."""
    DRAWN_FORWARD.add((case, dtype))

    def check():
        the_mosaic_forward_is_as_exact_as_its_reference(case, DTYPES[dtype])

    return lambda test: functools.update_wrapper(check, test)


@pytest.mark.check1_shards(seeds=256, per_task=16)
@drawn_forward('one sequence', 'fp32')
def test_the_mosaic_forward_over_one_fp32_sequence_is_as_exact_as_its_reference():
    """`the_mosaic_forward_is_as_exact_as_its_reference` at an fp32 call over one sequence, under either
    forms."""


@pytest.mark.check1_shards(seeds=256, per_task=16)
@drawn_forward('a ragged length', 'bf16')
def test_the_mosaic_forward_over_a_ragged_length_is_as_exact_as_its_reference():
    """`the_mosaic_forward_is_as_exact_as_its_reference` at a 16-bit call over a ragged length."""


@pytest.mark.check1_shards(seeds=256, per_task=16)
@drawn_forward('packed documents', 'bf16')
def test_the_mosaic_forward_over_packed_documents_is_as_exact_as_its_reference():
    """`the_mosaic_forward_is_as_exact_as_its_reference` at a 16-bit call over packed documents."""


@pytest.mark.check1_shards(seeds=128, per_task=8)
@drawn_forward('a decay a key dimension', 'bf16')
def test_the_mosaic_forward_over_a_decay_a_key_dimension_is_as_exact_as_its_reference():
    """`the_mosaic_forward_is_as_exact_as_its_reference` at a 16-bit call under a decay a key dimension."""


@pytest.mark.check1_shards(seeds=128, per_task=8)
@drawn_forward('a decay a key dimension, packed', 'bf16')
def test_the_packed_mosaic_forward_over_a_decay_a_key_dimension_is_as_exact_as_its_reference():
    """`the_mosaic_forward_is_as_exact_as_its_reference` at a 16-bit call of packed documents under a decay a
    key dimension."""


@pytest.mark.parametrize(('case', 'dtype'), interpreted(FORWARD_GRID, INTERPRETED_FORWARD - DRAWN_FORWARD))
def test_an_fp32_call_under_a_decay_a_key_dimension_is_refused_by_the_mainloop(case, dtype):
    """An fp32 call under a decay a key dimension: the template's route and the mainloop's rule both refuse
    it, under either forms (`on_the_mainloop`, `mainloop.per_key_refusal`); its arm is the portable math's
    (test_recurrence_chunk's)."""
    hooks = case_hooks(case)
    assert on_the_mainloop(hooks, case, dtype, forms='accurate') is None
    assert on_the_mainloop(hooks, case, dtype) is None


def the_mosaic_forward_is_as_exact_as_its_reference(case: str, dtype) -> None:
    """The mainloop's output and final memory on par from float64 with its reference's over the case's draws
    (check 1, truth the recurrence a token at a time in float64): the gated delta rule with a decay a head or
    a key dimension (Kimi delta attention's), its queries scaled by `Dk^-1/2`. An fp32 call's reference
    under the accurate forms is the recurrence a token at a time in fp32, and under the fast forms, as a
    16-bit call's, recorded fla-core in the call's dtype (`fla_sites`), whose numbers it runs: the values
    between launches rounded to bf16 put the final memory 1e-3 from float64, which the fp32 recurrence holds
    to 6e-8. A call the template refuses is checked refused and its arithmetic on the mainloop's rule
    (`on_the_mainloop`)."""
    hooks = case_hooks(case)
    label = f'the mosaic forward, {case} in {jnp.dtype(dtype).name}'
    exact = on_the_mainloop(hooks, case, dtype, forms='accurate') if dtype == jnp.float32 else None
    if exact is not None:
        truth = jax.jit(functools.partial(token_by_token, hooks))
        assert_as_exact(jax.jit(exact), truth, functools.partial(draw, dtype, **geometry(case)),
                        f'{label} under the accurate forms', seeds=SEEDS, truth_fn=truth)
    on = on_the_mainloop(hooks, case, dtype)
    if on is not None:
        the_template_is_as_exact_as_fla(on, recorded_site(case, dtype), ('out', 'final'), label)


UNSEEN = generation.GpuTarget("sm87", "NVIDIA Jetson AGX Orin", 163 * 1024, multiprocessors=16,
                              clock_hz=1.3e9, memory_bandwidth=204.8e9)
"""A GPU the facts table names no row of: Jetson AGX Orin's Ampere (mma.sync's family), 163 KB of shared
memory a block, 16 multiprocessors, its clock and its memory's bandwidth, as its device reports them."""


def unseen_call(dtype) -> tuple:
    """The case 'one sequence' in `dtype` on the GPU no row names (`UNSEEN`): its hooks and its call bound to
    the template (`template.MOSAIC.bind`)."""
    case = 'one sequence'
    hooks = recurrence.Hooks(decay='scalar', scale=SHAPES[case]['key_width'] ** -0.5, write_bound=1.0)
    (operands,) = draw(dtype, 0, **geometry(case))
    bound = template.MOSAIC.bind(operands.query, operands.key, operands.value, operands.log_decay,
                                 operands.beta, operands.state, operands.starts, bound=hooks.inverse_bound)
    return hooks, bound


@pytest.mark.parametrize('dtype', [jnp.bfloat16, jnp.float32], ids=['bf16', 'fp32'])
def test_an_unseen_gpus_facts_derive_the_templates_tiles(dtype):
    """The recurrence template on a GPU no row names (`UNSEEN`): its facts and its family's bundle's
    (mma.sync's) alone derive the call's tiles (`template.Mosaic.derived`): every launch's shared memory as
    the bundle declares it within a block of it (`template.planned`), with two chunks in flight
    unless two are past it or leave a multiprocessor fewer of the states launch's programs
    (`Mosaic.states_held`). Its variants, which the opt-in tuner times (`Mosaic.variants`), lead with those
    tiles, each legal there. An fp32 call it refuses: a launch's registers at its narrowest tiles pass a
    thread's (the fp32 solve's merges hold the inverse's three parts)."""
    assert generation.family(UNSEEN.generation) == 'mma.sync'
    _, bound = unseen_call(dtype)
    budget = generation.shared_budget(UNSEEN)
    with generation.assumed(UNSEEN):
        if dtype == jnp.float32:
            assert template.MOSAIC.derived(bound, UNSEEN) is None
            narrowest = mainloop.Tiles(bundles.of('sm80', jnp.bfloat16).narrowest_columns(jnp.bfloat16), 1)
            assert template.MOSAIC.registers(bound, narrowest) > template.MOSAIC.register_budget(narrowest)
            return
        picked = template.MOSAIC.derived(bound, UNSEEN)
        assert picked is not None and template.MOSAIC.fitting(bound, UNSEEN) == (picked,)
        assert template.MOSAIC.staged_bytes(bound, picked, UNSEEN) <= budget
        assert template.MOSAIC.registers(bound, picked) <= template.MOSAIC.register_budget(picked)
        if picked.stages == 1:
            wider = mainloop.Tiles(picked.block_v, 2)
            assert (template.MOSAIC.staged_bytes(bound, wider, UNSEEN) > budget
                    or template.MOSAIC.states_held(bound, wider, UNSEEN)
                    < template.MOSAIC.states_held(bound, picked, UNSEEN))
        variants = template.MOSAIC.variants(bound)
        assert variants[0] == picked and len(set(variants)) == len(variants)
        assert all(template.MOSAIC.legal(bound, tiles, UNSEEN) for tiles in variants)


@pytest.mark.check1_shards(seeds=256, per_task=32)
def test_the_template_on_an_unseen_gpu_is_as_exact_as_its_reference():
    """The bf16 forward on the GPU no row names (`UNSEEN`) at each of its variants (`Mosaic.variants`, which
    lead with its derived tiles, `test_an_unseen_gpus_facts_derive_the_templates_tiles`), on its bundle
    interpreted: on par from float64 with recorded fla-core (check 1, as
    `the_mosaic_forward_is_as_exact_as_its_reference`'s). Each variant's value block and stages change which
    program computes a value column. Its site's 256 recorded seeds at three variants took 267 to 306 s whole
    on the CPU (int-g1-each at 7683420535), so its draws are sharded over map tasks (`check1_shards`)."""
    hooks, bound = unseen_call(jnp.bfloat16)
    with generation.assumed(UNSEEN):
        variants = template.MOSAIC.variants(bound)

    def dew(given: Chunk, tiles: mainloop.Tiles):
        return mosaic_rule(given.query, given.key, given.value, given.log_decay, given.beta, given.state,
                           tiles=tiles, starts=given.starts, scale=hooks.scale, bound=hooks.inverse_bound,
                           family=bundles.of('sm80', jnp.bfloat16))

    for tiles in variants:
        the_template_is_as_exact_as_fla(functools.partial(dew, tiles=tiles),
                                        recorded_site('one sequence', jnp.bfloat16), ('out', 'final'),
                                        f'the mosaic forward on an unseen GPU at {tiles}, in bfloat16')


@pytest.mark.parametrize('training', [False, True], ids=['forward', 'training'])
def test_the_mosaic_launches_state_their_work(training):
    """A bf16 gated delta rule's forward, or its forward and gradient, through the template, lowered for the
    GPU the test runs on, or off one an H100 (Mosaic GPU lowers no Hopper copy for an A100's device):
    every launch states its cost (`_kernel`'s, `templates.cost.estimate`), which Dew's FLOP count reads to
    size it (`lowered.kernel_calls`' `flops`: no launch unsized, row 1895's kernel report), their products
    together at least the model's recurrent function's (`flops.delta_flops`: the chunked form's preparation
    and inversion are work past it)."""
    from dew.nn.flops import delta_flops
    from dew.nn.kernels import lowered

    rows, length, key_heads, value_heads, width = 1, 256, 2, 4, 128
    operands = [jax.ShapeDtypeStruct((rows, length, key_heads, width), jnp.bfloat16)] * 2 + [
        jax.ShapeDtypeStruct((rows, length, value_heads, width), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, length, value_heads), jnp.float32),
        jax.ShapeDtypeStruct((rows, length, value_heads), jnp.bfloat16)]

    def rule(*operands):
        # The bound a layer declares of its sigmoid write gates (`recurrence.Hooks.write_bound`).
        return mosaic_rule(*operands, tiles=mainloop.Tiles(32, 1), bound=1.0)[0].astype(jnp.float32).sum()

    lowered_for = generation.device_generation() if mainloop.family_of() is not None else "sm90"
    with generation.assumed(lowered_for):
        fn = jax.grad(rule, argnums=(0, 1, 2)) if training else rule
        program = jax.jit(fn).trace(*operands).lower(lowering_platforms=("cuda",))
    launches = [call for call in lowered.kernel_calls(program) if call.kind == "mosaic_gpu"]
    assert launches and all(call.flops is not None for call in launches)
    model = delta_flops(np.asarray([length] * rows), value_heads, width, width)
    assert sum(call.flops or 0.0 for call in launches) >= model * (2 if training else 1)


@lowers('sm90')
@pytest.mark.parametrize(("key_width", "value_width", "dtype", "block"),
                         [(128, 128, jnp.bfloat16, 16), (128, 128, jnp.bfloat16, 32),
                          (576, 128, jnp.bfloat16, None), (128, 256, jnp.float32, None)],
                         ids=["bf16 16 wide", "bf16 32 wide", "bf16 576 keys derived", "fp32 derived"])
def test_an_output_launch_lowers_at_a_value_block_narrower_than_its_keys(key_width, value_width, dtype,
                                                                         block):
    """A gated delta rule's forward lowered for an H100 at value blocks narrower than its keys' rows, pinned
    or as its tiles derive (a 576-wide bf16 key's, an fp32 call's): the output launch's queries are the left
    operand of its scores' product and of its memory's (`Q Kᵀ`, `Q H`), and warpgroup MMA reads a product's
    shared operands at one transform, so the keys' swizzle is one the memory's value-block rows fill too. At
    the keys' own 128 bytes Mosaic GPU found no layout for a 32-wide block's (from 99c198a208)."""
    operands = [jax.ShapeDtypeStruct((1, 256, 2, key_width), dtype)] * 2 + [
        jax.ShapeDtypeStruct((1, 256, 4, value_width), dtype), jax.ShapeDtypeStruct((1, 256, 4), jnp.float32),
        jax.ShapeDtypeStruct((1, 256, 4), dtype)]
    # The run's fast forms, under which an fp32 call derives tiles at all (`fast_rounding`).
    with generation.assumed('sm90'):
        tiles = mainloop.Tiles(block, 1) if block is not None else template.MOSAIC.derived(
            template.MOSAIC.bind(*operands, None, None, floor=None, bound=1.0, rounded=True),
            generation.gpu_target())
        jax.jit(lambda *given: mosaic_rule(*given, tiles=tiles, bound=1.0, rounded=True)[0]).trace(
            *operands).lower(lowering_platforms=("cuda",))


@lowers('sm80')
@pytest.mark.parametrize('target', ['sm80', 'sm86'])
def test_a_narrow_key_dimensions_gradient_lowers_its_copies_for_an_ampere_gpu(target):
    """A decay a key dimension 16 wide over values 8 wide (the recurrence stability sites' geometry), its
    floored gradient through the template lowered for an Ampere GPU: the key memory backward copies the
    entered memory's 8 rows of 64 bf16 columns in with cp.async, one 8-row tile at its rows' 128-byte
    swizzle, whose copy layout Mosaic GPU refuses ('Number of tiles is not a multiple of 4',
    stab-floorless-a100-deba5b564); staged at the swizzle the copy tiles it lowers."""
    rows, length, value_heads, width, value_width = 2, 70, 16, 16, 8
    operands = [jax.ShapeDtypeStruct((rows, length, 1, width), jnp.bfloat16)] * 2 + [
        jax.ShapeDtypeStruct((rows, length, value_heads, value_width), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, length, value_heads, width), jnp.float32),
        jax.ShapeDtypeStruct((rows, length, value_heads), jnp.bfloat16)]
    hooks = recurrence.Hooks(decay='vector', scale=width ** -0.5, floor=FLOOR, write_bound=1.0)

    def rule(*given):
        return chunk_on('mosaic_gpu', hooks, Chunk(*given[:3], log_decay=given[3], beta=given[4]))

    def loss(*given):
        out, final = rule(*given)
        return out.astype(jnp.float32).sum() + final.sum()

    # Abstract devices stating the GPU's capability, which Mosaic GPU lowers cp.async's waits for.
    mesh = generation.gpu_mesh(jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",)), target)
    with generation.assumed(target), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4))).trace(*operands).lower(lowering_platforms=("cuda",))


@pytest.mark.parametrize('dtype', [jnp.bfloat16, jnp.float32], ids=['bf16', 'fp32-accurate'])
def test_a_launch_states_the_part_pairs_it_issues_and_declares_the_models_products(dtype):
    """A gated delta rule's forward through the template, lowered for an H100: each launch
    states the products it issues, its operands' part pairs (`product.terms`: one for bf16, six for an fp32
    operand's three bf16 parts under the accurate forms), and declares the model's products apart
    (`lowered.KernelCall.model_flops`), which Dew's FLOP count reads for the model's work."""
    from dew.nn.kernels import lowered
    from dew.nn.kernels.templates import product
    from dew.nn.precision import ACCURATE

    rows, length, key_heads, value_heads, width = 1, 256, 2, 4, 64
    operands = [jax.ShapeDtypeStruct((rows, length, key_heads, width), dtype)] * 2 + [
        jax.ShapeDtypeStruct((rows, length, value_heads, width), dtype),
        jax.ShapeDtypeStruct((rows, length, value_heads), jnp.float32),
        jax.ShapeDtypeStruct((rows, length, value_heads), dtype)]

    rule = functools.partial(mosaic_rule, tiles=mainloop.Tiles(32, 1), bound=1.0)
    mesh = generation.gpu_mesh(jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), generation.run_precision(ACCURATE), jax.sharding.use_abstract_mesh(mesh):
        program = jax.jit(rule).trace(*operands).lower(lowering_platforms=("cuda",))
    launches = [call for call in lowered.kernel_calls(program) if call.kind == "mosaic_gpu"]
    assert launches and all(call.model_flops is not None for call in launches)
    ratios = {(call.flops or 0.0) / call.model_flops for call in launches if call.model_flops}
    if dtype == jnp.bfloat16:
        assert ratios == {1.0}
    else:
        # A launch over fp32 operands issues every part pair; one over the rule's bf16 gates, one.
        pairs = float(len(product.terms(mainloop.PARTS, mainloop.PARTS)))
        assert pairs in ratios and ratios <= {1.0, pairs}


def test_a_ranged_calls_inverse_states_the_fp32_parts_its_merges_issue():
    """A bf16 Kimi delta attention forward with no bound on its inverse, lowered for an H100: its chunk
    inverse's doubling merges in fp32 (`Call.ranged`), two bf16 parts on warpgroup products, which take no
    transposed TF32 (`mainloop.FAST_PARTS`), so the intra solve, all doubling, states their three leading
    pairs a model product, and no launch more; each stated the held bf16 operand's one (row 4032)."""
    from dew.nn.kernels import lowered
    from dew.nn.kernels.templates import product

    rows, length, key_heads, value_heads, width = 1, 256, 2, 4, 64
    operands = [jax.ShapeDtypeStruct((rows, length, key_heads, width), jnp.bfloat16)] * 2 + [
        jax.ShapeDtypeStruct((rows, length, value_heads, width), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, length, value_heads, width), jnp.float32),
        jax.ShapeDtypeStruct((rows, length, value_heads), jnp.bfloat16)]

    rule = functools.partial(mosaic_rule, tiles=mainloop.Tiles(32, 1), floor=FLOOR)
    mesh = generation.gpu_mesh(jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        program = jax.jit(rule).trace(*operands).lower(lowering_platforms=("cuda",))
    launches = [call for call in lowered.kernel_calls(program) if call.kind == "mosaic_gpu"]
    ratios = {(call.flops or 0.0) / call.model_flops for call in launches if call.model_flops}
    pairs = float(len(product.terms(mainloop.FAST_PARTS, mainloop.FAST_PARTS)))
    assert pairs in ratios and max(ratios) == pairs


def test_each_launch_lowers_once_however_many_layers_call_it():
    """Two layers' bf16 gated delta rules of one shape, their forward and gradient, lowered for the GPU the
    test runs on, or off one an H100: each of the template's distinct launches is one Mosaic GPU call in the
    lowered program, which both layers call (`mainloop.lowered_once`), as many as one layer's. Each call
    site traced, lowered and serialized its own kernel: a Qwen3.5-4B step's first update waited 30 minutes
    on its layers' identical kernels (row 2102)."""
    from dew.nn.kernels import lowered

    rows, length, key_heads, value_heads, width = 1, 256, 2, 4, 128
    operands = [jax.ShapeDtypeStruct((rows, length, key_heads, width), jnp.bfloat16)] * 2 + [
        jax.ShapeDtypeStruct((rows, length, value_heads, width), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, length, value_heads), jnp.float32),
        jax.ShapeDtypeStruct((rows, length, value_heads), jnp.bfloat16)]

    def layers(count: int):
        def run(query, key, value, g, beta):
            total = jnp.float32(0.0)
            for layer in range(count):
                # Each layer's own queries, as each layer projects its own.
                scaled = query * (layer + 1)
                out = mosaic_rule(scaled, key, value, g, beta, tiles=mainloop.Tiles(32, 1), bound=1.0)[0]
                total = total + out.astype(jnp.float32).sum()
            return total

        return jax.grad(run, argnums=(0, 1, 2))

    # The GPU the test runs on where its family takes the call, else an H100 (Mosaic GPU lowers no Hopper
    # copy for an A100's device).
    lowered_for = generation.device_generation() if mainloop.family_of() is not None else "sm90"

    def kernels(count: int) -> int:
        # The program's Mosaic GPU calls, each lowered function's once (`lowered.custom_calls`).
        with generation.assumed(lowered_for):
            program = jax.jit(layers(count)).trace(*operands).lower(lowering_platforms=("cuda",))
        return sum(call.target.startswith("mosaic_gpu") for call in lowered.custom_calls(program))

    one = kernels(1)
    assert one > 0 and kernels(2) == one


def diverging(width: int = 64) -> Chunk:
    """One 64-token head of keys `2 e0` under a write gate of one and no decay, its first value alone
    nonzero: each write's transition `I - beta k k^T` has the eigenvalue -3, the chunk's inverse entries
    `-4 (-3)^(d-1)` at lag `d`, past fp16's largest at lag 10, and the output grows to about 5.7e29."""
    tokens = 64
    keys = jnp.zeros((1, tokens, 1, width), jnp.bfloat16).at[..., 0].set(2.0)
    values = jnp.zeros((1, tokens, 1, width), jnp.bfloat16).at[:, 0, 0, 1].set(1.0)
    return Chunk(keys, keys, values, log_decay=jnp.zeros((1, tokens, 1), jnp.float32),
                 beta=jnp.ones((1, tokens, 1), jnp.bfloat16),
                 state=jnp.zeros((1, 1, width, width), jnp.float32))


def test_a_16bit_delta_rule_whose_form_proves_no_range_solves_at_fp32s_range():
    """The diverging rule (`diverging`), its keys drawn within 1e-3 of `2 e0`, in a 16-bit call whose form
    proves no bound on its chunk's inverse (unnormalized keys, no declared write gate): the template's solve,
    whose fp16 merges gave NaN there where the portable math gave 5.7e29 (row 2381), merges at fp32's range
    in bf16 parts (`mainloop._merged`): its output on par from float64 with the portable math's, the token
    recurrence the truth (check 1); and the delta rule's selector on an H100 gives the call the template."""
    from dew.nn.kernels.recurrence_chunk import port_kernel

    hooks = recurrence.Hooks(decay='scalar', scale=diverging().key.shape[-1] ** -0.5)

    def draw(seed):
        drawn = diverging()
        noise = 1e-3 * jax.random.normal(jax.random.key(59 + seed), drawn.key.shape)
        return ((drawn.key.astype(jnp.float32) + noise).astype(jnp.bfloat16),)

    def output(rule):
        def run(key):
            drawn = diverging()
            found = rule(Chunk(key, key, drawn.value.astype(key.dtype), log_decay=drawn.log_decay,
                               beta=drawn.beta.astype(key.dtype),
                               state=drawn.state.astype(jnp.promote_types(key.dtype, jnp.float32))))
            return found[0] if isinstance(found, tuple) else found
        return run

    assert_as_exact(output(functools.partial(chunk_on, 'mosaic_gpu', hooks)),
                    output(functools.partial(chunk_on, 'xla', hooks)), draw,
                    "the template's solve at fp32's range", seeds=SEEDS,
                    truth_fn=output(functools.partial(token_by_token, hooks)))
    drawn = diverging()
    operands = (drawn.query, drawn.key, drawn.value, drawn.log_decay, drawn.beta, drawn.state)
    with generation.assumed('sm90'):
        assert port_kernel(*operands, report=False) == 'mosaic_gpu'


NEAR_TWO_HOOKS = recurrence.Hooks(decay='scalar', scale=64 ** -0.5, write_bound=2.0)
"""The gated delta rule under write gates near two (twice a sigmoid's) on keys 64 wide, no decay."""


def near_two(seed: int) -> Chunk:
    """Unit keys within 1e-3 of one direction over 64 tokens under write gates near two, as queries too, bf16;
    no decay, no memory entering."""
    tokens, width = 64, 64
    keys = jax.random.split(jax.random.key(53 + seed), 4)
    base = jax.random.normal(keys[0], (1, 1, 1, width))
    key = base + 1e-3 * jax.random.normal(keys[1], (1, tokens, 1, width))
    key = (key / jnp.linalg.norm(key, axis=-1, keepdims=True)).astype(jnp.bfloat16)
    value = jax.random.normal(keys[2], (1, tokens, 1, width)).astype(jnp.bfloat16)
    beta = (2.0 * jax.nn.sigmoid(6.0 + jax.random.normal(keys[3], (1, tokens, 1)))).astype(jnp.bfloat16)
    return Chunk(key, key, value, log_decay=jnp.zeros((1, tokens, 1), jnp.float32), beta=beta,
                 state=jnp.zeros((1, 1, width, width), jnp.float32))


def test_the_solve_holds_a_negative_eigenvalue_rules_inverse():
    """Unit keys within 1e-3 of one direction under write gates near two (twice a sigmoid's, the
    negative-eigenvalue delta rule's), declared (`write_bound`): each transition's eigenvalue near -1, the
    chunk's inverse alternating in sign at entries up to two, within fp16's range (`recurrence.Hooks.
    inverse_bound`). The template's 16-bit call on par from float64 with recorded fla's (`fla_sites`), the
    token recurrence the truth (check 1): no transition damps
    them, and the fp32 portable math stays 14 times nearer."""
    the_template_is_as_exact_as_fla(functools.partial(chunk_on, 'mosaic_gpu', NEAR_TWO_HOOKS), 'near two',
                                    ('out',), "the template's solve under write gates near two")


def test_a_gated_delta_net_named_onto_the_template_matches_the_portable_rule():
    """A layer naming the template (`generation.chosen(gated_delta_rule='mosaic_gpu')`): its output on par
    from float64 with the portable rule's (check 1, the layer's own float64 forward the truth). The layer
    computes in bf16, the call the template's launches hold on every family (an fp32 call's solve holds
    more registers than a thread has)."""
    from dew.nn.linear import GatedDeltaNet

    layer = GatedDeltaNet(emb_features=64, num_k_heads=1, num_v_heads=2, head_k_dim=64, head_v_dim=64,
                          dtype=jnp.bfloat16)

    def draw(seed):
        x = jax.random.normal(jax.random.key(23 + seed), (1, 72, 64)).astype(jnp.bfloat16)
        with generation.chosen(gated_delta_rule='xla'):
            return layer.init(jax.random.key(29 + seed), x), x

    def applied(kernel):
        def run(params, x):
            with generation.chosen(gated_delta_rule=kernel):
                return layer.apply(params, x)
        return run

    def truth(params, x):
        return layer.clone(dtype=jnp.float64).apply(params, x)

    assert_as_exact(applied('mosaic_gpu'), applied('xla'), draw, 'a layer named onto the template',
                    seeds=SEEDS, truth_fn=truth)


def test_an_fp32_operand_splits_into_finite_parts_summing_to_it():
    """An fp32 operand's three bf16 parts (`product.formed`) are finite and sum to it exactly, at fp32's
    largest finite values too, past bf16's: its first part saturates there, where rounding would make it
    infinite and its remainder NaN."""
    largest = float(jnp.finfo(jnp.float32).max)
    x = jnp.array([largest, -largest, 1.0 + 2.0 ** -20, -3.0e-30, 0.0], jnp.float32)
    parts = product.formed(x, mainloop.PARTS)
    assert all(bool(jnp.all(jnp.isfinite(part))) for part in parts)
    total = sum(np.asarray(part, np.float64) for part in parts)
    np.testing.assert_array_equal(total, np.asarray(x, np.float64))


TAKEN = HOOKS + FORMED
"""The chunked op's hook combinations, each run where the mainloop takes it and checked refused where not.
Their draws are fp32, whose delta-rule and per-key calls the template refuses (an fp32 solve's registers, a
per-key decay's factoring): the cases that run check 1 are the fp32 calls without the delta rule, a decay a
head or none, natively (template numerics review 2026-10-26, P2-3: the other forms, the delta rule without
a decay, the gated linear attentions, the rated scan in bf16 and the formed hooks, hold no float64
evidence on the template; their reference arm in a 16-bit call is open). Under the run's own forms: a call
without the delta rule takes three parts under either (`recurrence_chunk.chunk_rounding`)."""


def every_hook_the_mosaic_mainloop_takes_is_as_exact_as_the_recurrence_token_by_token(hooks, packed):
    """The chunked op on the mainloop under the run's own forms (the fast forms), each hook combination it
    takes, its output and final memory on par with the recurrence a token at a time's from float64 over the
    site's declared draws (check 1), as test_recurrence_chunk holds the portable math; a call the template
    refuses (a launch's registers past a thread's) is checked refused, its arithmetic the forward check's
    (`on_the_mainloop`). The draws are fp32; a call without the delta rule, whose bar is the portable math,
    takes its products in three bf16 parts, an fp32 product's accuracy, under the fast forms too
    (`recurrence_chunk.chunk_rounding`): in two (`mainloop.FAST_PARTS`, the delta rule's, held to recorded
    fla-core, `the_mosaic_forward_is_as_exact_as_its_reference`) this check put the template 42 to 83 times
    the fp32 recurrence's distance from float64, interpreted and on an H100 alike (flow-cells-cpu,
    gpuflow-a), and row 184's write gate cell 30 to 69 times the portable math's on every leaf."""
    (sample,) = hooked_draw(hooks, packed, 0)
    if not taken(sample, hooks):
        with pytest.raises(REFUSED):
            jax.eval_shape(functools.partial(chunk_on, 'mosaic_gpu', hooks), sample)
        return
    dew = jax.jit(functools.partial(chunk_on, 'mosaic_gpu', hooks))
    reference = jax.jit(functools.partial(token_by_token, hooks))
    assert_as_exact(dew, reference, functools.partial(hooked_draw, hooks, packed),
                    f'the mosaic forward, {hooks} packed={packed}', seeds=HOOK_SEEDS[hooks, packed])


HOOK_CELLS = [(hooks, packed) for hooks in TAKEN for packed in (False, True)]
"""The mainloop's hook cells, each hook combination over one sequence and packed documents."""

LONG_HOOK_CELLS = {(FORMED[1], True), (FORMED[4], True), (FORMED[5], True)}
"""The hook cells past two minutes interpreted, each a check-1 site of its own in map tasks (`check1_shards`):
the step write gate's packed documents (128 declared seeds, 384 s whole) and the rated scan's (64, 172 s),
hooks-accurate at 136c07a5a, and the rated scan's with its output norm (167.6 s in train 630's CI, main
f424ab7d8)."""


@pytest.mark.parametrize(('hooks', 'packed'), [cell for cell in HOOK_CELLS if cell not in LONG_HOOK_CELLS])
def test_every_hook_the_mosaic_mainloop_takes_is_as_exact_as_the_recurrence_interpreted(hooks, packed):
    """`every_hook_the_mosaic_mainloop_takes_is_as_exact_as_the_recurrence_token_by_token` interpreted off a
    GPU, check 1 from float64 at each cell's declared seeds: the arithmetic the GPU flow alone judged, which
    the CPU suite took no accuracy check of (its fast-form numbers 42 to 83 times the fp32 recurrence's
    distance reached main unseen, row 3962)."""
    every_hook_the_mosaic_mainloop_takes_is_as_exact_as_the_recurrence_token_by_token(hooks, packed)


@pytest.mark.check1_shards(seeds=8, per_task=4)
def test_a_bf16_gated_delta_nets_training_step_with_its_norm_is_on_par_interpreted():
    """Row 184's gated delta net cell (tests/recurrence_matrix.py: the template, bf16, one sequence)
    interpreted off a GPU, its reference the portable math there (fla's Triton kernels run on a GPU alone):
    its output, final memory and every operand's gradient on par from float64 (check 1). The chunked op's l2
    norm rounds the normalized queries and keys to the call's own dtype; kept fp32, the call reached the
    template as an fp32 one, its final memory 30 to 41 times the portable math's distance in two parts
    (acb1301632c), 29 times fla's on a GH200 (M-JS-9)."""
    import recurrence_matrix

    recurrence_matrix._on_par("recurrence", ('sm90', 'gated delta net', 'mosaic_gpu', 'bfloat16', 'one'))


@pytest.mark.check1_shards(seeds=128, per_task=16)
def test_the_step_gated_hook_over_packed_documents_is_as_exact_as_the_recurrence_interpreted():
    """`every_hook_the_mosaic_mainloop_takes_is_as_exact_as_the_recurrence_token_by_token` at the step write
    gate over packed documents (`LONG_HOOK_CELLS`), sixteen draws a task: 32 took 77 to 122 s a task
    (jsf-slow7 at 55dbf3627)."""
    every_hook_the_mosaic_mainloop_takes_is_as_exact_as_the_recurrence_token_by_token(FORMED[1], packed=True)


@pytest.mark.parametrize('hooks', [FORMED[4], FORMED[5]], ids=['scan', 'with its output norm'])
@pytest.mark.check1_shards(seeds=64, per_task=16)
def test_the_rated_hook_over_packed_documents_is_as_exact_as_the_recurrence_interpreted(hooks):
    """`every_hook_the_mosaic_mainloop_takes_is_as_exact_as_the_recurrence_token_by_token` at the rated scan,
    and with its output norm, over packed documents (`LONG_HOOK_CELLS`), sixteen draws a task: 32 took 192 to
    235 s a task for the two (jsf-slow5 at c8f5a12e7)."""
    every_hook_the_mosaic_mainloop_takes_is_as_exact_as_the_recurrence_token_by_token(hooks, packed=True)


RESOURCE_SHAPES = {'qwen3.5-27b-gdn': (16, 48, 128, 128, False), 'kimi-linear-kda': (32, 32, 128, 128, True),
                   'kimi-linear-kda, no floor': (32, 32, 128, 128, True),
                   'nemotron-h-4b-ssd': (8, 96, 128, 80, False),
                   'a 192-wide value head': (16, 16, 128, 192, False)}
"""The scoreboard's geometries the resource gate compiles (`tools/benchmark_recurrence.SHAPES`), and a gated
delta rule's value head 192 wide, whose training launches take a value block of whole staged blocks (a
bf16 output gradient's 64-column swizzle; 96 columns cut one): key heads, value heads, key and value widths,
and whether the decay is a key dimension's."""

FLOORLESS = frozenset({'kimi-linear-kda, no floor'})
"""The gate's geometries under a decay a key dimension that declare no floor: each sub-chunk's own pairs
each their own decay (`algebra.key_diagonal_pairs`), the forward's staging and the backward's beside them."""


def resource_floor(name: str) -> float | None:
    """`name`'s declared floor: `FLOOR` under a decay a key dimension, none where it declares none
    (`FLOORLESS`) or the decay is a head's."""
    return FLOOR if RESOURCE_SHAPES[name][4] and name not in FLOORLESS else None

RATED = frozenset({'nemotron-h-4b-ssd'})
"""The gate's rated scans (Mamba2's SSD): no delta rule, its 16-bit call `rounded` (the fast forms), its
launches the keys-gated ones (`mainloop.Call.gated_keys`)."""


def resource_operands(name: str, dtype, rows: int, length: int) -> list[jax.ShapeDtypeStruct]:
    """`(q, k, v, g, beta, h0)` of `RESOURCE_SHAPES[name]`'s geometry over `rows` of `length` tokens, in
    `dtype` (the decay and the initial memory fp32), as `mainloop.forward` takes them."""
    key_heads, value_heads, key_width, value_width, per_key = RESOURCE_SHAPES[name]

    def array(*shape, kind=dtype):
        return jax.ShapeDtypeStruct(shape, kind)

    decay = (rows, length, value_heads) + ((key_width,) if per_key else ())
    return [array(rows, length, key_heads, key_width), array(rows, length, key_heads, key_width),
            array(rows, length, value_heads, value_width), array(*decay, kind=jnp.float32),
            array(rows, length, value_heads), array(rows, value_heads, key_width, value_width,
                                                    kind=jnp.float32)]


@lowers('sm90')
def test_the_resource_gate_reads_each_launch_of_one_kernel_apart():
    """The 27B gated delta net's forward launched twice in one program on an H100, at its derived tiles and
    at a variant: a kernel's name compiled under both launches (`report.launching`) is gated on each
    launch's evidence, the variant's shared memory its own; a gate that read a kernel by its name alone kept
    one launch's and dropped the other's."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels

    target = 'sm90'
    family, facts = bundles.of(target, jnp.bfloat16), generation.gpu_target(target)
    args = resource_operands('qwen3.5-27b-gdn', jnp.bfloat16, 1, 256)
    bound = template.MOSAIC.bind(*args, None, max_segments=None, floor=None, bound=1.0)
    with generation.assumed(facts):
        derived, variant = template.MOSAIC.variants(bound)[:2]

    def call():
        def fn(*operands):
            launched = []
            for tiles in (derived, variant):
                with report.launching(template.MOSAIC, bound, tiles):
                    launched.append(mainloop.forward(*operands, tiles=tiles, starts=None, max_segments=None,
                                                     family=family))
            return launched

        return fn, tuple(args)

    (record,) = compile_mosaic_kernels([KernelCase('two launches', call)], target)
    launches: dict[str, list] = {}
    for kernel in record['kernels']:
        assert not kernel['unread'], f"{kernel['kernel']}: {kernel['unread']}"
        launches.setdefault(kernel['kernel'], []).append((kernel['launch'], kernel['shared']))
    assert launches
    assert all(len({launch for launch, _ in found}) == 2 for found in launches.values()), launches
    assert any(len({shared for _, shared in found}) == 2 for found in launches.values()), launches


@lowers('sm90')
def test_two_launches_whose_padded_lengths_meet_name_their_own_kernels():
    """The 27B gated delta net's forward over 4089 tokens and over 4096 in one program on an H100, each at its
    derived tiles under its own launch (`report.launching`): the mainloop pads both lengths to the same
    chunks, so its launches' statics meet, and each launch's kernels still carry its own launch's digest
    (`tuning.lowered_once` keys a trace on the launch tracing it). They shared one trace and the second's
    kernels were named for the first, a spill of one refused as the other (row 2941)."""
    from dew.nn.kernels.lowered import kernel_calls

    target = 'sm90'
    family, facts = bundles.of(target, jnp.bfloat16), generation.gpu_target(target)
    lengths = (4089, 4096)
    operands = {length: resource_operands('qwen3.5-27b-gdn', jnp.bfloat16, 1, length) for length in lengths}
    with generation.assumed(facts):
        launches = {length: (bound := template.MOSAIC.bind(*operands[length], None, max_segments=None,
                                                           floor=None, bound=1.0),
                             template.MOSAIC.variants(bound)[0])
                    for length in lengths}

    def both(*flat):
        outputs, at = [], 0
        for length in lengths:
            bound, tiles = launches[length]
            with report.launching(template.MOSAIC, bound, tiles):
                outputs.append(mainloop.forward(*flat[at:at + 6], tiles=tiles, starts=None, max_segments=None,
                                                family=family))
            at += 6
        return outputs

    with generation.assumed(facts):
        lowered = jax.jit(both).trace(*operands[4089], *operands[4096]).lower(lowering_platforms=('cuda',))
    named: dict[str, set] = {}
    for call in kernel_calls(lowered):
        if call.kind == 'mosaic_gpu':
            named.setdefault(call.name, set()).add(call.launch)
    assert named and None not in set().union(*named.values())
    assert all(len(found) == 2 for found in named.values()), named


@pytest.mark.parametrize('kernel', ['recurrence_scan', 'recurrence_states', 'recurrence_reverse'],
                         ids=['forward', 'training forward', 'backward'])
def test_a_compile_refusing_the_joined_tiles_steps_the_derivation_down(kernel):
    """A compile that refused the 27B gated delta net's derived tiles on an H100 (`report.spilled`: a kernel
    spilled), its forward's walk (`mainloop._scan`), its training forward's or its backward's: the derivation
    steps the refused tiles down, which the call takes (`fits`); a refusal of the joined tiles once matched
    none of the rules' candidates, and the call found none (`Unfitted`)."""
    args = resource_operands('qwen3.5-27b-gdn', jnp.bfloat16, 1, 4096)
    bound = template.MOSAIC.bind(*args, None, bound=1.0)
    with report.refusals(), generation.assumed('sm90'):
        target = template.Mosaic.launched()
        first = template.MOSAIC.derived(bound, target)
        report.refuse([(report._identity(template.MOSAIC, bound), first, {kernel})])
        stepped = template.MOSAIC.derived(bound, target)
        taken = template.MOSAIC.fits(bound, target)
    assert stepped is not None and taken == (stepped,)
    # The refused tiles' block or ring stepped down (`_staged` takes the next ring a refusal leaves).
    assert (stepped.block_v, stepped.stages) < (first.block_v, first.stages)


def test_a_refusal_under_the_remat_rung_reaches_the_route():
    """Under the remat rung recomputing the delta rule's memory (`recurrence_chunk.recomputing`), a refusal of
    every tile of the run's training call's launch (its identity as its lowering for an H100 names it,
    `report.traced`) reaches the route: a call naming the template is refused first for its launch (code
    'launch'; its backward with it) where it was taken before. The route's bind is the run's launch, the
    rung's flag resolved alike (`recurrence_chunk.recomputed_memory`); bound as kept, it judged another
    launch than the run's (row 4133)."""
    from dew.nn.kernels import recurrence_chunk
    from dew.nn.kernels.lowered import kernel_calls
    from dew.nn.kernels.selection import KernelRefused

    operands = [jnp.zeros(shape, dtype) for shape, dtype in (
        ((1, 64, 1, 64), jnp.bfloat16), ((1, 64, 1, 64), jnp.bfloat16), ((1, 64, 2, 64), jnp.bfloat16),
        ((1, 64, 2), jnp.float32), ((1, 64, 2), jnp.bfloat16))]
    # The run's call as the chunked op's route makes it (`recurrence_chunk.chunk`'s options).
    options = {'max_segments': None, 'scale': None, 'delta': True, 'wide': False, 'floor': None, 'bound': 1.0,
               'rounded': True}

    def loss(*given):
        out, final = recurrence_chunk.template_rule(*given, **options)
        return out.astype(jnp.float32).sum() + final.sum()

    def named() -> str:
        return recurrence_chunk.port_kernel(*operands, bound=1.0, product='training', report=False)

    with (report.refusals(), generation.assumed('sm90'), recurrence_chunk.recomputing(('delta_memory',)),
          generation.chosen(gated_delta_rule='mosaic_gpu')):
        target = template.Mosaic.launched()
        named()
        training = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
        lowered = training.trace(*operands).lower(lowering_platforms=('cuda',))
        (identity,) = {report.traced(call.launch).identity for call in kernel_calls(lowered)
                       if call.kind == 'mosaic_gpu' and call.launch is not None}
        # The run's launch bound as the run binds it, its identity the lowering's.
        bound = template.MOSAIC.bind(*operands, None, None, firsts=None, skip=None, recomputed=True,
                                     **options)
        assert report._identity(template.MOSAIC, bound) == identity
        while (tiles := template.MOSAIC.derived(bound, target)) is not None:
            names = {launch.name for launch in template.MOSAIC._launches(bound, tiles, target, backward=True)}
            report.refuse([(identity, tiles, names)])
        with pytest.raises(KernelRefused) as refused:
            named()
        assert refused.value.refusals[0].code == 'launch'


def _model(launch: mainloop.Launch, family: mosaic.Family) -> resources.RegisterModel:
    """A launch's register model as the compile gate decides on it (`resources.register_model`,
    `kernel_compile.decisions`): its stated registers (`Launch.registers`) at its consumer roles
    (`Launch.consumers`, a walk's `Tiles.walkers`), by the bundle it issues by (`Launch.family`; the solve
    launch's warpgroup MMA layout without a producer), `family` where it names none."""
    return resources.register_model(launch.family or family, launch.consumers, launch.registers)


class Unbudgeted(template.Mosaic):
    """The recurrence template with every launch's registers held: its derivation takes the tile it would
    where only registers refuse one (`compiled_cases`' refused tiles), a states launch no multiprocessor
    holds at its registers (`Launch.resident` zero) one program a multiprocessor, as it would hold were its
    registers held."""

    def _held(self, ba, config) -> bool:
        del ba, config
        return True

    def states_held(self, ba, tiles, target, *, training: bool = False) -> int:
        return max(super().states_held(ba, tiles, target, training=training), 1)


NEIGHBOURS = 3
"""The most variants a call's derived tiles have beside them (`Mosaic.neighbours`: the value block either
side of the derived one and its stages' other depth), the resource gate's variant groups (`GATE_GROUPS`)."""


TWO_ROLES = 'two roles'
"""A training group's variant: its walks at two consumer roles a program (`Tiles.walkers`), compiled apart
from its derived tiles."""

GATE_GROUPS = [(name, dtype, training, packed, variant) for name in RESOURCE_SHAPES
               for dtype in (jnp.bfloat16, jnp.float32) for packed in (False, True)
               for training in (False, True) for variant in (None, TWO_ROLES, *range(NEIGHBOURS))
               if not (RESOURCE_SHAPES[name][4] and dtype == jnp.float32)
               and (name != 'a 192-wide value head' or (training and dtype == jnp.bfloat16 and not packed))
               and not (training and (name, dtype) not in {('qwen3.5-27b-gdn', jnp.bfloat16),
                                                           ('qwen3.5-27b-gdn', jnp.float32),
                                                           ('kimi-linear-kda', jnp.bfloat16),
                                                           ('kimi-linear-kda, no floor', jnp.bfloat16),
                                                           ('nemotron-h-4b-ssd', jnp.bfloat16),
                                                           ('a 192-wide value head', jnp.bfloat16)})
               and not (variant == TWO_ROLES and not (training and not packed))
               and not (isinstance(variant, int) and (training or packed or dtype == jnp.float32))]
"""The resource gate's groups (`compiled_cases`' `only`), a test each so each compiles in its own group: every
geometry's forward in a 16-bit and an fp32 call, one row and packed documents, and the training pass of the
gated delta rule in both and of Kimi delta attention and the SSD in 16 bits (Kimi's keys, 128 wide, past what
an fp32 call takes under a decay a key dimension), and the 192-wide value head's 16-bit training pass alone;
one row's 16-bit forward at its derived tiles apart from
each of its variants (`variant`, a test each: an fp32 forward, which every family refuses, has none), and one
row's training pass at its derived tiles apart from its walks at two roles (`TWO_ROLES`). One test
compiling them all took 674 seconds on sm90, a group with its variants 132 on sm80, and Kimi Linear's three
variants 117 on sm80 at 9e65ee222's gate, each about the forward's 89 (armada); the 27B's fp32 training pass
with its two-role walks 130.2 s on sm90 (train 630's CI, main f424ab7d8)."""



def _form(name: str, dtype) -> dict:
    """A resource case's call form beside its operands under the run's fast forms (`rounded`, `fast_rounding`,
    an fp32 call's operands in two parts), as its route gives it: a rated scan's without the delta rule and
    wide for its skip."""
    return {'delta': False, 'wide': True, 'rounded': True} if name in RATED else {'rounded': True}


def _skipped(name: str, dtype) -> dict:
    """The skip a rated scan's 16-bit call fuses into the template's output and reverse walk under the run's
    fast forms (`recurrence_chunk._rated_scan`), its value heads' `[HV]`, as its route hands it."""
    if name not in RATED or jnp.dtype(dtype).itemsize != 2:
        return {}
    return {'skip': jax.ShapeDtypeStruct((RESOURCE_SHAPES[name][1],), jnp.float32)}


def _two_roles(name: str, bound, derived, family, facts, *, packed: bool) -> list:
    """The walks at two consumer roles a program (`Tiles.walkers`) as a gate case's tiles, which the
    derivation gives a call whose walks fill more than a wave and `RESOURCE_SHAPES` never do: a role's
    registers are its share of the program's; none where the family's walks take one role
    (`template.walk_roles`), packed documents', a value width of an odd number of blocks, or two roles' tiles
    past the GPU's resources (`Mosaic._staged` None)."""
    value_width = RESOURCE_SHAPES[name][3]
    if template.walk_roles(family) < 2 or packed or value_width // derived.block_v % 2:
        return []
    two = template.MOSAIC._staged(bound, derived.block_v, facts, walkers=2,
                                  wide=derived.block_o or derived.block_v)
    return [] if two is None else [(two, 'two roles ')]


def compiled_cases(target: str, plans: dict, refused: list, apart: dict, only: tuple | None = None) -> list:
    """The resource gate's cases on `target` (`KernelCase`s), each with its launches' register models (row
    1971): each `RESOURCE_SHAPES` geometry's forward in a 16-bit and an fp32 call, one row and packed
    documents, and the gated delta rule's 16-bit training pass, at the template's derived tiles and, one row's
    forward, at each of their variants; beside them the tiles the derivation takes where every register held
    (`Template._held`) and refused for registers alone, and the fast backward its launches' registers refuse,
    so a model over ptxas's need fails as one under it does. Each case's plan by its label into `plans`
    (`template.planned`), each call the template refuses into `refused`, and the launches a case's
    model refuses by its label into `apart` where they are held apart (`apart_failures`)."""
    from fla_gdn_support import KernelCase

    family, facts = bundles.of(target, jnp.bfloat16), generation.gpu_target(target)
    rows, length = 1, 256
    shared = generation.shared_budget(facts)

    def cases(name: str, dtype, *, training: bool, packed: bool, variant: int | None) -> list:
        """The case's kernels at its derived tiles, or, one row's forward, at its `variant`-th other variant
        (`Mosaic.variants`)."""
        args = resource_operands(name, dtype, rows, length)
        starts = jax.ShapeDtypeStruct((rows, length), jnp.bool_) if packed else None
        # A delta rule's layer declares unit keys and sigmoid write gates (`recurrence.Hooks.inverse_bound`).
        bound = template.MOSAIC.bind(*args, starts, max_segments=8 if packed else None,
                                     floor=resource_floor(name), bound=None if name in RATED else 1.0,
                                     **_form(name, dtype), **_skipped(name, dtype))
        with generation.assumed(facts):
            derived = template.MOSAIC.derived(bound, facts)
            held = Unbudgeted().derived(bound, facts)
            # The tile the derivation takes where every register held, refused for its registers alone.
            unheld = [] if held is None or held == derived else [(held, 'refused ')]
            if derived is None and variant is not None:
                return []
            if derived is None:
                # Refused: a launch's registers or shared memory past the device's at every tile.
                refused.append(f"{name} {jnp.dtype(dtype).name}{' training' if training else ''}")
                assert not template.MOSAIC.eligible(bound, facts)
                return [case(name, dtype, args, starts, bound, tiles, training=training, packed=packed,
                             kind=kind) for tiles, kind in unheld]
            tiled = template.MOSAIC.variants(bound)
            if isinstance(variant, int):
                assert len(tiled) - 1 <= NEIGHBOURS, f"{name} has {len(tiled) - 1} variants, past NEIGHBOURS"
                return [case(name, dtype, args, starts, bound, tiles, training=training, packed=packed,
                             kind='') for tiles in tiled[1:][variant:variant + 1]]
            unwide = {key: value for key, value in _form(name, dtype).items() if key != 'wide'}
            if training and not template.backward_holds(
                    *args, tiles=derived, family=family, target=facts, starts=starts,
                    max_segments=8 if packed else None, floor=resource_floor(name), **unwide,
                    **_skipped(name, dtype)):
                # The rule's gradient is the port's or the portable math's where the fast backward's launches
                # do not hold (`template.backward_holds`); where only their registers refuse them,
                # the gate compiles them, held to ptxas's need.
                refused.append(f"{name} {jnp.dtype(dtype).name} fast backward{' packed' if packed else ''}")
                launches = template.MOSAIC._launches(bound, derived, facts, backward=True)
                if any(launch.shared > shared for launch in launches):
                    return []
                return [case(name, dtype, args, starts, bound, derived, training=True, packed=packed,
                             kind='refused ')]
        assert tiled[0] == derived, f"{name} on {target}: variants {tiled} lead with no {derived}"
        derived_tiles = [(derived, '')]
        walked = _two_roles(name, bound, derived, family, facts, packed=packed)
        if variant == TWO_ROLES:
            return [case(name, dtype, args, starts, bound, tiles, training=training, packed=packed, kind=kind)
                    for tiles, kind in walked]
        # A training pass's two-role walks a group of their own (`TWO_ROLES`).
        walked = [] if training else walked
        # A forward without a gradient at its own tiles (`Tiles.forward`), which no fast backward narrows.
        alone = [] if training or derived.forward is None else [(derived.forward, 'alone ')]
        return [case(name, dtype, args, starts, bound, tiles, training=training, packed=packed, kind=kind)
                for tiles, kind in derived_tiles + walked + alone + ([] if training else unheld)]

    def case(name: str, dtype, args, starts, bound, tiles, *, training: bool, packed: bool, kind: str):
        _, value_heads, key_width, value_width, _ = RESOURCE_SHAPES[name]

        def array(*shape, kind=dtype):
            return jax.ShapeDtypeStruct(shape, kind)

        label = (f"{name} {jnp.dtype(dtype).name} {'training' if training else 'forward'}"
                 f"{' packed' if packed else ''} {kind}{tiles}")
        plans[label] = template.MOSAIC._launches(bound, tiles, facts, backward=training)
        unwide = {key: value for key, value in _form(name, dtype).items() if key != 'wide'}
        with generation.assumed(facts):
            held = not training or kind != 'two roles ' or template.backward_holds(
                *args, tiles=tiles, family=family, target=facts, starts=starts,
                floor=resource_floor(name), **unwide, **_skipped(name, dtype))
        if not held:
            # A gradient at two roles its fast backward's launches do not hold is no run's (`backward_holds`,
            # as at the derived tiles): the launches its model refuses held apart (`apart_failures`).
            apart[label] = {launch.name for launch in plans[label]
                            if launch.registers > _model(launch, family).budget}

        def call():
            given = list(args)
            if training:
                given += [array(rows, length, value_heads, value_width),
                          array(rows, value_heads, key_width, value_width, kind=jnp.float32)]
            formed = {key: value for key, value in _form(name, dtype).items()
                      if key != 'wide' or not training}
            run = functools.partial(mainloop_backward.gradients if training else mainloop.forward, **formed,
                                    floor=resource_floor(name))
            skip = list(_skipped(name, dtype).values())

            def fn(*operands):
                *operands, documents = operands if packed else (*operands, None)
                *operands, fused = (*operands, None) if not skip else operands
                return run(*operands, tiles=tiles, starts=documents, max_segments=8 if packed else None,
                           family=family, skip=fused)

            return fn, (*given, *skip, *([starts] if packed else []))

        # A refused tile compiled for the model's evidence alone, which no call runs (`KernelCase.required`).
        return KernelCase(label, call, {launch.name: _model(launch, family) for launch in plans[label]},
                          required=not kind.startswith('refused'))

    return [kernel for group in (GATE_GROUPS if only is None else [only])
            for kernel in cases(group[0], group[1], training=group[2], packed=group[3], variant=group[4])]


def planned_failures(found: list, plans: dict) -> list[str]:
    """Each compiled kernel past its plan's shared memory (`template.planned`'s launch of its name,
    from which the tiles are derived): the recurrence's shared-memory model, beside the register model the
    shared gate holds to ptxas (`kernel_compile.mosaic_failures`)."""
    def planned(record: dict, kernel: dict) -> int:
        # A kernel's launches the longest names that begin its own: `recurrence_intra_solve` is no
        # `recurrence_intra`.
        named = [launch for launch in plans[record['name']] if kernel['kernel'].startswith(launch.name)]
        longest = max((len(launch.name) for launch in named), default=0)
        return max((launch.shared for launch in named if len(launch.name) == longest), default=0)

    return [f"{record['name']} {kernel['kernel']}: {kernel['shared']} shared bytes past its plan's "
            f"{planned(record, kernel)}" for record in found for kernel in record.get('kernels', ())
            if kernel['shared'] is not None and not kernel['shared'] <= planned(record, kernel)]


def apart_failures(found: list, apart: dict) -> list[str]:
    """Each compiled case's launches held apart (`apart`, the names its model refuses by its label): every
    launch its model admits as a required case's (`kernel_compile.mosaic_failures`: spill-free, every report
    read), and every one it refuses free to spill, its model's decision beside ptxas's in the record's
    register evidence, advisory where ptxas holds it spill-free (as `mosaic_failures` keeps a spill-free
    disagreement)."""
    from kernel_compile import _register_evidence, mosaic_failures

    failed = []
    for record in found:
        refused = {kernel["kernel"] for kernel in record.get("kernels", ()) for name in apart[record["name"]]
                   if kernel["kernel"] == name or kernel["kernel"].startswith(f"{name}__")}
        admitted = [kernel for kernel in record.get("kernels", ()) if kernel["kernel"] not in refused]
        failed += mosaic_failures([{**record, "kernels": admitted}])
        record["evidence"] = _register_evidence(record)
    return failed


def test_a_kernel_is_held_to_its_own_launchs_plan_where_another_launchs_name_begins_its_own():
    """`planned_failures` holds `recurrence_intra_solve` to its own launch's shared memory, not to the larger
    plan of `recurrence_intra`, whose name begins its own, and `recurrence_intra` to its own."""
    import types

    plans = {"case": [types.SimpleNamespace(name="recurrence_intra", shared=200_000),
                      types.SimpleNamespace(name="recurrence_intra_solve", shared=65_568)]}
    found = [{"name": "case", "kernels": [{"kernel": "recurrence_intra_solve", "shared": 100_000},
                                          {"kernel": "recurrence_intra", "shared": 150_000}]}]
    failed = planned_failures(found, plans)
    assert [failure.split(":")[0] for failure in failed] == ["case recurrence_intra_solve"]


def _gated(target: str, group: tuple):
    """The gate's test of `target` and `group`, named by them."""
    name, dtype, training, packed, variant = group
    label = (f"{target}-{name}-{jnp.dtype(dtype).name}-{'training' if training else 'forward'}"
             f"{'-packed' if packed else ''}{'' if variant is None else f'-variant-{variant}'}")
    return pytest.param(target, group, id=label)


@pytest.mark.parametrize('target, group', [_gated(target, group) for target in ('sm80', 'sm90')
                                           for group in GATE_GROUPS])
def test_the_mosaic_mainloop_compiles_within_each_gpus_resources_as_its_model_decides(target, group):
    """The mainloop's launches (`compiled_cases`), compiled by Mosaic for a compile-only A100 and H100 on this
    CPU (tests/kernel_compile.py): every launch compiles within the multiprocessor's shared memory (163 KB,
    227 KB) and tensor memory and at most its plan's shared memory (`template.planned`, from which
    the tiles are derived, `planned_failures`), no launch the template's legality holds spills a register (a
    refused tile's compiled as the model's evidence, `KernelCase.required`), and each launch's register model
    decides as ptxas's need does, admitting a tile or refusing it (row 1971,
    `kernel_compile.mosaic_failures`). Lowering alone takes no device's budget: a launch past an A100's shared
    memory lowered and failed on the GPU (the fast backward's readout's third staging tile). The gated delta
    rule's 16-bit call, the production cell, derives tiles on every family. A GPU host's ptxas builds for its
    own GPU and those before it, not a newer one's (an A100's refuses sm_90a), so there the newer targets are
    the CPU suite's alone. A test a group of cases (`GATE_GROUPS`)."""
    if jax.default_backend() == 'gpu':
        local = generation.capability(generation.device_generation())
        if local is not None and local < generation.capability(target):
            pytest.skip(f"this host's ptxas builds for its {local} GPU and before, not {target}")
    from kernel_compile import compile_mosaic_kernels, failures, mosaic_failures

    plans, refused, apart = {}, [], {}
    cases = compiled_cases(target, plans, refused, apart, only=group)
    found = compile_mosaic_kernels(cases, target)
    # Every case compiles and reports; a refused tile's spill is the model's refusal, a required one's fails,
    # and a case whose launches are held apart fails as either does, launch by launch.
    failed = failures(found) + planned_failures(found, plans)
    failed += mosaic_failures([record for case, record in zip(cases, found, strict=True)
                               if case.required and case.name not in apart])
    failed += apart_failures([record for record in found if record["name"] in apart], apart)
    failed += [f"{case}: no tiles derived" for case in refused
               if case.startswith("qwen3.5-27b-gdn bfloat16") and "fast backward" not in case]
    assert not failed, "\n".join(failed)


def test_a_gated_delta_nets_ring_and_residency_are_one_choice():
    """The 27B gated delta net at one row of 4,096 tokens on an H100 (sm90 stated) derives its value blocks 32
    wide in a ring more than one deep, its walk one program a multiprocessor: its plans priced by the bytes
    they hold in flight (`resources.hidden`), where the most programs first took two one deep and then
    narrowed the block to 16, the forward 1.4 to 1.5 times as long (FS-RESIDENT-WINDOW1-h100)."""
    op = template.MOSAIC
    with generation.assumed("sm90"):
        target = generation.gpu_target()
        ba = op.bind(*resource_operands("qwen3.5-27b-gdn", jnp.bfloat16, 1, 4096), None, max_segments=None,
                     floor=None)
        tiles = op.derived(ba, target)
        assert tiles is not None and tiles.block_v == 32 and tiles.stages > 1
        assert op.states_held(ba, tiles, target) == 1


@lowers('sm90')
def test_a_training_launchs_backward_kernels_carry_its_launch():
    """The gated delta rule's 16-bit training step on the template lowered for an H100: its fast backward's
    kernels, which JAX traces on the pullback after the launch's scope ended and a forward alone does not
    run, carry the launch that traced its forward (`report.relaunched`), so a compile that finds one
    spilling refuses that launch and steps its training tiles (row 3851)."""
    from kernel_compile import lowered_for

    from dew.nn.kernels.lowered import kernel_calls

    name = 'qwen3.5-27b-gdn'
    args = resource_operands(name, jnp.bfloat16, 1, 256)

    def call(*given):
        return template.MOSAIC(*given, None, bound=1.0, **_form(name, jnp.bfloat16))[0]

    def trained(*given):
        out, pull = jax.vjp(call, *given)
        return pull(out)

    def launches(fn) -> dict:
        lowered = lowered_for(fn, args, 'sm90')
        return {found.name: found.launch for found in kernel_calls(lowered, unrun=True)
                if found.kind == 'mosaic_gpu'}

    forward, training = launches(call), launches(trained)
    assert set(training) - set(forward), (forward, training)
    assert len(set(training.values())) == 1 and None not in training.values(), training

