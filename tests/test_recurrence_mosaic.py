"""The recurrence template's Mosaic GPU mainloop (`dew.nn.kernels.templates.recurrence.mainloop`) through the
chunked op's explicit route (`recurrence_chunk.chunk(..., mainloop='mosaic')`), compiled on a GPU of an
instruction family one of its bundles claims (mma.sync, wgmma): check 1 (`reference_error.assert_as_exact`) of
its output and final memory against its reference's (the recurrence a token at a time,
`test_recurrence_chunk.token_by_token`, in an fp32 call, the fla port in a 16-bit one), both from float64,
over one sequence a row and packed documents, value heads over fewer key heads, keys 64 and 128 wide, values
off a block's width, bf16 and fp32 operands and lengths off the chunks, and every hook combination the
mainloop takes (a decay a head or none, with and without the delta rule, Mamba2's rated decay, the skip and
the norm) on the chunked op's own draws (`test_recurrence_chunk`). Off such a GPU the cases run in Mosaic
GPU's interpreter on the wgmma bundle (`mainloop.INTERPRETED`) at one row of a shorter length
(`INTERPRETED_LENGTH`), whose products are the dots they compute: coverage of the mainloop's algebra and its
roles' streams, not proof of its kernels, which the GPU gate is;
the stability suite holds the mainloop under its regimes there (tests/test_recurrence_stability.py)."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, assert_as_exact_as_the_reference, float64_forward
from test_recurrence_chunk import (
    FORMED,
    HOOKS,
    SEEDS as HOOK_SEEDS,
    chunk_on,
    draw as hooked_draw,
    token_by_token,
)

from dew.nn.kernels import generation, recurrence
from dew.nn.kernels.recurrence_chunk import Chunk, chunk, mosaic_rule
from dew.nn.kernels.templates import bundles, mosaic, resources
from dew.nn.kernels.templates.base import Unfitted
from dew.nn.kernels.templates.recurrence import mainloop, template

ON_ITS_FAMILY = pytest.mark.skipif(
    mainloop.family_of() is None,
    reason="the mainloop's bundles run compiled on a GPU of the family each claims (an A100's, an H100's)")

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


NATIVE = pytest.mark.gpu(reason="checked natively: interpreted, its draws take a CPU task's share alone, "
                                "and each mainloop it runs is checked interpreted at a case beside it")
"""A case the GPU tier checks alone: the CPU's interpreted check 1 of every mainloop takes the cases
`INTERPRETED_FORWARD` names, and `test_recurrence_mosaic_gradient.INTERPRETED_GRADIENT`."""

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
    """`grid`'s parameters, each id its values joined by '-', those not in `chosen` the GPU tier's alone
    (`NATIVE`); a test's dtype taken by its id."""
    return [pytest.param(*(DTYPES.get(value, value) for value in values), id='-'.join(values),
                         marks=() if values in chosen else NATIVE)
            for values in grid]


INTERPRETED_HEADS = {'one sequence': {'key_heads': 1, 'value_heads': 2, 'value_width': 64},
                     'packed documents': {'value_heads': 2}}
"""Off a GPU, the cases' heads and values narrowed to what each still checks (one sequence two key blocks and
a key head's group of value heads, packed documents their starts on and off the chunks' grid): the
interpreted gradient of one sequence took 300 to 390 s and of packed documents 535 to 688, past the suite's
240 a test (row 1850); narrowed, 131 s and 336 s: one value head, or two chunks, left the packed log decay
unresolved within 40 seeds, its extra seeds taking 784 and 934 s."""


def geometry(case: str) -> dict:
    """Case `case`'s operands as `SHAPES` gives them, shortened and narrowed off a GPU of a family a bundle
    claims (`INTERPRETED_LENGTH`, `INTERPRETED_HEADS`)."""
    shape = SHAPES[case]
    if mainloop.family_of() is not None:
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
    hooks'."""
    heads = given.value.shape[:3]
    g = jnp.zeros(heads, jnp.float32) if given.log_decay is None else given.log_decay
    beta = jnp.ones(heads, given.value.dtype) if given.beta is None else given.beta
    # The l2 norm takes the queries and keys to fp32 at least before the rule reads them.
    query, key = ((x.astype(jnp.promote_types(x.dtype, jnp.float32)) for x in (given.query, given.key))
                  if hooks.l2norm else (given.query, given.key))
    return template.MOSAIC.bind(query, key, given.value, g, beta, given.state, given.starts,
                                delta=hooks.delta, wide=hooks.skip or hooks.output is not None,
                                floor=hooks.floor)


def taken(given: Chunk, hooks: recurrence.Hooks) -> bool:
    """Whether the template takes a call over `given`'s operands under `hooks` on the device its launches run
    for (`Mosaic.eligible`: every launch's shared memory and registers within a block's and a thread's, a
    decay a key dimension's declared floor)."""
    return template.MOSAIC.eligible(template_call(given, hooks), template.Mosaic.launched())


REFUSED = (Unfitted, NotImplementedError)
"""The template's refusals: no config whose launches the device holds, or a call its mainloop does not
take (a decay a key dimension without a floor its sub-chunks factor under)."""


def on_the_mainloop(hooks: recurrence.Hooks, case: str, dtype):
    """The chunked op on the mainloop for `case`'s calls in `dtype`: through the template's route where it
    takes them (`chunk(..., mainloop='mosaic')`, at its derived tiles); else, the route refusing them (a
    launch's registers past a thread's), the mainloop's rule at its narrowest value block, one chunk in
    flight (`mosaic_rule`), whose arithmetic is the same at any tiles. None where the mainloop's rule refuses
    the call too (checked): an fp32 call's decay a key dimension (`mainloop.per_key_refusal`); and on a GPU
    where its launches at those tiles pass the device's shared memory (an fp32 call's solve on an A100's
    block), whose arithmetic the CPU interprets."""
    (operands,) = draw(dtype, 0, **geometry(case))
    if taken(operands, hooks):
        return functools.partial(chunk, hooks, mainloop='mosaic')
    with pytest.raises(REFUSED):
        jax.eval_shape(functools.partial(chunk, hooks, mainloop='mosaic'), operands)
    family = mainloop.family_of() or mainloop.INTERPRETED
    if SHAPES[case].get('per_key') and mainloop.per_key_refusal(dtype, hooks.floor):
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
        launches = mainloop.planned(*given, tiles=narrowest, family=family, starts=starts, floor=hooks.floor)
        if max(launch.shared for launch in launches) > generation.shared_budget(template.Mosaic.launched()):
            return None

    def rule(given: Chunk, max_segments: int | None = None):
        return mosaic_rule(given.query, given.key, given.value, given.log_decay, given.beta, given.state,
                           tiles=narrowest, starts=given.starts, max_segments=max_segments, scale=hooks.scale,
                           floor=hooks.floor)

    return rule


@pytest.mark.parametrize(('case', 'dtype'), interpreted(
    [(case, dtype) for dtype in DTYPES for case in SHAPES], INTERPRETED_FORWARD))
def test_the_mosaic_forward_is_as_exact_as_its_reference(case, dtype):
    """The mainloop's output and final memory on par from float64 with its reference's over the case's draws
    (check 1, truth the recurrence a token at a time in float64): the gated delta rule with a decay a head or
    a key dimension (Kimi delta attention's), its queries scaled by `Dk^-1/2`. An fp32 call's reference is the
    recurrence a token at a time in fp32; a 16-bit call's is the fla port in the call's dtype
    (`recurrence_chunk.fla_rule`), whose numbers it runs: the values between launches rounded to bf16 put the
    final memory 1e-3 from float64, which the fp32 recurrence holds to 6e-8. A call the template refuses
    is checked refused and its arithmetic on the mainloop's rule (`on_the_mainloop`)."""
    per_key = SHAPES[case].get('per_key')
    hooks = recurrence.Hooks(decay='vector' if per_key else 'scalar', scale=SHAPES[case]['key_width'] ** -0.5,
                             floor=FLOOR if per_key else None)
    on = on_the_mainloop(hooks, case, dtype)
    if on is None:
        return
    dew = jax.jit(on)
    truth = jax.jit(functools.partial(token_by_token, hooks))
    reference = truth if dtype == jnp.float32 else jax.jit(functools.partial(chunk_on, 'fla', hooks))
    assert_as_exact(dew, reference, functools.partial(draw, dtype, **geometry(case)),
                    f'the mosaic forward, {case} in {jnp.dtype(dtype).name}', seeds=SEEDS, truth_fn=truth)


UNSEEN = generation.GpuTarget("sm87", "NVIDIA Jetson AGX Orin", 163 * 1024, multiprocessors=16,
                              clock_hz=1.3e9, memory_bandwidth=204.8e9)
"""A GPU the facts table names no row of: Jetson AGX Orin's Ampere (mma.sync's family), 163 KB of shared
memory a block, 16 multiprocessors, its clock and its memory's bandwidth, as its device reports them."""


@pytest.mark.parametrize('dtype', [jnp.bfloat16, jnp.float32], ids=['bf16', 'fp32'])
def test_an_unseen_gpus_facts_derive_the_templates_tiles_as_exact_as_its_reference(dtype):
    """The recurrence template on a GPU no row names (`UNSEEN`): its facts and its family's bundle's
    (mma.sync's) alone derive the call's tiles (`template.Mosaic.derived`): every launch's shared memory as
    the bundle declares it within a block of it (`mainloop.planned`), with two chunks in flight unless two
    are past it or leave a multiprocessor fewer of the states launch's programs (`Mosaic.states_held`); at
    those tiles the forward, on the bundle interpreted, is on par from float64 with its reference's (check 1,
    as `test_the_mosaic_forward_is_as_exact_as_its_reference`'s). Its variants, which
    the opt-in tuner times (`Mosaic.variants`), lead with those tiles, each legal there and on par too. An
    fp32 call it refuses: a launch's registers at its narrowest tiles pass a thread's (the fp32 solve's
    merges hold the inverse's three parts)."""
    assert generation.family(UNSEEN.generation) == 'mma.sync'
    case = 'one sequence'
    hooks = recurrence.Hooks(decay='scalar', scale=SHAPES[case]['key_width'] ** -0.5)
    (operands,) = draw(dtype, 0, **geometry(case))
    bound = template.MOSAIC.bind(operands.query, operands.key, operands.value, operands.log_decay,
                                 operands.beta, operands.state, operands.starts)
    budget = generation.shared_budget(UNSEEN)
    if dtype == jnp.float32:
        with generation.assumed(UNSEEN):
            assert template.MOSAIC.derived(bound, UNSEEN) is None
            narrowest = mainloop.Tiles(bundles.of('sm80', jnp.bfloat16).narrowest_columns(jnp.bfloat16), 1)
            assert template.MOSAIC.registers(bound, narrowest) > template.MOSAIC.register_budget(narrowest)
        return
    with generation.assumed(UNSEEN):
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

    def dew(given: Chunk, tiles: mainloop.Tiles):
        return mosaic_rule(given.query, given.key, given.value, given.log_decay, given.beta, given.state,
                           tiles=tiles, starts=given.starts, scale=hooks.scale,
                           family=bundles.of('sm80', jnp.bfloat16))

    truth = jax.jit(functools.partial(token_by_token, hooks))
    reference = truth if dtype == jnp.float32 else jax.jit(functools.partial(chunk_on, 'fla', hooks))
    # Each variant: its value block and stages change which program computes a value column.
    for tiles in variants:
        assert_as_exact(jax.jit(functools.partial(dew, tiles=tiles)), reference,
                        functools.partial(draw, dtype, **geometry(case)),
                        f'the mosaic forward on an unseen GPU at {tiles}, in {jnp.dtype(dtype).name}',
                        seeds=SEEDS, truth_fn=truth)


@pytest.mark.parametrize('training', [False, True], ids=['forward', 'training'])
def test_the_mosaic_launches_state_their_work(training):
    """A bf16 gated delta rule's forward, or its forward and gradient, through the template, lowered for the
    GPU the test runs on, or off one an H100 (Mosaic GPU lowers no Hopper copy for an A100's device):
    every launch states its cost (`_kernel`'s, `templates.gemm.cost`), which Dew's FLOP count reads to
    size it (`instrumentation._stated_flops`: no launch unsized, row 1895's kernel report), their products
    together at least the model's recurrent function's (`flops.delta_flops`: the chunked form's preparation
    and inversion are work past it)."""
    import re

    from dew.nn.flops import delta_flops
    from dew.telemetry.instrumentation import _stated_flops

    rows, length, key_heads, value_heads, width = 1, 256, 2, 4, 128
    operands = [jax.ShapeDtypeStruct((rows, length, key_heads, width), jnp.bfloat16)] * 2 + [
        jax.ShapeDtypeStruct((rows, length, value_heads, width), jnp.bfloat16),
        jax.ShapeDtypeStruct((rows, length, value_heads), jnp.float32),
        jax.ShapeDtypeStruct((rows, length, value_heads), jnp.bfloat16)]

    def rule(*operands):
        return mosaic_rule(*operands, tiles=mainloop.Tiles(32, 1))[0].astype(jnp.float32).sum()

    lowered_for = generation.device_generation() if mainloop.family_of() is not None else "sm90"
    with generation.assumed(lowered_for):
        fn = jax.grad(rule, argnums=(0, 1, 2)) if training else rule
        text = jax.jit(fn).trace(*operands).lower(lowering_platforms=("cuda",)).as_text(dialect="hlo")
    launches = [line for line in text.splitlines() if re.search(r'custom_call_target="mosaic_gpu', line)]
    assert launches and all(_stated_flops(line) is not None for line in launches)
    model = delta_flops(np.asarray([length] * rows), value_heads, width, width)
    assert sum(_stated_flops(line) or 0.0 for line in launches) >= model * (2 if training else 1)


def test_a_gated_delta_net_named_onto_the_template_matches_the_portable_rule():
    """A layer naming the template (`generation.chosen(gated_delta_rule='mosaic_gpu')`): its output on par
    from float64 with the portable rule's (the layer's own float64 forward the truth). The layer computes
    in bf16, the call the template's launches hold on every family (an fp32 call's solve holds more
    registers than a thread has)."""
    from dew.nn.linear import GatedDeltaNet

    x = jax.random.normal(jax.random.key(23), (1, 72, 64)).astype(jnp.bfloat16)
    layer = GatedDeltaNet(emb_features=64, num_k_heads=1, num_v_heads=2, head_k_dim=64, head_v_dim=64,
                          dtype=jnp.bfloat16)
    with generation.chosen(gated_delta_rule='xla'):
        params = layer.init(jax.random.key(29), x)
        want = layer.apply(params, x)
    with generation.chosen(gated_delta_rule='mosaic_gpu'):
        got = layer.apply(params, x)
    truth, terms = float64_forward(layer, params, x)
    assert_as_exact_as_the_reference(got, want, truth, 'a layer named onto the template', terms=terms)


def test_an_fp32_operand_splits_into_finite_parts_summing_to_it():
    """An fp32 operand's three bf16 parts (`mainloop._split`) are finite and sum to it exactly, at fp32's
    largest finite values too, past bf16's: its first part saturates there, where rounding would make it
    infinite and its remainder NaN."""
    largest = float(jnp.finfo(jnp.float32).max)
    x = jnp.array([largest, -largest, 1.0 + 2.0 ** -20, -3.0e-30, 0.0], jnp.float32)
    parts = mainloop._split(x)
    assert all(bool(jnp.all(jnp.isfinite(part))) for part in parts)
    total = sum(np.asarray(part, np.float64) for part in parts)
    np.testing.assert_array_equal(total, np.asarray(x, np.float64))


TAKEN = HOOKS + FORMED
"""The chunked op's hook combinations, each run where the mainloop takes it and checked refused where not.
Their draws are fp32, whose delta-rule and per-key calls the template refuses (an fp32 solve's registers, a
per-key decay's factoring): the cases that run check 1 are the fp32 calls without the delta rule, a decay a
head or none, natively (template numerics review 2026-10-26, P2-3: the other forms, the delta rule without
a decay, the gated linear attentions, the rated scan in bf16 and the formed hooks, hold no float64
evidence on the template; their reference arm in a 16-bit call is open)."""


@ON_ITS_FAMILY
@pytest.mark.parametrize('packed', (False, True), ids=['one sequence', 'packed'])
@pytest.mark.parametrize('hooks', TAKEN, ids=str)
def test_every_hook_the_mosaic_mainloop_takes_is_as_exact_as_the_recurrence_token_by_token(hooks, packed):
    """The chunked op on the mainloop, each hook combination it takes, its output and final memory on par with
    the recurrence a token at a time's from float64 over the site's declared draws (check 1), as
    test_recurrence_chunk holds the portable math; a call the template refuses (a launch's registers past a
    thread's) is checked refused, its arithmetic the forward check's (`on_the_mainloop`)."""
    (sample,) = hooked_draw(hooks, packed, 0)
    if not taken(sample, hooks):
        with pytest.raises(REFUSED):
            jax.eval_shape(functools.partial(chunk, hooks, mainloop='mosaic'), sample)
        return
    dew = jax.jit(functools.partial(chunk, hooks, mainloop='mosaic'))
    reference = jax.jit(functools.partial(token_by_token, hooks))
    assert_as_exact(dew, reference, functools.partial(hooked_draw, hooks, packed),
                    f'the mosaic forward, {hooks} packed={packed}', seeds=HOOK_SEEDS[hooks, packed])


RESOURCE_SHAPES = {'qwen3.5-27b-gdn': (16, 48, 128, 128, False), 'kimi-linear-kda': (32, 32, 128, 128, True)}
"""The scoreboard's delta-rule geometries the resource gate compiles (`tools/benchmark_recurrence.SHAPES`):
key heads, value heads, key and value widths, and whether the decay is a key dimension's."""


def _model(launch: mainloop.Launch, family: mosaic.Family) -> dict:
    """A launch's register model as the compile gate decides on it (`kernel_compile.decisions`): its stated
    registers (`Launch.registers`) against its family's budget at its consumer roles (`Launch.consumers`, a
    walk's `Tiles.walkers`), and the producer's beside them where the family streams from a role of its
    own."""
    roles = launch.consumers
    producer = family.warpgroups(roles) > roles
    return {"registers": launch.registers, "budget": family.registers(roles), "consumers": roles,
            "most": family.programs(roles), "register_step": family.register_step,
            "producer": resources.PRODUCER_REGISTERS if producer else 0}


class Unbudgeted(template.Mosaic):
    """The recurrence template with every launch's registers held: its derivation takes the tile it would
    where only registers refuse one (`compiled_cases`' refused tiles)."""

    def _held(self, ba, config) -> bool:
        del ba, config
        return True


def compiled_cases(target: str, plans: dict, refused: list) -> list:
    """The resource gate's cases on `target` (`KernelCase`s), each with its launches' register models (row
    1971): each `RESOURCE_SHAPES` geometry's forward in a 16-bit and an fp32 call, one row and packed
    documents, and the gated delta rule's 16-bit training pass, at the template's derived tiles and, one row's
    forward, at each of their variants; beside them the tiles the derivation takes where every register held
    (`Template._held`) and refused for registers alone, and the fast backward its launches' registers refuse,
    so a model over ptxas's need fails as one under it does. Each case's plan by its label into `plans`
    (`mainloop.planned`), each call the template refuses into `refused`."""
    from fla_gdn_support import KernelCase

    family, facts = bundles.of(target, jnp.bfloat16), generation.gpu_target(target)
    rows, length = 1, 256
    shared = generation.shared_budget(facts)

    def cases(name: str, dtype, *, training: bool, packed: bool) -> list:
        """The case's kernels at its derived tiles and, one row's forward, at each of its variants."""
        key_heads, value_heads, key_width, value_width, per_key = RESOURCE_SHAPES[name]

        def array(*shape, kind=dtype):
            return jax.ShapeDtypeStruct(shape, kind)

        decay = (rows, length, value_heads) + ((key_width,) if per_key else ())
        args = [array(rows, length, key_heads, key_width), array(rows, length, key_heads, key_width),
                array(rows, length, value_heads, value_width), array(*decay, kind=jnp.float32),
                array(rows, length, value_heads), array(rows, value_heads, key_width, value_width,
                                                        kind=jnp.float32)]
        starts = array(rows, length, kind=jnp.bool_) if packed else None
        bound = template.MOSAIC.bind(*args, starts, max_segments=8 if packed else None,
                                     floor=FLOOR if per_key else None)
        with generation.assumed(facts):
            derived = template.MOSAIC.derived(bound, facts)
            held = Unbudgeted().derived(bound, facts)
            # The tile the derivation takes where every register held, refused for its registers alone.
            unheld = [] if held is None or held == derived else [(held, 'refused ')]
            if derived is None:
                # Refused: a launch's registers or shared memory past the device's at every tile.
                refused.append(f"{name} {jnp.dtype(dtype).name}{' training' if training else ''}")
                assert not template.MOSAIC.eligible(bound, facts)
                return [case(name, dtype, args, starts, bound, tiles, training=training, packed=packed,
                             kind=kind) for tiles, kind in unheld]
            variants = template.MOSAIC.variants(bound)
            if training and not mainloop.backward_holds(
                    *args, tiles=derived, family=family, shared=shared, starts=starts,
                    max_segments=8 if packed else None):
                # The rule's gradient is the port's or the portable math's where the fast backward's
                # launches do not hold (`mainloop.backward_holds`); where only their registers refuse them,
                # the gate compiles them, held to ptxas's need.
                refused.append(f"{name} {jnp.dtype(dtype).name} fast backward{' packed' if packed else ''}")
                launches = template.MOSAIC._launches(bound, derived, facts, backward=True)
                if any(launch.shared > shared for launch in launches):
                    return []
                return [case(name, dtype, args, starts, bound, derived, training=True, packed=packed,
                             kind='refused ')]
        assert variants[0] == derived, f"{name} on {target}: variants {variants} lead with no {derived}"
        derived_tiles = [(tiles, '') for tiles in (variants[:1] if training or packed else variants)]
        # The walks at two consumer roles a program (`Tiles.walkers`), which the derivation gives a call
        # whose walks fill more than a wave and these shapes never do: a role's registers are its share
        # of the program's.
        value_width = RESOURCE_SHAPES[name][3]
        walked = ([(dataclasses.replace(derived, walkers=2), 'two roles ')]
                  if family.consumers >= 2 and not packed and value_width // derived.block_v % 2 == 0 else [])
        return [case(name, dtype, args, starts, bound, tiles, training=training, packed=packed, kind=kind)
                for tiles, kind in derived_tiles + walked + ([] if training else unheld)]

    def case(name: str, dtype, args, starts, bound, tiles, *, training: bool, packed: bool, kind: str):
        _, value_heads, key_width, value_width, per_key = RESOURCE_SHAPES[name]

        def array(*shape, kind=dtype):
            return jax.ShapeDtypeStruct(shape, kind)

        label = (f"{name} {jnp.dtype(dtype).name} {'training' if training else 'forward'}"
                 f"{' packed' if packed else ''} {kind}{tiles}")
        plans[label] = template.MOSAIC._launches(bound, tiles, facts, backward=training)

        def call():
            given = list(args)
            if training:
                given += [array(rows, length, value_heads, value_width),
                          array(rows, value_heads, key_width, value_width, kind=jnp.float32)]
            run = (mainloop.gradients if training
                   else functools.partial(mainloop.forward, floor=FLOOR if per_key else None))

            def fn(*operands):
                *operands, documents = operands if packed else (*operands, None)
                return run(*operands, tiles=tiles, starts=documents, max_segments=8 if packed else None,
                           family=family)

            return fn, (*given, *([starts] if packed else []))

        return KernelCase(label, call, {launch.name: _model(launch, family) for launch in plans[label]})

    # Kimi's keys, 128 wide, past what an fp32 call takes under a decay a key dimension (`forward`).
    compiled = [kernel for name in RESOURCE_SHAPES for dtype in (jnp.bfloat16, jnp.float32)
                for packed in (False, True) if not (RESOURCE_SHAPES[name][4] and dtype == jnp.float32)
                for kernel in cases(name, dtype, training=False, packed=packed)]
    return compiled + [kernel for packed in (False, True)
                       for kernel in cases('qwen3.5-27b-gdn', jnp.bfloat16, training=True, packed=packed)]


def planned_failures(found: list, plans: dict) -> list[str]:
    """Each compiled kernel past its plan's shared memory (`mainloop.planned`'s launch of its name, from
    which the tiles are derived): the recurrence's shared-memory model, beside the register model the shared
    gate holds to ptxas (`kernel_compile.mosaic_failures`)."""
    def planned(record: dict, kernel: dict) -> int:
        return max((launch.shared for launch in plans[record['name']]
                    if kernel['kernel'].startswith(launch.name)), default=0)

    return [f"{record['name']} {kernel['kernel']}: {kernel['shared']} shared bytes past its plan's "
            f"{planned(record, kernel)}" for record in found for kernel in record.get('kernels', ())
            if kernel['shared'] is not None and not kernel['shared'] <= planned(record, kernel)]


@pytest.mark.parametrize('target', ['sm80', 'sm90'])
def test_the_mosaic_mainloop_compiles_within_each_gpus_resources_as_its_model_decides(target):
    """The mainloop's launches (`compiled_cases`), compiled by Mosaic for a compile-only A100 and H100 on this
    CPU (tests/kernel_compile.py): every launch compiles within the multiprocessor's shared memory (163 KB,
    227 KB) and tensor memory and at most its plan's shared memory (`mainloop.planned`, from which the tiles
    are derived, `planned_failures`), no launch the template's legality holds spills a register, and each
    launch's register model decides as ptxas's need does, admitting a tile or refusing it (row 1971,
    `kernel_compile.mosaic_failures`). Lowering alone takes no device's budget: a launch past an A100's
    shared memory lowered and failed on the GPU (the fast backward's readout's third staging tile). The gated
    delta rule's 16-bit call, the production cell, derives tiles on every family. A GPU host's ptxas builds
    for its own GPU and those before it, not a newer one's (an A100's refuses sm_90a), so there the newer
    targets are the CPU suite's alone."""
    if jax.default_backend() == 'gpu':
        local = generation.capability(generation.device_generation())
        if local is not None and local < generation.capability(target):
            pytest.skip(f"this host's ptxas builds for its {local} GPU and before, not {target}")
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    plans, refused = {}, []
    found = compile_mosaic_kernels(compiled_cases(target, plans, refused), target)
    failed = mosaic_failures(found) + planned_failures(found, plans)
    failed += [f"{case}: no tiles derived" for case in refused
               if case.startswith("qwen3.5-27b-gdn bfloat16") and "fast backward" not in case]
    assert not failed, "\n".join(failed)
