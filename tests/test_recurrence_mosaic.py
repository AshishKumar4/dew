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
from reference_error import assert_as_exact, with_gradients
from test_recurrence_chunk import FORMED, HOOKS, SEEDS as HOOK_SEEDS, draw as hooked_draw, token_by_token

from dew.nn.kernels import generation, recurrence
from dew.nn.kernels.recurrence_chunk import Chunk, chunk
from dew.nn.kernels.templates.recurrence import mainloop

ON_ITS_FAMILY = pytest.mark.skipif(
    mainloop.family_of() is None,
    reason="the mainloop's bundles run compiled on a GPU of the family each claims (an A100's, an H100's)")

pytestmark = pytest.mark.kernel

SEEDS = 8
"""Draws a case takes, check 1's fewest (`reference_error.FEWEST_SEEDS`)."""

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


def geometry(case: str) -> dict:
    """Case `case`'s operands as `SHAPES` gives them, shortened off a GPU of a family a bundle claims
    (`INTERPRETED_LENGTH`)."""
    shape = SHAPES[case]
    if mainloop.family_of() is not None:
        return shape
    return {**shape, 'rows': 1, 'length': min(shape['length'], INTERPRETED_LENGTH)}


def draw(dtype, seed: int, *, rows: int, length: int, key_heads: int, value_heads: int, key_width: int,
         value_width: int, packed: bool = False, per_key: bool = False) -> tuple[Chunk]:
    """A draw's operands from `seed`, its keys' and values' dimensions each permuted by an order the seed
    draws (an exact symmetry of every head's recurrence): unit queries and keys, normal values, a log decay a
    head in (-1, 0) and a write gate in (0, 1), a memory entering; the queries, keys, values and write gate in
    `dtype`."""
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


@pytest.mark.parametrize('dtype', [jnp.bfloat16, jnp.float32], ids=['bf16', 'fp32'])
@pytest.mark.parametrize('case', SHAPES)
def test_the_mosaic_forward_is_as_exact_as_its_reference(case, dtype):
    """The mainloop's output and final memory on par from float64 with its reference's over the case's draws
    (check 1, truth the recurrence a token at a time in float64): the gated delta rule with a decay a head or
    a key dimension (Kimi delta attention's), its queries scaled by `Dk^-1/2`. An fp32 call's reference is the
    recurrence a token at a time in fp32; a 16-bit call's is the fla port in the call's dtype
    (`recurrence_chunk.fla_rule`), whose numbers it runs: the values between launches rounded to bf16 put the
    final memory 1e-3 from float64, which the fp32 recurrence holds to 6e-8."""
    hooks = recurrence.Hooks(decay='vector' if SHAPES[case].get('per_key') else 'scalar',
                             scale=SHAPES[case]['key_width'] ** -0.5)
    dew = jax.jit(functools.partial(chunk, hooks, mainloop='mosaic'))
    truth = jax.jit(functools.partial(token_by_token, hooks))
    reference = truth if dtype == jnp.float32 else jax.jit(functools.partial(chunk, hooks, ported=True))
    assert_as_exact(dew, reference, functools.partial(draw, dtype, **geometry(case)),
                    f'the mosaic forward, {case} in {jnp.dtype(dtype).name}', seeds=SEEDS, truth_fn=truth)


GRADIENT_CASES = [case for case, shape in SHAPES.items() if not shape.get('per_key')]
"""The cases whose gradient the mainloop takes: a decay a head's (the gated delta rule's)."""


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


@pytest.mark.parametrize('dtype', [jnp.bfloat16, jnp.float32], ids=['bf16', 'fp32'])
@pytest.mark.parametrize('case', GRADIENT_CASES)
def test_the_mosaic_gradient_is_as_exact_as_its_reference(case, dtype):
    """The gradient of the mainloop's rule on par from float64 with its reference's over the case's draws
    (check 1, truth the recurrence a token at a time's in float64): in a 16-bit call the mainloop's fast
    backward (`mainloop.gradients`, fla's DAG) against the fla port's, whose numbers it runs; in an fp32 call
    the portable math's adjoint against the fp32 recurrence's."""
    hooks = recurrence.Hooks(decay='scalar', scale=SHAPES[case]['key_width'] ** -0.5)

    def compiled(**route):
        return jax.jit(with_gradients(restarted(functools.partial(chunk, hooks, **route), case)))

    truth = jax.jit(with_gradients(restarted(functools.partial(token_by_token, hooks), case)))
    reference = truth if dtype == jnp.float32 else compiled(ported=True)
    assert_as_exact(compiled(mainloop='mosaic'), reference, functools.partial(drawn, dtype, **geometry(case)),
                    f'the mosaic gradient, {case} in {jnp.dtype(dtype).name}', seeds=SEEDS, truth_fn=truth)


def chosen(fn, **kernels):
    """`fn` called with `kernels` chosen (`generation.chosen`), which a jitted function's trace keys on."""
    def run(*args):
        with generation.chosen(**kernels):
            return fn(*args)

    return run


@ON_ITS_FAMILY
@pytest.mark.parametrize('case', GRADIENT_CASES)
def test_the_ports_forward_takes_the_mosaic_backward_as_exact_as_its_own(case):
    """The port's rule (`recurrence_chunk.fla_rule`) with the template's fast backward
    (`gated_delta_rule_backward='mosaic'`, `_mosaic_backward_rule`) in a 16-bit call: its gradient on par
    from float64 with the port's own backward's over the case's draws (check 1), the route a target's row
    takes it by."""
    hooks = recurrence.Hooks(decay='scalar', scale=SHAPES[case]['key_width'] ** -0.5)
    trained = jax.jit(with_gradients(restarted(functools.partial(chunk, hooks, ported=True), case)))
    truth = jax.jit(with_gradients(restarted(functools.partial(token_by_token, hooks), case)))
    assert_as_exact(chosen(trained, gated_delta_rule='fla', gated_delta_rule_backward='mosaic'),
                    chosen(trained, gated_delta_rule='fla', gated_delta_rule_backward='fla'),
                    functools.partial(drawn, jnp.bfloat16, **geometry(case)),
                    f"the port's forward with the mosaic backward, {case}", seeds=SEEDS, truth_fn=truth)


TAKEN = HOOKS + FORMED
"""The chunked op's hook combinations, every one of which the mainloop takes."""


@ON_ITS_FAMILY
@pytest.mark.parametrize('packed', (False, True), ids=['one sequence', 'packed'])
@pytest.mark.parametrize('hooks', TAKEN, ids=str)
def test_every_hook_the_mosaic_mainloop_takes_is_as_exact_as_the_recurrence_token_by_token(hooks, packed):
    """The chunked op on the mainloop, each hook combination it takes, its output and final memory on par with
    the recurrence a token at a time's from float64 over the site's declared draws (check 1), as
    test_recurrence_chunk holds the portable math."""
    dew = jax.jit(functools.partial(chunk, hooks, mainloop='mosaic'))
    reference = jax.jit(functools.partial(token_by_token, hooks))
    assert_as_exact(dew, reference, functools.partial(hooked_draw, hooks, packed),
                    f'the mosaic forward, {hooks} packed={packed}', seeds=HOOK_SEEDS[hooks, packed])


RESOURCE_SHAPES = {'qwen3.5-27b-gdn': (16, 48, 128, 128, False), 'kimi-linear-kda': (32, 32, 128, 128, True)}
"""The scoreboard's delta-rule geometries the resource gate compiles (`tools/benchmark_recurrence.SHAPES`):
key heads, value heads, key and value widths, and whether the decay is a key dimension's."""


@pytest.mark.parametrize('target', ['sm80', 'sm90'])
def test_the_mosaic_mainloop_compiles_within_each_gpus_resources(target):
    """The mainloop's launches, its forward in a 16-bit and an fp32 call and the gated delta rule's fast
    backward, at the scoreboard's widths, one row of 256 tokens and packed documents, compiled by Mosaic for a
    compile-only A100 and H100 on this CPU (tests/kernel_compile.py): within the multiprocessor's shared
    memory (163 KB, 227 KB), no register spilled. Lowering alone takes no device's budget: a launch past an
    A100's shared memory lowered and failed on the GPU (the fast backward's readout's third staging tile)."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, mosaic_failures

    from dew.nn.kernels.templates import mosaic_sm90
    from dew.nn.kernels.templates.recurrence import _sm80

    family = {'sm80': _sm80.MMA_SYNC, 'sm90': mosaic_sm90.WGMMA}[target]
    rows, length = 1, 256

    def case(name: str, dtype, *, training: bool, packed: bool):
        key_heads, value_heads, key_width, value_width, per_key = RESOURCE_SHAPES[name]

        def call():
            def array(*shape, kind=dtype):
                return jax.ShapeDtypeStruct(shape, kind)

            decay = (rows, length, value_heads) + ((key_width,) if per_key else ())
            args = [array(rows, length, key_heads, key_width), array(rows, length, key_heads, key_width),
                    array(rows, length, value_heads, value_width), array(*decay, kind=jnp.float32),
                    array(rows, length, value_heads), array(rows, value_heads, key_width, value_width,
                                                            kind=jnp.float32)]
            if training:
                args += [array(rows, length, value_heads, value_width),
                         array(rows, value_heads, key_width, value_width, kind=jnp.float32)]
            starts = array(rows, length, kind=jnp.bool_) if packed else None
            run = mainloop.gradients if training else mainloop.forward

            def fn(*operands):
                *operands, given = operands if packed else (*operands, None)
                return run(*operands, starts=given, max_segments=8 if packed else None, family=family)

            return fn, (*args, *([starts] if packed else []))

        label = f"{name} {jnp.dtype(dtype).name} {'training' if training else 'forward'}"
        return KernelCase(label + (' packed' if packed else ''), call)

    # Kimi's keys, 128 wide, past what an fp32 call takes under a decay a key dimension (`forward`).
    cases = [case(name, dtype, training=False, packed=packed) for name in RESOURCE_SHAPES
             for dtype in (jnp.bfloat16, jnp.float32) for packed in (False, True)
             if not (RESOURCE_SHAPES[name][4] and dtype == jnp.float32)]
    cases += [case('qwen3.5-27b-gdn', jnp.bfloat16, training=True, packed=packed) for packed in (False, True)]
    found = compile_mosaic_kernels(cases, target)
    assert all(record.get('kernels') for record in found), found
    assert not mosaic_failures(found), "\n".join(mosaic_failures(found))
