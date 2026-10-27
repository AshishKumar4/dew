"""The recurrence template's stability on its default path: the gated delta rule's 16-bit forward and fast
backward (`recurrence_chunk.mosaic_rule` through `chunk(..., mainloop='mosaic')`, an H100's training), under
the regimes that break chunked forms (test_recurrence_stability's `REGIMES`: decays near zero, steep then
mild, aligned keys with the write gate near one, resets on chunks' edges, a long sequence), held two ways as
test_recurrence_stability holds the portable math:

- finite wherever the recurrence a token at a time in float64 is;
- check 1 (`assert_as_exact`): on par from float64 with its reference arm, the fla port's
  (`chunk_on('fla', ...)`), whose numbers it runs.

Natively on a GPU of a family the mainloop claims (`NATIVE`: interpreted, one case's check 1 outran the CPU
tier's hour); off one, the CPU interprets the forward's finiteness under aligned keys and reads the
gradient's backward decision. The gradient's arm asserts which backward runs: the fast backward where its
launches hold at the call's derived tiles (`mainloop.backward_holds`, as `mosaic_rule` reads it; natively,
its reverse launch in the trace), and on an mma.sync GPU, where they do not, none: the rule refuses the
gradient (row 1926), so check 1 there holds the forward alone."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import (
    assert_as_exact,
    assert_as_exact_as_the_reference,
    equations,
    widened,
    with_gradients,
)
from test_recurrence_chunk import chunk_on, token_by_token
from test_recurrence_mosaic import NATIVE
from test_recurrence_stability import SEEDS, SITES, cotangents, restarting, stressed

from dew.nn.kernels import generation
from dew.nn.kernels.recurrence_chunk import Chunk, chunk
from dew.nn.kernels.templates.recurrence import mainloop, template

pytestmark = pytest.mark.kernel

HOOKS = SITES['scalar delta']
"""The template's default call: the gated delta rule, a decay a head."""

REGIMES = ('decays near zero', 'steep then mild', 'aligned keys', 'resets on chunk edges', 'a long sequence')
"""The regimes where the template's tile algebra departs from its reference arm's: `_scaled`'s clip and
`PAIR_FLOOR` (decays near zero, steep then mild), the inverse's fp16 doubling (aligned keys, the write gate
near one), the walk's resets (resets on chunks' edges) and its carried memory (a long sequence)."""

CASES = [pytest.param(regime, product, id=f'{regime}-{product}', marks=NATIVE)
         for regime in REGIMES for product in ('forward', 'gradient')]


def bf16(operands: Chunk) -> Chunk:
    """`operands` as a 16-bit call takes them: its queries, keys, values and write gate in bf16, its log decay
    and memory fp32."""
    return dataclasses.replace(operands, **{name: getattr(operands, name).astype(jnp.bfloat16)
                                            for name in ('query', 'key', 'value', 'beta')})


def fast_backward(operands: Chunk) -> bool:
    """Whether the template's rule takes a gradient over `operands` on its fast backward: its launches hold at
    the call's derived tiles on the device's bundle, or interpreted, the interpreted one's (`mosaic_rule`)."""
    target = template.Mosaic.launched()
    bound = template.MOSAIC.bind(operands.query, operands.key, operands.value, operands.log_decay,
                                 operands.beta, operands.state, operands.starts)
    tiles = template.MOSAIC.derived(bound, target)
    return mainloop.backward_holds(operands.query, operands.key, operands.value, operands.log_decay,
                                   operands.beta, operands.state, tiles=tiles,
                                   family=mainloop.family_of() or mainloop.INTERPRETED,
                                   shared=generation.shared_budget(target), starts=operands.starts)


def launched(fn, *args) -> set[str]:
    """The names of the Pallas launches `fn`'s trace over `args` holds: none where the kernels interpret (a
    CPU's trace inlines their bodies)."""
    graph = jax.make_jaxpr(fn)(*args)
    return {str(equation.params.get('name') or equation.params.get('name_and_src_info'))
            for equation in equations(graph) if equation.primitive.name == 'pallas_call'}


def assert_finite(fn, truth, operands, regime: str) -> None:
    """`fn`'s leaves finite wherever the float64 truth's are, on `operands`."""
    mine = jax.tree.leaves(jax.jit(fn)(*operands))
    with jax.enable_x64(new_val=True):
        exact = jax.tree.leaves(jax.jit(truth)(*widened(operands)))
    for ours, theirs in zip(mine, exact, strict=True):
        finite = np.isfinite(np.asarray(theirs))
        assert np.all(np.isfinite(np.asarray(ours, np.float64))[finite]), (
            f'{regime}: not finite where float64 is')


@pytest.mark.parametrize('regime, product', CASES)
def test_the_template_stays_finite_and_on_par_with_the_port_where_chunked_forms_break(regime, product):
    """The template's 16-bit forward, or its forward and gradient, finite wherever the float64 recurrence's
    are on the regime's first draws, and on par with the port's from float64 over its declared seeds
    (check 1); its gradient the fast backward's where its launches hold, refused on an mma.sync GPU."""
    template = functools.partial(chunk, HOOKS, mainloop='mosaic')
    port = functools.partial(chunk_on, 'fla', HOOKS)
    exact = functools.partial(token_by_token, HOOKS)
    if product == 'forward':
        dew, reference, truth = template, port, exact

        def draws(seed):
            return (bf16(stressed(HOOKS, regime, seed)[0]),)
    else:
        dew, reference, truth = (with_gradients(restarting(fn, regime)) for fn in (template, port, exact))

        def draws(seed):
            operands, (out, final) = cotangents(HOOKS, regime, seed)
            return bf16(operands), (out.astype(jnp.bfloat16), final)

        fast = generation.family(generation.device_generation()) != 'mma.sync'
        operands, _ = draws(0)
        assert fast_backward(restarting(lambda given: given, regime)(operands)) == fast, regime
        if not fast:
            # The rule takes no gradient its fast backward does not (row 1926): check 1 holds the forward
            # alone (its own case).
            with pytest.raises(NotImplementedError):
                jax.eval_shape(dew, *draws(0))
            return
        names = launched(dew, *draws(0))
        if names:
            assert any('recurrence_reverse' in name for name in names), (regime, sorted(names))
    for seed in range(2):
        assert_finite(dew, truth, draws(seed), regime)
    assert_as_exact(dew, reference, draws, f'the template, {regime}, its {product}',
                    seeds=SEEDS['scalar delta', regime], truth_fn=truth)


def test_the_template_forward_stays_finite_under_aligned_keys_interpreted():
    """The template's 16-bit forward, interpreted, finite wherever the float64 recurrence's is on aligned keys
    with the write gate near one, where the inverse's fp16 doubling grows most (`_merges`)."""
    regime = 'aligned keys'
    (operands,) = (bf16(stressed(HOOKS, regime, 0)[0]),)
    assert_finite(jax.jit(functools.partial(chunk, HOOKS, mainloop='mosaic')),
                  jax.jit(functools.partial(token_by_token, HOOKS)), (operands,), regime)


@pytest.mark.parametrize('regime', REGIMES)
def test_the_template_gradient_takes_the_fast_backward_where_its_launches_hold(regime):
    """The template's gradient over each regime's draw takes its fast backward where its launches hold at the
    call's derived tiles (`fast_backward`): off a GPU by the interpreted bundle, on a warpgroup-MMA GPU by
    its own; on an mma.sync GPU they do not, and the rule refuses the gradient (row 1926)."""
    operands, _ = cotangents(HOOKS, regime, 0)
    fast = generation.family(generation.device_generation()) != 'mma.sync'
    assert fast_backward(restarting(lambda given: given, regime)(bf16(operands))) == fast


def test_the_fast_backward_decay_gradient_holds_under_decays_near_zero_interpreted():
    """The fast backward's log decay gradient, interpreted, as near float64 as the port's under decays near
    zero on a draw: a chunk's ranges sum each token's terms before it (`mainloop.gradients`), never a token's
    own order-one term added and taken away, which rounded away the `exp(-30)` terms before it (65 to 169
    times the port's distance on three draws, the H100's check 1 140 times, SE-numerics-h100-fd61e11eeb;
    0.98 to 1.00 times after), held on one draw by the gate's one-draw rule."""
    regime = 'decays near zero'
    operands, (out, final) = cotangents(HOOKS, regime, 0)
    drawn = (bf16(operands), (out.astype(jnp.bfloat16), final))
    template = with_gradients(functools.partial(chunk, HOOKS, mainloop='mosaic'))
    port = with_gradients(functools.partial(chunk_on, 'fla', HOOKS))
    with jax.enable_x64(new_val=True):
        exact = jax.jit(with_gradients(functools.partial(token_by_token, HOOKS)))(*widened(drawn))

    def leaf(fn):
        return np.asarray(jax.jit(fn)(*drawn)[1].log_decay)

    assert_as_exact_as_the_reference(leaf(template), leaf(port), np.asarray(exact[1].log_decay),
                                     f'the template, {regime}, its log decay gradient on a draw')
