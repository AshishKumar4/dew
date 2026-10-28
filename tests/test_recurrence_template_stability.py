"""The recurrence template's stability on the path a call takes: its Mosaic GPU mainloop
(`recurrence_chunk.mosaic_rule` through `chunk(..., mainloop='mosaic')`), forward and fast backward, over
every form the stability suite holds (test_recurrence_stability's `SITES`: the gated delta rule a decay a
head and a key dimension, a write gate without the delta rule, the gated delta net's formed gates and
Mamba2's rated decay) under its regimes (`REGIMES`: decays near one and near zero, the decay's bound,
aligned keys with the write gate near one, resets on chunks' edges, saturated and spread gates, long and
fading memories, steep then mild decays), held as test_recurrence_stability holds the portable math:

- finite wherever the recurrence a token at a time in float64 is;
- check 1 (`assert_as_exact`): on par from float64 with its reference arm, the fla port where it runs the
  form (the delta rule with a decay), else the portable math, in the call's 16-bit dtype.

Natively on a GPU of a family the mainloop claims (`NATIVE`: interpreted, one case's check 1 outran the CPU
tier's hour); off one, the CPU interprets each form's forward on the regimes that grow its tiles most for
finiteness, and its fast backward's decay gradient under decays near zero. A form or gradient the template
refuses (`Mosaic.eligible`, `Mosaic.trains`) is checked refused and skipped, a refusal and not numerical
coverage (row 1968: the suite held only the portable arms)."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import (
    FEWEST_SEEDS,
    assert_as_exact,
    assert_as_exact_as_the_reference,
    widened,
    with_gradients,
)
from test_recurrence_chunk import chunk_on, token_by_token
from test_recurrence_mosaic import NATIVE, REFUSED, taken, template_call
from test_recurrence_stability import GRADIENT_SEEDS, REGIMES, SEEDS, SITES, cotangents, restarting, stressed

from dew.nn.kernels import generation
from dew.nn.kernels.recurrence_chunk import Chunk, chunk
from dew.nn.kernels.templates.recurrence import mainloop, template

pytestmark = pytest.mark.kernel

HOOKS = SITES['scalar delta']
"""The template's default call: the gated delta rule, a decay a head."""

CASES = [pytest.param(name, regime, product, id=f'{name}-{regime}-{product}', marks=NATIVE)
         for name in SITES for regime in REGIMES for product in ('forward', 'gradient')]

INTERPRETED = ('aligned keys', 'decays near zero')
"""The regimes the CPU interprets each form's forward under: where the template's tiles grow most, the
inverse's fp16 doubling (aligned keys, the write gate near one) and its pairs' clip (`_scaled`,
`PAIR_FLOOR`) over chunks spanning tens of e-folds."""


def bf16(operands: Chunk, hooks) -> Chunk:
    """`operands` as a 16-bit call under `hooks` takes them: its queries, keys and values in bf16, its write
    gate in bf16 with the delta rule (as fla's layer holds it) and fp32 without (Mamba-2's step, which the
    template takes unrounded), its log decay and memory fp32."""
    names = ('query', 'key', 'value', 'beta') if hooks.delta else ('query', 'key', 'value')
    return dataclasses.replace(operands, **{name: getattr(operands, name).astype(jnp.bfloat16)
                                            for name in names})


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


def assert_finite(fn, truth, operands, regime: str) -> None:
    """`fn`'s leaves finite wherever the float64 truth's are, on `operands`."""
    mine = jax.tree.leaves(jax.jit(fn)(*operands))
    with jax.enable_x64(new_val=True):
        exact = jax.tree.leaves(jax.jit(truth)(*widened(operands)))
    for ours, theirs in zip(mine, exact, strict=True):
        finite = np.isfinite(np.asarray(theirs))
        assert np.all(np.isfinite(np.asarray(ours, np.float64))[finite]), (
            f'{regime}: not finite where float64 is')


def reference_arm(hooks):
    """The arm the template is held against for a form: the fla port where it runs it (the delta rule with a
    decay), else the portable math, each in the call's dtype."""
    return functools.partial(chunk_on, 'fla' if hooks.delta and hooks.decay != 'none' else 'xla', hooks)


def template_trains(operands: Chunk, hooks) -> bool:
    """Whether the template's fast backward takes a gradient over `operands` under `hooks` on the device its
    launches run for (`Mosaic.trains`)."""
    return template.MOSAIC.trains(template_call(operands, hooks), template.Mosaic.launched())


def runs(fn, hooks, operands: Chunk, drawn, product: str) -> bool:
    """Whether the template runs the case `fn` traces over `drawn`: a form it does not take (`taken`), or a
    gradient its fast backward does not (`Mosaic.trains`: an fp32 call's, a decay a key dimension's, an
    mma.sync GPU's), checked refused as `fn` traces, where it does not."""
    if not taken(operands, hooks):
        with pytest.raises(REFUSED):
            jax.eval_shape(fn, *drawn)
        return False
    if product == 'gradient' and not template_trains(operands, hooks):
        with pytest.raises(NotImplementedError):
            jax.eval_shape(fn, *drawn)
        return False
    return True


@pytest.mark.parametrize('name, regime, product', CASES)
def test_the_template_stays_finite_and_on_par_where_chunked_forms_break(name, regime, product):
    """Each form's 16-bit forward through the template, or its forward and gradient, finite wherever the
    float64 recurrence's are on the regime's first draws, and on par with its reference arm's from float64
    over the site's declared seeds (check 1); a form or gradient it refuses is checked refused and skipped."""
    hooks = SITES[name]
    mosaic = functools.partial(chunk, hooks, mainloop='mosaic')
    reference, exact = reference_arm(hooks), functools.partial(token_by_token, hooks)
    if product == 'forward':
        dew, truth, seeds = mosaic, exact, SEEDS[name, regime]

        def draws(seed):
            return (bf16(stressed(hooks, regime, seed)[0], hooks),)
    else:
        dew, reference, truth = (with_gradients(restarting(fn, regime)) for fn in (mosaic, reference, exact))
        seeds = GRADIENT_SEEDS[name, regime]

        def draws(seed):
            operands, (out, final) = cotangents(hooks, regime, seed)
            return bf16(operands, hooks), (out.astype(jnp.bfloat16), final)

    operands = restarting(lambda given: given, regime)(draws(0)[0])
    if not runs(dew, hooks, operands, draws(0), product):
        pytest.skip(f"the template refuses {name}'s {product}: a refusal, not numerical coverage")
    for seed in range(2):
        assert_finite(dew, truth, draws(seed), regime)
    assert_as_exact(dew, reference, draws, f'the template, {name} under {regime}, its {product}',
                    seeds=seeds, truth_fn=truth)


def test_a_ragged_chunks_last_real_write_holds_the_final_memory_interpreted():
    """The rated scan's 16-bit forward through the template under the decay bound, interpreted off a GPU, on
    par with the portable math from float64 over check 1's fewest draws: its rows end in a ragged chunk (70
    tokens), and its last real token's write, which weighs most under a steep decay, is held outside the
    product as the portable math peels it; the padded last slot's put the final memory 2.0 times the
    portable math's distance, interpreted and on an H100."""
    hooks = SITES['rated']
    assert_as_exact(functools.partial(chunk, hooks, mainloop='mosaic'), reference_arm(hooks),
                    lambda seed: (bf16(stressed(hooks, 'the decay bound', seed)[0], hooks),),
                    'the template, rated under the decay bound, its forward', seeds=FEWEST_SEEDS,
                    truth_fn=functools.partial(token_by_token, hooks))


@pytest.mark.parametrize('name', SITES)
@pytest.mark.parametrize('regime', INTERPRETED)
def test_each_form_the_template_takes_stays_finite_interpreted(name, regime):
    """Each form's 16-bit forward through the template, interpreted off a GPU, finite wherever the float64
    recurrence's is on the regime's first draw; a form it refuses is checked refused and skipped."""
    hooks = SITES[name]
    mosaic = functools.partial(chunk, hooks, mainloop='mosaic')
    drawn = (bf16(stressed(hooks, regime, 0)[0], hooks),)
    if not runs(mosaic, hooks, drawn[0], drawn, 'forward'):
        pytest.skip(f'the template refuses {name}: a refusal, not numerical coverage')
    assert_finite(jax.jit(mosaic), jax.jit(functools.partial(token_by_token, hooks)), drawn, regime)


@pytest.mark.parametrize('regime', REGIMES)
def test_the_template_gradient_takes_the_fast_backward_where_its_launches_hold(regime):
    """The template's gradient over each regime's draw takes its fast backward where its launches hold at the
    call's derived tiles (`fast_backward`): off a GPU by the interpreted bundle, on a warpgroup-MMA GPU by
    its own; on an mma.sync GPU they do not, and the rule refuses the gradient (row 1926)."""
    operands, _ = cotangents(HOOKS, regime, 0)
    fast = generation.family(generation.device_generation()) != 'mma.sync'
    assert fast_backward(restarting(lambda given: given, regime)(bf16(operands, HOOKS))) == fast


def test_the_fast_backward_decay_gradient_holds_under_decays_near_zero_interpreted():
    """The fast backward's log decay gradient, interpreted, as near float64 as the port's under decays near
    zero on a draw: a chunk's ranges sum each token's terms before it (`mainloop.gradients`), never a token's
    own order-one term added and taken away, which rounded away the `exp(-30)` terms before it (65 to 169
    times the port's distance on three draws, the H100's check 1 140 times, SE-numerics-h100-fd61e11eeb;
    0.98 to 1.00 times after), held on one draw by the gate's one-draw rule. On a GPU whose launches do not
    hold the fast backward (an mma.sync one's) the rule refuses the gradient (row 1926)."""
    regime = 'decays near zero'
    operands, (out, final) = cotangents(HOOKS, regime, 0)
    drawn = (bf16(operands, HOOKS), (out.astype(jnp.bfloat16), final))
    template = with_gradients(functools.partial(chunk, HOOKS, mainloop='mosaic'))
    if not fast_backward(restarting(lambda given: given, regime)(drawn[0])):
        with pytest.raises(NotImplementedError):
            jax.eval_shape(template, *drawn)
        return
    port = with_gradients(functools.partial(chunk_on, 'fla', HOOKS))
    with jax.enable_x64(new_val=True):
        exact = jax.jit(with_gradients(functools.partial(token_by_token, HOOKS)))(*widened(drawn))

    def leaf(fn):
        return np.asarray(jax.jit(fn)(*drawn)[1].log_decay)

    assert_as_exact_as_the_reference(leaf(template), leaf(port), np.asarray(exact[1].log_decay),
                                     f'the template, {regime}, its log decay gradient on a draw')
