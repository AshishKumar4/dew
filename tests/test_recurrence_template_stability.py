"""The recurrence template's stability on the path a call takes: its Mosaic GPU mainloop
(`template.mosaic_rule`, the template named to the selectors, `chunk_on('mosaic_gpu', ...)`), forward
and fast backward, over every form the stability suite holds (test_recurrence_stability's `SITES`: the gated
delta rule a decay a head and a key dimension, a write gate without the delta rule, the gated delta net's
formed gates and Mamba2's rated decay) under its regimes (`REGIMES`: decays near one and near zero, the
decay's bound, aligned keys with the write gate near one, resets on chunks' edges, saturated and spread gates,
long and fading memories, steep then mild decays), held as test_recurrence_stability holds the portable math:

- finite wherever the recurrence a token at a time in float64 is;
- check 1 (`assert_as_exact`): on par from float64 with its reference arm, the fla port where it runs the
  form (the delta rule with a decay), else the portable math, in the call's 16-bit dtype. Mamba2's rated
  decay runs under both forms (`forms_of`): the accurate ones held to the portable math, the fast ones, its
  16-bit numbers mamba_ssm's (`chunk_forward.prepared`), to the port of mamba_ssm's kernels, as recorded
  mamba_ssm holds them at the same inputs (test_mamba_ssd).

Natively on a GPU of a family the mainloop claims, in its GPU flow (tests/test_gpu_recurrence.py: interpreted,
one case's check 1 outran the CPU tier's hour); off one, the CPU interprets each form's forward on the regimes
that grow its tiles most for finiteness, and its fast backward's decay gradient under decays near zero. A form
or gradient the template refuses (`Mosaic.eligible`, `Mosaic.trains`) is checked refused and skipped, a
refusal and not numerical coverage (row 1968: the suite held only the portable arms)."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import reference_error as gate
from fla_sites import LEAVES, RECORDED, the_template_is_as_exact_as_fla
from reference_error import FEWEST_SEEDS, assert_as_exact, recording, widened, with_gradients
from test_recurrence_chunk import chunk_on, token_by_token
from test_recurrence_mosaic import REFUSED, refuses_its_gradient, taken, template_call
from test_recurrence_stability import GRADIENT_SEEDS, REGIMES, SEEDS, SITES, cotangents, restarting, stressed

from dew.nn.kernels import generation
from dew.nn.kernels.recurrence_chunk import Chunk
from dew.nn.kernels.templates import bundles
from dew.nn.kernels.templates.recurrence import mainloop, template

pytestmark = pytest.mark.kernel

HOOKS = SITES['scalar delta']
"""The template's default call: the gated delta rule, a decay a head."""

def forms_of(hooks) -> tuple[str | None, ...]:
    """The run's forms a form's checks run under: the rated scan's fast and accurate both, whose numbers
    differ (mamba_ssm's rounding, `chunk_forward.prepared`, and three fp32 parts); the accurate ones for a
    form whose queries and keys its l2 norm widens, an fp32 call, whose fast numbers take its operands in two
    bf16 parts past TF32's bits (`mainloop.FAST_PARTS`), held to recorded fla-core's TF32 by
    test_recurrence_mosaic and not to an fp32 arm (the gated net's final memory 42 to 52 times its fp32 port's
    distance from float64, interpreted and on an H100, row 3962); any other form's the run's own (None), which
    its numbers do not read."""
    if hooks.rated:
        return ('fast', 'accurate')
    return ('accurate',) if hooks.l2norm else (None,)


CASES = [(name, regime, product, forms) for name in SITES for forms in forms_of(SITES[name])
         for regime in REGIMES for product in ('forward', 'gradient')]
"""Every form's regimes, forward and gradient, under each of its forms: the GPU flow's alone
(tests/test_gpu_recurrence.py)."""

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
                                 operands.beta, operands.state, operands.starts, bound=HOOKS.inverse_bound)
    tiles = template.MOSAIC.derived(bound, target)
    return template.backward_holds(operands.query, operands.key, operands.value, operands.log_decay,
                                   operands.beta, operands.state, tiles=tiles,
                                   family=mainloop.family_of() or bundles.INTERPRETED, target=target,
                                   starts=operands.starts)


def assert_finite(fn, truth, operands, regime: str) -> None:
    """`fn`'s leaves finite wherever the float64 truth's are, on `operands`."""
    mine = jax.tree.leaves(jax.jit(fn)(*operands))
    with jax.enable_x64(new_val=True):
        exact = jax.tree.leaves(jax.jit(truth)(*widened(operands)))
    for ours, theirs in zip(mine, exact, strict=True):
        finite = np.isfinite(np.asarray(theirs))
        assert np.all(np.isfinite(np.asarray(ours, np.float64))[finite]), (
            f'{regime}: not finite where float64 is')


def reference_arm(hooks, forms: str | None):
    """The arm the template is held against for a form under `forms`, each in the call's dtype, where it runs
    here: the fla port for the gated delta net's formed gates, the port of mamba_ssm's kernels for the rated
    scan under the fast forms (its numbers mamba_ssm's), else the portable math. The delta rule with a decay
    on its operands as given is held to recorded fla-core (`fla_recorded`)."""
    if hooks.delta and hooks.decay != 'none':
        return functools.partial(chunk_on, 'fla', hooks)
    return functools.partial(chunk_on, 'pallas' if hooks.rated and forms == 'fast' else 'xla', hooks)


def fla_recorded(hooks) -> bool:
    """Whether a form's reference is recorded fla-core (`fla_sites`): the delta rule with a decay on its
    operands as given, no gates formed in the rule."""
    return hooks.delta and hooks.decay != 'none' and hooks.gates is None


def under(forms: str | None):
    """The run's numerics under `forms`, the run's own where None."""
    numerics = generation.numerics()
    return generation.run_precision(numerics if forms is None else dataclasses.replace(numerics, forms=forms))


def template_trains(operands: Chunk, hooks) -> bool:
    """Whether the template's fast backward takes a gradient over `operands` under `hooks` on the device its
    launches run for (`Mosaic.trains`)."""
    return template.MOSAIC.trains(template_call(operands, hooks), template.Mosaic.launched())


def runs(fn, hooks, operands: Chunk, drawn, product: str) -> bool:
    """Whether the template runs the case `fn` traces over `drawn`: a form it does not take (`taken`), or a
    gradient its fast backward does not (`Mosaic.trains`: an fp32 call's, a decay a key dimension's, or one
    whose launches the device does not hold), checked refused as `fn` traces, where it does not."""
    if not taken(operands, hooks):
        with pytest.raises(REFUSED):
            jax.eval_shape(fn, *drawn)
        return False
    if product == 'gradient' and not template_trains(operands, hooks):
        refuses_its_gradient(fn, *drawn)
        return False
    return True


def the_template_stays_finite_and_on_par_where_chunked_forms_break(name, regime, product, forms):
    """Each form's 16-bit forward through the template, or its forward and gradient, under `forms`
    (`forms_of`), finite wherever the float64 recurrence's are on the regime's first draws, and on par with
    its reference arm's from float64 over the site's declared seeds (check 1); a form or gradient it refuses
    is checked refused and skipped."""
    hooks = SITES[name]
    mosaic = functools.partial(chunk_on, 'mosaic_gpu', hooks)
    reference, exact = reference_arm(hooks, forms), functools.partial(token_by_token, hooks)
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
    with under(forms):
        if not runs(dew, hooks, operands, draws(0), product):
            pytest.skip(f"the template refuses {name}'s {product}: a refusal, not numerical coverage")
        for seed in range(2):
            assert_finite(dew, truth, draws(seed), regime)
        label = f'the template, {name} under {regime}, its {product}'
        if fla_recorded(hooks):
            the_template_is_as_exact_as_fla(mosaic, f"stability/{name}/{regime}/{product}",
                                            ('out', 'final') if product == 'forward' else LEAVES, label)
            return
        assert_as_exact(dew, reference, draws, label, seeds=seeds, truth_fn=truth)


@pytest.mark.parametrize('forms', forms_of(SITES['rated']))
def test_a_ragged_chunks_last_real_write_holds_the_final_memory_interpreted(forms):
    """The rated scan's 16-bit forward through the template under the decay bound and `forms`, interpreted
    off a GPU, on par from float64 over check 1's fewest draws with its reference arm (`reference_arm`): the
    port of mamba_ssm's kernels under the fast forms, whose numbers the template's then are, the portable
    math under the accurate ones, which hold them in fp32. Its rows end in a ragged chunk (70 tokens), and
    its last real token's write, which weighs most under a steep decay, is held outside the product as the
    portable math peels it; the padded last slot's put the final memory 2.0 times the portable math's
    distance, interpreted and on an H100."""
    hooks = SITES['rated']
    with under(forms):
        assert_as_exact(functools.partial(chunk_on, 'mosaic_gpu', hooks), reference_arm(hooks, forms),
                        lambda seed: (bf16(stressed(hooks, 'the decay bound', seed)[0], hooks),),
                        f'the template, rated under the decay bound, its forward ({forms} forms)',
                        seeds=FEWEST_SEEDS, truth_fn=functools.partial(token_by_token, hooks))


@pytest.mark.parametrize('forms', forms_of(SITES['rated']))
def test_an_fp32_rated_scan_through_the_template_stays_finite_interpreted(forms):
    """The rated scan's fp32 forward through the template under `forms`, interpreted off a GPU, finite
    wherever the float64 recurrence's is: its skip added after the walk where the template holds no 16-bit
    values (an fp32 call's two parts under the fast forms; fused there, its output launch failed the gated
    keys' check)."""
    hooks = SITES['rated']
    mosaic = functools.partial(chunk_on, 'mosaic_gpu', hooks)
    drawn = stressed(hooks, 'the decay bound', 0)
    with under(forms):
        assert runs(mosaic, hooks, drawn[0], drawn, 'forward')
        assert_finite(jax.jit(mosaic), jax.jit(functools.partial(token_by_token, hooks)), drawn,
                      'the decay bound')


@pytest.mark.parametrize(('name', 'regime', 'forms'),
                         [(name, regime, forms) for name, regime, product, forms in CASES
                          if product == 'forward' and regime in INTERPRETED])
@pytest.mark.check1_shards(seeds=256, per_task=16)
def test_each_form_the_template_takes_stays_finite_and_on_par_interpreted(name, regime, forms):
    """Each form's 16-bit forward through the template, interpreted off a GPU under the regimes that grow its
    tiles most, held as its GPU flow cell is: finite, and its output and final memory on par from float64
    with its reference arm. Held finite alone, a write gate without the delta rule rounded as the SSD is
    (`recurrence_chunk.chunk_rounding`) put its final memory 2.1e4 times the portable math's distance from
    float64 unseen off a GPU (row 3962). Whole, the cells took 27 to 151 s (Kimi delta attention's aligned
    keys 128, the rated scan's decays near zero under the accurate forms 151, stab-case at d51a2d2cf): a site
    each, sharded over the most seeds a cell declares, sixteen draws a task, where every cell's 64 a task
    took 966.6 s, Kimi delta attention's decays near zero 329 of it (train 625's CI, armada job
    20261105212801-7b615468)."""
    the_template_stays_finite_and_on_par_where_chunked_forms_break(name, regime, 'forward', forms)


@pytest.mark.parametrize('regime', REGIMES)
def test_the_template_gradient_takes_the_fast_backward_where_its_launches_hold(regime):
    """The template's gradient over each regime's draw takes its fast backward, its launches holding at the
    call's derived tiles (`fast_backward`): off a GPU by the interpreted bundle, on a GPU by its own, an
    A100's too (its keys launch a key block a program, its query/key launch's memory sum a product: their
    loops' operand fragments spilled 64 to 200 bytes a thread past its 255 registers)."""
    operands, _ = cotangents(HOOKS, regime, 0)
    assert fast_backward(restarting(lambda given: given, regime)(bf16(operands, HOOKS)))


@pytest.mark.whole_suite  # 130 s interpreted, past a group's two minutes (ledger 2145).
def test_the_fast_backward_decay_gradient_holds_under_decays_near_zero_interpreted():
    """The fast backward's log decay gradient, interpreted, as near float64 as recorded fla-core's
    (`fla_sites`) under decays near zero on a draw: a chunk's ranges sum each token's terms before it
    (`mainloop_backward.gradients`), never a token's own order-one term added and taken away, which rounded
    away the `exp(-30)` terms before it (65 to 169 times the port's distance on three draws, the H100's
    check 1 140 times, SE-numerics-h100-fd61e11eeb; 0.98 to 1.00 times after), held on one draw by the gate's
    one-draw rule, pooled (the recording keeps no rows of it). On a GPU whose launches do not hold the fast
    backward the rule refuses the gradient (row 1926)."""
    regime = 'decays near zero'
    operands, (out, final) = cotangents(HOOKS, regime, 0)
    drawn = (bf16(operands, HOOKS), (out.astype(jnp.bfloat16), final))
    template = with_gradients(functools.partial(chunk_on, 'mosaic_gpu', HOOKS))
    if not fast_backward(restarting(lambda given: given, regime)(drawn[0])):
        refuses_its_gradient(template, *drawn)
        return
    with jax.enable_x64(new_val=True):
        exact = jax.jit(with_gradients(functools.partial(token_by_token, HOOKS)))(*widened(drawn))
    mine = gate.distance(np.asarray(jax.jit(template)(*drawn)[1].log_decay), np.asarray(exact[1].log_decay))
    # fla-core's own distance on the draw, recorded (`fla_sites`): its mean squared error from float64.
    theirs = float(np.sqrt(recording(RECORDED)[f"stability/scalar delta/{regime}/gradient.mse.dg"][0]))
    assert gate.within(mine, gate.FACTOR * theirs), (
        f"the template, {regime}, its log decay gradient on a draw: {mine:.3e} from float64 (rms), recorded "
        f"fla-core {theirs:.3e} (ratio {mine / theirs:.2f}, allowed {gate.FACTOR})")
