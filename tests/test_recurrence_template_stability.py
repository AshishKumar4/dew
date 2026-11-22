"""The recurrence template's stability on the path a call takes: its Mosaic GPU mainloop
(`template.mosaic_rule`, the template named to the selectors, `chunk_on('mosaic_gpu', ...)`), forward
and fast backward, over every form the stability suite holds (test_recurrence_stability's `SITES`: the gated
delta rule a decay a head and a key dimension, a write gate without the delta rule, the gated delta net's
formed gates and Mamba2's rated decay) under its regimes (`REGIMES`: decays near one and near zero, the
decay's bound, aligned keys with the write gate near one, resets on chunks' edges, saturated and spread gates,
long and fading memories, steep then mild decays), held as test_recurrence_stability holds the portable math:

- finite wherever the recurrence a token at a time in float64 is;
- check 1 (`assert_as_exact`): on par from float64 with its reference arm in the call's 16-bit dtype, its
  family's independent oracle (docs/design/testing.md): recorded fla-core where it runs the form (the delta
  rule with a decay; a write gate without it under a decay a head, fla's `chunk_simple_gla` on its keys
  gated, whose chunked state gradient takes the chunk's `q^T do` in one product as the template's does, 1.57
  times the token loop's distance, row 6755), the recurrence a token at a time where no fla release runs it,
  its distance from the portable math reported beside the verdict. Mamba2's rated
  decay runs under both forms (`forms_of`): the accurate ones held to the portable math, the fast ones, its
  16-bit numbers mamba_ssm's (`chunk_forward.prepared`), to the port of mamba_ssm's kernels, as recorded
  mamba_ssm holds them at the same inputs (test_mamba_ssd).

Natively on a GPU of a family the mainloop claims, in its GPU flow (tests/test_gpu_recurrence.py: interpreted,
one case's check 1 outran the CPU tier's hour); off one, the CPU interprets each form's forward on the regimes
that grow its tiles most for finiteness, and its fast backward's decay gradient under decays near zero. A form
or gradient the template refuses (`Mosaic.eligible`, `Mosaic.trains`) is checked refused and skipped, a
refusal and not numerical coverage (row 1968: the suite held only the portable arms)."""

import contextlib
import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fla_sites import FAINT_CROSS, FLOOR_GATES, LEAVES, the_template_is_as_exact_as_fla
from generation_support import COMPILES_ON_CPU
from map_judge import sharded
from reference_error import FEWEST_SEEDS, assert_as_exact, widened, with_gradients
from test_recurrence_chunk import against_the_portable_math, chunk_on, oracle, token_by_token
from test_recurrence_mosaic import REFUSED, refuses_its_gradient, taken, template_call
from test_recurrence_stability import (
    GRADIENT_SEEDS,
    REGIMES,
    SEEDS,
    SITES,
    bounded,
    cotangents,
    restarting,
    stressed,
)

from dew.nn.kernels import generation
from dew.nn.kernels.recurrence_chunk import Chunk
from dew.nn.kernels.templates.recurrence import template

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
inverse's fp16 doubling (aligned keys, the write gate near one) and its pairs' clip (`algebra.between`'s
and `algebra.decay`'s, zero from a wholes' difference of -127) over chunks spanning tens of e-folds."""


def bf16(operands: Chunk, hooks) -> Chunk:
    """`operands` as a 16-bit call under `hooks` takes them: its queries, keys and values in bf16, its write
    gate in bf16 with the delta rule (as fla's layer holds it) and fp32 without (Mamba-2's step, which the
    template takes unrounded), its log decay and memory fp32."""
    names = ('query', 'key', 'value', 'beta') if hooks.delta else ('query', 'key', 'value')
    return dataclasses.replace(operands, **{name: getattr(operands, name).astype(jnp.bfloat16)
                                            for name in names})


FINITE_SEEDS = 2
"""The first draws a site holds finite wherever the float64 truth is (`assert_finite`), before its check 1."""


def assert_finite(fn, truth, operands, regime: str) -> None:
    """`fn`'s leaves finite wherever the float64 truth's are, on `operands`."""
    mine = jax.tree.leaves(jax.jit(fn)(*operands))
    with jax.enable_x64(new_val=True):
        exact = jax.tree.leaves(jax.jit(truth)(*widened(operands)))
    for ours, theirs in zip(mine, exact, strict=True):
        finite = np.isfinite(np.asarray(theirs))
        assert np.all(np.isfinite(np.asarray(ours, np.float64))[finite]), (
            f'{regime}: not finite where float64 is')


def reference_arm(hooks, forms: str | None, **bound):
    """The arm the template is held against for a form under `forms`, each in the call's 16-bit dtype, its
    family's oracle under the run's numerics (`oracle`): the fla port for the delta rule (the gated delta
    net's formed gates; with a decay on its operands as given, recorded fla-core, `fla_recorded`); for the
    rated scan the port of mamba_ssm's kernels under the fast forms (its numbers mamba_ssm's) and the
    portable math under the accurate ones (row 6781: recorded mamba_ssm, which waits on the fixture store);
    the recurrence a token at a time, in fp32 with its output in the call's dtype, where no release computes
    the call's numbers. `bound` is a chunked arm's (`bounded`): the recurrence reads its documents' starts a
    token at a time."""
    upstream = oracle(hooks, jnp.bfloat16)
    if upstream is None:
        return functools.partial(token_by_token, hooks)
    if upstream == 'fla':
        return functools.partial(chunk_on, 'fla', hooks, **bound)
    return functools.partial(chunk_on, 'pallas' if forms == 'fast' else 'xla', hooks, **bound)


def fla_recorded(hooks) -> bool:
    """Whether a form's reference is recorded fla-core (`fla_sites`): the delta rule with a decay, or a write
    gate without it under a decay a head (`chunk_simple_gla` on its keys gated), on its operands as given, no
    gates formed in the rule."""
    held = (hooks.delta and hooks.decay != 'none') or (hooks.write_gate and hooks.decay == 'scalar')
    return held and hooks.gates is None and not hooks.rated


@contextlib.contextmanager
def under(forms: str | None):
    """The run's numerics under `forms`, the run's own where None; the fast forms at the default product
    precision they round at (`rounds_products`; the suite's lanes configure 'highest')."""
    numerics = generation.numerics()
    with (generation.run_precision(numerics if forms is None else dataclasses.replace(numerics, forms=forms)),
          jax.default_matmul_precision('default') if forms == 'fast' else contextlib.nullcontext()):
        yield


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


RESTARTED = 'resets on chunk edges'
"""The one regime whose arms are its own (`_arms`): its documents restart on chunk edges (`restarting`), its
chunked arms bounded by them (`bounded`)."""


@functools.cache
def _arms(name: str, regime: str, product: str, forms: str) -> tuple:
    """A case's arms (`the_template_stays_finite_and_on_par_where_chunked_forms_break`'s): the template's,
    its reference arm, the recurrence a token at a time and the portable math's, their gradients' where
    `product` is; one function each for every regime but `RESTARTED`, which run them alike, so a form's
    regimes of one shape trace its arms once, where each case built and traced its own (the GPU tier's 120
    traces of the template's at assert_finite)."""
    if regime != RESTARTED:
        regime = next(other for other in REGIMES if other != RESTARTED)
    hooks = SITES[name]
    mosaic, portable = (functools.partial(chunk_on, kernel, hooks, **bounded(regime))
                        for kernel in ('mosaic_gpu', 'xla'))
    exact = functools.partial(token_by_token, hooks)
    with under(forms):
        reference = reference_arm(hooks, forms, **bounded(regime))
    if product == 'gradient':
        return tuple(with_gradients(restarting(fn, regime)) for fn in (mosaic, reference, exact, portable))
    return mosaic, reference, exact, portable


def the_template_stays_finite_and_on_par_where_chunked_forms_break(name, regime, product, forms):
    """Each form's 16-bit forward through the template, or its forward and gradient, under `forms`
    (`forms_of`), finite wherever the float64 recurrence's are on the regime's first draws, and on par with
    its reference arm's from float64 over the site's declared seeds (check 1); a form or gradient it refuses
    is checked refused and skipped."""
    hooks = SITES[name]
    if product == 'forward':
        seeds = SEEDS[name, regime]

        def draws(seed):
            return (bf16(stressed(hooks, regime, seed)[0], hooks),)
    else:
        seeds = GRADIENT_SEEDS[name, regime]

        def draws(seed):
            operands, (out, final) = cotangents(hooks, regime, seed)
            return bf16(operands, hooks), (out.astype(jnp.bfloat16), final)

    operands = restarting(lambda given: given, regime)(draws(0)[0])
    mosaic = _arms(name, regime, 'forward', forms)[0]
    dew, reference, truth, portable = _arms(name, regime, product, forms)
    with under(forms):
        if not runs(dew, hooks, operands, draws(0), product):
            pytest.skip(f"the template refuses {name}'s {product}: a refusal, not numerical coverage")
        label = f'the template, {name} under {regime}, its {product}'

        def finite(start: int, stop: int) -> None:
            for seed in range(start, stop):
                assert_finite(dew, truth, draws(seed), regime)

        # Its first draws held finite once a site: by the map task that draws them, the judge pooling its
        # word (`sharded`), where every map task and the judge held them again, each a compile and two runs
        # of every case's arms, Kimi delta attention's about 8 s a run interpreted (row 5906).
        sharded(f'{label}, finite', finite, FINITE_SEEDS, upto=FINITE_SEEDS)
        if fla_recorded(hooks):
            the_template_is_as_exact_as_fla(mosaic, f"stability/{name}/{regime}/{product}",
                                            ('out', 'final') if product == 'forward' else LEAVES, label)
            return
        if oracle(hooks, jnp.bfloat16) is None and (distance := against_the_portable_math(
                label, dew, portable, draws, truth, FEWEST_SEEDS)) is not None:
            print(f"{label}: its distance against the portable math's: {distance}", flush=True)
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


CHECK1_SHARDS = {
    "test_each_form_the_template_takes_stays_finite_and_on_par_interpreted[gated-aligned_keys-accurate]":
        {"seeds": 128, "per_task": 32},
    "test_each_form_the_template_takes_stays_finite_and_on_par_interpreted[scalar_delta-decays_near_zero]":
        {"seeds": 64, "per_task": 32},
    "test_each_form_the_template_takes_stays_finite_and_on_par_interpreted[vector_delta-decays_near_zero]":
        {"seeds": 64, "per_task": 16},
    "test_each_form_the_template_takes_stays_finite_and_on_par_interpreted[scalar_write-decays_near_zero]":
        {"seeds": 64, "per_task": 32},
    "test_each_form_the_template_takes_stays_finite_and_on_par_interpreted[gated-decays_near_zero-accurate]":
        {"seeds": 128, "per_task": 32},
    "test_each_form_the_template_takes_stays_finite_and_on_par_interpreted[rated-decays_near_zero-fast]":
        {"seeds": 256, "per_task": 64},
    "test_each_form_the_template_takes_stays_finite_and_on_par_interpreted[rated-decays_near_zero-accurate]":
        {"seeds": 256, "per_task": 64},
}
"""The interpreted forwards whose check-1 seeds run as map tasks, by node id, a literal the CI plan reads as
it reads a mark's (tools/armada/ci.py): their seeds and a task's share (`interpreted_cases`)."""


def interpreted_cases(test):
    """`test` parametrized by `CASES`' forwards under the regimes the CPU interprets (`INTERPRETED`), each
    marked to run its seeds as map tasks where `CHECK1_SHARDS` names its node, by `test`'s own name: a form
    or regime `SITES`, `forms_of` or `INTERPRETED` adds reaches the CPU cell (rows 9716, 11837)."""
    cases = []
    for name, regime, product, forms in CASES:
        if product != 'forward' or regime not in INTERPRETED:
            continue
        ident = '-'.join(part.replace(' ', '_') for part in (name, regime, forms) if part is not None)
        held = CHECK1_SHARDS.get(f"{test.__name__}[{ident}]")
        cases.append(pytest.param(name, regime, forms, id=ident,
                                  marks=[pytest.mark.check1_shards(**held)] if held else []))
    return pytest.mark.parametrize(('name', 'regime', 'forms'), cases)(test)


@interpreted_cases
def test_each_form_the_template_takes_stays_finite_and_on_par_interpreted(name, regime, forms):
    """Each form's 16-bit forward through the template, interpreted off a GPU under the regimes that grow its
    tiles most, held as its GPU flow cell is: finite, and its output and final memory on par from float64
    with its reference arm. Held finite alone, a write gate without the delta rule rounded as the SSD is
    (`recurrence_chunk.chunk_rounding`) put its final memory 2.1e4 times the portable math's distance from
    float64 unseen off a GPU (row 3962). Whole, the cells took 27 to 151 s (Kimi delta attention's aligned
    keys 128, the rated scan's decays near zero under the accurate forms 151, stab-case at d51a2d2cf): a site
    each, sharded over the most seeds a cell declares, eight draws a task, where every cell's 64 a task
    took 966.6 s, Kimi delta attention's decays near zero 329 of it (train 625's CI, armada job
    20261105212801-7b615468), and a cell's sixteen 638 s, past a train's bound (train de27b8e32's CI, row
    7458)."""
    the_template_stays_finite_and_on_par_where_chunked_forms_break(name, regime, 'forward', forms)


ORDINARY = tuple(regime for regime, (_, tokens) in REGIMES.items()
                 if tokens is None and regime not in INTERPRETED)
"""The regimes of a draw's own tokens beside `INTERPRETED`'s, which the CPU interprets each form's forward
under too (`test_each_form_the_template_takes_stays_on_par_in_its_ordinary_regimes_interpreted`):
a long sequence's and a fading memory's (`LONG` tokens) and decays near zero's (`NEAR_ZERO`) are past a task
interpreted, and their gradients everywhere, which the GPU tier runs whole (tests/test_gpu_recurrence.py's
`STRESS`, row 5461's ruling)."""


SHARDED_ORDINARY = (('vector delta', 'resets on chunk edges'), ('vector delta', 'the decay bound'),
                    ('rated', 'resets on chunk edges'))
"""The ordinary regimes' cases past a task whole, each a check-1 site of its own recorded seeds over map
tasks: Kimi delta attention's 16 seeds of 23 interpreted chunks (its 18 documents laid apart) at 10 s a draw,
and its 64 seeds under the decay bound at 2.2 s (179 and 146 s whole at CI's rate, train of 2026-11-14
05:20Z); the rated scan's 64 seeds under resets on chunk edges, each of its forms (121 s whole under the
accurate ones, train 861's CI). Their marks are literal parameters (`pytest.param`), as the plan reads a
site's from the source."""


@pytest.mark.parametrize(('name', 'regime', 'forms'), [
    *((name, regime, forms) for name, regime, product, forms in CASES
      if product == 'forward' and regime in ORDINARY and (name, regime) not in SHARDED_ORDINARY),
    pytest.param('vector delta', 'resets on chunk edges', None, id='vector_delta-resets_on_chunk_edges',
                 marks=pytest.mark.check1_shards(seeds=16, per_task=8)),
    pytest.param('vector delta', 'the decay bound', None, id='vector_delta-the_decay_bound',
                 marks=pytest.mark.check1_shards(seeds=64, per_task=32)),
    pytest.param('rated', 'resets on chunk edges', 'fast', id='rated-resets_on_chunk_edges-fast',
                 marks=pytest.mark.check1_shards(seeds=64, per_task=32)),
    pytest.param('rated', 'resets on chunk edges', 'accurate', id='rated-resets_on_chunk_edges-accurate',
                 marks=pytest.mark.check1_shards(seeds=64, per_task=32))])
def test_each_form_the_template_takes_stays_on_par_in_its_ordinary_regimes_interpreted(name, regime, forms):
    """`test_each_form_the_template_takes_stays_finite_and_on_par_interpreted` under the regimes of a draw's
    own tokens (`ORDINARY`): the CPU twin of each form's forward cell of the GPU tier's, each case a test of
    its own seeds, those past a task sharded apart (`SHARDED_ORDINARY`). Sharded together over the most seeds
    a case declares, every map task set up all 36 cases and drew each one's second stage past its own seeds
    whatever its first decided: 590 of a 714 s task's seconds drawing (train of 2026-11-14 05:20Z)."""
    the_template_stays_finite_and_on_par_where_chunked_forms_break(name, regime, 'forward', forms)


def compiled_gradient(hooks, name: str) -> list[dict]:
    """The template's bf16 gradient under `hooks` at the suite's narrow widths, compiled for a compile-only
    H100 as a start compiles it (`kernel_compile.compile_mosaic_kernels`: each launch's derived config
    certified as it traces, `report.certified`), its records."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels

    operands, (d_out, d_final) = cotangents(hooks, 'decays near one', 0)
    operands = bf16(operands, hooks)

    def loss(operands):
        out, final = chunk_on('mosaic_gpu', hooks, operands)
        return jnp.sum(out * d_out) + jnp.sum(final * d_final)

    def call():
        return jax.grad(loss), (jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), operands),)

    return compile_mosaic_kernels([KernelCase(name, call)], 'sm90')


def kimis_unit_keys_under_tiny_gates_at_the_floor_are_as_exact_as_fla():
    """Kimi delta attention's 16-bit gradient through the template under decays at its floor and write gates
    down to 2^-60 (fla_sites' `FLOOR_GATES`): its output, final memory and every operand's gradient on par
    with recorded fla-core's from float64 (check 1, 512 seeds), run natively in `test_gpu_recurrence`
    (interpreted off a GPU at 3 draws a task took 171 CI tasks / ~14.6 CPU-h, row 8835)."""
    mosaic = _arms('vector delta', 'the decay bound', 'forward', None)[0]
    the_template_is_as_exact_as_fla(mosaic, FLOOR_GATES, LEAVES,
                                    'the template, Kimi delta attention under tiny gates at the floor')


@COMPILES_ON_CPU
def test_kimis_unit_keys_under_tiny_gates_at_the_floor_stays_finite_interpreted():
    """Kimi delta attention's 16-bit gradient under `FLOOR_GATES`, interpreted off a GPU on `FINITE_SEEDS`
    draws: every leaf finite wherever float64 is (its 512-seed check 1 runs natively in
    `test_gpu_recurrence.template_stability_cells`)."""
    from fla_sites import chunk_of, sites

    site = sites()[FLOOR_GATES]
    mosaic = _arms('vector delta', 'the decay bound', 'forward', None)[0]
    truth = functools.partial(token_by_token, site.hooks)
    for seed in range(FINITE_SEEDS):
        operands = chunk_of(site.arrays(seed))
        assert_finite(with_gradients(mosaic), with_gradients(truth), operands, FLOOR_GATES)


@COMPILES_ON_CPU
@pytest.mark.check1_shards(seeds=8, per_task=2)
def test_kimis_faint_cross_terms_are_as_exact_as_fla_interpreted():
    """Kimi delta attention's 16-bit gradient through the template, interpreted, over one key under faint
    write gates beside large ones and no decay (fla_sites' `FAINT_CROSS`): its output, final memory and
    every operand's gradient on par with recorded fla-core's from float64 (check 1). A token's own scores
    pair taken out of the log decay's terms after they were summed in fp32 lost the faint cross term
    beside it (base review 29b07fb60 F1, row 7906), which a cell held bitwise outside the gate (row 8569)."""
    mosaic = _arms('vector delta', 'the decay bound', 'forward', None)[0]
    the_template_is_as_exact_as_fla(mosaic, FAINT_CROSS, LEAVES,
                                    'the template, Kimi delta attention under faint cross terms')


@COMPILES_ON_CPU
def test_a_gradients_kernels_are_each_certified_as_its_program_lowers():
    """The gated delta net's bf16 gradient compiled for a compile-only H100 as a start compiles it
    (`compiled_gradient`): every Mosaic GPU kernel its program runs, its training forward's and its fast
    backward's, is one certification compiled alone, its launch certified as reverse mode runs it
    (`report.pulled`), so a spilling kernel of it refuses its config there, not in a retrace of the whole
    program (row 5905)."""
    record, = compiled_gradient(SITES['gated'], 'the gated delta net gradient')
    assert record["modules"] and set(record["modules"]) <= set(record["certified"]), record


@COMPILES_ON_CPU
def test_the_narrow_gated_delta_gradient_compiles_for_an_h100():
    """The gated delta rule's bf16 gradient at the suite's narrow widths (16-wide keys, 8-wide values, its
    gradient arms' heads), compiled for an H100 on this CPU (`compiled_gradient`) with no kernel failing: its
    values' backward, 8 programs a multiprocessor at its model, launched each thread at 32 registers, under
    the 34 its 16-column warpgroup MMAs assemble at (`Issuer.entry_floor`, M-JS-63)."""
    from kernel_compile import mosaic_failures

    assert mosaic_failures(compiled_gradient(HOOKS, 'the narrow gated delta gradient')) == []
