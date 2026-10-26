"""The recurrence template's stability on its default path: the gated delta rule's 16-bit forward and fast
backward (`recurrence_chunk.mosaic_rule` through `chunk(..., mainloop='mosaic')`, an H100's training), under
the regimes that break chunked forms (test_recurrence_stability's `REGIMES`: decays near zero, steep then
mild, aligned keys with the write gate near one, resets on chunks' edges, a long sequence), held two ways as
test_recurrence_stability holds the portable math:

- finite wherever the recurrence a token at a time in float64 is;
- check 1 (`assert_as_exact`): on par from float64 with its reference arm, the fla port's
  (`chunk_on('fla', ...)`), whose numbers it runs.

Natively on a GPU of a family the mainloop claims, interpreted off one at the cases `INTERPRETED` names. The
gradient's arm asserts which backward ran: the fast backward (its reverse launch in the trace) where its
launches hold (`mainloop.backward_holds`), and on an mma.sync GPU, where they do not, the port's, whose
gradient its reference arm computes too, so check 1 there holds the forward alone."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, equations, widened, with_gradients
from test_recurrence_chunk import chunk_on, token_by_token
from test_recurrence_mosaic import NATIVE
from test_recurrence_stability import SEEDS, SITES, cotangents, restarting, stressed

from dew.nn.kernels import generation
from dew.nn.kernels.recurrence_chunk import Chunk, chunk

pytestmark = pytest.mark.kernel

HOOKS = SITES['scalar delta']
"""The template's default call: the gated delta rule, a decay a head."""

REGIMES = ('decays near zero', 'steep then mild', 'aligned keys', 'resets on chunk edges', 'a long sequence')
"""The regimes where the template's tile algebra departs from its reference arm's: `_scaled`'s clip and
`PAIR_FLOOR` (decays near zero, steep then mild), the inverse's fp16 doubling (aligned keys, the write gate
near one), the walk's resets (resets on chunks' edges) and its carried memory (a long sequence)."""

INTERPRETED = {('aligned keys', 'forward'), ('resets on chunk edges', 'gradient')}
"""The cases the CPU interprets: the inverse's fp16 doubling under aligned keys, and the fast backward's
launches over documents reset on chunks' edges; the rest are the GPU tier's (`NATIVE`)."""

CASES = [pytest.param(regime, product, id=f'{regime}-{product}',
                      marks=() if (regime, product) in INTERPRETED else NATIVE)
         for regime in REGIMES for product in ('forward', 'gradient')]


def bf16(operands: Chunk) -> Chunk:
    """`operands` as a 16-bit call takes them: its queries, keys, values and write gate in bf16, its log decay
    and memory fp32."""
    return dataclasses.replace(operands, **{name: getattr(operands, name).astype(jnp.bfloat16)
                                            for name in ('query', 'key', 'value', 'beta')})


def launched(fn, *args) -> set[str]:
    """The names of the Pallas launches `fn`'s trace over `args` holds."""
    graph = jax.make_jaxpr(fn)(*args)
    return {str(equation.params.get('name') or equation.params.get('name_and_src_info'))
            for equation in equations(graph) if equation.primitive.name == 'pallas_call'}


def assert_finite(fn, truth, operands, regime: str) -> None:
    """`fn`'s leaves finite wherever the float64 truth's are, on `operands`."""
    mine = jax.tree.leaves(fn(*operands))
    with jax.enable_x64(new_val=True):
        exact = jax.tree.leaves(truth(*widened(operands)))
    for ours, theirs in zip(mine, exact, strict=True):
        finite = np.isfinite(np.asarray(theirs))
        assert np.all(np.isfinite(np.asarray(ours, np.float64))[finite]), (
            f'{regime}: not finite where float64 is')


@pytest.mark.parametrize('regime, product', CASES)
def test_the_template_stays_finite_and_on_par_with_the_port_where_chunked_forms_break(regime, product):
    """The template's 16-bit forward, or its forward and gradient, finite wherever the float64 recurrence's
    are on the regime's first draws, and on par with the port's from float64 over its declared seeds
    (check 1); its gradient the fast backward's where its launches hold, the port's on an mma.sync GPU."""
    template = functools.partial(chunk, HOOKS, mainloop='mosaic')
    port = functools.partial(chunk_on, 'fla', HOOKS)
    exact = functools.partial(token_by_token, HOOKS)
    if product == 'forward':
        dew, reference, truth = (jax.jit(fn) for fn in (template, port, exact))

        def draws(seed):
            return (bf16(stressed(HOOKS, regime, seed)[0]),)
    else:
        dew, reference, truth = (jax.jit(with_gradients(restarting(fn, regime)))
                                 for fn in (template, port, exact))

        def draws(seed):
            operands, (out, final) = cotangents(HOOKS, regime, seed)
            return bf16(operands), (out.astype(jnp.bfloat16), final)

        fast = generation.family(generation.device_generation()) != 'mma.sync'
        reversed_ = any('recurrence_reverse' in name for name in launched(dew, *draws(0)))
        assert reversed_ == fast, f'{regime}: the fast backward ran: {reversed_}, expected {fast}'
        if not fast:
            # Both arms take the port's gradient: check 1 holds the forward alone (its own case).
            assert_finite(dew, truth, draws(0), regime)
            return
    for seed in range(2):
        assert_finite(dew, truth, draws(seed), regime)
    assert_as_exact(dew, reference, draws, f'the template, {regime}, its {product}',
                    seeds=SEEDS['scalar delta', regime], truth_fn=truth)
