"""The recurrence template's Mosaic GPU mainloop's gradient (`mainloop.gradients`, its fast backward)
through the chunked op's explicit route, as test_recurrence_mosaic holds its forward: check 1 of the rule's
gradient against its reference's, both from float64, over a GPU's cases and, interpreted off one, the cases
that cover each launch (`INTERPRETED_GRADIENT`). A file of its own: interpreted, the forward's and the
gradient's checks together took 37 minutes in one process, past a CI task's deadline."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import pytest
from reference_error import assert_as_exact, with_gradients
from test_recurrence_chunk import chunk_on, token_by_token
from test_recurrence_mosaic import DTYPES, SEEDS, SHAPES, draw, geometry, interpreted, on_the_mainloop

from dew.nn.kernels import recurrence
from dew.nn.kernels.recurrence_chunk import Chunk, recomputing

pytestmark = pytest.mark.kernel

INTERPRETED_GRADIENT = {('a ragged length', 'bf16', 'kept'), ('a ragged length', 'bf16', 'recomputed'),
                        ('a ragged length', 'fp32', 'kept'), ('one sequence', 'bf16', 'kept')}
"""The gradient's cases the CPU interprets: the fast backward's launches over a ragged chunk, two key blocks
and two key heads' groups, its forward's memories kept and recomputed, and an fp32 call's adjoint (the
portable math's). Interpreted, the whole grid took 5.4 times these (7828 against 1452 seconds). Packed
documents' gradient is the GPU tier's: interpreted it took 352 s, past the suite's 240-second split, and
narrower draws left its log decay unresolved within 40 seeds (row 1850); the CPU interprets their forward
(tests/test_recurrence_mosaic.py) and reads their backward decision
(tests/test_recurrence_template_stability.py's resets on chunk edges)."""

GRADIENT_CASES = [case for case, shape in SHAPES.items() if not shape.get('per_key')]
"""The cases whose gradient the mainloop takes: a decay a head's (the gated delta rule's)."""

REMATS = {'kept': (), 'recomputed': ('delta_memory',)}
"""The remat rungs a gradient case runs under: the forward's memories kept, or recomputed in the backward."""


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


@pytest.mark.parametrize(('case', 'dtype', 'remat'), interpreted(
    [(case, dtype, remat) for remat in ('kept', 'recomputed') for dtype in DTYPES for case in GRADIENT_CASES],
    INTERPRETED_GRADIENT))
def test_the_mosaic_gradient_is_as_exact_as_its_reference(case, dtype, remat):
    """The gradient of the mainloop's rule on par from float64 with its reference's over the case's draws
    (check 1, truth the recurrence a token at a time's in float64): in a 16-bit call the mainloop's fast
    backward (`mainloop.gradients`, fla's DAG) against the fla port's, whose numbers it runs, from the
    forward's memories kept or, under a remat rung that recomputes them (`recomputing`), recomputed; in an
    fp32 call the portable math's adjoint against the fp32 recurrence's. A call the template refuses is
    checked refused and its gradient on the mainloop's rule (`on_the_mainloop`)."""
    hooks = recurrence.Hooks(decay='scalar', scale=SHAPES[case]['key_width'] ** -0.5)

    # Plain functions: the gate compiles both arms alike (`reference_error._arms`).
    def compiled(fn):
        def traced(*arguments):
            with recomputing(REMATS[remat]):
                return with_gradients(restarted(fn, case))(*arguments)

        return traced

    truth = with_gradients(restarted(functools.partial(token_by_token, hooks), case))
    reference = truth if dtype == jnp.float32 else compiled(functools.partial(chunk_on, 'fla', hooks))
    on = on_the_mainloop(hooks, case, dtype)
    if on is None:
        return
    assert_as_exact(compiled(on), reference,
                    functools.partial(drawn, dtype, **geometry(case)),
                    f'the mosaic gradient, {case} in {jnp.dtype(dtype).name}', seeds=SEEDS, truth_fn=truth)
