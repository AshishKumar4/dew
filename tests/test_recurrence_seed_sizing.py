"""Offline declarations for the recurrence accuracy gate, sized on armada."""

import functools

import jax
import jax.numpy as jnp
import pytest
from reference_error import seeds_for
from test_recurrence import HOOKS, draw, one_tap_draw

from dew.nn.kernels import recurrence


@pytest.mark.parametrize('group', range(12))
def test_hook_seed_counts(group):
    counts = []
    start = 40 * group
    for index, hooks in enumerate(HOOKS[start:start + 40], start=start):
        reference = jax.jit(functools.partial(recurrence.step_reference, hooks))
        count = seeds_for(reference, functools.partial(draw, hooks), f'hook {index} {hooks}')
        counts.append(count)
        jax.clear_caches()
    print(f'HOOK_SEEDS {start}: {counts}', flush=True)


def test_bf16_seed_counts():
    counts = []
    for hooks in (HOOKS[0], HOOKS[-1]):
        reference = jax.jit(functools.partial(recurrence.step_reference, hooks))
        counts.append(seeds_for(reference, functools.partial(draw, hooks, state_dtype=jnp.bfloat16),
                                f'bf16 {hooks}'))
        jax.clear_caches()
    print(f'BF16_SEEDS {counts}', flush=True)


def test_one_tap_seed_counts():
    hooks = recurrence.Hooks(conv=recurrence.Conv(None), decay='none', delta=False)
    counts = {}
    for dtype in (jnp.float32, jnp.bfloat16):
        reference = jax.jit(functools.partial(recurrence.step_reference, hooks))
        inputs = functools.partial(one_tap_draw, dtype)
        count = seeds_for(reference, inputs, f'one-tap {jnp.dtype(dtype).name}')
        counts[jnp.dtype(dtype).name] = count
        jax.clear_caches()
    print(f'ONE_TAP_SEEDS {counts}', flush=True)
