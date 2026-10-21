"""Scratch: size test_recurrence_chunk's sites (not committed)."""
import functools

import jax
import pytest
from reference_error import seeds_for

import test_recurrence_chunk as t


@pytest.mark.parametrize('packed', t.PACKED)
@pytest.mark.parametrize('hooks', t.HOOKS, ids=str)
def test_size(hooks, packed):
    reference = jax.jit(functools.partial(t.token_by_token, hooks))
    n = seeds_for(reference, functools.partial(t.draw, hooks, packed), f'{hooks} {packed}')
    print(f'SIZE|{t.HOOKS.index(hooks)}|{packed}|{n}', flush=True)
