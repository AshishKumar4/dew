"""Scratch: size test_recurrence_chunk's sites (not committed)."""
import functools

import jax
import pytest
from reference_error import seeds_for

import test_recurrence_chunk as t

SITES = [(hooks, packed) for hooks in t.HOOKS for packed in t.PACKED]


@pytest.mark.parametrize('index', range(len(SITES)), ids=[f's{i}' for i in range(len(SITES))])
def test_size(index):
    hooks, packed = SITES[index]
    reference = jax.jit(functools.partial(t.token_by_token, hooks))
    n = seeds_for(reference, functools.partial(t.draw, hooks, packed), f'{hooks} {packed}')
    raise AssertionError(f'SIZE|{index}|{n}')
