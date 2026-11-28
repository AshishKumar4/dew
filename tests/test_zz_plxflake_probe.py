"""Reproduce XLA:CPU's corrupted cotangent at the first key of a peeled scan's tail."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from arm_arithmetic import summed_as_run
from arm_forms import _arms, _ynn_decided

import test_attention_parallax as tp
from dew.nn.kernels import generation
from dew.nn.kernels.templates import attention as template, attention_parallax


def _covariance():
    logits = (-128.0, -128.0) + (0.0,) * 127
    seconds = (2.0 ** 127, -(2.0 ** 127)) + (0.0,) * 127
    query, second, key, second_key, value, _ = tp._witness(logits, seconds, seconds)
    valid = jnp.zeros((1, 129), bool).at[0, jnp.asarray([0, 1, 128])].set(True)
    return (query, second, key, second_key, value[..., :1]), template.spans(1, 129, 1, valid=valid)


def _far():
    *operands, spans = tp._tiled_keys({0: (0.0, 0.0, 0.0), 128: (-20.0, 2.0 ** 100, 0.0)}, 129)
    return operands, spans


def _blocked(step, start, count):
    carry, first = step(start, 0)
    carry, rest = jax.lax.scan(step, carry, jnp.arange(1, count))
    if first is not None and generation.placed_platform() == "cpu":
        first, rest = jax.lax.optimization_barrier((first, rest))
    return carry, jax.tree.map(lambda a, b: jnp.concatenate((a[None], b)), first, rest)


@pytest.mark.parametrize("barrier", [False, True], ids=["stock", "barrier"])
def test_check_arm_repeats(monkeypatch, barrier):
    if barrier:
        monkeypatch.setattr(attention_parallax, "_tiled", _blocked)
    failures = []
    for make in (_covariance, _far):
        operands, spans = make()
        ops = (*operands, jnp.ones((1, 1, 1, 1), jnp.float32))
        run = jax.jit(tp._every_cotangent(attention_parallax.attend, spans))
        vanilla = jax.jit(tp._every_cotangent(tp._plain, spans))
        dew, ref = _arms(run, vanilla)
        with summed_as_run():
            _ynn_decided(make.__name__, (dew, ref), ops)
            previous = np.asarray(dew(*ops)[4]).ravel().view(np.uint32).copy()
            for attempt in range(160):
                one = dew(*ops)
                ref(*ops)
                two = dew(*ops)
                for result in (one, two):
                    actual = np.asarray(result[4]).ravel().view(np.uint32)
                    indices = np.flatnonzero(previous != actual)
                    if indices.size and len(failures) < 10:
                        failures.append((make.__name__, attempt, indices[:3].tolist(),
                                         [hex(int(x)) for x in previous[indices[:3]]],
                                         [hex(int(x)) for x in actual[indices[:3]]]))
                    previous = actual.copy()
            print("Parallax replay", make.__name__, len(failures), failures)
    assert not failures


@pytest.mark.mesh(devices=2)
def test_proposed_barrier_preserves_checked_map(monkeypatch):
    monkeypatch.setattr(attention_parallax, "_tiled", _blocked)
    tp.test_parallax_runs_under_a_checked_map_with_its_values_split_alone()
