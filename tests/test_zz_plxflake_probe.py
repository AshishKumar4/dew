import jax
import jax.numpy as jnp
import numpy as np
import pytest
from arm_arithmetic import summed_as_run
from arm_forms import _arms, _ynn_decided

import test_attention_parallax as tp
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


def _scan(step, start, count):
    return jax.lax.scan(step, start, jnp.arange(count))


@pytest.mark.parametrize('peeled', [True, False])
def test_check_arm_repeats(monkeypatch, peeled):
    if not peeled:
        monkeypatch.setattr(attention_parallax, '_tiled', _scan)
    failures = []
    for make in (_covariance, _far):
        operands, spans = make()
        ops = (*operands, jnp.ones((1, 1, 1, 1), jnp.float32))
        dew, ref = _arms(jax.jit(tp._every_cotangent(attention_parallax.attend, spans)),
                         jax.jit(tp._every_cotangent(tp._plain, spans)))
        with summed_as_run():
            _ynn_decided(make.__name__, (dew, ref), ops)
            print('PLX-MODE', make.__name__, peeled, dew.pairing.refused, flush=True)
            expected = np.asarray(dew(*ops)[4]).ravel().view(np.uint32).copy()
            previous = expected
            for attempt in range(120):
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
            print('PLX-REPEAT', make.__name__, peeled, len(failures), failures, flush=True)
    assert not failures
