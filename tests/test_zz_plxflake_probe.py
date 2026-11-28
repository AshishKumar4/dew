import jax
import jax.numpy as jnp
import numpy as np
import pytest
from arm_arithmetic import summed_as_run
from arm_forms import _arms, _ynn_decided, without_ynn

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


_PEELED = attention_parallax._tiled


def _derivative_scan(step, start, count):
    if step.__qualname__ == 'derivative.<locals>.step':
        return _scan(step, start, count)
    return _PEELED(step, start, count)


def _two_steps(step, start, count):
    if count != 2 or step.__qualname__ != 'derivative.<locals>.step':
        return _PEELED(step, start, count)
    carry, first = step(start, 0)
    carry, second = step(carry, 1)
    return carry, jax.tree.map(lambda a, b: jnp.stack((a, b)), first, second)


def _blocked(step, start, count):
    if step.__qualname__ != 'derivative.<locals>.step':
        return _PEELED(step, start, count)
    carry, first = step(start, 0)
    carry, rest = jax.lax.scan(step, carry, jnp.arange(1, count))
    first, rest = jax.lax.optimization_barrier((first, rest))
    return carry, jax.tree.map(lambda a, b: jnp.concatenate((a[None], b)), first, rest)


@pytest.mark.parametrize('variant', [
    'peeled', 'unpeeled', 'derivative_scan', 'two_steps', 'barrier', 'no_cpu_fusion', 'no_fusion', 'ynn',
])
def test_check_arm_repeats(monkeypatch, variant):
    tiling = {'unpeeled': _scan, 'derivative_scan': _derivative_scan, 'two_steps': _two_steps,
              'barrier': _blocked}
    if variant in tiling:
        monkeypatch.setattr(attention_parallax, '_tiled', tiling[variant])
    failures = []
    for make in (_covariance, _far):
        operands, spans = make()
        ops = (*operands, jnp.ones((1, 1, 1, 1), jnp.float32))
        run = jax.jit(tp._every_cotangent(attention_parallax.attend, spans))
        vanilla = jax.jit(tp._every_cotangent(tp._plain, spans))
        disabled = {'no_cpu_fusion': 'cpu-instruction-fusion', 'no_fusion': 'fusion,cpu-instruction-fusion'}
        if variant in disabled:
            run = without_ynn(run, compiler_options={'xla_disable_hlo_passes': disabled[variant]})
        if variant == 'ynn':
            dew, ref = run, vanilla
        else:
            dew, ref = _arms(run, vanilla)
        with summed_as_run():
            if variant != 'ynn':
                _ynn_decided(make.__name__, (dew, ref), ops)
            print('PLX-MODE', make.__name__, variant, flush=True)
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
            print('PLX-REPEAT', make.__name__, variant, len(failures), failures, flush=True)
    assert not failures
