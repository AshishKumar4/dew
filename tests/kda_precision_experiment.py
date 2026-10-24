"""Process-local precision experiments; never changes the production kernels."""

import sys
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from dew.nn.kernels import chunked_launch, recurrence


def install_dot(algorithm='bf16x6'):
    from dew.nn.kernels import fla_gdn, fla_kda

    original = fla_gdn._dot
    presets = {
        'bf16x6': jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X6,
        'bf16x9': jax.lax.DotAlgorithmPreset.BF16_BF16_F32_X9,
        'ieee': jax.lax.Precision.HIGHEST,
    }

    def dot(a, b, *, exact, ieee=False, solve=None):
        if a.dtype == jnp.float32 and not exact and not ieee and (
                solve is None or jnp.dtype(solve).itemsize == 4):
            return jnp.dot(a, b, preferred_element_type=jnp.float32, precision=presets[algorithm])
        return original(a, b, exact=exact, ieee=ieee, solve=solve)

    fla_gdn._dot = fla_kda._dot = dot


def log_measurements():
    import reference_error

    original = reference_error._judgement

    def logged(draws, label, tails=(reference_error.LOWER_TAIL, reference_error.UPPER_TAIL), *,
               contained=True):
        for index, name in enumerate(draws.names):
            ratio, low, high = reference_error._judged(draws, index, tails)
            print(json.dumps({'label': label + name, 'seeds': len(draws.mine),
                              'ratio': ratio, 'interval': [low, high]}), flush=True)
        return original(draws, label, tails, contained=contained)

    reference_error._judgement = logged


def install_prefix64(mode='prefix64'):
    original = recurrence._cumsum_kernel

    def prefix(chunk_ref, s_ref, o_ref, *, chunk, aligned, scale, reverse):
        if reverse:
            return original(chunk_ref, s_ref, o_ref, chunk=chunk, aligned=aligned, scale=scale,
                            reverse=reverse)
        start, count = chunked_launch._chunk_of(chunk_ref, chunk, aligned)

        @pl.when(count > 0)
        def _():
            tokens = chunked_launch.read_rows(chunked_launch._rows(s_ref, start, chunk), count)
            with jax.enable_x64():
                dtype = jnp.float32 if mode == 'scale64' else jnp.float64
                total = jnp.cumsum(tokens.astype(dtype), axis=0)
                if mode == 'sum64':
                    total = total.astype(jnp.float32)
                # RCP_LN2's float32 approximation is part of the suspected double rounding.
                factor = ((scale if mode == 'sum64' else 1 / np.log(2.0))
                          if scale is not None else 1.0)
                if mode == 'scale64':
                    total = total.astype(jnp.float64)
                result = (total * factor).astype(jnp.float32)
            chunked_launch.write_rows(chunked_launch._rows(o_ref, start, chunk), count, result)

    recurrence._cumsum_kernel = prefix


@pytest.mark.gpu(reason='the loss is the A100 tensor-core product, not interpreted float32')
def test_a_single_representable_product_is_not_missing_its_low_cross_term():
    from dew.nn.kernels import fla_gdn
    from jax.experimental.pallas import triton as plgpu

    def kernel(a_ref, b_ref, result_ref):
        result_ref[...] = fla_gdn._dot(a_ref[...], b_ref[...], exact=False)

    run = pl.pallas_call(
        kernel, out_shape=jax.ShapeDtypeStruct((16, 16), jnp.float32),
        compiler_params=plgpu.CompilerParams(num_warps=4))
    a, b = np.zeros((16, 16), np.float32), np.zeros((16, 16), np.float32)
    value = np.float32(1 + 2 ** -11)
    a[:, 0], b[0, :] = value, value
    wanted = np.full((16, 16), np.float32(np.float64(value) ** 2))
    got = np.asarray(run(jnp.asarray(a), jnp.asarray(b)))
    print(json.dumps({'value': float(value), 'got': float(got[0, 0]),
                      'wanted': float(wanted[0, 0]), 'error': float(got[0, 0] - wanted[0, 0])}),
          flush=True)
    np.testing.assert_array_equal(got, wanted)


@pytest.mark.parametrize('mode', ['baseline', 'sum64', 'scale64', 'prefix64'])
def test_short_forward_prefix_precision(mode, monkeypatch):
    from functools import partial

    import reference_error
    from reference_error import assert_as_exact
    from test_kda import operands
    from tools.kda_reference import oracle_chunk
    from dew.nn.kernels.recurrence_chunk import chunk_gated_delta_rule
    from dew.nn.precision import l2norm

    def run(q, k, v, g, beta, state):
        return chunk_gated_delta_rule(l2norm(q), l2norm(k), v, g, beta, state,
                                      chunk_size=4, ported=True)

    monkeypatch.setattr(recurrence, '_cumsum_kernel', recurrence._cumsum_kernel)
    monkeypatch.setattr(reference_error, '_judgement', reference_error._judgement)
    if mode != 'baseline':
        install_prefix64(mode)
    log_measurements()
    assert_as_exact(jax.jit(run), jax.jit(partial(oracle_chunk, chunk_size=4)), operands,
                    f'forced-port-{mode}', seeds=64)


if __name__ == '__main__':
    if sys.argv[1] in ('prefix64', 'sum64', 'scale64'):
        install_prefix64(sys.argv[1])
    elif sys.argv[1] in ('bf16x6', 'bf16x9', 'ieee'):
        install_dot(sys.argv[1])
    elif sys.argv[1] == 'prefix64-bf16x6':
        install_prefix64()
        install_dot()
    elif sys.argv[1] == 'baseline':
        pass
    else:
        raise ValueError(sys.argv[1])
    log_measurements()
    raise SystemExit(pytest.main(sys.argv[2:]))
