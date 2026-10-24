"""Process-local precision experiments; never changes the production kernels."""

import sys
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from dew.nn.kernels import chunked_launch, recurrence


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
            dtype = jnp.float32 if mode == 'scale64' else jnp.float64
            total = jnp.cumsum(tokens.astype(dtype), axis=0)
            if mode == 'sum64':
                total = total.astype(jnp.float32)
            # RCP_LN2's float32 approximation is part of the suspected double rounding.
            factor = ((scale if mode == 'sum64' else 1 / np.log(2.0))
                      if scale is not None else 1.0)
            if mode == 'scale64':
                total = total.astype(jnp.float64)
            result = total * factor
            chunked_launch.write_rows(chunked_launch._rows(o_ref, start, chunk), count, result)

    recurrence._cumsum_kernel = prefix


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
    with jax.enable_x64():
        if mode != 'baseline':
            install_prefix64(mode)
        log_measurements()
        assert_as_exact(jax.jit(run), jax.jit(partial(oracle_chunk, chunk_size=4)), operands,
                        f'forced-port-{mode}', seeds=64)


if __name__ == '__main__':
    jax.config.update('jax_enable_x64', True)
    if sys.argv[1] in ('prefix64', 'sum64', 'scale64'):
        install_prefix64(sys.argv[1])
    elif sys.argv[1] == 'baseline':
        pass
    else:
        raise ValueError(sys.argv[1])
    log_measurements()
    raise SystemExit(pytest.main(sys.argv[2:]))
