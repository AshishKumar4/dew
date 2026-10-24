"""Process-local precision experiments; never changes the production kernels."""

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from dew.nn.kernels import chunked_launch, recurrence


def install_prefix64():
    original = recurrence._cumsum_kernel

    def prefix(chunk_ref, s_ref, o_ref, *, chunk, aligned, scale, reverse):
        if reverse:
            return original(chunk_ref, s_ref, o_ref, chunk=chunk, aligned=aligned, scale=scale,
                            reverse=reverse)
        start, count = chunked_launch._chunk_of(chunk_ref, chunk, aligned)

        @pl.when(count > 0)
        def _():
            tokens = chunked_launch.read_rows(chunked_launch._rows(s_ref, start, chunk), count)
            total = jnp.cumsum(tokens.astype(jnp.float64), axis=0)
            # RCP_LN2's float32 approximation is part of the suspected double rounding.
            factor = (1 / np.log(2.0)) if scale is not None else 1.0
            chunked_launch.write_rows(chunked_launch._rows(o_ref, start, chunk), count, total * factor)

    recurrence._cumsum_kernel = prefix


if __name__ == '__main__':
    jax.config.update('jax_enable_x64', True)
    if sys.argv[1] == 'prefix64':
        install_prefix64()
    else:
        raise ValueError(sys.argv[1])
    raise SystemExit(pytest.main(sys.argv[2:]))
