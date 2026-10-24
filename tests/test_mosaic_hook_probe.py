"""Diagnostic controls for compiled score-hook backward failures."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl

from test_attention_mosaic import ON_HOPPER, _check_hooked_backward

from dew.nn.kernels.templates import attention_mosaic, mosaic, mosaic_sm90

pytestmark = [pytest.mark.kernel, ON_HOPPER,
              pytest.mark.gpu(reason="isolates the H100 score-hook backward failures")]


@pytest.mark.parametrize("hook,accurate,consumers,approx", [
    ("key-bias", False, 2, True),
    ("key-bias", False, 1, True),
    ("key-bias", True, 1, True),
    ("softcap-sinks-alibi", False, 2, True),
    ("softcap-sinks-alibi", False, 2, False),
    ("softcap-sinks-alibi", True, 1, False),
], ids=["key-red", "key-single", "key-accurate", "initial-red", "initial-exact",
        "initial-accurate-exact"])
def test_hook_control(monkeypatch, hook, accurate, consumers, approx):
    backward = attention_mosaic.backward
    params = mosaic.compiler_params
    monkeypatch.setattr(attention_mosaic, "backward",
                        functools.partial(backward, consumers=consumers))
    monkeypatch.setattr(mosaic, "compiler_params",
                        lambda **_: dataclasses.replace(params(), approx_math=approx))
    _check_hooked_backward(hook, accurate=accurate, interpret=False)


@pytest.mark.parametrize("consumers", [1, 2])
def test_column_reduction(consumers):
    family = mosaic_sm90.WGMMA
    count, rows, columns = 128, 64, 64
    source = jnp.arange(consumers * count * rows * columns, dtype=jnp.float32)
    source = jnp.sin(source.reshape(consumers, count, rows, columns) * 0.001)

    def body(src, out):
        wg = lax.axis_index("wg")

        @pl.when(wg < consumers)
        def _():
            @pl.loop(0, count)
            def _(step):
                tile = family.tile_of(src.at[wg, step])
                out[wg, step, :] = tile.sum(axis=0)

    kernel = family.kernel(body, out_type=jax.ShapeDtypeStruct((consumers, count, columns), jnp.float32),
                           grid=(), grid_names=(), consumers=consumers, interpret=False)
    result = kernel(source)
    expected = jnp.sum(source, axis=2)
    error = np.asarray(result - expected)
    print(f"COLUMN consumers={consumers} max_error={np.max(np.abs(error))} "
          f"bad_columns={np.count_nonzero(np.abs(error) > 0.001)}")
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected), rtol=0, atol=0.001)
