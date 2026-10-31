"""Mamba's selective scan on the recurrence template (`templates.recurrence.selective`): its output and every
input's gradient on par from float64 with the portable scan's (`dew.nn.mixers.mamba.selective_scan`, check
1), one row and packed documents, interpreted off a GPU over two channel blocks (a block's view of a wider
row strided in global memory), and at Falcon Mamba 7B's widths on a GPU (`tests/test_gpu_recurrence.py`)."""
import functools

import jax
import jax.numpy as jnp
import pytest
from reference_error import FEWEST_SEEDS, assert_as_exact, with_gradients

from dew.nn.kernels import generation
from dew.nn.kernels.templates import bundles, mosaic_sm90
from dew.nn.kernels.templates.recurrence import selective
from dew.nn.mixers.mamba import selective_scan as portable

pytestmark = pytest.mark.kernel

INTERPRETED = jax.default_backend() != "gpu"
"""The template interprets off a GPU, by the wgmma bundle, whose rings and barriers the interpreter models."""

SHAPES = {"interpreted": (1, 70, 256, 16), "falcon-mamba-7b": (2, 2048, 8192, 16)}
"""Each case's rows, tokens, channels and states: a ragged chunk over two channel blocks, and Falcon Mamba
7B's widths."""


def draw(dtype, seed: int, rows: int, length: int, channels: int, states: int, packed: bool):
    """A draw's inputs, their outputs' cotangents last: steps after a softplus, decays `-exp(N(0, 1/4))`,
    input and readout gates normal, `x`, `dt`, `B`, `C` and the cotangents in `dtype`, documents starting at a
    third and two thirds of each row where `packed`."""
    keys = jax.random.split(jax.random.key(seed), 7)
    x = jax.random.normal(keys[0], (rows, length, channels)).astype(dtype)
    dt = jax.nn.softplus(jax.random.normal(keys[1], (rows, length, channels)) - 3.0).astype(dtype)
    a = -jnp.exp(0.5 * jax.random.normal(keys[2], (channels, states)))
    b, c = (jax.random.normal(key, (rows, length, states)).astype(dtype) for key in keys[3:5])
    skip = jax.random.normal(keys[5], (channels,))
    starts = None
    if packed:
        starts = jnp.zeros((rows, length), bool).at[:, (0, length // 3, 2 * length // 3)].set(True)
    dy = jax.random.normal(keys[6], (rows, length, channels)).astype(dtype)
    return (x, dt, a, b, c, skip), starts, dy


def scanned(fn, starts):
    """`fn`'s output alone, its documents' starts `starts`."""
    return lambda *inputs: fn(*inputs, starts=starts)[0]


def the_template_is_as_exact_as_the_portable_scan(case: str, packed: bool, dtype=jnp.bfloat16):
    """`case`'s output and every input's gradient through the template on par from float64 with the portable
    scan's, over check 1's fewest draws."""
    shape = SHAPES[case]
    family = mosaic_sm90.WGMMA if INTERPRETED else bundles.of(generation.device_generation(), dtype)
    template = functools.partial(selective.selective_scan, family=family, interpret=INTERPRETED)

    def draws(seed):
        inputs, _, dy = draw(dtype, seed, *shape, packed)
        return (*inputs, dy)

    starts = draw(dtype, 0, *shape, packed)[1]
    assert_as_exact(jax.jit(with_gradients(scanned(template, starts))),
                    jax.jit(with_gradients(scanned(portable, starts))), draws,
                    f"the selective template, {case}{', packed' if packed else ''}", seeds=FEWEST_SEEDS,
                    truth_fn=jax.jit(with_gradients(scanned(portable, starts))))


@pytest.mark.parametrize("packed", [False, True], ids=["one row", "packed documents"])
def test_the_template_is_as_exact_as_the_portable_scan_interpreted(packed):
    the_template_is_as_exact_as_the_portable_scan("interpreted", packed)
