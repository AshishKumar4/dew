"""An epilogue's hook over a Hopper tile read out a chunk of columns at a time where its caller declares
it column-local (`hooks.ColumnLocal`, `pricing.hook_chunk`): its outputs staged at different widths each
fill whole buffers, and a hook reading every column of its row, undeclared, still sees them all; each
product, the template named at a pinned 128-column tile, interpreted, as exact as XLA's from float64."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact, with_gradients

from dew.nn.kernels import gemm, generation, grouped_matmul, tuning
from dew.nn.kernels.templates.gemm import tiling
from dew.nn.kernels.templates.gemm.hooks import ColumnLocal
from dew.nn.kernels.templates.hooks import Hook

SEEDS = 8
ROWS, DEPTH, WIDTH = 128, 128, 128
TILE = tiling.Tiles(64, WIDTH, 64, 2, 1, warpgroups=1, family="wgmma")
"""A tile whose columns are the product's: a hook's row is the tile's row."""


def _draw(seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((ROWS, DEPTH)) / 8
    w = rng.standard_normal((DEPTH, WIDTH)) / 8
    # Each row's halves apart in size, so a row's maximum lies in its second half.
    w[:, WIDTH // 2:] *= 4
    return jnp.asarray(x, jnp.bfloat16), jnp.asarray(w, jnp.bfloat16)


def _centred(dtypes):
    """A hook reading its row's every column: each entry less its row's maximum, in `dtypes`' one."""
    def hook(acc):
        return ((acc - jnp.max(acc, axis=1, keepdims=True)).astype(dtypes[0]),)
    return hook


def _two_widths(dtypes):
    """A column-local hook storing two values in `dtypes`, each staged at its own width."""
    def hook(acc):
        return acc.astype(dtypes[0]), (2 * acc).astype(dtypes[1])
    return ColumnLocal(hook)


def _template(hook, out_dtypes):
    def run(x, w):
        with (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"),
              tuning.pinned({grouped_matmul.MosaicGmm: TILE})):
            return tuple(gemm.gemm(x, w, None, epilogue=Hook(hook(out_dtypes)), out_dtype=out_dtypes[0],
                                   out_dtypes=out_dtypes, interpret_on_cpu=True))
    return run


def _reference(hook, out_dtypes, wide=jnp.float32):
    def run(x, w):
        return hook(out_dtypes)(jnp.dot(x.astype(wide), w.astype(wide), preferred_element_type=wide))
    return run


@pytest.mark.parametrize(("hook", "out_dtypes"), [
    (_centred, (jnp.float32,)), (_two_widths, (jnp.bfloat16, jnp.float32)),
    (_two_widths, (jnp.bfloat16, jnp.float8_e4m3fn))], ids=["row-maximum", "bf16-fp32", "bf16-fp8"])
def test_a_hooked_product_over_a_chunked_tile_is_its_programs(hook, out_dtypes):
    """Through the template at a 128-column tile, interpreted as an H100 runs it, as exact as XLA's product
    and hook from float64: a row-maximum hook (undeclared: its tile read out whole); a column-local hook of a
    bf16 and an fp32 output (read out 64 columns at a time, each chunk in its turn's buffer); and one of a
    bf16 and an FP8 output (the FP8 one staged 128 columns wide, so its tile is read out whole)."""
    assert_as_exact(jax.jit(_template(hook, out_dtypes)), jax.jit(_reference(hook, out_dtypes)), _draw,
                    f"chunked hook {hook.__name__}", seeds=SEEDS,
                    truth_fn=_reference(hook, (jnp.float64,) * len(out_dtypes), wide=jnp.float64))


def _width_scaled(x):
    """A prologue scaling each entry by its tile's width: entrywise, but reading the extent it is traced at,
    so no slice of its tile computes as the tile does (undeclared: `hooks.ColumnLocal`)."""
    return x * x.shape[-1]


def _draw_rows(seed):
    rng = np.random.default_rng(seed)
    bf16 = functools.partial(jnp.asarray, dtype=jnp.bfloat16)
    return (bf16(rng.standard_normal((ROWS, DEPTH)) / 8), bf16(rng.standard_normal((DEPTH, WIDTH)) / 8),
            bf16(rng.standard_normal((ROWS, WIDTH))))


def test_an_undeclared_prologues_pullback_reads_its_tile_whole():
    """An undeclared prologue's pulled cotangent (`gemm._prologue_pulled`), here one reading its tile's
    width, through the template at a pinned 128-column tile its depth's width: read out whole, its rows'
    gradient as exact as XLA's from float64 (read out 64 columns at a time, it would scale by 64)."""
    def product(x, w):
        return gemm.gemm(x, w, None, prologue=Hook(_width_scaled), applied=_width_scaled(x),
                         out_dtype=jnp.bfloat16,
                         interpret_on_cpu=True)

    def dew(*arguments):
        # Forward and reverse pass traced on the GPU named, the template pinned at the tile.
        with (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"),
              tuning.pinned({grouped_matmul.MosaicGmm: TILE})):
            return with_gradients(product)(*arguments)

    def plain(x, w, wide=jnp.float32):
        formed = _width_scaled(x.astype(wide)).astype(x.dtype if wide == jnp.float32 else wide)
        return jnp.dot(formed.astype(wide), w.astype(wide)).astype(x.dtype if wide == jnp.float32 else wide)

    assert_as_exact(dew, jax.jit(with_gradients(plain)), _draw_rows,
                    "undeclared width-scaled prologue's pullback", seeds=SEEDS,
                    truth_fn=with_gradients(functools.partial(plain, wide=jnp.float64)))
