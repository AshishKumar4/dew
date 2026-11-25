"""An epilogue's hook over a Hopper tile read out a chunk of columns at a time (`pricing.hook_chunk`,
`hooks.column_local`): a hook reading every column of its row still sees them all, and outputs staged at
different widths each fill whole buffers; each product, the template named at a pinned 128-column tile,
interpreted, as exact as XLA's from float64."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact

from dew.nn.kernels import gemm, generation, grouped_matmul, tuning
from dew.nn.kernels.templates.gemm import tiling

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


def _centred(acc):
    """A hook reading its row's every column: each entry less its row's maximum."""
    return acc - jnp.max(acc, axis=1, keepdims=True)


def _two_widths(acc):
    """A hook storing a bf16 value and an FP8 one, staged at different widths."""
    return acc.astype(jnp.bfloat16), (2 * acc).astype(jnp.float8_e4m3fn)


def _template(hook, out_dtypes):
    def run(x, w):
        with (generation.assumed("sm90"), generation.chosen(grouped_matmul="mosaic"),
              tuning.pinned({grouped_matmul.MosaicGmm: TILE})):
            return gemm.gemm(x, w, None, epilogue=hook, out_dtype=out_dtypes[0], out_dtypes=out_dtypes,
                             interpret_on_cpu=True)
    return run


def _reference(hook, out_dtypes, wide=jnp.float32):
    def run(x, w):
        outs = hook(jnp.dot(x.astype(wide), w.astype(wide), preferred_element_type=wide))
        outs = outs if isinstance(outs, tuple) else (outs,)
        return tuple(out.astype(dtype if wide == jnp.float32 else out.dtype)
                     for out, dtype in zip(outs, out_dtypes, strict=True))
    return run


@pytest.mark.parametrize(("hook", "out_dtypes"), [(_centred, (jnp.float32,)),
                                                  (_two_widths, (jnp.bfloat16, jnp.float8_e4m3fn))],
                         ids=["row-maximum", "two-widths"])
def test_a_hooked_product_over_a_chunked_tile_is_its_programs(hook, out_dtypes):
    """A row-maximum hook (no column-local one: it takes the tile whole) and a two-dtype hook (its FP8 output
    staged 128 columns wide where its bf16 one is 64) through the template at a 128-column tile, interpreted
    as an H100 runs it: as exact as XLA's product and hook, from float64."""
    assert_as_exact(jax.jit(_template(hook, out_dtypes)), jax.jit(_reference(hook, out_dtypes)), _draw,
                    f"chunked hook {hook.__name__}", seeds=SEEDS,
                    truth_fn=_reference(hook, out_dtypes, wide=jnp.float64))
