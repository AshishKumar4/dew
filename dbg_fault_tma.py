"""Diagnostic: the key TMA ring without any matrix multiply or residual stores."""
import functools
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import mosaic

mode = sys.argv[1]
width = 128
x = jnp.asarray(np.random.default_rng(0).standard_normal((2, 4096, 8, width)),
                jnp.bfloat16)

def entry(x_ref, out_ref):
    batch = jax.lax.axis_index("batch")
    head = jax.lax.axis_index("head")
    query = jax.lax.axis_index("query")
    wg = jax.lax.axis_index("wg")
    @functools.partial(pl.run_scoped,
                       **mosaic.ring((64, width), x.dtype, 2, swizzle=128),
                       collective_axes="wg")
    def body(smem, full, consumed):
        ring = mosaic.Ring(smem, full, consumed)
        def producer():
            @pl.loop(0, 64)
            def load(step):
                ring.load(step, x_ref.at[batch, pl.ds(step * 64, 64), head // 4, :])
            ring.drain(jnp.int32(64))
        def consumer(index):
            del index
            def add(step, total):
                tile = plgpu.layout_cast(ring.acquire(step)[...], plgpu.Layout.WGMMA)
                total = total + tile.astype(jnp.float32)
                ring.release(step)
                return total
            total = jax.lax.fori_loop(0, 64, add, jnp.zeros((64, width), jnp.float32))
            out_ref[batch, query, head, :, :] = total
        if mode == "roles":
            mosaic.roles(wg, 1, consumer, producer)
        else:
            pl.when(wg == 0)(lambda: consumer(wg))
            pl.when(wg == 1)(producer)
    body()

run = plgpu.kernel(entry, out_type=jax.ShapeDtypeStruct((2, 64, 32, 64, width), jnp.float32),
                   grid=(2, 64, 32), grid_names=("batch", "query", "head"),
                   num_threads=2, thread_name="wg", kernel_name="dbg_tma_only",
                   compiler_params=mosaic.compiler_params())
for repeat in range(2):
    out = jax.block_until_ready(jax.jit(run)(x))
    print("ITERATION", repeat, flush=True)
print("OK", mode, bool(jnp.isfinite(out).all()), flush=True)
