"""Diagnostic: the key TMA ring without any matrix multiply or residual stores."""
import functools
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels.templates import mosaic
from dew.nn.kernels.templates import mosaic_sm90

mode = sys.argv[1]
if mode == "badcoord":
    from jaxlib.mlir.dialects import llvm
    original_asm = llvm.inline_asm
    def bad_coordinate(result_type, operands, asm, constraints, **kwargs):
        if "cp.async.bulk.tensor.5d" in asm and "shared::cta.global" in asm:
            operands = list(operands)
            operands[4] = original_asm(
                operands[4].type, [], "mov.u32 $0, 1;", "=r",
                has_side_effects=True, asm_dialect=0)
        return original_asm(result_type, operands, asm, constraints, **kwargs)
    llvm.inline_asm = bad_coordinate
width = 128
x = jnp.asarray(np.random.default_rng(0).standard_normal((2, 4096, 8, width)),
                jnp.bfloat16)
q = jnp.asarray(np.random.default_rng(1).standard_normal((2, 4096, 32, width)),
                jnp.bfloat16)

def entry(x_ref, q_ref, out_ref):
    batch = jax.lax.axis_index("batch")
    head = jax.lax.axis_index("head")
    query_index = jax.lax.axis_index("query")
    wg = jax.lax.axis_index("wg")
    @functools.partial(pl.run_scoped,
                       **mosaic.ring((64, width), x.dtype, 2, swizzle=128),
                       query=mosaic.staged((64, width), x.dtype),
                       query_full=plgpu.Barrier(),
                       collective_axes="wg")
    def body(smem, full, consumed, query, query_full):
        ring = mosaic.Ring(smem, full, consumed)
        def producer():
            @pl.loop(0, 64)
            def load(step):
                ring.load(step, x_ref.at[batch, pl.ds(step * 64, 64), head // 4, :])
            ring.drain(jnp.int32(64))
        def consumer(index):
            del index
            if mode in ("qonly", "mma"):
                mosaic.loaded(((q_ref.at[batch, pl.ds(query_index * 64, 64), head, :],
                                query, query_full),))
            def add(step, total):
                slot = ring.acquire(step)
                if mode == "mma":
                    tile = mosaic_sm90.WGMMA.product(query, slot.transpose((1, 0)), (64, 64))
                else:
                    tile = plgpu.layout_cast(slot[...], plgpu.Layout.WGMMA).astype(jnp.float32)
                total = total + tile
                ring.release(step)
                return total
            columns = 64 if mode == "mma" else width
            total = jax.lax.fori_loop(0, 64, add, jnp.zeros((64, columns), jnp.float32))
            out_ref[batch, query_index, head, :, :] = total
        if mode == "roles":
            mosaic.roles(wg, 1, consumer, producer)
        else:
            pl.when(wg == 0)(lambda: consumer(wg))
            pl.when(wg == 1)(producer)


columns = 64 if mode == "mma" else width
run = plgpu.kernel(entry, out_type=jax.ShapeDtypeStruct((2, 64, 32, 64, columns), jnp.float32),
                   grid=(2, 64, 32), grid_names=("batch", "query", "head"),
                   num_threads=2, thread_name="wg", kernel_name="dbg_tma_only",
                   compiler_params=mosaic.compiler_params())
for repeat in range(2):
    out = jax.block_until_ready(jax.jit(run)(x, q))
    print("ITERATION", repeat, flush=True)
print("OK", mode, bool(jnp.isfinite(out).all()), flush=True)
