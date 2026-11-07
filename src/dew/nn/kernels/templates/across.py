"""Values a launch's warpgroups or programs each hold a part of, joined across them: the last of a launch's
programs to arrive at a semaphore (`arrived`), and the consumer warpgroups' parts of one tile summed in their
order and stored once (`summing`, `summed_across`)."""

from collections.abc import Callable

import jax
from jax import numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.mosaic.gpu import fragmented_array as mgpu_fragments, utils as mgpu_utils
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend.mlir import ir
from jax.extend.mlir.dialects import arith

from dew.nn.kernels.templates import resources


def arrived(counter, contributors) -> jax.Array:
    """1 where this consumer warpgroup is the last of `contributors` to arrive at `counter` (a GPU-scope
    semaphore each arrives at once, `Transport.fixed_up`'s), else 0, an int32 scalar: one of its threads adds
    one to the count by an atomic that releases the warpgroup's writes before it (a barrier first, so every
    thread's are) and acquires every earlier arrival's; the last, which reads `contributors - 1` before its
    add, sets the count back to zero for the next launch, and a barrier's reduction tells every thread of
    the warpgroup which it is. One atomic and no wait: a program spinning on a peer the GPU never scheduled
    hangs (row 4732), where an atomic ticket's last arriver merges (CUTLASS's and Triton's split-K
    semaphores alike)."""
    scalar = plgpu.Layout.WG_SPLAT(())

    def ticket(ctx, ref, count):
        del ctx
        i32 = ir.IntegerType.get_signless(32)
        mgpu_utils.warpgroup_barrier()
        thread = arith.remui(mgpu_utils.thread_idx(), mgpu_utils.c(resources.THREADS, i32))
        before_last = arith.subi(count.registers.item(), mgpu_utils.c(1, i32))
        last = mgpu_utils.inline_ptx(
            f"""{{
            .reg .pred elected, last;
            .reg .b32 before;
            setp.eq.u32 elected, $1, 0;
            mov.b32 before, 0;
            @elected atom.acq_rel.gpu.global.add.u32 before, [$2], 1;
            setp.eq.and.u32 last, before, $3, elected;
            @last st.relaxed.gpu.global.u32 [$2], 0;
            bar.red.or.pred last, $4, {resources.THREADS}, last;
            selp.u32 $0, 1, 0, last;
            }}""",
            thread, mgpu_utils.memref_ptr(ref), before_last, mgpu_utils.warpgroup_barrier_idx(sync=False),
            result_types=i32, has_side_effects=True, convergent=True)
        return mgpu_fragments.FragmentedArray.splat(last, (), is_signed=True)

    inline = plgpu.inline_mgpu(arg_types=(plgpu.RefType(), scalar),
                               return_type=plgpu.ShapeDtypeStruct((), jnp.dtype(jnp.int32), layout=scalar))
    return inline(ticket)(counter, jnp.asarray(contributors, jnp.int32))


def summing(shape: tuple[int, ...], consumers: int) -> dict:
    """`summed_across`' allocations for `pl.run_scoped` (collective across the consumers, in a scope after
    the walk's, whose shared memory it reuses): every consumer's fp32 part but the first's, the barrier they
    arrive at once theirs is written, and the one the first arrives at once it has read them; none for one
    consumer, whose part is the sum."""
    if consumers == 1:
        return {}
    return {'parts': plgpu.SMEM((consumers - 1, *shape), jnp.float32),
            'ready': plgpu.Barrier(num_arrivals=consumers - 1, num_barriers=1),
            'read': plgpu.Barrier(num_arrivals=1, num_barriers=1)}


def summed_across(value, scratch: dict, wg, consumers: int, store: Callable) -> None:
    """The consumers' fp32 parts of one tile (each consumer's `value`) added and stored once: every consumer
    but the first leaves its part in `scratch` (`summing`'s, scoped) and arrives at its `ready`, then waits
    until the first has read every part (`read`), so the scratch serves the next sum; the first waits, adds
    them in the consumers' order (one sum, the same each run), arrives at `read` and stores the total
    (`store(total)`). One consumer stores its own."""
    if consumers == 1:
        store(value)
        return
    parts, ready, read = scratch['parts'], scratch['ready'], scratch['read']

    @pl.when(wg > 0)
    def _():
        parts[wg - 1] = value
        plgpu.barrier_arrive(ready.at[0])
        plgpu.barrier_wait(read.at[0])

    @pl.when(wg == 0)
    def _():
        plgpu.barrier_wait(ready.at[0])
        total = value
        for other in range(consumers - 1):
            total = total + parts[other]
        plgpu.barrier_arrive(read.at[0])
        store(total)
