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
    semaphore each arrives at once, `Family.fixed_up`'s), else 0, an int32 scalar: one of its threads adds
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


MERGED_REGISTERS = 64
"""The registers a thread of a merging role holds the partials it reads ahead of combining them in
(`Family.fixed_up`): as many partials as fit, at least one (an H100's 16-row decode's at 64 columns 10, its
tile's 8 and its two statistics' 1 each; a GEMM's 128 x 128 fp32 tile 128). One partial read at a time
waited each read behind the last one's combine, 0.49 us a slice (a 16-row decode's 49, M-DD-12)."""


def merge_spans(summed, held: int) -> tuple[int | None, list[tuple[int, int | None]]]:
    """The place among a partial's leaves of its widest 2-D leaf and the column spans `(start, width)`
    `Family.fixed_up` merges it in: as few as bring its registers a thread (`held`) within
    `MERGED_REGISTERS`, each a whole number of the layout's 8-column atoms dividing its columns; none and one
    whole span `(0, None)` where it already fits or has no 2-D leaf."""
    leaves = jax.tree.leaves(summed)
    tiles = [place for place, leaf in enumerate(leaves) if leaf.ndim == 2]
    if held <= MERGED_REGISTERS or not tiles:
        return None, [(0, None)]
    widest = max(tiles, key=lambda place: leaves[place].size)
    columns = leaves[widest].shape[1]
    slices = -(-held // MERGED_REGISTERS)
    while slices < columns // 8 and (columns % slices or (columns // slices) % 8):
        slices += 1
    if columns % slices or (columns // slices) % 8:
        return None, [(0, None)]
    width = columns // slices
    return widest, [(start, width) for start in range(0, columns, width)]


def sliced(value, widest: int | None, start: int, width: int):
    """`value`'s leaves with its `widest`-th (`merge_spans`) cut to `width` columns from `start`."""
    leaves, tree = jax.tree.flatten(value)
    return jax.tree.unflatten(tree, [leaf[:, start:start + width] if place == widest else leaf
                                     for place, leaf in enumerate(leaves)])


def joined(parts: list, widest: int | None):
    """The merged slices `parts` (`sliced`'s) as one value: their `widest`-th leaves side by side, every other
    leaf the last slice's (a column-separable merge leaves them alike)."""
    flat = [jax.tree.flatten(part)[0] for part in parts]
    return jax.tree.unflatten(jax.tree.structure(parts[0]), [
        jnp.concatenate([each[place] for each in flat], axis=1) if place == widest else flat[-1][place]
        for place in range(len(flat[0]))])
