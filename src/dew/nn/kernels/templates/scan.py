"""A warpgroup's scan of a first-order linear recurrence over a block of positions (design §6's toolkit): each
position's state `h_t = a_t h_{t-1} + b_t` from its decays `a` and writes `b` and the state carried into
the block, its 128 threads each holding consecutive positions (`blocked`), as cub's BlockScan holds a
thread's items. A thread scans its own positions in registers, its warp's 32 threads their aggregates by
shuffles, and the warpgroup's four warps theirs through shared memory (`scratch`), the pairs `(a, b)`
combined as `(a1, b1) then (a2, b2) = (a1 a2, a2 b1 + b2)`; `reverse` scans from the last position, each
state `h_t = a_t h_{t+1} + b_t`. A chunked recurrence's backward walks its chunks on it (Mamba's selective
scan's, mamba_ssm's selective_scan_bwd_kernel.cuh, a BlockScan a state forward and a reverse one back).
Interpreted, the scan is XLA's (`lax.associative_scan`)."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import arith, memref, nvvm, vector
from jax.experimental.mosaic.gpu import fragmented_array as fa, utils
from jax.experimental.pallas import mosaic_gpu as plgpu

THREADS = 128
"""A warpgroup's threads, a block's positions spread over them."""

WARP = 32
"""A warp's threads, whose aggregates a shuffle combines."""


def blocked(positions: int) -> plgpu.Layout:
    """The layout a block of `positions` takes for its scan: each of the warpgroup's threads its
    `positions // THREADS` consecutive positions, the first thread's the first."""
    if positions % THREADS:
        raise ValueError(f"a block of {positions} positions spreads unevenly over {THREADS} threads")
    return plgpu.Layout.WG_STRIDED((positions,), vec_size=positions // THREADS)


def scratch() -> plgpu.SMEM:
    """The shared memory a scan's warps exchange their aggregates through: a decay and a write each warp."""
    return plgpu.SMEM((2, THREADS // WARP), jnp.float32)


def linear_scan(decays: jax.Array, writes: jax.Array, carried: jax.Array, exchanged, *,
                reverse: bool = False, exclusive: bool = False,
                interpret: bool = False) -> tuple[jax.Array, jax.Array]:
    """`(states, last)`: each position's state of the recurrence `h_t = a_t h_{t-1} + b_t` over `decays`
    `a` and `writes` `b`, fp32 `[T]` in `blocked(T)`'s layout, from `carried`, the state before the first
    position, an fp32 scalar; `last`, the state past the block, a scalar too.
    `reverse` runs from the last position, `h_t = a_t h_{t+1} + b_t`; `exclusive` gives each position the
    state before it in the scan's order (`h_{t-1}`, or `h_{t+1}` reversed) rather than its own. `exchanged`
    is `scratch()`'s, which the warpgroup's threads meet on (a barrier) before and after they read it."""
    if interpret:
        return _interpreted(decays, writes, carried, reverse, exclusive)
    positions = decays.shape[0]
    layout, scalar = blocked(positions), plgpu.Layout.WG_SPLAT(())
    scanned: Callable[..., tuple[jax.Array, jax.Array]] = plgpu.inline_mgpu(
        arg_types=(layout, layout, scalar, plgpu.RefType()),
        return_type=(plgpu.ShapeDtypeStruct((positions,), jnp.dtype(jnp.float32), layout=layout),
                     plgpu.ShapeDtypeStruct((), jnp.dtype(jnp.float32), layout=scalar)),
    )(lambda ctx, a, b, h, ref: _scanned(a, b, h, ref, reverse, exclusive))
    return scanned(decays, writes, carried, exchanged)


BANKS = 32
"""Shared memory's banks, a 4-byte word each (`cost.SHARED_BYTES_PER_CLOCK`'s 128 bytes a clock)."""


def striped(positions: int) -> plgpu.Layout:
    """The layout a block of `positions` takes across the threads: position `t + 128 i` thread `t`'s `i`-th,
    a warp's threads adjacent at each step, as a coalesced global access takes them."""
    return plgpu.Layout.WG_STRIDED((positions,), vec_size=1)


def exchange_scratch(positions: int) -> plgpu.SMEM:
    """The shared memory `to_striped` takes for a block of `positions`: a word each and one more each
    `BANKS`, so no two threads of a warp store or load the same bank at once (cub's padded BlockExchange)."""
    return plgpu.SMEM((positions + positions // BANKS,), jnp.float32)


def to_striped(values: jax.Array, staged, *, interpret: bool = False) -> jax.Array:
    """`values`, fp32 `[T]` in `blocked(T)`'s layout, in `striped(T)`'s through `staged`
    (`exchange_scratch`): a thread's positions stored a word at a time, each `BANKS` words one more apart,
    and read back across the threads; unpadded, a thread's 64 bytes stored at once conflicted on 70% of
    an H100's shared stores (mamba1-scan-h100-7d3152837's ncu, 252M of 360M wavefronts)."""
    if interpret:
        return values
    positions = values.shape[0]
    moved: Callable[..., jax.Array] = plgpu.inline_mgpu(
        arg_types=(blocked(positions), plgpu.RefType()),
        return_type=plgpu.ShapeDtypeStruct((positions,), jnp.dtype(jnp.float32), layout=striped(positions)),
    )(lambda ctx, x, ref: _to_striped(x, ref))
    return moved(values, staged)


def _to_striped(x: fa.FragmentedArray, staged):
    """`to_striped` in Mosaic GPU's terms."""
    index, i32 = ir.IndexType.get(), ir.IntegerType.get_signless(32)
    [vec] = x.registers.flat
    count = ir.VectorType(vec.type).shape[0]
    thread = arith.index_cast(index, arith.remui(utils.thread_idx(), arith.constant(i32, THREADS)))

    def padded(position):
        return arith.addi(position, arith.divui(position, arith.constant(index, BANKS)))

    first = arith.muli(thread, arith.constant(index, count))
    for i in range(count):
        position = arith.addi(first, arith.constant(index, i))
        memref.store(vector.extract(vec, [], [i]), staged, [padded(position)])
    utils.warpgroup_barrier()
    element = ir.VectorType.get((1,), ir.F32Type.get())
    registers = []
    for i in range(count):
        position = arith.addi(thread, arith.constant(index, i * THREADS))
        registers.append(vector.from_elements(element, [memref.load(staged, [padded(position)])]))
    utils.warpgroup_barrier()
    layout = fa.WGStridedFragLayout(shape=x.shape, vec_size=1)
    return fa.FragmentedArray(_registers=np.array(registers, dtype=object), _layout=layout, _is_signed=None)


def summed(values: jax.Array, exchanged, *, interpret: bool = False) -> jax.Array:
    """The sum of `values`, fp32 `[T]` in `blocked(T)`'s layout, a scalar: a thread's positions in its
    registers, its warp's by shuffles, the four warps' through `exchanged` (`scratch()`'s)."""
    positions = values.shape[0]
    if interpret:
        return values.sum()
    total: Callable[..., jax.Array] = plgpu.inline_mgpu(
        arg_types=(blocked(positions), plgpu.RefType()),
        return_type=plgpu.ShapeDtypeStruct((), jnp.dtype(jnp.float32), layout=plgpu.Layout.WG_SPLAT(())),
    )(lambda ctx, x, ref: _summed(x, ref))
    return total(values, exchanged)


def _summed(x: fa.FragmentedArray, exchanged):
    """`summed` in Mosaic GPU's terms."""
    index, i32 = ir.IndexType.get(), ir.IntegerType.get_signless(32)
    [vec] = x.registers.flat
    count = ir.VectorType(vec.type).shape[0]
    total = vector.extract(vec, [], [0])
    for i in range(1, count):
        total = arith.addf(total, vector.extract(vec, [], [i]))
    for distance in (16, 8, 4, 2, 1):
        total = arith.addf(total, utils.shfl_bfly(total, distance))
    thread = arith.remui(utils.thread_idx(), arith.constant(i32, THREADS))
    warp = arith.index_cast(index, arith.divui(thread, arith.constant(i32, WARP)))
    utils.warpgroup_barrier()
    with utils.when(arith.cmpi(arith.CmpIPredicate.eq, arith.remui(thread, arith.constant(i32, WARP)),
                               arith.constant(i32, 0))):
        memref.store(total, exchanged, [arith.constant(index, 0), warp])
    utils.warpgroup_barrier()
    total = memref.load(exchanged, [arith.constant(index, 0), arith.constant(index, 0)])
    for w in range(1, THREADS // WARP):
        column = [arith.constant(index, 0), arith.constant(index, w)]
        total = arith.addf(total, memref.load(exchanged, column))
    utils.warpgroup_barrier()
    return fa.FragmentedArray.splat(total, ())


def _interpreted(decays, writes, carried, reverse: bool, exclusive: bool):
    """`linear_scan`'s states by XLA's associative scan."""
    def combined(first, then):
        return first[0] * then[0], then[0] * first[1] + then[1]

    a, b = lax.associative_scan(combined, (decays, writes), reverse=reverse)
    states = a * carried + b
    last = states[0] if reverse else states[-1]
    if exclusive:
        # Each position's state the one before it in the scan's order, the carried state's at its first.
        before = jnp.reshape(carried, (1,)).astype(states.dtype)
        states = (jnp.concatenate([states[1:], before]) if reverse
                  else jnp.concatenate([before, states[:-1]]))
    return states, last


def _pair(first, then):
    """Two pairs `(a, b)` combined, `first`'s positions before `then`'s in the scan's order."""
    return arith.mulf(first[0], then[0]), arith.addf(arith.mulf(then[0], first[1]), then[1])


def _shuffled(pair, distance: int, reverse: bool):
    """`pair` from the lane `distance` before this one in the scan's order (its own where none is)."""
    i32 = ir.IntegerType.get_signless(32)
    # A shuffle down clamps at the warp's last lane, one up at its first (CUDA's __shfl_down_sync's and
    # __shfl_up_sync's operands at a warp's width).
    kind, clamp = (nvvm.ShflKind.down, WARP - 1) if reverse else (nvvm.ShflKind.up, 0)
    mask, offset, bound = (arith.constant(i32, value) for value in (0xFFFFFFFF, distance, clamp))
    return tuple(nvvm.shfl_sync(mask, value, offset, bound, kind) for value in pair)


def _scanned(a: fa.FragmentedArray, b: fa.FragmentedArray, carried: fa.FragmentedArray, exchanged,
             reverse: bool, exclusive: bool):
    """`linear_scan` in Mosaic GPU's terms: a thread's positions in its registers, its warp's by shuffles,
    the warpgroup's warps through `exchanged`."""
    f32, i32, index = ir.F32Type.get(), ir.IntegerType.get_signless(32), ir.IndexType.get()
    # A thread's positions one vector (`blocked`).
    [a_vector], [b_vector] = a.registers.flat, b.registers.flat
    count = ir.VectorType(a_vector.type).shape[0]
    order = range(count - 1, -1, -1) if reverse else range(count)
    one, zero = arith.constant(f32, 1.0), arith.constant(f32, 0.0)

    def element(vec, i):
        return vector.extract(vec, [], [i])

    # The thread's own positions, each pair the scan of the thread's so far, through it or before it.
    running, pairs = (one, zero), {}
    for i in order:
        through = _pair(running, (element(a_vector, i), element(b_vector, i)))
        pairs[i] = running if exclusive else through
        running = through
    # The warp's threads before this one: an inclusive scan of the threads' aggregates by shuffles.
    thread = arith.remui(utils.thread_idx(), arith.constant(i32, THREADS))
    lane = arith.remui(thread, arith.constant(i32, WARP))
    lane = arith.subi(arith.constant(i32, WARP - 1), lane) if reverse else lane
    warp_total = running
    for distance in (1, 2, 4, 8, 16):
        before = _shuffled(warp_total, distance, reverse)
        has = arith.cmpi(arith.CmpIPredicate.uge, lane, arith.constant(i32, distance))
        joined = _pair(before, warp_total)
        warp_total = tuple(arith.select(has, j, w) for j, w in zip(joined, warp_total, strict=True))
    # This thread's exclusive prefix within its warp: the inclusive one a lane before, the identity at the
    # warp's first lane.
    shifted = _shuffled(warp_total, 1, reverse)
    first_lane = arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(i32, 0))
    within = tuple(arith.select(first_lane, identity, s)
                   for identity, s in zip((one, zero), shifted, strict=True))
    # The warps' aggregates through shared memory, each warp's last lane in the scan's order its own.
    warp = arith.divui(thread, arith.constant(i32, WARP))
    warps = THREADS // WARP
    order_warp = arith.subi(arith.constant(i32, warps - 1), warp) if reverse else warp
    utils.warpgroup_barrier()
    last_lane = arith.cmpi(arith.CmpIPredicate.eq, lane, arith.constant(i32, WARP - 1))
    column = arith.index_cast(index, order_warp)
    with utils.when(last_lane):
        for row, value in enumerate(warp_total):
            memref.store(value, exchanged, [arith.constant(index, row), column])
    utils.warpgroup_barrier()
    before_warp, total = (one, zero), (one, zero)
    for w in range(warps):
        aggregate = tuple(memref.load(exchanged, [arith.constant(index, row), arith.constant(index, w)])
                          for row in (0, 1))
        earlier = arith.cmpi(arith.CmpIPredicate.ult, arith.constant(i32, w), order_warp)
        joined = _pair(before_warp, aggregate)
        before_warp = tuple(arith.select(earlier, j, p) for j, p in zip(joined, before_warp, strict=True))
        total = _pair(total, aggregate)
    utils.warpgroup_barrier()
    # The state before this thread's first position, then each of its positions'.
    [h] = carried.registers.flat
    prefix = _pair(before_warp, within)
    entering = arith.addf(arith.mulf(prefix[0], h), prefix[1])
    states = [arith.addf(arith.mulf(pairs[i][0], entering), pairs[i][1]) for i in range(count)]
    registers = a.registers.copy()
    registers.flat[0] = vector.from_elements(a_vector.type, states)
    last = arith.addf(arith.mulf(total[0], h), total[1])
    return (fa.FragmentedArray(_registers=registers, _layout=a.layout, _is_signed=None),
            fa.FragmentedArray.splat(last, ()))
