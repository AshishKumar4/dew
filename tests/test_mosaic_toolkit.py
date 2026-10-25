"""The Mosaic toolkit (dew.nn.kernels.templates.mosaic) under Mosaic GPU's interpreter: a ring streaming one
operand across persistent tiles of their own trip counts, one consumer or two sharing its slots, and the
wgmma family's walk keeping each step's slot a step past it. The interpreter is coverage, not proof (it
treats wgmma_wait as a no-op); each kernel also lowers for an H100 here, and runs on one in the GPU gate."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.sharding import Mesh
from reference_error import FEWEST_SEEDS, assert_as_exact

from dew.nn.kernels import generation
from dew.nn.kernels.templates import mosaic, mosaic_sm90

pytestmark = pytest.mark.kernel

BLOCK = (64, 64)
INTERPRETED = plgpu.InterpretGPUParams()
"""Mosaic GPU's own interpreter, which models warpgroups, barriers and shared memory."""
TILES = 3
STAGES = 2


def _first_step(tile):
    """A tile's first step when tile `t` takes `t + 1` steps: the steps the tiles before it took."""
    return tile * (tile + 1) // 2


def _ring_sums(x, *, interpret, consumers: int = 1):
    """Each tile's sum of its steps' blocks of `x` (tile `t` takes `t + 1`), streamed through a ring whose
    counter runs across the tiles; with two consumers each sums half the columns."""
    columns = BLOCK[1] // consumers

    def body(x_gmem, out_gmem):
        @functools.partial(pl.run_scoped, **mosaic.ring(BLOCK, x.dtype, STAGES, consumers=consumers),
                           out_smem=plgpu.SMEM((consumers, BLOCK[0], columns), jnp.float32),
                           collective_axes="wg")
        def _(smem, full, consumed, out_smem):
            ring = mosaic.Ring(smem, full, consumed)
            wg = jax.lax.axis_index("wg")

            def producer():
                @plgpu.nd_loop((TILES,), collective_axes="sm")
                def _(loop: plgpu.NDLoopInfo):
                    tile = loop.index[0]
                    first = _first_step(tile)

                    @pl.loop(0, tile + 1)
                    def _(step):
                        ring.load(first + step, x_gmem.at[pl.ds((first + step) * BLOCK[0], BLOCK[0])])

                ring.drain(jnp.int32(_first_step(TILES)))

            def consumer(wg):
                @plgpu.nd_loop((TILES,), collective_axes="sm")
                def _(loop: plgpu.NDLoopInfo):
                    tile = loop.index[0]
                    first = _first_step(tile)

                    def step_sum(step, total):
                        slot = ring.acquire(first + step)
                        block = slot[:, pl.ds(wg * columns, columns)].astype(jnp.float32)
                        ring.release(first + step)
                        return total + block

                    zero = jnp.zeros((BLOCK[0], columns), jnp.float32)
                    total = jax.lax.fori_loop(0, tile + 1, step_sum, zero)
                    plgpu.wait_smem_to_gmem(0)
                    # Each consumer's own staging buffer: the warpgroups share the scope's allocations.
                    out_smem[wg] = total
                    plgpu.commit_smem()
                    mine = out_gmem.at[tile, :, pl.ds(wg * columns, columns)]
                    plgpu.copy_smem_to_gmem(out_smem.at[wg], mine)
                plgpu.wait_smem_to_gmem(0)

            mosaic.roles(wg, consumers, consumer, producer)

    return plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((TILES, *BLOCK), jnp.float32), grid=(1,),
                        grid_names=("sm",), num_threads=consumers + 1, thread_name="wg",
                        compiler_params=mosaic.compiler_params(), interpret=interpret)(x)


def _expected(x):
    blocks = np.asarray(x, np.float32).reshape(-1, *BLOCK)
    return np.stack([blocks[_first_step(tile):_first_step(tile) + tile + 1].sum(0) for tile in range(TILES)])


def _stream():
    """Small integers, so any tile's sum is exact in fp32 whatever its order."""
    steps = _first_step(TILES)
    return jax.random.randint(jax.random.key(0), (steps * BLOCK[0], BLOCK[1]), -64, 64).astype(jnp.bfloat16)


@INTERPRETS
@pytest.mark.parametrize("consumers", [1, 2])
def test_a_ring_streams_tiles_of_their_own_trip_counts(consumers):
    """A ring's counter runs across persistent tiles of one, two and three steps over two slots: each
    tile sums exactly its own steps' blocks, one consumer or two sharing each slot and meeting at each
    step."""
    x = _stream()
    summed = _ring_sums(x, interpret=INTERPRETED, consumers=consumers)
    np.testing.assert_array_equal(np.asarray(summed), _expected(x))


@pytest.mark.parametrize("consumers", [1, 2])
def test_the_ring_lowers_for_an_h100(consumers):
    """The ring's kernel lowers for an H100 on any host, as the GPU gate runs it."""
    x = jax.ShapeDtypeStruct(_stream().shape, jnp.bfloat16)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(functools.partial(_ring_sums, interpret=False, consumers=consumers)).trace(x).lower(
            lowering_platforms=("cuda",))


RACES = plgpu.InterpretGPUParams(detect_races=True)
"""The interpreter with its happens-before race detector, which raises where a load overwrites a slot a
consumer still reads (a slot given back early) or a read meets a load it did not wait for."""


def _kept_sums(x, *, interpret, consumers: int = 1):
    """Each tile's sum of its steps' blocks of `x` (tile `t` takes `t + 1`), each consumer its rows, through
    the wgmma family's walk keeping each step's slot past it (`Family.walk`'s `kept`): the tile's first step
    reads nothing, each later step adds the block before its own, read from the kept slot by a product
    (`Slots.previous`), and a step past the last adds the last (`last`). The ring's counter runs across the
    tiles, so a tile's last kept slot goes back before the next tiles' loads reuse it."""
    family = mosaic_sm90.WGMMA
    rows, columns = family.role_rows, BLOCK[1]
    block = (consumers * rows, columns)
    steps = _first_step(TILES)

    def body(x_gmem, out_gmem):
        def scope(out_smem, *, streams):
            rings = family.opened(streams)

            def producer():
                @plgpu.nd_loop((TILES,), collective_axes="sm")
                def _(loop: plgpu.NDLoopInfo):
                    tile = loop.index[0]

                    @pl.loop(0, tile + 1)
                    def _(step):
                        index = _first_step(tile) + step
                        rings["x"].load(index, x_gmem.at[pl.ds(index * block[0], block[0])])

                rings["x"].drain(jnp.int32(steps))

            def consumer(wg):
                # The consumer's rows of a block, picked by a product.
                picked = family.iota((rows, block[0]), 1) == family.iota((rows, block[0]), 0) + wg * rows
                select = jnp.where(picked, 1.0, 0.0).astype(x.dtype)

                def opening(index, slots, carry, accumulators):
                    del index, slots
                    return carry, accumulators

                def added(index, slots, carry, accumulators):
                    del index
                    acc = family.register_products(accumulators[0], ((select, None),), slots.previous("x"),
                                                   bool(interpret))
                    return carry, (acc,)

                @plgpu.nd_loop((TILES,), collective_axes="sm")
                def _(loop: plgpu.NDLoopInfo):
                    tile = loop.index[0]
                    first = _first_step(tile)
                    segments = ((opening, first, first + 1), (added, first + 1, first + tile + 1))
                    _, (acc,) = family.walk(rings, segments, (), ((rows, columns),), bool(interpret),
                                            kept=("x",), last=added)
                    plgpu.wait_smem_to_gmem(0)
                    out_smem[wg] = acc
                    plgpu.commit_smem()
                    plgpu.copy_smem_to_gmem(out_smem.at[wg], out_gmem.at[tile, pl.ds(wg * rows, rows)])

                plgpu.wait_smem_to_gmem(0)

            family.roles(consumers, consumer, producer)

        family.scoped(scope, plgpu.SMEM((consumers, rows, columns), jnp.float32),
                      streams=family.streams({"x": (block, x.dtype)}, STAGES, consumers))

    return family.kernel(body, out_type=jax.ShapeDtypeStruct((TILES, *block), jnp.float32), grid=(1,),
                         grid_names=("sm",), consumers=consumers, interpret=interpret)(x)


def _kept_stream(consumers: int, seed: int = 0):
    """Small integers in blocks of each consumer's rows, drawn by `seed`, so any tile's sum is exact in
    fp32."""
    rows = mosaic_sm90.WGMMA.role_rows * consumers
    shape = (_first_step(TILES) * rows, BLOCK[1])
    return jax.random.randint(jax.random.key(seed), shape, -64, 64).astype(jnp.bfloat16)


def _check_kept(consumers: int, *, interpret) -> None:
    """Check 1 (`assert_as_exact`) of the kept walk's tile sums against the blocks' sums in fp32: the
    stream's small integers make every sum exact, so the reference equals float64 and the walk must too."""
    def kept(x):
        return _kept_sums(x, interpret=interpret, consumers=consumers)

    def expected(x):
        rows = mosaic_sm90.WGMMA.role_rows * consumers
        blocks = x.astype(jnp.promote_types(x.dtype, jnp.float32)).reshape(-1, rows, BLOCK[1])
        return jnp.stack([blocks[_first_step(tile):_first_step(tile) + tile + 1].sum(0)
                          for tile in range(TILES)])

    assert_as_exact(jax.jit(kept), jax.jit(expected), lambda seed: (_kept_stream(consumers, seed),),
                    f"kept walk, {consumers} consumers", seeds=FEWEST_SEEDS)


@INTERPRETS
@pytest.mark.parametrize("consumers", [1, 2])
def test_a_walk_keeps_each_slot_a_step_across_tiles(consumers):
    """A walk keeping each step's slot past it, over persistent tiles of one, two and three steps through
    two slots: each tile sums exactly its own steps' blocks, read a step late from the kept slots, and the
    race detector finds no load overwriting a slot a consumer still reads, so each kept slot goes back once,
    the tile's last at its walk's end, and the ring's counter stays whole across the tiles."""
    _check_kept(consumers, interpret=RACES)


@pytest.mark.parametrize("consumers", [1, 2])
def test_the_kept_walk_lowers_for_an_h100(consumers):
    """The kept walk's kernel lowers for an H100 on any host, as the GPU gate runs it."""
    x = jax.ShapeDtypeStruct(_kept_stream(consumers).shape, jnp.bfloat16)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(functools.partial(_kept_sums, interpret=False, consumers=consumers)).trace(x).lower(
            lowering_platforms=("cuda",))


ON_HOPPER = pytest.mark.skipif(generation.device_generation() != "sm90",
                               reason="runs the toolkit's kernels compiled, on an H100")


@ON_HOPPER
@pytest.mark.gpu(reason="runs the toolkit's kernels compiled, on an H100")
@pytest.mark.parametrize("consumers", [1, 2])
def test_a_ring_streams_tiles_of_their_own_trip_counts_on_an_h100(consumers):
    """The ring's kernel compiled and run on an H100, where the interpreter's no-op waits are real."""
    x = _stream()
    summed = jax.jit(functools.partial(_ring_sums, interpret=False, consumers=consumers))(x)
    np.testing.assert_array_equal(np.asarray(summed), _expected(x))


@ON_HOPPER
@pytest.mark.gpu(reason="runs the toolkit's kernels compiled, on an H100")
@pytest.mark.parametrize("consumers", [1, 2])
def test_a_walk_keeps_each_slot_a_step_across_tiles_on_an_h100(consumers):
    """The kept walk's kernel compiled and run on an H100, where the waits and arrivals are real."""
    _check_kept(consumers, interpret=False)
