"""The Mosaic toolkit (dew.nn.kernels.templates.mosaic) under Mosaic GPU's interpreter: a ring streaming one
operand across persistent tiles of their own trip counts, one consumer or two sharing its slots. The
interpreter is coverage, not proof (it treats wgmma_wait as a no-op); each kernel also lowers for an H100
here, and runs on one in the GPU gate."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.sharding import Mesh

from dew.nn.kernels import generation
from dew.nn.kernels.templates import mosaic

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


FED_STEPS = 5
FED_ROWS, FED_WIDTH = 64, 128


def _fed_products(p, x, *, interpret, consumers: int = 2, lo: int = 0):
    """`p @ x` and `2 * p @ x` over `FED_STEPS` steps of 64 rows of `x` (`p`'s 64 columns a step), on
    warpgroup MMA without a producer (`mosaic_sm90.FED`): every consumer walks every step of one ring the
    first fills (`Family.fed`), each multiplying its share of the step's rows (two consumers: a 32-row
    contraction, its blocks staged at `mosaic.swizzle_for`'s 64 bytes), and their sums meet before one
    store each, twice in one scope reused (`mosaic.summed_across`), in a scope after the rings'
    (`Family.scoped` returns); the walk's steps counted from `lo`, its ring counters from 0 (`first`)."""
    from dew.nn.kernels.templates import mosaic_sm90
    family, share = mosaic_sm90.FED, FED_ROWS // consumers
    swizzle = mosaic.swizzle_for(jnp.bfloat16, share)

    def body(p_gmem, x_gmem, out_gmem, doubled_gmem):
        allocated = family.streams({'p': ((FED_ROWS, FED_ROWS), jnp.bfloat16),
                                    'x': ((FED_ROWS, FED_WIDTH), jnp.bfloat16)}, STAGES, consumers,
                                   swizzles={'p': swizzle, 'x': swizzle})
        wg = jax.lax.axis_index("wg")

        def walked(*, streams):
            first = jnp.int32(lo)
            # A plain walk's views take its own step indices, as a producer's stream gives them.
            views = {'p': (p_gmem, lambda j: (0, j - first)), 'x': (x_gmem, lambda j: (j - first, 0))}
            feed = family.fed(family.opened(streams), views, first, first + FED_STEPS, wg)

            def step(at, slots, carry, accumulators):
                del at
                (acc,) = accumulators
                block, rows = slots['p'], slots['x']
                product = family.products(acc, block.at[:, pl.ds(wg * share, share)],
                                          rows.at[pl.ds(wg * share, share)], interpret)
                return carry, (product,)

            return family.walk(feed, ((step, first, first + FED_STEPS),), (), ((FED_ROWS, FED_WIDTH),),
                               interpret, first=first)[1][0]

        summed = family.scoped(walked, streams=allocated)

        def total(**scratch):
            for value, out in ((summed, out_gmem), (summed * 2, doubled_gmem)):
                mosaic.summed_across(value, scratch, wg, consumers,
                                     lambda value, out=out: out.__setitem__(slice(None), value))

        pl.run_scoped(total, **mosaic.summing((FED_ROWS, FED_WIDTH), consumers), collective_axes="wg")

    out = jax.ShapeDtypeStruct((FED_ROWS, FED_WIDTH), jnp.float32)
    return family.kernel(body, out_type=(out, out), grid=(1,), grid_names=("programs",), consumers=consumers,
                         interpret=interpret, name="fed")(p, x)


def _fed_operands():
    """Small integers, so the product is exact in fp32 whatever its order."""
    keys = jax.random.split(jax.random.key(1))
    p = jax.random.randint(keys[0], (FED_ROWS, FED_STEPS * FED_ROWS), -4, 4).astype(jnp.bfloat16)
    x = jax.random.randint(keys[1], (FED_STEPS * FED_ROWS, FED_WIDTH), -4, 4).astype(jnp.bfloat16)
    return p, x


def _fed_expected(p, x):
    product = np.asarray(p, np.float32) @ np.asarray(x, np.float32)
    return product, 2 * product


@INTERPRETS
@pytest.mark.parametrize(("consumers", "lo"), [(1, 0), (2, 0), (2, 3)])
def test_consumers_feeding_their_own_ring_split_a_step_and_meet_before_the_store(consumers, lo):
    """Interpreted: consumers with no producer walk every step of one ring through slot reuse (five steps
    over two slots), each its share of a step's rows, and store their sum once, twice through one scratch:
    exactly the product and its double; a walk whose steps start past 0 (a later key block's), its ring
    counters from its own first (`first`), alike."""
    p, x = _fed_operands()
    got = _fed_products(p, x, interpret=INTERPRETED, consumers=consumers, lo=lo)
    for value, expected in zip(got, _fed_expected(p, x), strict=True):
        np.testing.assert_array_equal(np.asarray(value), expected)


@pytest.mark.parametrize("consumers", [1, 2])
def test_the_fed_walk_lowers_for_an_h100(consumers):
    """The fed walk's kernel, two consumers' 32-row contraction staged at 64 bytes, lowers for an H100 on any
    host."""
    p, x = (jax.ShapeDtypeStruct(each.shape, each.dtype) for each in _fed_operands())
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(functools.partial(_fed_products, interpret=False, consumers=consumers)).trace(p, x).lower(
            lowering_platforms=("cuda",))


@ON_HOPPER
@pytest.mark.gpu(reason="runs the toolkit's kernels compiled, on an H100")
@pytest.mark.parametrize(("consumers", "lo"), [(1, 0), (2, 0), (2, 3)])
def test_consumers_feeding_their_own_ring_split_a_step_on_an_h100(consumers, lo):
    """The fed walk compiled and run on an H100, where the interpreter's no-op waits are real."""
    p, x = _fed_operands()
    got = jax.jit(functools.partial(_fed_products, interpret=False, consumers=consumers, lo=lo))(p, x)
    for value, expected in zip(got, _fed_expected(p, x), strict=True):
        np.testing.assert_array_equal(np.asarray(value), expected)


def test_shared_bytes_counts_a_programs_allocations_by_lifetime():
    """A program's shared memory (`resources.shared_bytes`): each array rounded up to 1 KiB, its walks the
    largest of them, every barrier 8 bytes; a tile scoped after its walks beside the program's on a family
    whose walks share a scope (warpgroup MMA), the larger of it and the walks on one that scopes each walk
    apart (mma.sync's emit_pipeline)."""
    from dew.nn.kernels.templates import mosaic_sm80, mosaic_sm90, resources

    program = (plgpu.SMEM((100,), jnp.float32), plgpu.Barrier(num_barriers=3))
    walks = ((plgpu.SMEM((2, 1024), jnp.bfloat16),), (plgpu.SMEM((4, 1024), jnp.bfloat16),))
    scope = resources.Scope(program=program, walks=walks, after=(((512,), jnp.float32),))
    # 400 bytes -> 1 KiB; walks 4 KiB and 8 KiB -> 8 KiB; 3 barriers; the after tile's 2 KiB.
    assert resources.shared_bytes(mosaic_sm90.WGMMA, scope) == 1024 + 2048 + 8192 + 3 * 8
    assert resources.shared_bytes(mosaic_sm80.Mma(jnp.dtype(jnp.bfloat16)), scope) == 1024 + 8192 + 3 * 8
    # Three warpgroups reducing implicitly each take their share of Mosaic GPU's scratch on warpgroup MMA.
    reducing = dataclasses.replace(scope, reducing_warpgroups=3)
    assert resources.shared_bytes(mosaic_sm90.WGMMA, reducing) == 1024 + 3 * 2048 + 2048 + 8192 + 3 * 8


def test_a_plan_keeps_its_residency_and_takes_the_deepest_ring_it_holds():
    """`resources.plan` on an A100 for one mma.sync consumer (two programs a multiprocessor, 81 KiB each):
    a launch of 16 KiB a stage holds two programs at four stages; one of 40 KiB a stage holds two programs
    at its floor of two only where it fits, else falls to one program and deepens there; one past every
    residency at its floor plans nothing."""
    from dew.nn.kernels.templates import mosaic_sm80, resources

    family = mosaic_sm80.Mma(jnp.dtype(jnp.bfloat16))

    def launch(stage_kib: int) -> resources.Launch:
        return resources.Launch(lambda stages: resources.Scope(
            program=(plgpu.SMEM((stages * stage_kib * 1024,), jnp.uint8),)), consumers=1, registers=128)

    with generation.assumed("sm80"):
        target = generation.gpu_target()
        share = resources.program_shared(generation.shared_budget(target), 2)
        small, wide, past = (resources.plan([launch(kib)], family, target) for kib in (16, 40, 200))
    assert small is not None and (small[0].programs, small[0].stages) == (2, min(8, share // (16 * 1024)))
    assert wide is not None and wide[0].programs == (2 if share >= 2 * 40 * 1024 else 1)
    assert past is None
