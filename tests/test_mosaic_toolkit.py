"""The Mosaic toolkit (dew.nn.kernels.templates.mosaic) under Mosaic GPU's interpreter: a ring streaming one
operand across persistent tiles of their own trip counts, one consumer or two sharing its slots. The
interpreter is coverage, not proof (it treats wgmma_wait as a no-op); each kernel also lowers for an H100
here, and runs on one in the GPU gate."""

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
