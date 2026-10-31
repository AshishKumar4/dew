"""The toolkit's warpgroup scan (dew.nn.kernels.templates.scan) of a first-order linear recurrence over a
block of positions: forward and reverse, from a carried state, against the recurrence walked in float64.
Interpreted, the scan is XLA's; its Mosaic GPU form lowers for an A100 and an H100 here and runs on one in
the GPU gate (test_gpu_gemm)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.sharding import Mesh

from dew.nn.kernels import generation
from dew.nn.kernels.templates import mosaic, scan

pytestmark = pytest.mark.kernel

POSITIONS = (1024, 2048)
"""A block of 8 positions a thread and one of 16, mamba_ssm's selective scan backward's (kNItems)."""

CARRIED = 0.75


def _scanned(decays, writes, *, reverse: bool, exclusive: bool = False, interpret):
    """`scan.linear_scan`'s states of `decays` and `writes` `[T]` from `CARRIED`, the state past them, and
    `scan.summed`'s sum of the writes, in one warpgroup's kernel, its operands read from and its results
    written to global memory directly."""
    positions = decays.shape[0]

    def body(a_gmem, b_gmem, states_gmem, last_gmem, total_gmem, exchanged):
        layout = scan.blocked(positions)
        a, b = (plgpu.load(gmem, layout=layout, optimized=False) for gmem in (a_gmem, b_gmem))
        interpreted = interpret is not None
        states, last = scan.linear_scan(a, b, jnp.full((positions,), CARRIED, jnp.float32), exchanged,
                                        reverse=reverse, exclusive=exclusive, interpret=interpreted)
        plgpu.store(states_gmem, states, optimized=False)
        plgpu.store(last_gmem, last, optimized=False)
        plgpu.store(total_gmem, scan.summed(b, exchanged, interpret=interpreted), optimized=False)

    vector = jax.ShapeDtypeStruct((positions,), jnp.float32)
    return plgpu.kernel(body, out_type=(vector, vector, vector), grid=(1,), grid_names=("block",),
                        scratch_types=(scan.scratch(),), compiler_params=mosaic.compiler_params(),
                        interpret=interpret)(decays, writes)


def _recurrence(positions: int, seed: int = 0):
    """Decays in (0.9, 1) and writes of either sign: a state that grows over the block, as a decay near
    one carries it."""
    keys = jax.random.split(jax.random.key(seed))
    decays = jax.random.uniform(keys[0], (positions,), jnp.float32, 0.9, 1.0)
    writes = jax.random.normal(keys[1], (positions,), jnp.float32)
    return decays, writes


def _walked(decays, writes, reverse: bool, exclusive: bool) -> tuple[np.ndarray, float]:
    """The recurrence walked position by position in float64: each position's state (the one before it in
    the walk's order where `exclusive`) and the state past the block."""
    a, b = (np.asarray(x, np.float64) for x in (decays, writes))
    order = range(len(a) - 1, -1, -1) if reverse else range(len(a))
    states, h = np.empty_like(a), CARRIED
    for t in order:
        before, h = h, a[t] * h + b[t]
        states[t] = before if exclusive else h
    return states, h


def _assert_walked(found, decays, writes, reverse: bool, exclusive: bool) -> None:
    """The kernel's states, last state and sum within an fp32 scan's rounding of the walk's: each one's
    error against the largest state's size (the writes' largest for the sum), which a sum of `T` fp32
    products of decays below one keeps under `T` roundings."""
    states, last, total = (np.asarray(x, np.float64) for x in found)
    truth, past = _walked(decays, writes, reverse, exclusive)
    rounding = len(truth) * np.finfo(np.float32).eps
    size = max(np.abs(truth).max(), abs(past))
    np.testing.assert_array_less(np.abs(states - truth), rounding * size)
    np.testing.assert_array_less(np.abs(last - past), rounding * size)
    b = np.asarray(writes, np.float64)
    np.testing.assert_array_less(np.abs(total - b.sum()), rounding * np.abs(b).sum())


@pytest.mark.parametrize("exclusive", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("positions", POSITIONS)
def test_the_interpreted_scan_walks_the_recurrence(positions, reverse, exclusive):
    """Interpreted, the scan's states are the recurrence's from the carried state, forward and reverse,
    each position's own or the one before it, and the sum the writes'."""
    decays, writes = _recurrence(positions)
    found = _scanned(decays, writes, reverse=reverse, exclusive=exclusive,
                     interpret=plgpu.InterpretGPUParams())
    _assert_walked(found, decays, writes, reverse, exclusive)


@pytest.mark.parametrize("target", ["sm80", "sm90"])
@pytest.mark.parametrize("exclusive", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_the_scan_lowers_for_each_family(target, reverse, exclusive):
    """The scan's Mosaic GPU form lowers for an A100 and an H100 on any host, as the GPU gate runs it."""
    vector = jax.ShapeDtypeStruct((POSITIONS[-1],), jnp.float32)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), target)
    scanned = functools.partial(_scanned, reverse=reverse, exclusive=exclusive, interpret=None)
    with generation.assumed(target), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(scanned).trace(vector, vector).lower(lowering_platforms=("cuda",))


def the_scan_walks_the_recurrence_on_this_gpu(positions: int, reverse: bool, exclusive: bool) -> None:
    """The scan compiled and run on this GPU: its shuffles and its warps' exchange as the hardware runs
    them."""
    decays, writes = _recurrence(positions, seed=1)
    scanned = functools.partial(_scanned, reverse=reverse, exclusive=exclusive, interpret=None)
    _assert_walked(jax.jit(scanned)(decays, writes), decays, writes, reverse, exclusive)
