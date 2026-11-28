"""The Mosaic toolkit's consumer-fed walks (dew.nn.kernels.templates.fed, the sm80 cp.async family's shared
pipeline): consumers feeding their own ring or sharing one collective pipeline, each its share of a step's
rows, their sums meeting before one store; interpreted, lowered for their GPUs, and run on them in the GPU
gate (tests/test_gpu_gemm.py)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import INTERPRETS
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.sharding import Mesh
from reference_error import ahead_arm

from dew.nn.kernels import generation
from dew.nn.kernels.templates import across, bundles, swizzling

pytestmark = pytest.mark.kernel

INTERPRETED = plgpu.InterpretGPUParams()
"""Mosaic GPU's own interpreter, which models warpgroups, barriers and shared memory."""
STAGES = 2

FED_STEPS = 5
FED_ROWS, FED_WIDTH = 64, 128


def _fed_products(p, x, *, interpret, consumers: int = 2, lo: int = 0, unrolled: bool = False,
                  in_flight: int = 0, shared: bool = False, split: bool = False, agreed: bool = True,
                  common: bool = False):
    """`p @ x` and `2 * p @ x` over `FED_STEPS` steps of 64 rows of `x` (`p`'s 64 columns a step), on
    warpgroup MMA without a producer (`mosaic_sm90.FED`): every consumer walks every step of one ring the
    first fills (`Family.fed`), each multiplying its share of the step's rows (two consumers: a 32-row
    contraction, its blocks staged at `swizzling.swizzle_for`'s 64 bytes), and their sums meet before one
    store each, twice in one scope reused (`across.summed_across`), in a scope after the rings'
    (`Family.scoped` returns); the walk's steps counted from `lo`, its ring counters from 0 (`first`).
    `shared`: on mma.sync, its consumers running one cp.async pipeline together (`CpAsync.shared_consumers`,
    each copying its share of every block and meeting the other before a slot is read and refilled);
    `split`, its walk in segments of their own weights: an empty one, a first step's alone, a gap its next
    step falls in (walked by none), the rest past a boundary each consumer's own (a causal call's masked
    segment, which one consumer's keys end a step sooner in) at twice the weight (`_split_weights`), and an
    empty one past the operands' blocks; its steps loaded as `walk_own` agrees them (`agreed`, else walked
    directly); `common`, the boundary every consumer's alike, stated so (`walk_own`'s `common`)."""
    from dew.nn.kernels.templates import mosaic_sm90
    share = FED_ROWS // consumers
    family = bundles.named("mma.sync", jnp.bfloat16, None) if shared else mosaic_sm90.FED
    swizzle = swizzling.swizzle_for(jnp.bfloat16, share)
    threaded = consumers > 1 or not shared

    def body(p_gmem, x_gmem, out_gmem, doubled_gmem):
        allocated = family.streams({'p': ((FED_ROWS, FED_ROWS), jnp.bfloat16),
                                    'x': ((FED_ROWS, FED_WIDTH), jnp.bfloat16)}, STAGES, consumers,
                                   swizzles={'p': swizzle, 'x': swizzle})
        wg = jax.lax.axis_index("wg") if threaded else 0

        def walked(*, streams):
            first = jnp.int32(lo)
            # A plain walk's views take its own step indices, as a producer's stream gives them.
            views = {'p': (p_gmem, lambda j: (0, j - first)), 'x': (x_gmem, lambda j: (j - first, 0))}
            feed = (family.opened(streams, views) if shared
                    else family.fed(family.opened(streams), views, first, first + FED_STEPS, wg))

            def step(at, slots, carry, accumulators, *, weight: int = 1):
                del at
                (acc,) = accumulators
                block, rows = slots['p'], slots['x']
                for _ in range(weight):
                    acc = family.products(acc, block.at[:, pl.ds(wg * share, share)],
                                          rows.at[pl.ds(wg * share, share)])
                return carry, (acc,)

            middle = first + 3 + (0 if common else wg)
            twice = functools.partial(step, weight=2)
            whole = ((step, first, first + FED_STEPS),)
            # An empty segment past the operands' blocks last, whose tiles no trip may load.
            past = first + FED_STEPS + 3
            segments = ((step, first, first), (step, first, first + 1), (step, first + 2, middle),
                        (twice, middle, first + FED_STEPS), (step, past, past)) if split else whole
            def walk(streams, lo, hi, start):
                del lo, hi, start
                return family.walk(streams, segments, (), ((FED_ROWS, FED_WIDTH),), interpret, first=first,
                                   unrolled=unrolled, in_flight=in_flight)[1][0]

            if shared and agreed:
                streamed = (first, first + FED_STEPS)
                return family.walk_own(feed, views, streamed, streamed, wg, walk, common=common)
            return walk(feed, first, first + FED_STEPS, first)

        summed = family.scoped(walked, streams=allocated)

        def total(**scratch):
            for value, out in ((summed, out_gmem), (summed * 2, doubled_gmem)):
                across.summed_across(value, scratch, wg, consumers,
                                     lambda value, out=out: out.__setitem__(slice(None), value))

        swizzling.run_scoped(total, **across.summing((FED_ROWS, FED_WIDTH), consumers),
                             **({"collective_axes": "wg"} if threaded else {}))

    out = jax.ShapeDtypeStruct((FED_ROWS, FED_WIDTH), jnp.float32)
    return family.kernel(body, out_type=(out, out), grid=(1,), grid_names=("programs",), consumers=consumers,
                         interpret=interpret, name="shared" if shared else "fed")(p, x)


def _fed_staged_over_rings(p, x, *, interpret, ordered: bool):
    """Two consumers without a producer walk `_fed_products`' ring, each its share of a step's rows, and
    stage their partial products after their walks in a tile over the ring's slots
    (`Family.stages_over_rings`: the slots and the tile one union, `after_walk.over_rings_scoped`), each its
    own `[64, 128]` fp32 part, past the barrier every consumer passes first (`walked`), or, not `ordered`,
    without passing it: `[2, 64, 128]` partials, `p @ x` their sum."""
    from dew.nn.kernels.templates import mosaic_sm90
    family, consumers = mosaic_sm90.FED, 2
    share = FED_ROWS // consumers
    swizzle = swizzling.swizzle_for(jnp.bfloat16, share)

    def body(p_gmem, x_gmem, out_gmem):
        allocated = family.streams({'p': ((FED_ROWS, FED_ROWS), jnp.bfloat16),
                                    'x': ((FED_ROWS, FED_WIDTH), jnp.bfloat16)}, STAGES, consumers,
                                   swizzles={'p': swizzle, 'x': swizzle})
        wg = jax.lax.axis_index("wg")

        def program(*, streams, after, walked):
            views = {'p': (p_gmem, lambda j: (0, j)), 'x': (x_gmem, lambda j: (j, 0))}
            feed = family.fed(family.opened(streams), views, jnp.int32(0), jnp.int32(FED_STEPS), wg)

            def step(at, slots, carry, accumulators):
                del at
                (acc,) = accumulators
                block, rows = slots['p'], slots['x']
                return carry, (family.products(acc, block.at[:, pl.ds(wg * share, share)],
                                               rows.at[pl.ds(wg * share, share)]),)

            (part,) = family.walk(feed, ((step, 0, FED_STEPS),), (), ((FED_ROWS, FED_WIDTH),), interpret)[1]
            if ordered:
                walked()

            def staged(stage):
                stage.at[wg][...] = part
                out_gmem.at[wg][...] = stage.at[wg][...]

            after(staged)

        family.scoped(program, streams=allocated, over_rings=True,
                      after_walks=(plgpu.SMEM((consumers, FED_ROWS, FED_WIDTH), jnp.float32),))

    out = jax.ShapeDtypeStruct((consumers, FED_ROWS, FED_WIDTH), jnp.float32)
    return family.kernel(body, out_type=(out,), grid=(1,), grid_names=("programs",), consumers=consumers,
                         interpret=interpret, name="fed_over_rings")(p, x)


def _fed_operands():
    """Small integers, so the product is exact in fp32 whatever its order: NumPy's (a GPU tier's prepare
    computes on no device)."""
    rng = np.random.default_rng(1)
    p = rng.integers(-4, 4, (FED_ROWS, FED_STEPS * FED_ROWS)).astype(jnp.bfloat16)
    x = rng.integers(-4, 4, (FED_STEPS * FED_ROWS, FED_WIDTH)).astype(jnp.bfloat16)
    return p, x


def _fed_expected(p, x):
    product = np.asarray(p, np.float32) @ np.asarray(x, np.float32)
    return product, 2 * product


@INTERPRETS
def test_consumers_staging_over_their_rings_write_only_past_every_consumers_reads_interpreted():
    """Interpreted with the interpreter's race detection: two consumers that stage their partial products
    over their ring's slots after their walks (`_fed_staged_over_rings`: a consumer's tile shares bytes with
    the slots its partner reads in its last step) write none of it while either reads a slot, each passing
    the barrier of their walks' ends first (`walked`), and their partials sum to `p @ x` exactly (small
    integers); the same consumers not passing it write a tile a product still reads, which the detector
    names."""
    from jax._src.pallas.mosaic_gpu.interpret import interpret_pallas_call as interpreter

    p, x = _fed_operands()
    # A race is recorded and the kernel runs on (`on_race="warn"`), so each arm reads the detector's verdict.
    races = plgpu.InterpretGPUParams(detect_races=True, on_race="warn")
    # Each arm is run whole before its verdict is read: a dispatch returns before the kernel ran, and the
    # detector's state is the last kernel's.
    (parts,) = jax.block_until_ready(_fed_staged_over_rings(p, x, interpret=races, ordered=True))
    assert not interpreter.get_races().races_found
    np.testing.assert_array_equal(np.asarray(parts).sum(axis=0), _fed_expected(p, x)[0])
    jax.block_until_ready(_fed_staged_over_rings(p, x, interpret=races, ordered=False))
    assert interpreter.get_races().races_found


@INTERPRETS
@pytest.mark.parametrize("unrolled", [False, True])
@pytest.mark.parametrize("in_flight", [0, 1])
@pytest.mark.parametrize(("shared", "consumers", "lo"),
                         [(False, 1, 0), (False, 2, 0), (False, 2, 3), (True, 1, 0), (True, 2, 0)])
def test_consumers_feeding_their_own_ring_split_a_step_and_meet_before_the_store(shared, consumers, lo,
                                                                                  in_flight, unrolled):
    """Interpreted: consumers with no producer walk every step of one ring through slot reuse (five steps
    over two slots), each its share of a step's rows, and store their sum once, twice through one scratch:
    exactly the product and its double; a walk whose steps start past 0 (a later key block's), its ring
    counters from its own first (`first`), alike; a step's products running on past it or not, and the
    walk a step a trip or a rotation of its ring a trip (`mosaic.stepping`: two whole rotations and the
    fifth step alone). `shared`: two mma.sync consumers over one cp.async pipeline alike (row 5272: the
    A100 backward's dK/dV on two consumers; an mma.sync program took one alone)."""
    p, x = _fed_operands()
    got = _fed_products(p, x, interpret=INTERPRETED, consumers=consumers, lo=lo, unrolled=unrolled,
                        in_flight=in_flight, shared=shared)
    for value, expected in zip(got, _fed_expected(p, x), strict=True):
        np.testing.assert_array_equal(np.asarray(value), expected)


@INTERPRETS
@pytest.mark.parametrize("common", [False, True], ids=["own", "common"])
def test_two_mma_sync_consumers_walk_segments_whose_boundaries_differ_through_one_pipeline(common):
    """Interpreted: two mma.sync consumers sharing one pipeline walk segments from a step past 0 (empty ones,
    a gap, and a boundary each consumer's own), as one pipeline over the steps `walk_own` agrees, each
    consumer's step its own segment's or none: exactly each segment's weighted products (a pipeline of its
    own a segment met unpaired where their boundaries differ, and hung); segments every consumer's alike,
    stated so (`common`), a pipeline each. Walked directly, several segments are refused."""
    p, x = _fed_operands()
    with pytest.raises(NotImplementedError):
        _fed_products(p, x, interpret=INTERPRETED, consumers=2, lo=3, shared=True, split=True, agreed=False)
    got = _fed_products(p, x, interpret=INTERPRETED, consumers=2, lo=3, shared=True, split=True,
                        common=common)
    summed = sum(weight * np.asarray(p, np.float32)[:, rows] @ np.asarray(x, np.float32)[rows]
                 for rows, weight in _split_weights(consumers=2, common=common))
    for value, expected in zip(got, (summed, 2 * summed), strict=True):
        np.testing.assert_array_equal(np.asarray(value), expected)


def _split_weights(consumers: int, *, common: bool = False):
    """`_fed_products`' split walk's weights: each consumer's share of each step's rows, and the times its
    segment adds its product (none for the step its gap holds, two past its consumer's boundary, every
    consumer's alike where `common`)."""
    share = FED_ROWS // consumers
    for wg in range(consumers):
        for step in range(FED_STEPS):
            weight = 0 if step == 1 else 1 if step < 3 + (0 if common else wg) else 2
            yield slice(step * FED_ROWS + wg * share, step * FED_ROWS + (wg + 1) * share), weight


@pytest.mark.parametrize("consumers", [1, 2])
@pytest.mark.parametrize(("shared", "target", "common"), [(False, "sm90", False), (True, "sm80", False),
                                                          (True, "sm80", True)])
def test_the_fed_walk_lowers_for_its_gpu(shared, target, common, consumers):
    """The fed walk's kernel, two consumers' 32-row contraction staged at 64 bytes, lowers for an H100 on any
    host; two mma.sync consumers' shared cp.async walk for an A100 (`shared`), and its segments every
    consumer's alike, a collective pipeline each (`common`)."""
    p, x = (jax.ShapeDtypeStruct(each.shape, each.dtype) for each in _fed_operands())
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), target)
    walked = functools.partial(_fed_products, interpret=False, consumers=consumers, shared=shared,
                               split=common, common=common)
    with generation.assumed(target), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(walked).trace(p, x).lower(lowering_platforms=("cuda",))


def consumers_feeding_their_own_ring_split_a_step_on_an_h100(consumers, lo):
    """The fed walk compiled and run on an H100, where the interpreter's no-op waits are real, its program
    compiled in a tier's prepare (`ahead_arm`)."""
    p, x = _fed_operands()
    got = ahead_arm(jax.jit(functools.partial(_fed_products, interpret=False, consumers=consumers, lo=lo)),
                    f"a fed walk's products: {consumers} consumers from {lo}")(p, x)
    for value, expected in zip(got, _fed_expected(p, x), strict=True):
        np.testing.assert_array_equal(np.asarray(value), expected)


def consumers_sharing_one_pipeline_on_an_a100(consumers, split):
    """The shared mma.sync walk run on an A100, its pipeline's meets real: five steps over two slots, its
    segments each consumer's own where `split`, compiled in a tier's prepare (`ahead_arm`)."""
    p, x = _fed_operands()
    got = ahead_arm(jax.jit(functools.partial(_fed_products, interpret=False, consumers=consumers,
                                              lo=3 if split else 0, shared=True, split=split)),
                    f"a shared walk's products: {consumers} consumers{', split' * split}")(p, x)
    summed = sum(weight * np.asarray(p, np.float32)[:, rows] @ np.asarray(x, np.float32)[rows]
                 for rows, weight in _split_weights(consumers)) if split else None
    for value, expected in zip(got, (summed, 2 * summed) if split else _fed_expected(p, x), strict=True):
        np.testing.assert_array_equal(np.asarray(value), expected)
