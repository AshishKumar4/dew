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
                        block = x_gmem.at[pl.ds((first + step) * BLOCK[0], BLOCK[0])]
                        ring.load(ring.turn(first + step), block)

                ring.drain(jnp.int32(_first_step(TILES)))

            def consumer(wg):
                @plgpu.nd_loop((TILES,), collective_axes="sm")
                def _(loop: plgpu.NDLoopInfo):
                    tile = loop.index[0]
                    first = _first_step(tile)

                    def step_sum(step, total):
                        slot = ring.acquire(ring.turn(first + step))
                        block = slot[:, pl.ds(wg * columns, columns)].astype(jnp.float32)
                        ring.release(ring.turn(first + step))
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


def a_ring_streams_tiles_of_their_own_trip_counts_on_an_h100(consumers):
    """The ring's kernel compiled and run on an H100, where the interpreter's no-op waits are real."""
    x = _stream()
    summed = jax.jit(functools.partial(_ring_sums, interpret=False, consumers=consumers))(x)
    np.testing.assert_array_equal(np.asarray(summed), _expected(x))


def _multiplying(family, lhs: str, rhs: str, interpret):
    """A walk's step that multiplies its `lhs` slot by its `rhs` slot into its one accumulator."""
    def step(index, slots, carry, accumulators):
        del index
        (acc,) = accumulators
        return carry, (family.products(acc, slots[lhs], slots[rhs], interpret),)

    return step


FED_STEPS = 5
FED_ROWS, FED_WIDTH = 64, 128


def _fed_products(p, x, *, interpret, consumers: int = 2, lo: int = 0, unrolled: bool = False,
                  in_flight: int = 0):
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
                               interpret, first=first, unrolled=unrolled, in_flight=in_flight)[1][0]

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
@pytest.mark.parametrize("unrolled", [False, True])
@pytest.mark.parametrize("in_flight", [0, 1])
@pytest.mark.parametrize(("consumers", "lo"), [(1, 0), (2, 0), (2, 3)])
def test_consumers_feeding_their_own_ring_split_a_step_and_meet_before_the_store(consumers, lo, in_flight,
                                                                                  unrolled):
    """Interpreted: consumers with no producer walk every step of one ring through slot reuse (five steps
    over two slots), each its share of a step's rows, and store their sum once, twice through one scratch:
    exactly the product and its double; a walk whose steps start past 0 (a later key block's), its ring
    counters from its own first (`first`), alike; a step's products running on past it or not, and the
    walk a step a trip or a rotation of its ring a trip (`mosaic.stepping`: two whole rotations and the
    fifth step alone)."""
    p, x = _fed_operands()
    got = _fed_products(p, x, interpret=INTERPRETED, consumers=consumers, lo=lo, unrolled=unrolled,
                        in_flight=in_flight)
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


def consumers_feeding_their_own_ring_split_a_step_on_an_h100(consumers, lo):
    """The fed walk compiled and run on an H100, where the interpreter's no-op waits are real."""
    p, x = _fed_operands()
    got = jax.jit(functools.partial(_fed_products, interpret=False, consumers=consumers, lo=lo))(p, x)
    for value, expected in zip(got, _fed_expected(p, x), strict=True):
        np.testing.assert_array_equal(np.asarray(value), expected)


@pytest.mark.parametrize("unrolled", [False, True])
@pytest.mark.parametrize("stages", [2, 3, 4, 5])
def test_a_walk_takes_each_step_of_its_range_once_in_order_at_its_turn(stages, unrolled):
    """`mosaic.stepping`'s loops, run in turn as a walk's are, take steps `lo` to `hi` once each and in order,
    each at its counter's slot and rotation, for every start, length and ring origin over three rotations:
    rolled, a step a trip; unrolled, the steps taken one a trip are the fewer than `stages` before the whole
    rotations and the fewer after them, and a rotation's steps know the steps before them in it are the
    range's."""
    taken = []

    def turned(index, turn, *state, before=0):
        taken.append((index, turn.slot, turn.rotation, before))
        return state

    for first in (0, 3):
        for lo in range(first, first + 3 * stages):
            for hi in range(lo, lo + 3 * stages + 1):
                taken.clear()
                singly = 0
                for loop in mosaic.stepping(turned, lo, hi, first, stages, unrolled=unrolled):
                    run = loop.step
                    for trip in range(int(loop.lo), int(loop.hi)):
                        before = len(taken)
                        run(trip)
                        singly += len(taken) - before == 1
                assert [step[0] for step in taken] == list(range(lo, hi)), (first, lo, hi)
                for index, slot, rotation, before in taken:
                    assert (slot, rotation) == ((index - first) % stages, (index - first) // stages)
                    assert index - before >= lo, (first, lo, hi, index)
                if unrolled:
                    assert singly <= 2 * (stages - 1), (first, lo, hi)
                else:
                    assert singly == hi - lo


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
    """`resources.plan` on an A100 for one mma.sync consumer at 128 registers a thread, which leave a
    multiprocessor four programs (`Family.programs` at the launch's own registers): a launch of 16 KiB a
    stage holds all four at the deepest ring a quarter of its shared memory holds; one of 40 KiB a stage
    holds the most programs whose share holds its floor of two stages; one past every residency at its
    floor plans nothing."""
    from dew.nn.kernels.templates import mosaic_sm80, resources

    family = mosaic_sm80.Mma(jnp.dtype(jnp.bfloat16))

    def launch(stage_kib: int) -> resources.Launch:
        return resources.Launch(lambda stages: resources.Scope(
            program=(plgpu.SMEM((stages * stage_kib * 1024,), jnp.uint8),)), consumers=1, registers=128)

    with generation.assumed("sm80"):
        target = generation.gpu_target()
        block = generation.shared_budget(target)
        small, wide, past = (resources.plan([launch(kib)], family, target) for kib in (16, 40, 200))
    assert small is not None and (small[0].programs, small[0].stages) == (
        4, min(8, resources.program_shared(block, 4) // (16 * 1024)))
    assert wide is not None and wide[0].programs == max(
        programs for programs in (1, 2, 3, 4) if resources.program_shared(block, programs) >= 2 * 40 * 1024)
    assert past is None


ISSUED = 64


@dataclasses.dataclass(frozen=True)
class _Call:
    """A call an `operands.Issuer` reads: the bundle its products issue by, interpreted."""
    family: mosaic.Family
    interpret: bool = True


def _issued_sums(a, b, c, d, exponent, *, retired: bool):
    """`restored(a @ b, exponent) + c @ d` through one prepared-operand issuer (`operands.Issuer.summed`) on
    warpgroup MMA, interpreted: `a` a scaled register operand (`product.Scaled`), its product in an
    accumulator of its own and restored by `2**exponent`, `c` a plain register operand, each staged into its
    product's buffer, `b` and `d` shared tiles (`operands.Parts`); `retired`, each product done before the
    next is formed."""
    from dew.nn.kernels.templates import mosaic_sm90, operands, product

    family = mosaic_sm90.FED  # one warpgroup, no producer
    tile = ((ISSUED, ISSUED), jnp.bfloat16)
    call = _Call(family)

    def body(a_gmem, b_gmem, c_gmem, d_gmem, exponent_gmem, out_gmem):
        def scoped(staged, b_smem, d_smem):
            b_smem[...], d_smem[...] = b_gmem[...], d_gmem[...]
            issuer = operands.Issuer(call, {"scaled": (staged,), "plain": (staged,)},
                                     restored=lambda value, k: value * jnp.exp2(k.astype(jnp.float32)))
            products = iter([("scaled", product.Scaled(a_gmem[...], exponent_gmem[0]),
                              operands.Parts((b_smem,), jnp.bfloat16)),
                             ("plain", c_gmem[...], operands.Parts((d_smem,), jnp.bfloat16))])
            out_gmem[...] = issuer.summed(products, retired=retired)

        pl.run_scoped(scoped, *(plgpu.SMEM(*tile) for _ in range(3)), collective_axes="wg")

    out = jax.ShapeDtypeStruct((ISSUED, ISSUED), jnp.float32)
    return family.kernel(body, out_type=out, grid=(1,), grid_names=("programs",), consumers=1,
                         interpret=True, name="issued")(a, b, c, d, exponent)


@INTERPRETS
@pytest.mark.parametrize("retired", [False, True])
def test_a_scaled_operands_product_is_summed_apart_and_restored(retired):
    """Interpreted: a scaled operand's product restored by its power of two and a plain one's, summed by one
    issuer, the plain product staged into the same buffer the scaled one's operand used: exactly
    `2**k a @ b + c @ d`, products retired one at a time or not."""
    keys = jax.random.split(jax.random.key(2), 4)
    a, b, c, d = (jax.random.randint(key, (ISSUED, ISSUED), -4, 4).astype(jnp.bfloat16) for key in keys)
    exponent = jnp.array([3], jnp.int32)
    summed = _issued_sums(a, b, c, d, exponent, retired=retired)
    f32 = [np.asarray(x, np.float32) for x in (a, b, c, d)]
    np.testing.assert_array_equal(np.asarray(summed), 8 * f32[0] @ f32[1] + f32[2] @ f32[3])


def _carried_sums(a, b, c, d):
    """`a @ b + c @ d` issued in place into one accumulator a walk carries (`Issuer.summed`'s `into`),
    interpreted: `a` a register left operand its product takes in registers (its buffers a count, wgmma's
    register form), `c` staged into a shared buffer, `b` and `d` shared tiles, both products retired."""
    from dew.nn.kernels.templates import mosaic_sm90, operands

    family = mosaic_sm90.FED  # one warpgroup, no producer
    call = _Call(family)

    def body(a_gmem, b_gmem, c_gmem, d_gmem, out_gmem):
        def scoped(staged, b_smem, d_smem):
            b_smem[...], d_smem[...] = b_gmem[...], d_gmem[...]
            issuer = operands.Issuer(call, {"registered": 1, "plain": (staged,)})
            products = (("registered", a_gmem[...], operands.Parts((b_smem,), jnp.bfloat16)),
                        ("plain", c_gmem[...], operands.Parts((d_smem,), jnp.bfloat16)))
            out_gmem[...] = issuer.summed(products, retired=True, into=family.zeros((ISSUED, ISSUED)))

        pl.run_scoped(scoped, *(plgpu.SMEM((ISSUED, ISSUED), jnp.bfloat16) for _ in range(3)),
                      collective_axes="wg")

    out = jax.ShapeDtypeStruct((ISSUED, ISSUED), jnp.float32)
    return family.kernel(body, out_type=out, grid=(1,), grid_names=("programs",), consumers=1,
                         interpret=True, name="carried")(a, b, c, d)


@INTERPRETS
def test_products_issue_in_place_into_a_carried_accumulator():
    """Interpreted: a register-operand product and a staged one, issued into one carried accumulator in
    place: exactly `a @ b + c @ d`."""
    keys = jax.random.split(jax.random.key(3), 4)
    a, b, c, d = (jax.random.randint(key, (ISSUED, ISSUED), -4, 4).astype(jnp.bfloat16) for key in keys)
    f32 = [np.asarray(x, np.float32) for x in (a, b, c, d)]
    np.testing.assert_array_equal(np.asarray(_carried_sums(a, b, c, d)), f32[0] @ f32[1] + f32[2] @ f32[3])


def test_a_plan_holds_no_more_programs_than_its_target_runs_threads_for():
    """`resources.plan` for one mma.sync consumer at 32 registers a thread and 1 KiB a program, whose
    registers would hold sixteen programs: on an A10 and an RTX PRO 6000, twelve, their 1536 resident
    threads' worth of its one warpgroup; on an A100 with its 2048, sixteen."""
    from dew.nn.kernels.templates import mosaic_sm80, resources

    def scope(stages: int) -> resources.Scope:
        return resources.Scope(program=(plgpu.SMEM((stages * 1024,), jnp.uint8),))

    launch = resources.Launch(scope, consumers=1, registers=32, floor=1)
    family = mosaic_sm80.Mma(jnp.dtype(jnp.bfloat16))
    for name, most in (("sm86", 12), ("sm120", 12), ("sm80", 16)):
        with generation.assumed(name):
            found = resources.plan([launch], family, generation.gpu_target())
        assert found is not None and found[0].programs == most, name


def test_a_launch_with_a_producer_takes_the_plan_holding_the_most_bytes_in_flight():
    """`resources.plan` on an H100 for a warpgroup-MMA launch beside its producer shaped as the 27B gated
    delta net's states launch (41 KiB of its own, 36 KiB a stage, 96 registers): one program five deep, 184
    KiB in flight, where its most programs, two one deep, held 74 KiB and slowed its walk
    (`resources.hidden`); a launch of one stage, 64 KiB at 128 registers, its most programs, three (their
    75 KiB shares and 136-register raises hold it, four's 56 KiB and 104 do not), thrice one's bytes in
    flight."""
    from dew.nn.kernels.templates import bundles, resources

    def launch(fixed: int, step: int, registers: int, most: int) -> resources.Launch:
        def scope(stages: int) -> resources.Scope:
            return resources.Scope(program=(plgpu.SMEM((fixed + stages * step,), jnp.uint8),))

        return resources.Launch(scope, consumers=1, registers=registers, floor=1, most=most)

    with generation.assumed("sm90"):
        family, target = bundles.of("sm90", jnp.bfloat16), generation.gpu_target()
        (states,) = resources.plan([launch(42016, 36912, 96, 8)], family, target)
        (single,) = resources.plan([launch(65536, 0, 128, 1)], family, target)
    assert (states.programs, states.stages) == (1, 5)
    assert (single.programs, single.stages) == (3, 1)


def _vector_scaled(scale, x, axis: str, *, interpret: bool):
    """An sm80 program scaling a 64 x 128 accumulator by a side vector along `axis` ('rows' or 'columns'),
    the vector loaded as an epilogue's side tile is (`Family.loaded`, its axes stated)."""
    from dew.nn.kernels.templates import mosaic_sm80

    family = mosaic_sm80.Mma(jnp.dtype(jnp.bfloat16))

    def body(scale_gmem, x_gmem, out_gmem):
        def scoped(scale_smem, barrier):
            family.loaded([(scale_gmem, scale_smem, barrier.at[0])], axes=((axis,),))
            vector = scale_smem[...]
            acc = plgpu.load(x_gmem, layout=family.layout, optimized=False)
            out_gmem[...] = acc * (vector[:, None] if axis == "rows" else vector[None, :])

        pl.run_scoped(scoped, plgpu.SMEM(scale.shape, scale.dtype), plgpu.Barrier(num_barriers=1))

    out = jax.ShapeDtypeStruct(x.shape, jnp.float32)
    return family.kernel(body, out_type=out, grid=(1,), grid_names=("programs",), consumers=1,
                         interpret=interpret, name="vector_scaled")(scale, x)


@pytest.mark.parametrize("axis", ["rows", "columns"])
def test_an_sm80_side_vector_loads_in_the_layout_of_the_axis_it_follows(axis):
    """An mma.sync epilogue's side vector is loaded in the accumulator's layout of the axis it follows (a
    row scale `[rows]` in its rows' vector layout, read as `s[:, None]`; a column scale in its columns'),
    so a row-scaled epilogue lowers for an A100 (row 2100's K18: every vector was loaded in the columns'
    layout, and a row scale found no layout)."""
    rows, columns = 64, 128
    scale = jax.ShapeDtypeStruct((rows if axis == "rows" else columns,), jnp.float32)
    x = jax.ShapeDtypeStruct((rows, columns), jnp.float32)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm80")
    with generation.assumed("sm80"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(functools.partial(_vector_scaled, axis=axis, interpret=False)).trace(scale, x).lower(
            lowering_platforms=("cuda",))


def test_a_light_launch_is_held_within_the_threads_its_multiprocessor_runs():
    """A one-warpgroup mma.sync launch of a few registers a thread on an L4 (sm89), whose multiprocessor runs
    1536 threads in at most 24 blocks (CUDA C++ Programming Guide §5.1, Table 33): its registers would hold
    sixteen programs, its threads hold twelve, and the residency it is planned at is twelve, where an A100's
    2048 threads hold sixteen."""
    from dew.nn.kernels.templates import bundles, resources

    for generation_name, programs in (("sm89", 12), ("sm80", 16)):
        with generation.assumed(generation_name):
            family = bundles.of(generation_name, jnp.bfloat16)
            assert resources.register_programs(family, 1, 32) == programs


def test_a_launch_adding_into_two_refs_of_one_shape_is_refused_at_its_trace():
    """Two zeroed refs of one shape and dtype a launch adds into in place are one value to XLA, which a
    dynamic-slice fusion wrapping the launch passes once for both (row 3180): `mosaic.checked` refuses the
    launch where it traces them, and takes them stacked into one ref (`stacked_zeros`), each member its own
    place: a launch writing a value of its own into each view (`stacked_views`) reads each back apart
    (`stacked_read`), in its shapes' order."""
    shapes = [(2, 3), (4,), (2, 3)]

    def written(*refs):
        for index, view in enumerate(mosaic.stacked_views(refs, groups)):
            view[...] = jnp.full(view.shape, index + 1.0)

    launched = mosaic.checked(written, "probe")

    def apart():
        launched(*(jax.new_ref(jnp.zeros(shape, jnp.float32)) for shape in shapes))

    groups = mosaic.stacked_groups(shapes)
    with pytest.raises(ValueError):
        jax.jit(apart).trace()

    def stacked():
        refs = mosaic.stacked_zeros(shapes, groups)
        launched(*refs)
        return mosaic.stacked_read(refs, groups)

    for index, (shape, array) in enumerate(zip(shapes, jax.jit(stacked)(), strict=True)):
        np.testing.assert_array_equal(array, np.full(shape, index + 1.0))


PERSISTED_TILES, PERSISTED_STEPS, PERSISTED_DEPTH = 3, 2, 64


def _padded(rows: int, family) -> int:
    """A tile's rows for a launch of `rows`: padded to the least its family's products issue."""
    return -(-rows // family.rows) * family.rows


def _persisted_products(x, w, *, family, interpret, rows: int | None = None, resident: int = 1):
    """`x @ w` a tile at a time through `family`'s persistent walk (`Family.persist`): each tile the
    family's least rows covering `rows` (its own least where None; `_padded`), of which it stores `rows`,
    tile `t` the tiles' rows `t` on, each step `PERSISTED_DEPTH` of the contraction, the middle tile
    multiplying nothing (zeros); each tile issued (`issue`'s walk) and finished (`Issuer.chunks`, each
    chunk stored at its columns), on the family's roles, launched for `resident` programs a multiprocessor."""
    stored = family.rows if rows is None else rows
    rows, columns = _padded(stored, family), w.shape[1]
    family = family.specialized(rows, columns)
    shape = (rows, columns)
    blocks = {'x': ((rows, PERSISTED_DEPTH), x.dtype), 'w': ((PERSISTED_DEPTH, columns), w.dtype)}

    def body(x_gmem, w_gmem, o_gmem):
        def scope(*, streams):
            def schedule(index):
                tile = index[0]
                views = {'x': (x_gmem, lambda j: (tile, j)), 'w': (w_gmem, lambda j: (j, 0))}
                return tile != 1, PERSISTED_STEPS, views, tile

            def issue(wg, tile, walk, *, active):
                del wg, tile, active

                return walk(_multiplying(family, 'x', 'w', interpret), None, (shape,))[1]

            def finish(wg, tile, accs, store, *, active):
                del wg, active
                for first, (value,) in family.chunks(accs):
                    store(0, value, o_gmem, (tile * rows, 0, stored, first, rows))

            family.persist(streams, (PERSISTED_TILES,), schedule, issue, finish, consumers=1, axis="programs",
                           outputs=((shape, jnp.float32),), interpret=interpret, accumulators=(shape,))

        family.scoped(scope, streams=family.streams(blocks, 2, 1))

    out = jax.ShapeDtypeStruct((PERSISTED_TILES * rows, columns), jnp.float32)
    return family.kernel(body, out_type=out, grid=(1,), grid_names=("programs",), consumers=1,
                         interpret=interpret, name="persisted", resident=resident)(x, w)


def _persisted_operands(rows: int):
    """Small integers, so every tile's product is exact in fp32 whatever its order."""
    keys = jax.random.split(jax.random.key(2))
    depth = PERSISTED_STEPS * PERSISTED_DEPTH
    x = jax.random.randint(keys[0], (PERSISTED_TILES * rows, depth), -4, 4).astype(jnp.bfloat16)
    w = jax.random.randint(keys[1], (depth, 128), -4, 4).astype(jnp.bfloat16)
    return x, w


@INTERPRETS
def test_tcgen05_tiles_walk_into_tensor_memory_and_their_epilogue_reads_it_out_in_chunks():
    """Interpreted: a persistent walk on tcgen05's bundle (`mosaic_sm100.RINGS`), its product warp issuing
    each tile's steps into one of two accumulator turns in tensor memory and the epilogue reading each
    tile's out in chunks, over three tiles (the turns wrap) with the middle one multiplying nothing: each
    tile's rows exactly its product, the middle's zeros."""
    from dew.nn.kernels.templates import mosaic_sm100
    family = mosaic_sm100.RINGS
    x, w = _persisted_operands(family.rows)
    got = np.asarray(_persisted_products(x, w, family=family, interpret=INTERPRETED))
    expected = np.asarray(x, np.float32) @ np.asarray(w, np.float32)
    expected[family.rows:2 * family.rows] = 0
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("rows", [4, 8, 16, 24, 32])
@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
def test_a_few_rows_lower_on_every_bundle_their_device_issues(generation_name, rows):
    """A launch of a decode step's few rows lowers on each bundle its device issues for them
    (`bundles.layouts`' `rows`: mma.sync's on cp.async on an A100; on an H100 wgmma's two layouts and
    mma.sync's on TMA rings), its tile padded to the bundle's least rows and storing only its own."""
    from dew.nn.kernels.templates import bundles
    x, w = (jax.ShapeDtypeStruct(each.shape, each.dtype) for each in _persisted_operands(64))
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), generation_name)
    with generation.assumed(generation_name), jax.sharding.use_abstract_mesh(mesh):
        for family in bundles.layouts(generation_name, jnp.bfloat16, rows=rows):
            padded = (PERSISTED_TILES * _padded(rows, family), x.shape[1])
            lhs = jax.ShapeDtypeStruct(padded, x.dtype)
            jax.jit(functools.partial(_persisted_products, family=family, interpret=False, rows=rows)).trace(
                lhs, w).lower(lowering_platforms=("cuda",))


def test_the_tcgen05_persistent_walk_lowers_for_a_b200():
    """The persistent walk on tcgen05's bundle lowers for a B200 on any host, as the GPU gate runs it."""
    from dew.nn.kernels.templates import mosaic_sm100
    family = mosaic_sm100.RINGS
    x, w = (jax.ShapeDtypeStruct(each.shape, each.dtype) for each in _persisted_operands(family.rows))
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm100")
    with generation.assumed("sm100"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(functools.partial(_persisted_products, family=family, interpret=False)).trace(x, w).lower(
            lowering_platforms=("cuda",))


def _relayed_products(q, k, v, *, interpret):
    """FlashAttention 4's relay on tcgen05's bundle: `S = q k` into tensor memory by the product warp, read
    out by a role of the table (`Tcgen05Issuer.read`), doubled, narrowed and written back as `P`
    (`write`), and `O = P v` with `P` the left operand from tensor memory (the TS product); `O` read out
    and stored by the same role. Its roles wait on signals the products' commit and the role's publication
    arrive at (`signals`)."""
    from dew.nn.kernels.templates import mosaic_sm100
    family = mosaic_sm100.RINGS
    rows = q.shape[0]
    columns = v.shape[1]

    def body(q_gmem, k_gmem, v_gmem, o_gmem):
        def scoped(q_smem, k_smem, v_smem, o_smem, loaded, s_tmem, p_tmem, o_tmem, scored, weighed, summed):
            s = mosaic_sm100.Accumulator(s_tmem, (rows, k.shape[1]))
            p = mosaic_sm100.Accumulator(p_tmem, (rows, k.shape[1]), dtype=jnp.bfloat16)
            o = mosaic_sm100.Accumulator(o_tmem, (rows, columns))
            scored, weighed, summed = (family.signals().opened(each) for each in (scored, weighed, summed))
            first = mosaic.Turn.of(0, 1)

            def relay(index):
                del index
                plgpu.barrier_wait(scored.barriers.at[0], parity=0)
                family.write(p, family.read(s) * 2)
                weighed.published(first)
                plgpu.barrier_wait(summed.barriers.at[0], parity=0)
                o_smem[...] = family.read(o)
                plgpu.wait_load_tmem()
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(o_smem, o_gmem)
                plgpu.wait_smem_to_gmem(0)

            def load():
                for gmem, smem, at in ((q_gmem, q_smem, 0), (k_gmem, k_smem, 1), (v_gmem, v_smem, 2)):
                    plgpu.copy_gmem_to_smem(gmem, smem, loaded.at[at])

            def products():
                for at in range(3):
                    plgpu.barrier_wait(loaded.at[at], parity=0)
                family.products(s, q_smem, k_smem, interpret)
                scored.committed(first)
                plgpu.barrier_wait(weighed.barriers.at[0], parity=0)
                family.products(o, p, v_smem, interpret)
                summed.committed(first)

            family.roles_of(((1, relay),), load=load, products=products)

        smem = [plgpu.SMEM(each.shape, each.dtype, transforms=mosaic.swizzle_transforms(each.dtype, 128))
                for each in (q, k, v)]
        signals = [family.signals().scratch(), family.signals(by_products=False).scratch(),
                   family.signals().scratch()]
        pl.run_scoped(scoped, *smem, plgpu.SMEM((rows, columns), jnp.float32), plgpu.Barrier(num_barriers=3),
                      family.tensor_memory((rows, k.shape[1])),
                      family.tensor_memory((rows, k.shape[1]), jnp.bfloat16),
                      family.tensor_memory((rows, columns)), *signals, collective_axes="wg")

    out = jax.ShapeDtypeStruct((rows, columns), jnp.float32)
    return family.kernel(body, out_type=out, grid=(1,), grid_names=("programs",), consumers=1,
                         interpret=interpret, name="relayed")(q, k, v)


def _relayed_operands():
    """Small integers, so `2 (q k) v` is exact in fp32 and its weights exact in bf16."""
    keys = jax.random.split(jax.random.key(3), 3)
    q = jax.random.randint(keys[0], (128, 64), -2, 3).astype(jnp.bfloat16)
    k = jax.random.randint(keys[1], (64, 128), -2, 3).astype(jnp.bfloat16)
    v = jax.random.randint(keys[2], (128, 64), -2, 3).astype(jnp.bfloat16)
    return q, k, v


@INTERPRETS
def test_a_role_relays_scores_through_tensor_memory_into_a_product_that_reads_them_there():
    """Interpreted: scores issued into tensor memory, read out by a role, doubled and written back narrowed,
    and the next product reading them from tensor memory as its left operand (tcgen05's TS form, P·V):
    exactly `2 (q k) v`."""
    q, k, v = _relayed_operands()
    got = np.asarray(_relayed_products(q, k, v, interpret=INTERPRETED))
    weights = 2 * (np.asarray(q, np.float32) @ np.asarray(k, np.float32))
    np.testing.assert_array_equal(got, weights @ np.asarray(v, np.float32))


def test_the_relay_through_tensor_memory_lowers_for_a_b200():
    """The relay lowers for a B200 on any host, as the GPU gate runs it."""
    q, k, v = (jax.ShapeDtypeStruct(each.shape, each.dtype) for each in _relayed_operands())
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm100")
    with generation.assumed("sm100"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(functools.partial(_relayed_products, interpret=False)).trace(q, k, v).lower(
            lowering_platforms=("cuda",))


def test_a_tpu_core_walks_the_persistent_tiles_interpreted():
    """Interpreted on any host: the persistent walk on a TPU core's bundle (`mosaic_tpu.MXU`), each step's
    blocks DMA'd into its VMEM ring a step ahead and multiplied by the MXU, each tile's rows stored from
    VMEM, over three tiles with the middle one multiplying nothing: each tile's rows exactly its product,
    the middle's zeros."""
    from dew.nn.kernels.templates import mosaic_tpu
    family = mosaic_tpu.MXU
    x, w = _persisted_operands(family.rows)
    got = np.asarray(_persisted_products(x, w, family=family, interpret=True))
    expected = np.asarray(x, np.float32) @ np.asarray(w, np.float32)
    expected[family.rows:2 * family.rows] = 0
    np.testing.assert_array_equal(got, expected)


def test_the_tpu_persistent_walk_lowers_for_a_v5e():
    """The persistent walk on a TPU core's bundle lowers for a TPU v5e on any host, its devices abstract."""
    from jax.sharding import AbstractDevice, AbstractMesh

    from dew.nn.kernels.templates import mosaic_tpu
    family = mosaic_tpu.MXU
    x, w = (jax.ShapeDtypeStruct(each.shape, each.dtype) for each in _persisted_operands(family.rows))
    device = AbstractDevice(device_kind=generation.TARGETS["v5e"].device_kind, num_cores=1, platform="tpu")
    with jax.sharding.use_abstract_mesh(AbstractMesh((1,), ("data",), abstract_device=device)):
        jax.jit(functools.partial(_persisted_products, family=family, interpret=False)).trace(x, w).lower(
            lowering_platforms=("tpu",))


def _gemm_on_a_tpu_core():
    """The GEMM template's kernel bodies on a TPU core's bundle (`mosaic_tpu.MXU`), each `(call, operands,
    expected)` at small integers, so each product is exact in fp32: a dense forward with a bias epilogue,
    the grouped forward (three groups, one empty, rows past the routed ones zero) with its matrices stored
    as they come and transposed, and the grouped kernel gradient."""
    from dew.nn.kernels.templates.gemm import mosaic_gpu

    rows, depth, width = 384, 512, 256
    tiles = mosaic_gpu.Tiles(128, 128, 128, 2, 1, family="mxu")
    keys = jax.random.split(jax.random.key(4), 4)
    x = jax.random.randint(keys[0], (rows, depth), -3, 4).astype(jnp.bfloat16)
    w = jax.random.randint(keys[1], (3, depth, width), -3, 4).astype(jnp.bfloat16)
    dy = jax.random.randint(keys[2], (rows, width), -3, 4).astype(jnp.bfloat16)
    bias = jax.random.randint(keys[3], (width,), -3, 4).astype(jnp.float32)
    sizes = jnp.asarray([100, 0, 250], jnp.int32)
    xf, wf, dyf = (np.asarray(each, np.float32) for each in (x, w, dy))
    unrouted = np.zeros((rows - 350, width), np.float32)
    routed = np.concatenate([xf[:100] @ wf[0], xf[100:350] @ wf[2], unrouted])
    grouped = functools.partial(mosaic_gpu.gmm, tiles=tiles, out_dtype=jnp.float32, multiprocessors=2)
    return {
        "dense bias": (functools.partial(mosaic_gpu.gmm, tiles=tiles, transpose_rhs=False,
                                         out_dtype=jnp.float32, multiprocessors=1, dense=True,
                                         epilogue=lambda acc, b: acc + b[None, :], epilogue_operands=(bias,),
                                         epilogue_axes=(("columns",),)),
                       (x, w[:1], None), xf @ wf[0] + np.asarray(bias)[None, :]),
        "grouped": (functools.partial(grouped, transpose_rhs=False), (x, w, sizes), routed),
        "grouped transposed": (functools.partial(grouped, transpose_rhs=True),
                               (x, jnp.swapaxes(w, 1, 2), sizes), routed),
        "kernel gradient": (functools.partial(mosaic_gpu.tgmm, tiles=tiles, out_dtype=jnp.float32,
                                              multiprocessors=2), (x, dy, sizes),
                            np.stack([xf[:100].T @ dyf[:100], np.zeros((depth, width), np.float32),
                                      xf[100:350].T @ dyf[100:350]])),
    }


@pytest.mark.parametrize("case", ["dense bias", "grouped", "grouped transposed", "kernel gradient"])
def test_the_gemms_kernel_bodies_run_on_a_tpu_core(case):
    """Interpreted on any host, each of the GEMM template's kernel bodies on a TPU core's bundle gives its
    product exactly, and it lowers for a TPU v5e, its devices abstract."""
    from jax.sharding import AbstractDevice, AbstractMesh

    call, operands, expected = _gemm_on_a_tpu_core()[case]
    np.testing.assert_array_equal(np.asarray(call(*operands, interpret=True)), expected)
    abstract = [None if each is None else jax.ShapeDtypeStruct(each.shape, each.dtype) for each in operands]
    device = AbstractDevice(device_kind=generation.TARGETS["v5e"].device_kind, num_cores=1, platform="tpu")
    with jax.sharding.use_abstract_mesh(AbstractMesh((1,), ("data",), abstract_device=device)):
        jax.jit(call).trace(*abstract).lower(lowering_platforms=("tpu",))


def test_a_consumer_launched_at_each_residency_its_family_admits_raises_to_the_gates_cap_spill_free():
    """A persistent warpgroup-MMA walk, one consumer beside its producer, launched at each residency its
    family admits a multiprocessor on an H100 (`RegisterModel.most`) and compiled as the GPU gate compiles
    it: the registers its consumer raises to in its PTX (`setmaxnreg.inc`) are the gate's cap for it at that
    residency (`RegisterModel.cap`), and its binary holds its registers within them, spilling nothing."""
    import kernel_compile
    from fla_gdn_support import KernelCase

    from dew.nn.kernels.templates import bundles, resources

    with generation.assumed("sm90"):
        family = bundles.of("sm90", jnp.bfloat16)
        model = resources.register_model(family, 1, None)
    x, w = (jax.ShapeDtypeStruct(each.shape, each.dtype) for each in _persisted_operands(family.rows))
    for programs in range(1, model.most + 1):
        call = functools.partial(_persisted_products, family=family, interpret=False, resident=programs)
        case = KernelCase("persisted", lambda call=call: (call, (x, w)))
        [record] = kernel_compile.compile_mosaic_kernels([case], "sm90")
        [kernel] = record["kernels"]
        cap = model.cap(programs, honored=True)
        assert (kernel["raised"], kernel["spilled"]) == (cap, 0), (programs, record.get("error"), kernel)
        assert kernel["registers"] <= cap


RELAYED = 64
"""A relayed product's depth and columns."""


def _chained(a, b, c, *, m_warps: int, interpret: bool):
    """`(a @ b) @ c` by mma.sync on `m_warps` warps along the rows (`MmaIssuer.specialized`'s split), each
    product's left operand a register value (`a` read from a plain tile, then the first product's
    accumulator) relayed through a tile the consumer binds where the warps split its columns (`relaying`),
    stored through registers."""
    from dew.nn.kernels.templates import mosaic_sm80, resources

    family = mosaic_sm80.Mma(jnp.dtype(jnp.bfloat16), m_warps=m_warps)
    rows = a.shape[0]

    def body(a_gmem, b_gmem, c_gmem, out_gmem):
        def scoped(a_smem, b_smem, c_smem, relay, out_smem):
            for source, staged in ((a_gmem, a_smem), (b_gmem, b_smem), (c_gmem, c_smem)):
                mosaic.copied(source, staged)
            mosaic.synced(interpret)
            issuer = family.relaying(relay)
            first = issuer.product(plgpu.load(a_smem, optimized=False), b_smem, (rows, RELAYED),
                                   interpret=interpret)
            out = issuer.product(first.astype(jnp.bfloat16), c_smem, (rows, RELAYED), interpret=interpret)
            plgpu.store(out_smem, out, optimized=False)
            mosaic.synced(interpret)
            mosaic.store_rows(out_smem, out_gmem, 0, 0, jnp.int32(rows), 0, rows, copy=mosaic.coalesced,
                              widest=resources.copied_rows(RELAYED, jnp.float32))

        swizzle = resources.widest_swizzle(jnp.bfloat16, RELAYED)
        pl.run_scoped(scoped, mosaic.plain((rows, RELAYED), jnp.bfloat16),
                      *(mosaic.staged((RELAYED, RELAYED), jnp.bfloat16, swizzle) for _ in range(2)),
                      mosaic.plain((rows, RELAYED), jnp.bfloat16), mosaic.plain((rows, RELAYED), jnp.float32))

    out = jax.ShapeDtypeStruct((rows, RELAYED), jnp.float32)
    return family.kernel(body, out_type=out, grid=(1,), grid_names=("programs",), consumers=1,
                         interpret=interpret, name="chained")(a, b, c)


def _chained_operands(rows: int):
    """Small integers whose chained products are exact in fp32, the first's (at most 128) in bf16."""
    keys = jax.random.split(jax.random.key(5), 3)
    return (jax.random.randint(keys[0], (rows, RELAYED), -2, 3).astype(jnp.bfloat16),
            *(jax.random.randint(key, (RELAYED, RELAYED), -1, 2).astype(jnp.bfloat16) for key in keys[1:]))


@pytest.mark.parametrize(("rows", "m_warps"), [(16, 1), (32, 2)])
def test_an_accumulator_feeds_the_next_product_on_any_warp_split(rows, m_warps):
    """Register values read as mma.sync's left operands, an accumulator's among them, on warps that split
    its columns (one or two along its rows): lowered for an A100 on any host, and on a GPU exactly `(a @ b)
    @ c` (each warp's columns relayed through the bound tile, where no register cast crosses warps)."""
    a, b, c = _chained_operands(rows)
    if jax.default_backend() == "gpu":
        f32 = [np.asarray(x, np.float32) for x in (a, b, c)]
        np.testing.assert_array_equal(np.asarray(_chained(a, b, c, m_warps=m_warps, interpret=False)),
                                      (f32[0] @ f32[1]) @ f32[2])
        return
    shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (a, b, c)]
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm80")
    with generation.assumed("sm80"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(functools.partial(_chained, m_warps=m_warps, interpret=False)).trace(*shapes).lower(
            lowering_platforms=("cuda",))


CODES = (64, 64)
"""A quantized stage's rows and depth."""


def _converted(codes, scale, *, generation_name: str, interpret, scaled: bool = True):
    """`codes` (int8 or e4m3) converted into a bf16 tile through a family's `converted`, each row scaled by
    `scale`, and copied out: on an A100's mma.sync and an H100's fed warpgroup MMA."""
    from dew.nn.kernels.templates import bundles, resources

    family = bundles.of(generation_name, jnp.bfloat16)

    def body(codes_gmem, scale_gmem, out_gmem):
        def scoped(codes_smem, into):
            mosaic.copied(codes_gmem, codes_smem)
            mosaic.synced(interpret is not False)
            row_scale = scale_gmem[...] if interpret is not False else family.rows_of(scale_gmem)
            family.converted(codes_smem, into, interpret is not False,
                             row_scale=row_scale if scaled else None)
            mosaic.synced(interpret is not False)
            mosaic.store_rows(into, out_gmem, 0, 0, jnp.int32(CODES[0]), 0, CODES[0], copy=mosaic.coalesced,
                              widest=resources.copied_rows(CODES[1], jnp.bfloat16))

        # A fed family's kernel runs on a thread axis, which its scoped allocations name.
        threaded = {"collective_axes": "wg"} if generation_name == "sm90" else {}
        pl.run_scoped(scoped, mosaic.plain(CODES, codes.dtype), mosaic.plain(CODES, jnp.bfloat16), **threaded)

    out = jax.ShapeDtypeStruct(CODES, jnp.bfloat16)
    return family.kernel(body, out_type=out, grid=(1,), grid_names=("programs",), consumers=1,
                         interpret=interpret, name="converted")(codes, scale)


def _codes(dtype):
    """Codes of `dtype` and per-row power-of-two scales, so each converted value is exact in bf16."""
    keys = jax.random.split(jax.random.key(6), 2)
    codes = jax.random.randint(keys[0], CODES, -8, 8).astype(dtype)
    return codes, jnp.exp2(jax.random.randint(keys[1], (CODES[0],), -3, 4).astype(jnp.float32))


@pytest.mark.parametrize("scaled", [True, False], ids=["scaled", "unscaled"])
@pytest.mark.parametrize("dtype", [jnp.int8, jnp.float8_e4m3fn])
@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
def test_a_quantized_stage_converts_into_the_tile_its_products_read(generation_name, dtype, scaled):
    """Int8 and e4m3 codes in shared memory converted into a bf16 tile, each row scaled in fp32: exactly
    `codes * scale` interpreted (on a CPU) or on a GPU of the generation, and lowered for one on any host;
    FP8 codes refused where the device converts none (an A100)."""
    codes, scale = _codes(dtype)
    expected = np.asarray(codes, np.float32) * (np.asarray(scale)[:, None] if scaled else 1)
    with generation.assumed(generation_name):
        converts = mosaic.converts(dtype)
    if not converts:
        # An A100 converts no FP8 in registers: the conversion is refused, never assembled.
        with pytest.raises(ValueError), generation.assumed(generation_name):
            _converted(codes, scale, generation_name=generation_name, interpret=INTERPRETED)
        return
    if jax.default_backend() == "cpu":
        with generation.assumed(generation_name):
            converted = _converted(codes, scale, generation_name=generation_name, interpret=INTERPRETED,
                                   scaled=scaled)
        np.testing.assert_array_equal(np.asarray(converted, np.float32), expected)
        # Compiled for the device on any host, its PTX through ptxas (a process of the device's capability,
        # `compile_mosaic_kernels`): an integer conversion's fold into sm_90's packed int8-to-bf16 assembly
        # fails there, not in the lowering.
        from fla_gdn_support import KernelCase
        from kernel_compile import compile_mosaic_kernels, mosaic_failures

        shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (codes, scale)]
        compiled = functools.partial(_converted, generation_name=generation_name, interpret=False,
                                     scaled=scaled)
        found = compile_mosaic_kernels([KernelCase("converted", lambda: (compiled, shapes))], generation_name)
        assert not mosaic_failures(found), mosaic_failures(found)
    elif generation.device_generation() == generation_name:
        converted = _converted(codes, scale, generation_name=generation_name, interpret=False, scaled=scaled)
        np.testing.assert_array_equal(np.asarray(converted, np.float32), expected)


PAGE, PAGED_STEPS, PAGED_BLOCK = 16, 4, 64
"""A paged walk's page rows, steps and the rows of a step's block, its pages a block 4."""


def _paged_products(p, pool, table, *, interpret):
    """`p @ x` where `x`'s rows are the pool's pages in the table's order (each a 16-row page, four a
    64-row step's block): mma.sync's pipeline loads a step's block as four page copies at their table's
    entries (`mosaic.Rows`) and joins them into the one tile its products read (`CpAsync.walk`)."""
    from dew.nn.kernels.templates import mosaic_sm80

    family = mosaic_sm80.Mma(jnp.dtype(jnp.bfloat16))
    pieces = PAGED_BLOCK // PAGE

    def body(p_gmem, pool_gmem, table_gmem, out_gmem):
        blocks = {'p': ((FED_ROWS, PAGED_BLOCK), jnp.bfloat16), 'x': ((PAGED_BLOCK, FED_WIDTH), jnp.bfloat16)}
        allocated = family.streams(blocks, 2, 1, copies={'x': pieces})

        def walked(*, streams):
            views = {'p': (p_gmem, lambda j: (0, j)),
                     'x': (pool_gmem, lambda j: mosaic.Rows(table_gmem, j * pieces, (0,), PAGE))}

            step = _multiplying(family, 'p', 'x', interpret)
            return family.walk(family.opened(streams, views), ((step, 0, PAGED_STEPS),), (),
                               ((FED_ROWS, FED_WIDTH),), interpret)[1][0]

        summed = family.scoped(walked, streams=allocated)

        def stored(out_smem):
            plgpu.store(out_smem, summed, optimized=False)
            mosaic.synced(interpret)
            mosaic.copied(out_smem, out_gmem)

        pl.run_scoped(stored, mosaic.plain((FED_ROWS, FED_WIDTH), jnp.float32))

    out = jax.ShapeDtypeStruct((FED_ROWS, FED_WIDTH), jnp.float32)
    return family.kernel(body, out_type=out, grid=(1,), grid_names=("programs",), consumers=1,
                         interpret=interpret, name="paged")(p, pool, table)


def _paged_operands():
    """Small integers (the product exact in fp32) and a page table that reverses a pool of twice the pages a
    walk reads, each page at an odd place."""
    keys = jax.random.split(jax.random.key(7), 2)
    pages = 2 * PAGED_STEPS * PAGED_BLOCK // PAGE
    p = jax.random.randint(keys[0], (FED_ROWS, PAGED_STEPS * PAGED_BLOCK), -4, 4).astype(jnp.bfloat16)
    pool = jax.random.randint(keys[1], (pages * PAGE, FED_WIDTH), -4, 4).astype(jnp.bfloat16)
    table = jnp.arange(pages - 1, 0, -2, dtype=jnp.int32)
    return p, pool, table


@pytest.mark.parametrize("on_gpu", [False, True], ids=["lowered", "run"])
def test_a_cp_async_walk_joins_its_pages_into_the_tile_its_products_read(on_gpu):
    """An A100's walk over a paged operand, a step's 64 rows four 16-row pages at its table's entries:
    lowered for an A100 on any host, and on one exactly `p @ x` for `x` the pages in the table's order."""
    p, pool, table = _paged_operands()
    pages = np.asarray(pool, np.float32).reshape(-1, PAGE, FED_WIDTH)
    gathered = pages[np.asarray(table)].reshape(-1, FED_WIDTH)
    if on_gpu:
        if generation.device_generation() not in ("sm80", "sm86", "sm89"):
            pytest.skip("runs on an sm8x GPU")
        np.testing.assert_array_equal(np.asarray(_paged_products(p, pool, table, interpret=False)),
                                      np.asarray(p, np.float32) @ gathered)
        return
    shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (p, pool, table)]
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm80")
    with generation.assumed("sm80"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(functools.partial(_paged_products, interpret=False)).trace(*shapes).lower(
            lowering_platforms=("cuda",))
