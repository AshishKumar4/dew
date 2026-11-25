"""The Mosaic toolkit (dew.nn.kernels.templates.mosaic) under Mosaic GPU's interpreter: a ring streaming one
operand across persistent tiles of their own trip counts, one consumer or two sharing its slots. The
interpreter is coverage, not proof (it treats wgmma_wait as a no-op); each kernel also lowers for an H100
here, and runs on one in the GPU gate."""

import dataclasses
import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from generation_support import COMPILES_ON_CPU, INTERPRETS, here, on_a_v5e, on_the_host
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.sharding import Mesh
from kernel_compile import gate_generations, stated_outputs
from reference_error import ahead_arm, equations

from dew.nn.kernels import generation
from dew.nn.kernels.templates import (
    across,
    assembled,
    bundles,
    emission,
    handoffs,
    mosaic,
    mosaic_sm80,
    outputs,
    product,
    resources,
    rings,
    stacked,
    stores,
    swizzling,
)
from dew.nn.kernels.templates.cost import Stated

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
        @functools.partial(pl.run_scoped, **rings.ring(BLOCK, x.dtype, STAGES, consumers=consumers),
                           out_smem=plgpu.SMEM((consumers, BLOCK[0], columns), jnp.float32),
                           collective_axes="wg")
        def _(smem, full, consumed, out_smem):
            ring = rings.Ring(smem, full, consumed)

            def producer(index):
                del index  # one producer warpgroup
                @plgpu.nd_loop((TILES,), collective_axes="sm")
                def _(loop: plgpu.NDLoopInfo):
                    tile = loop.index[0]
                    first = _first_step(tile)

                    @pl.loop(0, tile + 1)
                    def _(step):
                        block = x_gmem.at[pl.ds((first + step) * BLOCK[0], BLOCK[0])]
                        ring.load(ring.turn(first + step), block)

                rings.drain((ring,), jnp.int32(_first_step(TILES)))

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

            roles = (handoffs.Role("consumer", consumers), handoffs.Role("producer", weight=0))
            handoffs.Schedule(roles).run(
                consumer=consumer, producer=producer)

    return plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((TILES, *BLOCK), jnp.float32), grid=(1,),
                        grid_names=("sm",), num_threads=consumers + 1, thread_name="wg",
                        compiler_params=mosaic.compiler_params(), interpret=interpret)(x)


def _expected(x):
    blocks = np.asarray(x, np.float32).reshape(-1, *BLOCK)
    return np.stack([blocks[_first_step(tile):_first_step(tile) + tile + 1].sum(0) for tile in range(TILES)])


def _stream():
    """Small integers, so any tile's sum is exact in fp32 whatever its order: NumPy's, as a site's draws are
    (a GPU tier's prepare computes on no device)."""
    steps = _first_step(TILES)
    return np.random.default_rng(0).integers(-64, 64, (steps * BLOCK[0], BLOCK[1])).astype(jnp.bfloat16)


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
    """The ring's kernel compiled and run on an H100, where the interpreter's no-op waits are real, its
    program compiled in a tier's prepare (`ahead_arm`)."""
    x = _stream()
    summed = ahead_arm(jax.jit(functools.partial(_ring_sums, interpret=False, consumers=consumers)),
                       f"a ring's tile sums: {consumers} consumers")(x)
    np.testing.assert_array_equal(np.asarray(summed), _expected(x))


def _multiplying(family, lhs: str, rhs: str, interpret):
    """A walk's step that multiplies its `lhs` slot by its `rhs` slot into its one accumulator."""
    def step(index, slots, carry, accumulators):
        del index
        (acc,) = accumulators
        return carry, (family.products(acc, slots[lhs], slots[rhs]),)

    return step


@pytest.mark.parametrize("unrolled", [False, True])
@pytest.mark.parametrize("stages", [2, 3, 4, 5])
def test_a_walk_takes_each_step_of_its_range_once_in_order_at_its_turn(stages, unrolled):
    """`mosaic.stepping`'s loops, run in turn as a walk's are, take steps `lo` to `hi` once each and in order,
    each at its counter's slot and rotation, for every start, length and ring origin over three rotations:
    rolled, a step a trip; unrolled, the steps taken one a trip are the fewer than `stages` before the whole
    rotations and the fewer after them, and a rotation's steps know the steps before them in it are the
    range's. Its eager steps the host CPU's (a tier's prepare computes on no device)."""
    taken = []

    def turned(index, turn, *state, before=0):
        taken.append((index, turn.slot, turn.rotation, before))
        return state

    with on_the_host():
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


@pytest.mark.parametrize("weights", [(2, 1, 0), (1, 1, 0), (3, 1, 1, 0), (1, 0)])
@pytest.mark.parametrize("programs", [1, 2])
def test_a_schedules_roles_share_the_launch_pool_by_weight(weights, programs):
    """Each raised role's registers (`resources.role_registers`, a `Schedule`'s): none past the launch's pool
    or setmaxnreg's ceiling, a heavier role never below a lighter one, and the pool spent to within a step
    a role unless every role is at the ceiling (a capped role's share goes to the rest)."""
    with generation.assumed("sm90"):
        held = resources.role_registers(weights, programs)
        pool = resources.entry_registers(len(weights), programs) * len(weights)
        step, ceiling = resources.facts().raises().step, resources.raise_ceiling()
        floor = resources.producer_floor()
    assert sum(held) <= pool
    raised = [(weight, budget) for weight, budget in zip(weights, held, strict=True) if weight]
    assert all(budget == floor for weight, budget in zip(weights, held, strict=True) if not weight)
    assert all(budget <= ceiling and budget % step == 0 for _, budget in raised)
    assert all(lighter <= heavier for (w1, heavier) in raised for (w2, lighter) in raised if w1 >= w2)
    assert (all(budget == ceiling for _, budget in raised)
            or pool - sum(held) < step * sum(weight for weight, _ in raised))


@pytest.mark.parametrize("generation_name", gate_generations(
    needs=lambda family, architecture: architecture.multiprocessor.register_raises is not None))
def test_a_launch_whose_producer_holds_more_lowers_it_to_its_own_floor_and_its_consumers_hold_the_rest(
        generation_name):
    """A launch whose producer lowers past setmaxnreg's least (its derivation's raise of one that spilled
    there, `resources.Producers`: the sm100 states launch's, 44 to 60 bytes at 24) runs its schedule's
    producer at them,
    its consumers sharing what remains of the launch's pool, and its register model states the same split:
    so the consumers' budget a derivation reads is the raise the kernel runs."""
    from dew.nn.kernels.templates import handoffs
    from dew.nn.kernels.templates.mosaic_sm90 import WGMMA

    with generation.assumed(generation_name):
        least, stated = resources.producer_floor(), 48
        for consumers in (1, 2):
            roles = WGMMA.roles(consumers, stated).registers()
            assert roles["producer"] == stated
            assert roles["consumer"] == WGMMA.registers(consumers, producer=stated)
            assert roles["consumer"] <= WGMMA.registers(consumers)
            assert WGMMA.roles(consumers).registers()["producer"] == least
            model = resources.register_model(WGMMA, consumers, roles["consumer"], producer=stated,
                                             target=generation.gpu_target())
            assert (model.producer, model.budget) == (stated, roles["consumer"])
            assert handoffs.Schedule(WGMMA.roles(consumers).roles, floor=stated).registers() == roles


@pytest.mark.parametrize("generation_name", gate_generations(
    needs=lambda family, architecture: architecture.multiprocessor.register_raises is not None))
def test_a_spilled_launchs_producer_steps_down_its_ladder_from_its_top(generation_name):
    """A derivation's rung (`resources.Producers.stepped`): a spilled launch's producer from the least to its
    stated cap where it has one (its consumers' statement's share), else to its top (its program's entry
    share), then from the stated cap to its top, then a setmaxnreg step lower each time, past the stated cap,
    asked already; every other launch's left where it was; none at the ladder's foot, a step above the least,
    so the derivation steps its tiles down instead; a top past setmaxnreg's most bounded at it."""
    with generation.assumed(generation_name):
        steps = resources.facts().raises()
        top = steps.lowest + 13 * steps.step
        first = resources.Producers().stepped({"states": top}, {})
        second = first.stepped({"states": top, "scan": top}, {})
        assert (first.of("states"), first.of("scan")) == (top, None)
        assert (second.of("states"), second.of("scan")) == (top - steps.step, top)
        stated = {"states": top - 2 * steps.step}
        rungs, rung = [], resources.Producers()
        for _ in range(4):
            rung = rung.stepped({"states": top}, stated)
            rungs.append(rung.of("states"))
        assert rungs == [top - 2 * steps.step, top, top - steps.step, top - 3 * steps.step]
        foot = resources.Producers((("states", steps.lowest + steps.step),))
        assert foot.stepped({"states": top}, {}) is None
        assert resources.Producers().stepped({"states": steps.lowest}, {}) is None
        assert resources.Producers().stepped({"states": steps.highest + steps.step}, {}).of("states") == (
            steps.highest)


@pytest.mark.parametrize("generation_name", gate_generations(
    needs=lambda family, architecture: architecture.multiprocessor.register_raises is not None))
def test_a_planned_launch_at_its_producers_registers_holds_the_programs_and_budget_its_schedule_runs(
        generation_name):
    """A launch planned at its producer's raised registers (`resources.Launch.producer`, as certification
    raises the sm100 states launch's) takes the programs a multiprocessor holds and the consumer budget its
    schedule runs at
    that split (`Family.roles`), which the planner, the walk's schedule and the gate's register model all
    read, so no decision holds a launch at the split it does not run."""
    from dew.nn.kernels.templates.mosaic_sm90 import WGMMA

    with generation.assumed(generation_name):
        stated = 48
        budget = WGMMA.roles(1, stated).registers()["consumer"]
        launch = resources.Launch(lambda stages: resources.Scope(), 1, budget, floor=1, most=1,
                                  producer=stated)
        (plan,) = resources.plan([launch], WGMMA, generation.gpu_target(generation_name))
        assert plan.programs == WGMMA.programs(1, budget, stated)
        over = dataclasses.replace(launch, registers=budget + WGMMA.register_step)
        assert resources.plan([over], WGMMA, generation.gpu_target(generation_name)) is None
        model = resources.register_model(WGMMA, 1, budget, producer=stated, target=generation.gpu_target())
        assert (model.budget, model.producer) == (budget, stated)


def test_a_launch_with_a_joined_correction_warpgroup_counts_its_registers_in_its_residency_and_budget():
    """A B200 launch of two softmax consumers beside a correction warpgroup joined at half their weight
    (`Family.role_schedule`'s `joined`, FlashAttention-4's three warpgroups and their producer): its program
    runs one warpgroup more, its consumers' budget is its schedule's (below two consumers' alone, the
    program's launch pool shared four ways), its correction role's half of it, and its register model counts
    the correction's registers in what a program allocates, so a multiprocessor holds no more programs of it
    than its registers do (row 5368); and its schedule refuses, as it traces, the role it was sized by run
    with no body, or a body of a role it was not (row 9785: a warpgroup launched beside no body hung)."""
    from dew.nn.kernels.templates import mosaic_sm100

    with generation.assumed("sm100"):
        family = mosaic_sm100.RINGS
        joined = ((handoffs.Role("correction", weight=0.5), 1),)
        held = family.role_schedule(2, joined=joined).registers()
        assert family.warpgroups(2, joined=joined) == family.warpgroups(2) + 1
        assert family.registers(2, joined=joined) == held["consumer"] < family.registers(2)
        assert 0 < held["correction"] < held["consumer"]
        model = resources.register_model(family, 2, held["consumer"], joined=joined,
                                         target=generation.gpu_target())
        alone = resources.register_model(family, 2, held["consumer"], target=generation.gpu_target())
        assert model.resident(held["consumer"]) <= alone.resident(held["consumer"])
        assert model.most == family.programs(2, held["consumer"], joined=joined)
        for given, bodies in ((joined, {}), ((), {"correction": print})):
            with pytest.raises(ValueError):
                family.scheduled(2, print, lambda: None, joined=given, bodies=bodies)


def test_a_pairs_step_is_the_pairs_block_its_ctas_split_and_its_bundle_is_a_configs():
    """A pair's step (`rings.Paired` indices) is the pair's block, the slot's twice along the axis its CTAs
    split, at the indices in the pair's blocks (`viewed`'s `Split`), where a plain step is the slot's block;
    and the pair's bundle (`mosaic_sm100.PAIR`, its products over a cluster of two, each CTA a half of
    tcgen05's widest product's columns) is composed for a Blackwell launch stating its accumulators and
    reached by a config's name, where a device without tcgen05 composes none. Its ref the host CPU's (a
    tier's prepare computes on no device)."""
    from dew.nn.kernels.templates import bundles, mosaic_sm100

    with on_the_host():
        ref = jax.new_ref(jnp.arange(256 * 512, dtype=jnp.float32).reshape(256, 512))
        block = (64, 128)
        split = rings.viewed((ref, lambda step: rings.Paired((1, step), along=0)), block, 2)
        assert isinstance(split, rings.Split) and split.along == 0 and split.ref.shape == (128, 128)
        np.testing.assert_array_equal(np.asarray(split.ref[...]), np.asarray(ref[128:256, 256:384]))
        plain = rings.viewed((ref, lambda step: (1, step)), block, 2)
        np.testing.assert_array_equal(np.asarray(plain[...]), np.asarray(ref[64:128, 256:384]))

    pair = mosaic_sm100.PAIR
    assert (pair.cluster, pair.pairs_product, pair.widest_product) == (2, True, 128)
    assert (mosaic_sm100.RINGS.cluster, mosaic_sm100.RINGS.pairs_product) == (1, False)
    assert bundles.named("tcgen05.pair", jnp.bfloat16, None) is pair
    assert pair in bundles.groups("sm100", jnp.bfloat16, stated=True)[0]
    assert pair not in [family for group in bundles.groups("sm90", jnp.bfloat16, stated=True)
                        for family in group]


def test_a_launchs_register_model_runs_the_programs_its_shared_memory_holds():
    """A launch's register model states the programs its plan runs: its registers' residency bounded by the
    programs its staged bytes hold on the target it derives for (row 5523: a B200's gathered input gradient
    staging 230,580 bytes was stated three programs, which its registers held, and ran one)."""
    from dew.nn.kernels.templates.mosaic_sm90 import WGMMA

    with generation.assumed("sm90"):
        unstaged = resources.register_model(WGMMA, 1, 112, target=generation.gpu_target())
        staged = resources.register_model(WGMMA, 1, 112, shared=200 * 1024, target=generation.gpu_target())
    assert unstaged.most > 1
    assert staged.most == 1


def test_a_register_models_caps_are_its_targets_wherever_they_are_asked():
    """The caps an H100's register model states (`RegisterModel.cap`), asked after the model is made with
    an A100 in effect, are those asked under the H100: its entry share and its consumers' raise beside its
    producer its own multiprocessor's (row 12028: they read the GPU in effect as they were asked, and an
    A100's takes no setmaxnreg)."""
    from dew.nn.kernels.templates.mosaic_sm90 import WGMMA

    model = resources.register_model(WGMMA, 2, 168, moves=True, target=generation.gpu_target("sm90"))
    assert model.producer and model.moves
    with generation.assumed("sm90"):
        own = [model.cap(programs, honored=honored) for programs in (1, 2) for honored in (False, True)]
    with generation.assumed("sm80"):
        asked = [model.cap(programs, honored=honored) for programs in (1, 2) for honored in (False, True)]
    assert asked == own


def test_a_register_model_decides_on_its_target_whatever_target_is_in_effect():
    """A register model asked for an H100 (`register_model`'s `target`) under an A100's derivation, whose
    multiprocessor has no setmaxnreg steps, decides as it does under the H100's own: its producer, file,
    residency and shared memory each its target's (row 11987's cross-generation checks raised reading the
    A100's setmaxnreg steps for an H100's producer)."""
    from dew.nn.kernels.templates.mosaic_sm90 import WGMMA

    h100 = generation.gpu_target("sm90")
    with generation.assumed("sm90"):
        own = resources.register_model(WGMMA, 2, 168, shared=100 * 1024, target=h100)
    with generation.assumed("sm80"):
        asked = resources.register_model(WGMMA, 2, 168, shared=100 * 1024, target=h100)
    assert asked == own


def test_a_kernel_named_with_its_register_model_states_it_on_its_launch():
    """A kernel named with its register model (`report.named`'s `model`) records it on the launch tracing it
    (`report.Launch.kernels`), by its name less the launch's suffix, where the compile point reads the
    producer its schedule lowers to (`kernel_compile.producer_failures`); named outside any launch, it states
    nothing."""
    from dew.nn.kernels.templates import attention, report
    from dew.nn.kernels.templates.attention_mosaic import MosaicForward, Tiles
    from dew.nn.kernels.templates.mosaic_sm90 import WGMMA

    op = MosaicForward()
    q = jax.ShapeDtypeStruct((1, 16, 1, 16), jnp.bfloat16)
    starts = jax.ShapeDtypeStruct((1, 16), jnp.int32)
    bound = op.bind(q, q, q, starts, starts, None, None, (), (), None, residuals=attention.Kept.NONE,
                    interpret=True)
    with generation.assumed("sm90"):
        stated = resources.register_model(WGMMA, 2, 168, target=generation.gpu_target())
        with report.launching(op, bound, Tiles(16, 16, 2, 1, family="wgmma")):
            name = report.named("attention", model=stated)
        kernel, launch, _ = assembled.kernel_of(name)
        assert kernel == "attention" and report.traced(launch).kernels == {"attention": stated}
        assert report.named("attention", model=stated) == "attention"


def test_a_handoffs_barriers_count_the_arrivals_of_its_roles():
    """A hand-off's barriers (`Schedule.scratch`) count the arrivals of the roles that fill and give it
    back: a signal two softmax warpgroups fill counts two, and an accumulator ring the product warp fills
    and two epilogue warpgroups give back counts one done arrival and two read ones. A role of weight 0
    beside a warp role is refused, the last warpgroup holding one or the other."""
    from dew.nn.kernels.templates import mosaic_sm100

    schedule = handoffs.Schedule(
        (handoffs.Role("softmax", count=2), handoffs.Role("epilogue", count=2), *mosaic_sm100.ISSUING),
        (handoffs.HandOff("scales", mosaic_sm100.Signals(count=2, tensor_core=False), "softmax"),
         handoffs.HandOff("output", mosaic_sm100.Accumulators(2, 128, 64), "mma", ("epilogue",))))
    scales, (_, done, read) = schedule.scratch(generation.gpu_target("sm100"))
    assert scales.num_arrivals == 2
    assert (done.num_arrivals, read.num_arrivals) == (1, 2)
    assert schedule.warpgroups() == 5
    with pytest.raises(ValueError):
        handoffs.Schedule((handoffs.Role("consumer"), handoffs.Role("producer", weight=0),
                           *mosaic_sm100.ISSUING))


def test_warp_roles_take_their_warps_in_turn_within_one_warpgroup():
    """A schedule's warp roles each take `count` warps of its last warpgroup in turn (`Schedule.places`):
    tcgen05's load warps every warp its product warp leaves, then the product warp; warp roles of more
    warps than a warpgroup holds are refused."""
    from dew.nn.kernels.templates import mosaic_sm100

    schedule = handoffs.Schedule((handoffs.Role("epilogue"), *mosaic_sm100.ISSUING))
    assert schedule.places() == {"load": 0, "mma": handoffs.WARPS - 1}
    assert schedule.role("load").count + schedule.role("mma").count == handoffs.WARPS
    with pytest.raises(ValueError):
        every_warp = handoffs.Role("load", handoffs.WARPS, warp=True, weight=0)
        handoffs.Schedule((handoffs.Role("epilogue"), every_warp, mosaic_sm100.PRODUCTS)).places()


def test_a_joined_warpgroup_role_takes_its_weights_share_of_the_raised_registers():
    """A warpgroup role joined to a schedule (`Schedule.joined`: a correction warpgroup beside two consumers,
    FlashAttention-4's) runs after its raised roles and before its last warpgroup's, its warpgroups counted
    in the program's, its registers its weight's share of the raise beside the consumers' (lighter, no
    more than theirs), and the hand-offs it fills or gives back the schedule's; a role of weight 0 joins
    only as a warp of the last warpgroup (row 5368)."""
    from dew.nn.kernels.templates import mosaic_sm100

    base = handoffs.Schedule((handoffs.Role("consumer", 2, weight=2), *mosaic_sm100.ISSUING))
    factors = handoffs.HandOff("factors", None, "consumer", ("correction",))
    joined = base.joined(handoffs.Role("correction", weight=1), 1, factors)
    assert [role.name for role in joined.roles] == ["consumer", "correction", "load", "mma"]
    assert joined.warpgroups() == base.warpgroups() + 1
    assert joined.handoffs == (factors,)
    with generation.assumed("sm100"):
        held = joined.registers()
    assert held["correction"] < held["consumer"] and held["correction"] > held["mma"]
    with pytest.raises(ValueError):
        base.joined(handoffs.Role("correction", weight=0), 1)


def test_shared_bytes_counts_a_programs_allocations_by_lifetime():
    """A program's shared memory (`resources.shared_bytes`): each array rounded up to 1 KiB, its walks the
    largest of them, every barrier 8 bytes, a named CTA barrier none (the block's hardware barrier,
    `plgpu.CtaBarrier`, a packed decode's consumers' meet); a tile scoped after its walks beside the program's
    on a family whose walks share a scope (warpgroup MMA), the larger of it and the walks on one that scopes
    each walk apart (mma.sync's emit_pipeline)."""
    from dew.nn.kernels.templates import mosaic_sm90

    program = (plgpu.SMEM((100,), jnp.float32), plgpu.Barrier(num_barriers=3),
               plgpu.CtaBarrier(num_arrivals=2))
    walks = ((plgpu.SMEM((2, 1024), jnp.bfloat16),), (plgpu.SMEM((4, 1024), jnp.bfloat16),))
    scope = resources.Scope(program=program, walks=walks, after=(((512,), jnp.float32),))
    # 400 bytes -> 1 KiB; walks 4 KiB and 8 KiB -> 8 KiB; 3 barriers; the after tile's 2 KiB.
    assert resources.shared_bytes(mosaic_sm90.WGMMA, scope) == 1024 + 2048 + 8192 + 3 * 8
    assert resources.shared_bytes(mosaic_sm80.Mma(jnp.dtype(jnp.bfloat16)), scope) == 1024 + 8192 + 3 * 8
    # Three warpgroups reducing implicitly each take their share of Mosaic GPU's scratch on warpgroup MMA.
    reducing = dataclasses.replace(scope, reducing_warpgroups=3)
    assert resources.shared_bytes(mosaic_sm90.WGMMA, reducing) == 1024 + 3 * 2048 + 2048 + 8192 + 3 * 8


def _fits(found, launches, family, target) -> bool:
    """Whether each plan `resources.plan` returned for `launches` fits `target`: its stages within its
    launch's floor and most, its shared bytes its scope's at those stages and within a program's share of
    the block at its residency, its programs within what the family's registers and the multiprocessor's
    resident threads hold at the launch's registers."""
    block = generation.shared_budget(target)
    return all(launch.floor <= plan.stages <= launch.most
               and plan.shared == resources.shared_bytes(family, launch.scope(plan.stages))
               and plan.shared <= resources.program_shared(block, plan.programs)
               and 1 <= plan.programs <= family.programs(launch.consumers, launch.registers)
               and plan.programs * launch.consumers <= resources.resident_warpgroups()
               for plan, launch in zip(found, launches, strict=True))


def test_a_plan_fits_its_target_or_none_is_made():
    """`resources.plan` on an A100 for one mma.sync consumer at 128 registers a thread: a launch of 16 KiB
    a stage and one of 40 KiB each plan within the block's shared memory, the registers and the resident
    threads (`_fits`); one whose floor of two stages no residency's share holds (200 KiB a stage) plans
    nothing."""
    family = mosaic_sm80.Mma(jnp.dtype(jnp.bfloat16))

    def launch(stage_kib: int) -> resources.Launch:
        return resources.Launch(lambda stages: resources.Scope(
            program=(plgpu.SMEM((stages * stage_kib * 1024,), jnp.uint8),)), consumers=1, registers=128)

    with generation.assumed("sm80"):
        target = generation.gpu_target()
        for kib in (16, 40):
            found = resources.plan([launch(kib)], family, target)
            assert found is not None and _fits(found, [launch(kib)], family, target), kib
        assert resources.plan([launch(200)], family, target) is None


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


def _merged(partials, combine):
    """Each of `partials`' leading slices a program's partial of one tile, merged across the programs by
    `combine` (`Family.fixed_up`), interpreted: the tile's first program, walked last, combines them."""
    from dew.nn.kernels.templates import mosaic_sm90

    family = mosaic_sm90.FED  # one warpgroup, no producer
    slices, block = partials.shape[0], partials.shape[1:]

    def body(partials_gmem, out_gmem, slots_gmem):
        # The interpreter walks the grid in order, so the tile's first program is its last.
        program = slices - 1 - jax.lax.axis_index("programs")

        def store(merged):
            out_gmem[...] = merged

        family.fixed_up(plgpu.load(partials_gmem.at[program], optimized=False), slots_gmem, None, program,
                        0, slices, store, combine=combine)

    out = (jax.ShapeDtypeStruct(block, jnp.float32), jax.ShapeDtypeStruct(partials.shape, jnp.float32))
    return family.kernel(body, out_type=out, grid=(slices,), grid_names=("programs",), consumers=1,
                         interpret=True, name="merged")(partials)[0]


@INTERPRETS
@pytest.mark.parametrize("slices", [2, 3, 5, 9])
@pytest.mark.parametrize("block", [(16, 8), (64, 32), (128, 64)])
def test_a_tiles_partials_merge_in_the_programs_order_whatever_their_count(block, slices):
    """Interpreted: the programs' partials of one tile merged across the programs (`Family.fixed_up`) by
    a merge that neither commutes nor associates, `3 * running + partial`: exactly its fold over the programs
    in order, at tiles whose merger reads one partial at a time or several, and at counts of partials those
    reads do not divide."""
    def combine(running, partial):
        return 3 * running + partial

    partials = jax.random.randint(jax.random.key(4), (slices, *block), -4, 4).astype(jnp.float32)
    np.testing.assert_array_equal(np.asarray(_merged(partials, combine)),
                                  functools.reduce(combine, np.asarray(partials)))


@COMPILES_ON_CPU
def test_a_tile_its_programs_share_merges_without_serializing_their_warpgroup_mmas():
    """M-FS-74's Qwen3.5-4B q input gradient (2048 x 8192 against 2560 x 8192 bf16, the right operand
    transposed) at a 128 x 256 tile on an H100: 160 tiles over its 132 multiprocessors, so programs share a
    tile's depth and its last arriver merges their partials (`Family.fixed_up`, `across.arrived`). Its merge's
    branch on a verdict ptxas held lane by lane put a compiler-inserted warpgroup arrive in a divergent path
    and serialized every HGMMA of the walk (C7520: 1.23x XLA's time on a GH200, row 10151): the binary its
    compile assembles notes no C7520, nor does the gate's own assembly of its PTX by each CUDA build's ptxas
    note any serialization (`assembled.SERIALIZED`)."""
    import kernel_compile
    from fla_gdn_support import KernelCase

    from dew.nn.kernels import gemm, grouped_matmul, tuning
    from dew.nn.kernels.templates.gemm.tiling import Tiles
    from dew.nn.kernels.templates.resources import Producers

    tile = Tiles(block_m=128, block_n=256, block_k=64, stages=4, grid_block_n=8, warpgroups=2, cluster=2,
                 family="wgmma", programs=1, splits=1, swapped=False, pingpong=False,
                 producers=Producers(caps=()), staging=None)
    operands = (jax.ShapeDtypeStruct((2048, 8192), jnp.bfloat16),
                jax.ShapeDtypeStruct((2560, 8192), jnp.bfloat16))
    call = functools.partial(gemm.gemm, transpose_rhs=True, out_dtype=jnp.bfloat16)
    with generation.chosen(grouped_matmul="mosaic"), tuning.pinned({grouped_matmul.MosaicGmm: tile}):
        [record] = kernel_compile.compile_mosaic_kernels([KernelCase("q-dx", lambda: (call, operands))],
                                                         "sm90")
    assert record.get("binaries"), record.get("error")
    diverged = [(binary["kernel"], note["message"]) for binary in record["binaries"]
                for note in binary["notes"] if note["code"] == "C7520"]
    serialized = [(kernel["kernel"], code) for kernel in record["kernels"] for code in kernel["losses"]
                  if code in assembled.SERIALIZED]
    assert (diverged, serialized) == ([], []), (diverged, serialized)


def test_a_plan_holds_no_more_programs_than_its_target_runs_threads_for():
    """`resources.plan` for one mma.sync consumer at 32 registers a thread and 1 KiB a program, whose
    registers alone would hold sixteen programs: on an A10, an RTX PRO 6000 and an A100 its plan fits each
    part's resident threads (`_fits`), whatever their count (1536 on the first two, 2048 on an A100)."""
    def scope(stages: int) -> resources.Scope:
        return resources.Scope(program=(plgpu.SMEM((stages * 1024,), jnp.uint8),))

    launch = resources.Launch(scope, consumers=1, registers=32, floor=1)
    family = mosaic_sm80.Mma(jnp.dtype(jnp.bfloat16))
    for name in ("sm86", "sm120", "sm80"):
        with generation.assumed(name):
            target = generation.gpu_target()
            found = resources.plan([launch], family, target)
            assert found is not None and _fits(found, [launch], family, target), name


def test_a_launch_with_a_producer_plans_within_its_target():
    """`resources.plan` on an H100 for a warpgroup-MMA launch beside its producer, shaped as the 27B gated
    delta net's states launch (41 KiB of its own, 36 KiB a stage, 96 registers) and as a one-stage launch
    of 64 KiB at 128 registers: each plan fits the block, the registers with the producer's and the
    resident threads (`_fits`); a launch whose own bytes exceed the block plans nothing."""
    from dew.nn.kernels.templates import bundles

    def launch(fixed: int, step: int, registers: int, most: int) -> resources.Launch:
        def scope(stages: int) -> resources.Scope:
            return resources.Scope(program=(plgpu.SMEM((fixed + stages * step,), jnp.uint8),))

        return resources.Launch(scope, consumers=1, registers=registers, floor=1, most=most)

    with generation.assumed("sm90"):
        family, target = bundles.of("sm90", jnp.bfloat16), generation.gpu_target()
        for made in (launch(42016, 36912, 96, 8), launch(65536, 0, 128, 1)):
            found = resources.plan([made], family, target)
            assert found is not None and _fits(found, [made], family, target)
        past = launch(generation.shared_budget(target) + 1, 0, 96, 1)
        assert resources.plan([past], family, target) is None


def _vector_scaled(scale, x, axis: str, *, interpret: bool):
    """An sm80 program scaling a 64 x 128 accumulator by a side vector along `axis` ('rows' or 'columns'),
    the vector loaded as an epilogue's side tile is (`Family.loaded`, its axes stated)."""
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


def test_a_launch_adding_into_two_refs_of_one_shape_is_refused_at_its_trace():
    """Two zeroed refs of one shape and dtype a launch adds into in place are one value to XLA, which a
    dynamic-slice fusion wrapping the launch passes once for both (row 3180): `mosaic.checked` refuses the
    launch where it traces them, and takes them stacked into one ref (`stacked.zeros`), each member its own
    place: a launch writing a value of its own into each view (`stacked.views`) reads each back apart
    (`stacked.read`), in its shapes' order."""
    shapes = [(2, 3), (4,), (2, 3)]

    def written(*refs):
        for index, view in enumerate(stacked.views(refs, groups)):
            view[...] = jnp.full(view.shape, index + 1.0)

    launched = mosaic.checked(written, "probe")

    def apart():
        launched(*(jax.new_ref(jnp.zeros(shape, jnp.float32)) for shape in shapes))

    groups = stacked.groups(shapes)
    with pytest.raises(ValueError):
        jax.jit(apart).trace()

    def together():
        refs = stacked.zeros(shapes, groups, jnp.zeros(()))
        launched(*refs)
        return stacked.read(refs, groups)

    for index, (shape, array) in enumerate(zip(shapes, ahead_arm(jax.jit(together), "stacked refs")(),
                                               strict=True)):
        np.testing.assert_array_equal(array, np.full(shape, index + 1.0))


PERSISTED_TILES, PERSISTED_STEPS, PERSISTED_DEPTH = 3, 2, 64


def _padded(rows: int, family) -> int:
    """A tile's rows for a launch of `rows`: padded to the least its family's products issue."""
    return -(-rows // family.rows) * family.rows


def _persisted_products(x, w, *, family, interpret, rows: int | None = None, resident: int = 1,
                        staging: tuple[int, int] | None = None, consumers: int = 1):
    """`x @ w` a tile at a time through `family`'s persistent walk (`Family.persist`): each tile the
    family's least rows covering `rows` (its own least where None; `_padded`), of which it stores `rows`,
    tile `t` the tiles' rows `t` on, each step `PERSISTED_DEPTH` of the contraction, the middle tile
    multiplying nothing (zeros); each tile issued (`issue`'s walk) and finished (`Issuer.chunks`, each
    chunk stored at its columns), on the family's roles, launched for `resident` programs a multiprocessor,
    its output staged in `staging`'s chunk columns and buffers (`stores.output_staging`'s by default).
    Two `consumers` each walk every tile, and the first alone stores it, as a balanced tail's last program
    to arrive does."""
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
                del active

                def stored_tile():
                    for first, (value,) in family.chunks(accs):
                        store(0, value, o_gmem, (tile * rows, 0, stored, first, rows))

                pl.when(wg == 0)(stored_tile)

            family.persist(streams, (PERSISTED_TILES,), schedule, issue, finish, consumers=consumers,
                           axis="programs", outputs=((shape, jnp.float32),), interpret=interpret,
                           accumulators=(shape,), chunks=staging)

        family.scoped(scope, streams=family.streams(blocks, 2, consumers))

    # Its outputs as its operands' axes where its rows are `x`'s (`outputs.Output`), padded past them a shape.
    out = (outputs.output(jnp.float32, outputs.axis(x, 0), outputs.axis(w, 1))
           if x.shape[0] == PERSISTED_TILES * rows else
           jax.ShapeDtypeStruct((PERSISTED_TILES * rows, columns), jnp.float32))
    return family.kernel(body, out_type=out, grid=(1,), grid_names=("programs",), consumers=consumers,
                         interpret=interpret, name="persisted", resident=resident)(x, w)


def _persisted_operands(rows: int):
    """Small integers, so every tile's product is exact in fp32 whatever its order: NumPy's (a GPU tier's
    prepare computes on no device, row 8932)."""
    rng = np.random.default_rng(2)
    depth = PERSISTED_STEPS * PERSISTED_DEPTH
    x = rng.integers(-4, 4, (PERSISTED_TILES * rows, depth)).astype(jnp.bfloat16)
    w = rng.integers(-4, 4, (depth, 128)).astype(jnp.bfloat16)
    return x, w


def _persisted_expected(x, w, rows: int) -> np.ndarray:
    """`_persisted_products`' value: each tile's rows its product, the middle tile's zeros."""
    expected = np.asarray(x, np.float32) @ np.asarray(w, np.float32)
    expected[rows:2 * rows] = 0
    return expected


@INTERPRETS
def test_two_mma_sync_consumers_finish_each_tile_when_one_alone_stores_it():
    """Interpreted: two mma.sync consumers' persistent tiles, each finished by both and stored by the first
    alone (a balanced tail's last program to arrive for one consumer): every tile stored, exactly `x @ w`
    (the store's own collective scope waited for a consumer that never stores, and hung)."""
    family = bundles.named("mma.sync", jnp.bfloat16, None)
    x, w = _persisted_operands(family.rows)
    got = _persisted_products(x, w, family=family, interpret=INTERPRETED, consumers=2)
    np.testing.assert_array_equal(np.asarray(got), _persisted_expected(x, w, family.rows))


def test_a_kernels_outputs_stated_as_its_operands_axes_are_stated_on_its_call():
    """A kernel whose output its operands' axes state (`outputs.output`: `x`'s rows by `w`'s columns),
    lowered for a B200 on any host: its shape is theirs, and its call states the dimensions each output
    descends from (`outputs.OUTPUT_DIMS`, its operands by their place), which a compile's live buffers read;
    the TPU core's, interpreted, takes the same statement as its shape."""
    from dew.nn.kernels.templates import mosaic_sm100

    family = mosaic_sm100.RINGS
    x, w = (jax.ShapeDtypeStruct(each.shape, each.dtype) for each in _persisted_operands(family.rows))
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm100")
    with generation.assumed("sm100"), jax.sharding.use_abstract_mesh(mesh):
        lowered = jax.jit(functools.partial(_persisted_products, family=family, interpret=False)).trace(
            x, w).lower(lowering_platforms=("cuda",))
    assert lowered.out_info.shape == (x.shape[0], w.shape[1])
    assert [dims for _, dims in stated_outputs(lowered) if dims is not None] == [[[[[0, 0]], [[1, 1]]]]]


def test_a_pallas_calls_outputs_stated_as_its_operands_axes_are_stated_on_its_call():
    """A Pallas call (`launches.pallas_call`, Triton's) whose output its operand's axes state, transposed,
    lowered for an H100 on any host: its shape is theirs, and its call states each dimension's axis
    (`outputs.OUTPUT_DIMS`), as a Mosaic launch's does."""
    from dew.nn.kernels import launches

    def copied(x):
        def body(x_ref, o_ref):
            o_ref[...] = x_ref[...].T

        transposed = outputs.output(x.dtype, outputs.axis(x, 1), outputs.axis(x, 0))
        return launches.pallas_call(body, name="copied", grid=(1,), out_shape=transposed)(x)

    x = jax.ShapeDtypeStruct((16, 32), jnp.float32)
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        lowered = jax.jit(copied).trace(x).lower(lowering_platforms=("cuda",))
    assert lowered.out_info.shape == (32, 16)
    assert [dims for _, dims in stated_outputs(lowered) if dims is not None] == [[[[[0, 1]], [[0, 0]]]]]


@INTERPRETS
@pytest.mark.parametrize("staging", [None, (32, 1), (16, 2)], ids=["default", "one buffer", "narrower"])
def test_tcgen05_tiles_walk_into_tensor_memory_and_their_epilogue_reads_it_out_in_chunks(staging):
    """Interpreted: a persistent walk on tcgen05's bundle (`mosaic_sm100.RINGS`), its product warp issuing
    each tile's steps into one of two accumulator turns in tensor memory and the epilogue reading each
    tile's out in chunks, over three tiles (the turns wrap) with the middle one multiplying nothing, as a
    Blackwell's tensor memory holds them, its output staged in the default chunks, one buffer of them, or
    narrower chunks double-buffered (`Family.persist`'s `staging`, row 5634: a block a ring stage short):
    each tile's rows exactly its product, the middle's zeros."""
    from dew.nn.kernels.templates import mosaic_sm100
    family = mosaic_sm100.RINGS
    x, w = _persisted_operands(family.rows)
    with generation.assumed("sm100"):
        got = np.asarray(_persisted_products(x, w, family=family, interpret=INTERPRETED, staging=staging))
    np.testing.assert_array_equal(got, _persisted_expected(x, w, family.rows))


def test_a_tcgen05_tiles_stated_chunk_holds_no_more_than_the_piece_its_epilogue_reads_out():
    """A 128-column tcgen05 tile, read out of tensor memory 64 columns at a time (`Issuer.chunk_columns`),
    stages its output in a stated chunk of those 64 columns, and is refused one of 128, which holds two of
    its pieces at once (theme K01-g's review: the tile's width alone was checked)."""
    from dew.nn.kernels.templates import mosaic_sm100

    outputs, family = (((mosaic_sm100.RINGS.rows, 128), jnp.bfloat16),), mosaic_sm100.RINGS
    assert family.output_staging(outputs, 1, (64, 1))
    pytest.raises(ValueError, family.output_staging, outputs, 1, (128, 1))


@pytest.mark.parametrize("rows", [4, 8, 16, 24, 32])
@pytest.mark.parametrize("generation_name", gate_generations())
def test_a_few_rows_lower_on_every_bundle_their_device_issues(generation_name, rows):
    """A launch of a decode step's few rows lowers on each bundle its device composes for them
    (`bundles.layouts`' `rows`: mma.sync's on cp.async on an A100; on an H100 wgmma's two layouts and
    mma.sync's on TMA rings; on a B200, for a launch stating no accumulators, and on an RTX PRO 6000,
    mma.sync's on TMA rings, so neither leaves Mosaic, row 4727), its tile padded to the bundle's least rows
    and storing only its own."""
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


def _relayed_products(q, k, v, *, interpret, weighed=lambda scores, side: scores * 2, side=None):
    """FlashAttention 4's relay on tcgen05's bundle: `S = q k` into tensor memory by the product warp, read
    out by a role of the table in tensor memory's own layout (`Tcgen05Issuer.read`), its weights
    `weighed(S, side)` (doubled by default; `side`, where given, a vector over its columns the role loads in
    the layout that takes each column's from its rows, `native_column`) narrowed and written back as `P`
    (`write`), and `O = P v` with `P` the left operand from tensor memory (the TS product); `O` read out
    and stored by the same role. Its roles wait on signals the products' commit and the role's publication
    arrive at (`signals`). A keys-first backward's relay is the same, its `q` the keys and its `k` the
    queries' transpose (row 4839)."""
    from dew.nn.kernels.templates import mosaic_sm100
    family = mosaic_sm100.RINGS
    rows = q.shape[0]
    columns = v.shape[1]
    sides = () if side is None else (side,)

    def body(q_gmem, k_gmem, v_gmem, *refs):
        *side_gmem, o_gmem = refs

        def scoped(q_smem, k_smem, v_smem, o_smem, loaded, s_tmem, p_tmem, o_tmem, scored, weighed_, summed,
                   *side_smem):
            s = mosaic_sm100.Accumulator(s_tmem, (rows, k.shape[1]))
            p = mosaic_sm100.Accumulator(p_tmem, (rows, k.shape[1]), dtype=jnp.bfloat16)
            o = mosaic_sm100.Accumulator(o_tmem, (rows, columns))
            scored, weighed_, summed = (family.signals().opened(each) for each in (scored, weighed_, summed))
            first = rings.Turn.of(0, 1)

            def relay(index):
                del index
                taken = None
                for smem in side_smem:
                    plgpu.barrier_wait(loaded.at[3], phase=0)
                    taken = plgpu.load(smem, layout=family.native_column)
                plgpu.barrier_wait(scored.barriers.at[0], phase=0)
                family.write(p, weighed(family.read(s, layout=mosaic_sm100.TMEM), taken))
                weighed_.published(first)
                plgpu.barrier_wait(summed.barriers.at[0], phase=0)
                o_smem[...] = family.read(o)
                plgpu.wait_load_tmem()
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(o_smem, o_gmem)
                plgpu.wait_smem_to_gmem(0)

            def load(index):
                if index:  # the first load warp's copies alone: one arrival each at `loaded`
                    return
                for at, (gmem, smem) in enumerate(zip((q_gmem, k_gmem, v_gmem, *side_gmem),
                                                      (q_smem, k_smem, v_smem, *side_smem), strict=True)):
                    plgpu.copy_gmem_to_smem(gmem, smem, loaded.at[at])

            def products(index):
                del index  # one product warp
                for at in range(3):
                    plgpu.barrier_wait(loaded.at[at], phase=0)
                family.products(s, q_smem, k_smem)
                scored.committed(first)
                plgpu.barrier_wait(weighed_.barriers.at[0], phase=0)
                family.products(o, p, v_smem)
                summed.committed(first)

            handoffs.Schedule((handoffs.Role("relay"), *mosaic_sm100.ISSUING)).run(relay=relay, load=load,
                                                                                mma=products)

        smem = [plgpu.SMEM(each.shape, each.dtype, transforms=swizzling.swizzle_transforms(each.dtype, 128))
                for each in (q, k, v)]
        signals = [family.signals().scratch(), family.signals(by_products=False).scratch(),
                   family.signals().scratch()]
        pl.run_scoped(scoped, *smem, plgpu.SMEM((rows, columns), jnp.float32),
                      plgpu.Barrier(num_barriers=3 + len(sides)),
                      family.tensor_memory((rows, k.shape[1])),
                      family.tensor_memory((rows, k.shape[1]), jnp.bfloat16),
                      family.tensor_memory((rows, columns)), *signals,
                      *(plgpu.SMEM(each.shape, each.dtype) for each in sides), collective_axes="wg")

    out = jax.ShapeDtypeStruct((rows, columns), jnp.float32)
    return family.kernel(body, out_type=out, grid=(1,), grid_names=("programs",), consumers=1,
                         interpret=interpret, name="relayed")(q, k, v, *sides)


def _relayed_operands():
    """Small integers, so `2 (q k) v` is exact in fp32 and its weights exact in bf16: NumPy's (a GPU tier's
    prepare computes on no device)."""
    rng = np.random.default_rng(3)
    return tuple(rng.integers(-2, 3, shape).astype(jnp.bfloat16)
                 for shape in ((128, 64), (64, 128), (128, 64)))


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


def _keys_first_products(k, q_t, statistic, do, *, interpret):
    """A tcgen05 attention backward's keys-first relay (row 4839): `Sᵀ = k qᵀ` into tensor memory with keys as
    its rows, each query's `statistic` taken from its column (`_relayed_products`' `side`), and
    `dV = Pᵀ dO` with `Pᵀ` the left operand from tensor memory."""
    return _relayed_products(k, q_t, do, interpret=interpret, side=statistic,
                             weighed=lambda scores, taken: scores - taken[None, :])


def _keys_first_operands():
    """Small integers, so `(k qᵀ - statistic) dO` is exact in fp32 and `Pᵀ` exact in bf16: NumPy's."""
    rng = np.random.default_rng(5)
    k, q_t = (rng.integers(-1, 2, shape).astype(jnp.bfloat16) for shape in ((128, 64), (64, 64)))
    statistic = rng.integers(-2, 3, (64,)).astype(np.float32)
    return k, q_t, statistic, rng.integers(-2, 3, (64, 64)).astype(jnp.bfloat16)


@INTERPRETS
def test_a_keys_first_role_takes_each_querys_statistic_from_its_column_into_a_product_from_tensor_memory():
    """Interpreted: transposed scores issued into tensor memory with keys as rows, read out by a role in
    tensor memory's own layout, each query's statistic taken from its column (`native_column`), written back
    narrowed and read as the left operand of `dV = Pᵀ dO` from tensor memory: exactly `(k qᵀ - l) dO`."""
    k, q_t, statistic, do = _keys_first_operands()
    got = np.asarray(_keys_first_products(k, q_t, statistic, do, interpret=INTERPRETED))
    weights = np.asarray(k, np.float32) @ np.asarray(q_t, np.float32) - np.asarray(statistic)[None, :]
    np.testing.assert_array_equal(got, weights @ np.asarray(do, np.float32))


def test_the_keys_first_relay_lowers_for_a_b200():
    """The keys-first relay lowers for a B200 on any host, as the GPU gate runs it."""
    operands = tuple(jax.ShapeDtypeStruct(each.shape, each.dtype) for each in _keys_first_operands())
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm100")
    with generation.assumed("sm100"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(functools.partial(_keys_first_products, interpret=False)).trace(*operands).lower(
            lowering_platforms=("cuda",))


def _backward_tensor_memory(queries: int, block: int) -> tuple:
    """A keys-first attention backward's tensor memory at `queries` a step and value columns `block` a launch
    (row 4839): `Sᵀ` and `dPᵀ` in fp32, `Pᵀ` and `dSᵀ` in bf16, and dK's and dV's `block` columns."""
    from dew.nn.kernels.templates import mosaic_sm100
    issuer = mosaic_sm100.RINGS
    return (*(issuer.tensor_memory((128, queries)) for _ in range(2)),
            *(issuer.tensor_memory((128, queries), jnp.bfloat16) for _ in range(2)),
            *(issuer.tensor_memory((128, block)) for _ in range(2)))


@pytest.mark.parametrize(("width", "block"), [(128, 128), (256, 128)])
def test_a_launch_walks_the_widest_column_block_its_tensor_memory_holds_as_pallas_allocates_it(width, block):
    """The widest of a launch's value-column blocks whose tensor memory a B200 holds
    (`resources.widest_columns`): a keys-first backward at 64 queries a step holds dK's and dV's 128 columns
    beside its scores and their gradients' 192 (448 of 512) and not 256 (704), and Pallas allocates a kernel
    scoping them at the block taken and refuses one at the block above it, so the derivation and the allocator
    agree."""
    from dew.nn.kernels.templates import mosaic_sm100
    held = functools.partial(_backward_tensor_memory, 64)
    assert resources.widest_columns(width, 16, held, generation.gpu_target("sm100")) == block
    # A GPU without tensor memory holds no block of it.
    assert resources.widest_columns(width, 16, held, generation.gpu_target("sm90")) is None

    def scoping(columns):
        def body(x_gmem, o_gmem):
            pl.run_scoped(lambda *tmem: None, *_backward_tensor_memory(64, columns), collective_axes="wg")
        return mosaic_sm100.RINGS.kernel(body, out_type=jax.ShapeDtypeStruct((8,), jnp.float32), grid=(1,),
                                         grid_names=("programs",), consumers=1, interpret=False,
                                         name="scoping")

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm100")
    operand = jax.ShapeDtypeStruct((8,), jnp.float32)
    with generation.assumed("sm100"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(scoping(block)).trace(operand).lower(lowering_platforms=("cuda",))
        if block < width:
            with pytest.raises(ValueError):
                jax.jit(scoping(2 * block)).trace(operand).lower(lowering_platforms=("cuda",))


def test_a_four_bit_tile_takes_the_tensor_memory_columns_pallas_allocates_it():
    """A packed 4-bit tile takes eight elements a column (`resources.tensor_memory_columns`): beside 448 fp32
    columns, a 512-wide fp4 tile's 64 fill a B200's 512, and Pallas allocates a kernel scoping both."""
    from dew.nn.kernels.templates import mosaic_sm100
    issuer = mosaic_sm100.RINGS
    held = (issuer.tensor_memory((128, 448)), issuer.tensor_memory((128, 512), jnp.float4_e2m1fn))
    assert resources.tensor_memory_columns(held) == generation.gpu_target("sm100").tmem_columns

    def body(x_gmem, o_gmem):
        pl.run_scoped(lambda *tmem: None, *held, collective_axes="wg")

    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm100")
    launch = issuer.kernel(body, out_type=jax.ShapeDtypeStruct((8,), jnp.float32), grid=(1,),
                           grid_names=("programs",), consumers=1, interpret=False, name="four_bit")
    with generation.assumed("sm100"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(launch).trace(jax.ShapeDtypeStruct((8,), jnp.float32)).lower(lowering_platforms=("cuda",))


def test_a_tpu_core_walks_the_persistent_tiles_interpreted():
    """Interpreted on any host: the persistent walk on a TPU core's bundle (`mosaic_tpu.MXU`), each step's
    blocks DMA'd into its VMEM ring a step ahead and multiplied by the MXU, each tile's rows stored from
    VMEM, over three tiles with the middle one multiplying nothing: each tile's rows exactly its product,
    the middle's zeros. On the host's CPU (`on_the_host`)."""
    from dew.nn.kernels.templates import mosaic_tpu
    family = mosaic_tpu.MXU
    x, w = _persisted_operands(family.rows)
    with on_the_host():
        got = np.asarray(_persisted_products(x, w, family=family, interpret=True))
    np.testing.assert_array_equal(got, _persisted_expected(x, w, family.rows))


def test_the_tpu_persistent_walk_lowers_for_a_v5e():
    """The persistent walk on a TPU core's bundle lowers for a TPU v5e on any host, its devices abstract."""
    from dew.nn.kernels.templates import mosaic_tpu
    family = mosaic_tpu.MXU
    x, w = (jax.ShapeDtypeStruct(each.shape, each.dtype) for each in _persisted_operands(family.rows))
    with on_a_v5e():
        jax.jit(functools.partial(_persisted_products, family=family, interpret=False)).trace(x, w).lower(
            lowering_platforms=("tpu",))


@COMPILES_ON_CPU
def test_a_consumer_launched_at_each_residency_its_family_admits_raises_to_the_gates_cap_spill_free():
    """A persistent warpgroup-MMA walk, one consumer beside its producer, launched at each residency its
    family admits a multiprocessor on an H100 (`RegisterModel.most`) and compiled as the GPU gate compiles
    it: the registers its consumer raises to in its PTX (`setmaxnreg.inc`) are the gate's cap for it at that
    residency (`RegisterModel.cap`), the cap its record states its launch runs at, and its binary holds its
    registers within them, spilling nothing."""
    import kernel_compile
    from fla_gdn_support import KernelCase

    from dew.nn.kernels.templates import bundles

    with generation.assumed("sm90"):
        family = bundles.of("sm90", jnp.bfloat16)
        model = resources.register_model(family, 1, None, target=generation.gpu_target())
    x, w = (jax.ShapeDtypeStruct(each.shape, each.dtype) for each in _persisted_operands(family.rows))
    for programs in range(1, model.most + 1):
        call = functools.partial(_persisted_products, family=family, interpret=False, resident=programs)
        case = KernelCase("persisted", lambda call=call: (call, (x, w)))
        [record] = kernel_compile.compile_mosaic_kernels([case], "sm90")
        [kernel] = record["kernels"]
        cap = model.cap(programs, honored=True)
        assert (kernel["raised"], kernel["cap"], kernel["spilled"]) == (cap, cap, 0), (programs, kernel)
        assert kernel["registers"] <= cap


RELAYED = 64
"""A relayed product's depth and columns."""


def _chained(a, b, c, *, m_warps: int, interpret: bool):
    """`(a @ b) @ c` by mma.sync on `m_warps` warps along the rows (`MmaIssuer.specialized`'s split), each
    product's left operand a register value (`a` read from a plain tile, then the first product's
    accumulator) relayed through a tile the consumer binds where the warps split its columns (`relaying`),
    stored through registers."""
    family = mosaic_sm80.Mma(jnp.dtype(jnp.bfloat16), m_warps=m_warps)
    rows = a.shape[0]

    def body(a_gmem, b_gmem, c_gmem, out_gmem):
        def scoped(a_smem, b_smem, c_smem, relay, out_smem):
            for source, staged in ((a_gmem, a_smem), (b_gmem, b_smem), (c_gmem, c_smem)):
                swizzling.copied(source, staged)
            mosaic.synced(interpret)
            issuer = family.relaying(relay)
            first = issuer.product(plgpu.load(a_smem, optimized=False), b_smem, (rows, RELAYED))
            out = issuer.product(first.astype(jnp.bfloat16), c_smem, (rows, RELAYED))
            plgpu.store(out_smem, out, optimized=False)
            mosaic.synced(interpret)
            stores.store_rows(out_smem, out_gmem, 0, 0, jnp.int32(rows), 0, rows, copy=swizzling.coalesced,
                              widest=resources.copied_rows(RELAYED, jnp.float32))

        swizzle = resources.widest_swizzle(jnp.bfloat16, RELAYED)
        pl.run_scoped(scoped, swizzling.plain((rows, RELAYED), jnp.bfloat16),
                      *(swizzling.staged((RELAYED, RELAYED), jnp.bfloat16, swizzle) for _ in range(2)),
                      swizzling.plain((rows, RELAYED), jnp.bfloat16),
                      swizzling.plain((rows, RELAYED), jnp.float32))

    out = jax.ShapeDtypeStruct((rows, RELAYED), jnp.float32)
    return family.kernel(body, out_type=out, grid=(1,), grid_names=("programs",), consumers=1,
                         interpret=interpret, name="chained")(a, b, c)


LANDED = 64
"""A landed product's rows, depth and columns: one warpgroup's tile."""

READS = ("product", "summed", "read", "accumulated", "added_into")
"""The warpgroup accumulator's read paths (`mosaic_sm90.WgmmaIssuer`): a scoped product's value, a summed
one's, a walk's accumulator read in it, the walk's results, and a value added into an accumulator."""


def _negated(a, b, *, path: str, interpret: bool):
    """`-(a @ b)` by warpgroup MMA (`mosaic_sm90.FED`, one warpgroup), the accumulator read along `path`
    (`READS`) and negated before it is stored: a sign op on the accumulator's registers, which ptxas 12.9
    folded through their value before the MMA where it was an integer op, storing -0.0 (row 4346; the fork's
    float-sign-ops emits it as a float instruction)."""
    from dew.nn.kernels.templates import mosaic_sm90

    family = mosaic_sm90.FED
    shape = (LANDED, LANDED)

    def body(a_gmem, b_gmem, out_gmem):
        def scoped(a_smem, b_smem, out_smem):
            swizzling.copied(a_gmem, a_smem)
            swizzling.copied(b_gmem, b_smem)
            if not interpret:
                # Written through registers (the generic proxy), read by warpgroup MMA (the async one).
                plgpu.commit_smem()
            mosaic.synced(interpret)
            if path == "product":
                value = family.product(a_smem, b_smem, shape)
            elif path == "summed":
                value = family.summed(shape, lambda acc: family.products(acc, a_smem, b_smem))
            else:
                def step(index, carry, accumulators):
                    del index
                    (acc,) = accumulators
                    acc = family.products(acc, a_smem, b_smem, accumulate=False)
                    if path == "read":
                        # Stored where it is read, inside the walk.
                        plgpu.store(out_smem, -family.read(acc), optimized=False)
                    elif path == "added_into":
                        acc = family.added_into(acc, jnp.zeros(shape, jnp.float32), interpret)
                    return carry, (acc,)

                walk = (mosaic.Loop(step, 0, 1),)
                _, (value,) = family.accumulated(walk, (), (shape,), interpret)
            if path != "read":
                plgpu.store(out_smem, -value, optimized=False)
            if not interpret:
                # Written through registers, read by the store's copy.
                plgpu.commit_smem()
            mosaic.synced(interpret)
            stores.store_rows(out_smem, out_gmem, 0, 0, jnp.int32(LANDED), 0, LANDED,
                              copy=swizzling.coalesced, widest=resources.copied_rows(LANDED, jnp.float32))

        swizzle = resources.widest_swizzle(jnp.bfloat16, LANDED)
        pl.run_scoped(scoped, *(swizzling.staged(shape, jnp.bfloat16, swizzle) for _ in range(2)),
                      swizzling.plain(shape, jnp.float32), collective_axes="wg")

    out = jax.ShapeDtypeStruct((LANDED, LANDED), jnp.float32)
    return family.kernel(body, out_type=out, grid=(1,), grid_names=("programs",), consumers=1,
                         interpret=interpret, name="negated")(a, b)


def every_warpgroup_read_lands_its_products_on_an_h100(path: str):
    """On an H100: each read path of a warpgroup accumulator (`READS`) gives the products' value to a sign op
    on its registers, `-(a @ b)` exactly over small integers, where ptxas dropped the MMA and stored -0.0
    (row 4346); its program compiled in a tier's prepare (`ahead_arm`), its operands NumPy's."""
    rng = np.random.default_rng(7)
    a, b = (rng.integers(-2, 3, (LANDED, LANDED)).astype(jnp.bfloat16) for _ in range(2))
    expected = -(np.asarray(a, np.float32) @ np.asarray(b, np.float32))
    negated = ahead_arm(jax.jit(functools.partial(_negated, path=path, interpret=False)),
                        f"a warpgroup read's negated products: {path}")
    np.testing.assert_array_equal(np.asarray(negated(a, b)), expected)


@INTERPRETS
@pytest.mark.parametrize("path", READS)
def test_every_warpgroup_read_path_gives_the_products_value(path):
    """Interpreted (Mosaic GPU's interpreter, its waits no-ops): each warpgroup accumulator read path
    (`READS`) negates `a @ b` exactly; on an H100 the same kernel is the GPU flow's
    (`every_warpgroup_read_lands_its_products_on_an_h100`)."""
    keys = jax.random.split(jax.random.key(7))
    a, b = (jax.random.randint(key, (LANDED, LANDED), -2, 3).astype(jnp.bfloat16) for key in keys)
    np.testing.assert_array_equal(np.asarray(_negated(a, b, path=path, interpret=True)),
                                  -(np.asarray(a, np.float32) @ np.asarray(b, np.float32)))


def _chained_operands(rows: int):
    """Small integers whose chained products are exact in fp32, the first's (at most 128) in bf16: NumPy's."""
    rng = np.random.default_rng(5)
    return (rng.integers(-2, 3, (rows, RELAYED)).astype(jnp.bfloat16),
            *(rng.integers(-1, 2, (RELAYED, RELAYED)).astype(jnp.bfloat16) for _ in range(2)))


@pytest.mark.parametrize(("rows", "m_warps"), [(16, 1), (32, 2)])
def test_an_accumulator_feeds_the_next_product_on_any_warp_split(rows, m_warps):
    """Register values read as mma.sync's left operands, an accumulator's among them, on warps that split
    its columns (one or two along its rows): lowered for an A100 on any host, and on a GPU exactly `(a @ b)
    @ c` (each warp's columns relayed through the bound tile, where no register cast crosses warps)."""
    a, b, c = _chained_operands(rows)
    if here() != "cpu":
        f32 = [np.asarray(x, np.float32) for x in (a, b, c)]
        chained = ahead_arm(jax.jit(functools.partial(_chained, m_warps=m_warps, interpret=False)),
                            f"chained products: {rows} rows, {m_warps} warps")(a, b, c)
        np.testing.assert_array_equal(np.asarray(chained), (f32[0] @ f32[1]) @ f32[2])
        return
    shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (a, b, c)]
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm80")
    with generation.assumed("sm80"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(functools.partial(_chained, m_warps=m_warps, interpret=False)).trace(*shapes).lower(
            lowering_platforms=("cuda",))


def _bounded(a, b, c, bounds, *, family, interpret):
    """`c + a @ b` by the mma.sync `family`, bounded at each row bound `bounds` holds (`Issuer.bounded`, read
    at run time), then as `family` itself, in one trace."""
    rows, count, widest = a.shape[0], bounds.shape[0] + 1, resources.copied_rows(RELAYED, jnp.float32)

    def body(a_gmem, b_gmem, c_gmem, bounds_gmem, out_gmem):
        def scoped(a_smem, b_smem, *outs):
            swizzling.copied(a_gmem, a_smem)
            swizzling.copied(b_gmem, b_smem)
            mosaic.synced(interpret)
            acc = plgpu.load(c_gmem, layout=family.layout, optimized=False)
            issuers = [*(family.bounded(bounds_gmem[index]) for index in range(count - 1)), family]
            for issuer, out in zip(issuers, outs, strict=True):
                plgpu.store(out, issuer.products(acc, a_smem, b_smem), optimized=False)
            mosaic.synced(interpret)
            for index, out in enumerate(outs):
                stores.store_rows(out, out_gmem.at[index], 0, 0, jnp.int32(rows), 0, rows,
                                  copy=swizzling.coalesced, widest=widest)

        staged = (swizzling.staged(shape, jnp.bfloat16, resources.widest_swizzle(jnp.bfloat16, RELAYED))
                  for shape in ((rows, RELAYED), (RELAYED, RELAYED)))
        pl.run_scoped(scoped, *staged, *(swizzling.plain((rows, RELAYED), jnp.float32) for _ in range(count)))

    out = jax.ShapeDtypeStruct((count, rows, RELAYED), jnp.float32)
    return family.kernel(body, out_type=out, grid=(1,), grid_names=("programs",), consumers=1,
                         interpret=interpret, name="bounded")(a, b, c, bounds)


@pytest.mark.parametrize("m_warps", [1, 2, 4])
def test_mma_sync_products_within_a_row_bound_keep_the_accumulator_past_it(m_warps):
    """mma.sync's products by a family bounded at a tile's rows (`Issuer.bounded`: a ragged group's last tile)
    issue for the 16-row blocks before the bound alone, a nonzero accumulator, each row its own, kept in the
    rest: two run-time bounds in one trace (-1 keeping every row, 65 issuing all), and a constant one a family
    carries into one jitted helper at each bound (M-FS-56b; exact on a GPU, its program compiled in a tier's
    prepare, `ahead_arm`; rows 7091, 12108, 12109)."""
    family = mosaic_sm80.Mma(jnp.dtype(jnp.bfloat16), m_warps=m_warps)
    rng = np.random.default_rng(9)
    a, b = (rng.integers(-2, 3, shape).astype(jnp.bfloat16) for shape in ((64, RELAYED), (RELAYED, RELAYED)))
    c = np.broadcast_to(np.arange(1, 65, dtype=np.float32)[:, None] * 0.5, (64, RELAYED)).copy()
    full = c + np.asarray(a, np.float32) @ np.asarray(b, np.float32)
    starts, on_gpu = np.arange(64)[:, None] // 16 * 16, here() != "cpu"
    run = jax.jit(_bounded, static_argnames=("family", "interpret"))
    for bounds, constant in (((-1, 65), -1), ((0, 64), 65), ((1, 47), 16), ((16, 17), 47)):
        operands, bounded = (a, b, c, np.asarray(bounds, np.int32)), family.bounded(constant)
        if on_gpu:
            got = ahead_arm(jax.jit(functools.partial(_bounded, family=bounded, interpret=False)),
                            f"bounded products: {m_warps} warps at {constant}")(*operands)
        else:
            with generation.assumed("sm80"):
                got = run(*operands, family=bounded, interpret=INTERPRETED)
        expected = [np.where(starts < bound, full, c) for bound in (*bounds, constant)]
        np.testing.assert_array_equal(np.asarray(got), np.stack(expected))
    if on_gpu:
        return
    shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (a, b, c, np.zeros(2, np.int32))]
    mesh = generation.gpu_mesh(Mesh(np.array(jax.devices()[:1]), ("data",)), "sm80")
    with generation.assumed("sm80"), jax.sharding.use_abstract_mesh(mesh):
        run.trace(*shapes, family=family, interpret=False).lower(lowering_platforms=("cuda",))


CODES = (64, 64)
"""A quantized stage's rows and depth."""


def _converted(codes, scale, *, generation_name: str, interpret, scaled: bool = True):
    """`codes` (int8 or e4m3) converted into a bf16 tile through the generation's family's `converted`, each
    row scaled by `scale`, and copied out."""
    from dew.nn.kernels.templates import bundles

    family = bundles.of(generation_name, jnp.bfloat16)

    def body(codes_gmem, scale_gmem, out_gmem):
        def scoped(codes_smem, into):
            swizzling.copied(codes_gmem, codes_smem)
            mosaic.synced(interpret is not False)
            row_scale = scale_gmem[...] if interpret is not False else family.rows_of(scale_gmem)
            family.converted(codes_smem, into, interpret is not False,
                             row_scale=row_scale if scaled else None)
            mosaic.synced(interpret is not False)
            stores.store_rows(into, out_gmem, 0, 0, jnp.int32(CODES[0]), 0, CODES[0],
                              copy=swizzling.coalesced, widest=resources.copied_rows(CODES[1], jnp.bfloat16))

        # Scoped as the family scopes a program's allocations (`Family.program_scoped`: across its thread
        # axis where its kernel runs on one).
        family.program_scoped(lambda *refs, streams: scoped(*refs), swizzling.plain(CODES, codes.dtype),
                              swizzling.plain(CODES, jnp.bfloat16), streams={})

    out = jax.ShapeDtypeStruct(CODES, jnp.bfloat16)
    return family.kernel(body, out_type=out, grid=(1,), grid_names=("programs",), consumers=1,
                         interpret=interpret, name="converted")(codes, scale)


def _codes(dtype):
    """Codes of `dtype` and per-row power-of-two scales, so each converted value is exact in bf16: NumPy's."""
    rng = np.random.default_rng(6)
    codes = rng.integers(-8, 8, CODES).astype(dtype)
    # ldexp is exact where a GPU's exp2 is not (0.12499999 for 2 ** -3 on an A100).
    return codes, np.ldexp(np.ones((CODES[0],), np.float32), rng.integers(-3, 4, (CODES[0],)))


@pytest.mark.parametrize("scaled", [True, False], ids=["scaled", "unscaled"])
@pytest.mark.parametrize("dtype", [jnp.int8, jnp.float8_e4m3fn])
@pytest.mark.parametrize("generation_name", gate_generations())
def test_a_quantized_stage_converts_into_the_tile_its_products_read(generation_name, dtype, scaled):
    """Int8 and e4m3 codes in shared memory converted into a bf16 tile, each row scaled in fp32: exactly
    `codes * scale` interpreted (on a CPU) or on a GPU of the generation, and lowered for one on any host;
    FP8 codes refused where the device converts none (an A100). A host neither a CPU nor a GPU of the
    generation (a TPU, another GPU) runs none of it (row 9003). On a GPU its program is compiled in a tier's
    prepare (`here`, `ahead_arm`)."""
    if here() not in ("cpu", generation_name):
        pytest.skip(f"converts interpreted on a CPU or natively on a {generation_name} GPU, not on this "
                    f"{here()}")
    codes, scale = _codes(dtype)
    expected = np.asarray(codes, np.float32) * (np.asarray(scale)[:, None] if scaled else 1)
    with generation.assumed(generation_name):
        converts = mosaic.converts(dtype)
    if not converts:
        # An A100 converts no FP8 in registers: the conversion is refused, never assembled.
        with pytest.raises(ValueError), generation.assumed(generation_name):
            _converted(codes, scale, generation_name=generation_name, interpret=INTERPRETED)
        return
    if here() == "cpu":
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
    else:
        native = jax.jit(functools.partial(_converted, generation_name=generation_name, interpret=False,
                                           scaled=scaled))
        converted = ahead_arm(native, f"a {jnp.dtype(dtype).name} stage: scaled {scaled}")(codes, scale)
        np.testing.assert_array_equal(np.asarray(converted, np.float32), expected)


CHUNK_DEPTH = 64


def _chunks_products(x, w, *, interpret):
    """`x @ w` on tcgen05's chunks (`mosaic_sm100.CHUNKS`) three ways, each from one walk step's slots: both
    operands from shared memory (`product`); `x` from registers (`registers_product`: staged into the
    issuer's tile, `product_scratch`'s lefts); and `x` from registers into a sum of ones the walk carries
    in registers (`register_products`)."""
    from dew.nn.kernels.templates import mosaic_sm100
    family = mosaic_sm100.CHUNKS
    rows, columns = family.rows, w.shape[1]

    def body(x_gmem, w_gmem, ss_gmem, staged_gmem, carried_gmem):
        allocated = family.streams({'x': ((rows, CHUNK_DEPTH), jnp.bfloat16),
                                    'w': ((CHUNK_DEPTH, columns), jnp.bfloat16)}, 2, 1)

        def scope(*, streams):
            rings = family.opened(streams)
            views = {'x': (x_gmem, lambda j: (0, j)), 'w': (w_gmem, lambda j: (j, 0))}

            def step(index, slots, carry, accumulators):
                del index
                held, ones = family.tile_of(slots['x']), jnp.ones((rows, columns), jnp.float32)
                ss_gmem[...] = family.product(slots['x'], slots['w'], (rows, columns))
                staged_gmem[...] = family.registers_product(held, slots['w'])
                carried_gmem[...] = family.register_products(ones, (held,), slots['w'])
                return carry, accumulators

            family.scheduled(1, lambda wg: family.walk(rings, ((step, 0, 1),), None, (), interpret),
                              lambda: family.stream(rings, views, 0, 1))

        # The issuer's scratch, its staged left operand's tile among it (`Family.scoped`'s `products`).
        family.scoped(scope, streams=allocated, products=(1, (columns,), ((CHUNK_DEPTH, jnp.bfloat16),)))

    out = jax.ShapeDtypeStruct((rows, columns), jnp.float32)
    return family.kernel(body, out_type=(out, out, out), grid=(1,), grid_names=("programs",), consumers=1,
                         interpret=interpret, name="chunks_left")(x, w)


def _kept_sums(x, w, *, interpret):
    """`3 + sum_j x_j @ w_j` and `5 + sum_j 2 x_j @ w_j` over a walk of `x`'s and `w`'s depth blocks on
    tcgen05's chunks, two sums of one width each kept in its own accumulator of the issuer's tensor memory
    across the walk (`product_scratch`'s `sums`, `accumulated`) from its initial value, each step's register
    left operand's product added onto it in place."""
    from dew.nn.kernels.templates import mosaic_sm100
    family = mosaic_sm100.CHUNKS
    rows, columns = family.rows, w.shape[1]
    steps = x.shape[1] // CHUNK_DEPTH

    def body(x_gmem, w_gmem, out_gmem, twice_gmem):
        allocated = family.streams({'x': ((rows, CHUNK_DEPTH), jnp.bfloat16),
                                    'w': ((CHUNK_DEPTH, columns), jnp.bfloat16)}, 2, 1)

        def scope(*, streams):
            rings = family.opened(streams)
            views = {'x': (x_gmem, lambda j: (0, j)), 'w': (w_gmem, lambda j: (j, 0))}

            def step(index, slots, carry, accumulators):
                del index
                held = family.tile_of(slots['x'])
                return carry, (family.register_products(accumulators[0], (held,), slots['w']),
                               family.register_products(accumulators[1], (held * 2,), slots['w']))

            def walked(wg):
                del wg
                _, (sums, twice) = family.walk(rings, ((step, 0, steps),), None, ((rows, columns),) * 2,
                                               interpret, initial=(family.zeros((rows, columns)) + 3.0,
                                                                   family.zeros((rows, columns)) + 5.0))
                out_gmem[...] = sums
                twice_gmem[...] = twice

            family.scheduled(1, walked, lambda: family.stream(rings, views, 0, steps))

        family.scoped(scope, streams=allocated,
                      products=(1, (), ((CHUNK_DEPTH, jnp.bfloat16),), (), (columns, columns)))

    out = jax.ShapeDtypeStruct((rows, columns), jnp.float32)
    return family.kernel(body, out_type=(out, out), grid=(1,), grid_names=("programs",), consumers=1,
                         interpret=interpret, name="chunks_kept")(x, w)


def _stepped_once(family, rings, step, outs, shape, interpret):
    """A consumer's walk of `rings` for one `step` from a fresh accumulator of `shape` an output of `outs`,
    each sum stored into its output."""
    def walked(wg):
        del wg
        found, _ = family.walk(rings, ((step, 0, 1),), (family.zeros(shape),) * len(outs), (), interpret)
        for out_gmem, value in zip(outs, found, strict=True):
            out_gmem[...] = value

    return walked


def _summed_together(x, w, *, interpret, awaiting: bool = False):
    """`x_0 @ w_0`, `x_1 @ w_1` and `x_0 @ w_1` (`x`'s and `w`'s two depth blocks, each a stream of its own)
    on tcgen05's chunks from shared memory, three groups of one width summed together in one step
    (`summed_together`): each issued into its own accumulator of the issuer's together scratch, one commit and
    one wait for all three, each read where it is stored. `awaiting`, the last group adds a value into its
    accumulator, which awaits its own products (`added_into`)."""
    from dew.nn.kernels.templates import mosaic_sm100
    family = mosaic_sm100.CHUNKS
    rows, columns = family.rows, w.shape[1]

    def body(x_gmem, w_gmem, *outs):
        blocks = dict.fromkeys(('x0', 'x1'), ((rows, CHUNK_DEPTH), jnp.bfloat16))
        blocks |= dict.fromkeys(('w0', 'w1'), ((CHUNK_DEPTH, columns), jnp.bfloat16))

        def scope(*, streams):
            rings = family.opened(streams)
            views = {f'x{k}': (x_gmem, lambda j, k=k: (0, k)) for k in range(2)}
            views |= {f'w{k}': (w_gmem, lambda j, k=k: (k, 0)) for k in range(2)}

            def step(index, slots, carry, accumulators):
                del index, carry
                # Every slot awaited first, so the step's only waits after its first product are the groups'.
                slots.acquire_all()
                pairs = (('x0', 'w0'), ('x1', 'w1'), ('x0', 'w1'))
                issues = [lambda acc, x=x, w=w: family.products(acc, slots[x], slots[w]) for x, w in pairs]
                if awaiting:
                    issues[-1] = lambda acc: family.added_into(family.products(acc, slots['x0'], slots['w1']),
                                                               jnp.zeros((rows, columns), jnp.float32),
                                                               interpret=False)
                readers = family.summed_together(((rows, columns),) * 3, tuple(issues))
                return tuple(read() for read in readers), accumulators

            family.scheduled(1, _stepped_once(family, rings, step, outs, (rows, columns), interpret),
                             lambda: family.stream(rings, views, 0, 1))

        family.scoped(scope, streams=family.streams(blocks, 2, 1),
                      products=(1, (), (), (), (), (columns,) * 3))

    out = jax.ShapeDtypeStruct((rows, columns), jnp.float32)
    return family.kernel(body, out_type=(out, out, out), grid=(1,), grid_names=("programs",), consumers=1,
                         interpret=interpret, name="chunks_together")(x, w)


@INTERPRETS
def test_groups_summed_together_in_tcgen05s_tensor_memory_are_each_its_own_sum():
    """Interpreted: on tcgen05's chunks, three groups of one width summed together (`summed_together`: each
    its own accumulator, one commit and one wait for all) are exactly each group's product, none another's."""
    keys = jax.random.split(jax.random.key(5))
    x = jax.random.randint(keys[0], (64, 2 * CHUNK_DEPTH), -4, 4).astype(jnp.bfloat16)
    w = jax.random.randint(keys[1], (2 * CHUNK_DEPTH, 128), -4, 4).astype(jnp.bfloat16)
    first, second, crossed = (np.asarray(out) for out in _summed_together(x, w, interpret=INTERPRETED))
    x32, w32 = np.asarray(x, np.float32), np.asarray(w, np.float32)
    np.testing.assert_array_equal(first, x32[:, :CHUNK_DEPTH] @ w32[:CHUNK_DEPTH])
    np.testing.assert_array_equal(second, x32[:, CHUNK_DEPTH:] @ w32[CHUNK_DEPTH:])
    np.testing.assert_array_equal(crossed, x32[:, :CHUNK_DEPTH] @ w32[CHUNK_DEPTH:])


def test_groups_summed_together_on_tcgen05_issue_every_product_before_one_commit_and_wait():
    """Traced for a B200 (`summed_together` on tcgen05's chunks): the three groups' products are all issued
    before the role commits its tensor-core barrier once and waits on it once, no commit or wait between
    them, so no group's wait stands between another's issue and its own."""
    x = jax.ShapeDtypeStruct((64, 2 * CHUNK_DEPTH), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((2 * CHUNK_DEPTH, 128), jnp.bfloat16)
    with generation.assumed("sm100"):
        traced = jax.make_jaxpr(lambda x, w: _summed_together(x, w, interpret=False))(x, w)
    order = [equation.primitive.name for equation in equations(traced.jaxpr)
             if equation.primitive.name in ("tcgen05_mma", "tcgen05_commit_arrive", "barrier_wait")]
    first = order.index("tcgen05_mma")
    issued = [index for index, name in enumerate(order) if name == "tcgen05_mma"]
    assert len(issued) == 3 and issued == list(range(first, first + 3)), order
    assert order[first + 3:first + 5] == ["tcgen05_commit_arrive", "barrier_wait"], order


def _staged_three(x, *, interpret: bool, batched: bool, aliased: bool = False):
    """`x` (16 x 64), twice and thrice it, staged in bf16 by an mma.sync issuer whose warps split a tile's
    columns (its stores bracketed by its own barriers), in one batch (`Family.staged_all`) or a store apiece
    (`Family.staged`), and each read back; `aliased`, its first buffer given twice."""
    family = bundles.named("mma.sync", jnp.bfloat16, None).specialized(16, 64)

    def body(x_gmem, *outs):
        def scoped(*buffers):
            value = x_gmem[...] if interpret else family.zeros((16, 64))
            targets = (buffers[0], buffers[0], buffers[2]) if aliased else buffers
            pairs = [(value * times, buffer) for times, buffer in zip((1, 2, 3), targets, strict=True)]
            if batched:
                family.staged_all(pairs)
            else:
                for each, buffer in pairs:
                    family.staged(each, buffer)
            if interpret:
                for out, buffer in zip(outs, buffers, strict=True):
                    out[...] = family.tile_of(buffer).astype(jnp.float32)

        pl.run_scoped(scoped, *(plgpu.SMEM((16, 64), jnp.bfloat16) for _ in range(3)))

    out = jax.ShapeDtypeStruct((16, 64), jnp.float32)
    return family.kernel(body, out_type=(out,) * 3, grid=(1,), grid_names=("programs",), consumers=1,
                         interpret=interpret, name="staged")(x)


@INTERPRETS
def test_a_staged_batch_holds_each_value_in_its_buffer():
    """Interpreted: three values staged in one batch (`Family.staged_all`) read back exactly from their own
    buffers."""
    x = jax.random.randint(jax.random.key(7), (16, 64), -4, 4).astype(jnp.float32)
    got = _staged_three(x, interpret=INTERPRETED, batched=True)
    for times, value in zip((1, 2, 3), got, strict=True):
        np.testing.assert_array_equal(np.asarray(value), times * np.asarray(x))


def _barriers(call) -> int:
    """The barriers (`gpu.barrier`, `nvvm.barrier`) in the typed module `call` lowers to for an A100."""
    import jax.experimental.mosaic.gpu as mgpu
    from kernel_compile import lowered_for

    from dew.nn.kernels.lowered import kernel_calls

    lowered = lowered_for(call, (jax.ShapeDtypeStruct((16, 64), jnp.float32),), "sm80")
    (kernel,) = [each for each in kernel_calls(lowered) if each.kind == "mosaic_gpu"]
    module = mgpu.deserialized(kernel.ir)
    pending, found = [module.operation], 0
    with module.context:
        while pending:
            for region in pending.pop().regions:
                for block in region.blocks:
                    for inner in block.operations:
                        found += inner.operation.name in ("gpu.barrier", "nvvm.barrier")
                        pending.append(inner.operation)
    return found


@COMPILES_ON_CPU
def test_an_mma_sync_staged_batch_takes_one_barrier_pair_where_its_stores_took_one_each():
    """Lowered for an A100: three tiles staged by an mma.sync issuer whose warps split a tile's columns take
    one pair of warpgroup barriers in one batch (`Family.staged_all`), four fewer than a store apiece's pair
    each (`staged`; a Kimi intra step's 20 a trip, FS-INTRA-BAR1)."""
    apiece, batch = (_barriers(functools.partial(_staged_three, interpret=False, batched=batched))
                     for batched in (False, True))
    assert batch >= 2 and apiece - batch == 4, (apiece, batch)


def test_a_staged_batch_refuses_a_buffer_twice():
    """A batch given one shared-memory buffer twice is refused as it traces, before any store: its stores go
    unbracketed against each other."""
    with generation.assumed("sm80"), pytest.raises(ValueError):
        jax.make_jaxpr(functools.partial(_staged_three, interpret=False, batched=True, aliased=True))(
            jax.ShapeDtypeStruct((16, 64), jnp.float32))


def _staged_together(x, w, *, interpret, shared: bool):
    """`x_0 @ w_0` and `2 x_0 @ w_0` on tcgen05's chunks, summed together by one prepared-operand issuer
    (`operands.Issuer.summed_together`): each group's left operand a register value its product name stages
    into a shared tile of its buffers, one tile for both where `shared`, `w_0` the stream's slot; each read
    where it is stored, in the program's together scratch alone."""
    from dew.nn.kernels.templates import mosaic_sm100, operands
    family = mosaic_sm100.CHUNKS
    rows, columns = family.rows, w.shape[1]
    call = _Call(family, interpret)

    def body(x_gmem, w_gmem, *outs):
        blocks = {'x': ((rows, CHUNK_DEPTH), jnp.bfloat16), 'w': ((CHUNK_DEPTH, columns), jnp.bfloat16)}

        def scope(staging, other, *, streams):
            rings = family.opened(streams)
            views = {'x': (x_gmem, lambda j: (0, 0)), 'w': (w_gmem, lambda j: (0, 0))}

            def step(index, slots, carry, accumulators):
                del index, carry
                slots.acquire_all()
                held, right = family.tile_of(slots['x']), operands.Parts((slots['w'],), jnp.bfloat16)
                issuer = operands.Issuer(call, {"left": (staging,), "twice": (staging if shared else other,)})
                readers = issuer.summed_together([[("left", held, right)], [("twice", held * 2, right)]])
                return tuple(read() for read in readers), accumulators

            family.scheduled(1, _stepped_once(family, rings, step, outs, (rows, columns), interpret),
                             lambda: family.stream(rings, views, 0, 1))

        tile = plgpu.SMEM((rows, CHUNK_DEPTH), jnp.bfloat16)
        family.scoped(scope, tile, tile, streams=family.streams(blocks, 2, 1),
                      products=(1, (), (), (), (), (columns,) * 2))

    out = jax.ShapeDtypeStruct((rows, columns), jnp.float32)
    return family.kernel(body, out_type=(out, out), grid=(1,), grid_names=("programs",), consumers=1,
                         interpret=interpret, name="staged_together")(x, w)


@pytest.mark.parametrize("shared", [True, False], ids=["one tile", "a tile each"])
def test_groups_summed_together_stage_no_operand_into_a_tile_a_product_in_flight_reads(shared):
    """Traced for a B200: two groups summed by one issuer (`operands.Issuer.summed_together`), each left
    operand a register value staged into a shared tile of its product's buffers, one tile for both or a tile
    each, in the program's together scratch alone: no staging into shared memory stands between a product's
    issue and the wait that retires it. A tcgen05 product reads its operands after it issues, until its
    commit's barrier, so a group staged while another's product is in flight overwrites that product's
    operand under it."""
    x = jax.ShapeDtypeStruct((64, CHUNK_DEPTH), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((CHUNK_DEPTH, 128), jnp.bfloat16)
    with generation.assumed("sm100"):
        traced = jax.make_jaxpr(lambda x, w: _staged_together(x, w, interpret=False, shared=shared))(x, w)
    order = [equation.primitive.name if equation.primitive.name != "swap" else "staged"
             for equation in equations(traced.jaxpr)
             if equation.primitive.name in ("tcgen05_mma", "barrier_wait")
             or (equation.primitive.name == "swap"
                 and getattr(equation.invars[0].aval, "memory_space", None) == plgpu.SMEM)]
    assert order.count("tcgen05_mma") == 2 and order.count("staged") >= 2, order
    in_flight = False
    for event in order:
        assert not (in_flight and event == "staged"), order
        in_flight = (in_flight or event == "tcgen05_mma") and event != "barrier_wait"


@pytest.mark.gpu(reason="a tcgen05 product's late operand read is the device's alone")
@pytest.mark.generations("sm100")
@pytest.mark.skipif(generation.device_generation() != "sm100",
                    reason="a tcgen05 product's late operand read is a B200's alone")
@pytest.mark.parametrize("shared", [True, False], ids=["one tile", "a tile each"])
def test_groups_summed_together_from_staged_operands_are_each_its_own_sum(shared):
    """On a B200: two groups summed by one issuer on tcgen05's chunks (`operands.Issuer.summed_together`),
    each left operand a register value staged into a shared tile, one tile for both or a tile each, its
    operands distinct small integers: each group's sum is exactly its own, `x @ w` and `2 x @ w`. A tcgen05
    product reads its operands after it issues, so a group staged while another's product still read the
    tile gave that product the second group's operand (row 6527: the order-only test held no sum). Mosaic
    GPU's interpreter runs a product as it issues, so only the device's run can tell (the fix undone, the
    interpreted sums stay exact)."""
    x = jax.random.randint(jax.random.key(11), (64, CHUNK_DEPTH), -4, 4).astype(jnp.bfloat16)
    w = jax.random.randint(jax.random.key(12), (CHUNK_DEPTH, 128), -4, 4).astype(jnp.bfloat16)
    with generation.assumed("sm100"):
        first, twice = (np.asarray(out) for out in _staged_together(x, w, interpret=False, shared=shared))
    product = np.asarray(x, np.float32) @ np.asarray(w, np.float32)
    np.testing.assert_array_equal(first, product)
    np.testing.assert_array_equal(twice, 2 * product)


def test_a_group_summed_together_that_awaits_its_own_products_is_refused():
    """A group summed together on tcgen05 whose issue awaits its own products (a value added into its
    accumulator, `added_into`) is refused as it traces: its wait would leave the groups' one commit behind
    nothing pending (row 5671)."""
    x = jax.ShapeDtypeStruct((64, 2 * CHUNK_DEPTH), jnp.bfloat16)
    w = jax.ShapeDtypeStruct((2 * CHUNK_DEPTH, 128), jnp.bfloat16)
    with generation.assumed("sm100"), pytest.raises(ValueError):
        jax.make_jaxpr(lambda x, w: _summed_together(x, w, interpret=False, awaiting=True))(x, w)


@INTERPRETS
def test_a_walks_sums_kept_in_tcgen05s_tensor_memory_are_its_products_onto_its_initial_value():
    """Interpreted: on tcgen05's chunks, a walk's two sums of one width kept in tensor memory across three
    steps (`Issuer.accumulated`), each in its own accumulator, each step's register product added onto it in
    place and none read before the walk's end, are exactly each one's initial value plus its products."""
    keys = jax.random.split(jax.random.key(4))
    x = jax.random.randint(keys[0], (64, 3 * CHUNK_DEPTH), -4, 4).astype(jnp.bfloat16)
    w = jax.random.randint(keys[1], (3 * CHUNK_DEPTH, 128), -4, 4).astype(jnp.bfloat16)
    sums, twice = (np.asarray(out) for out in _kept_sums(x, w, interpret=INTERPRETED))
    product_ = np.asarray(x, np.float32) @ np.asarray(w, np.float32)
    np.testing.assert_array_equal(sums, 3 + product_)
    np.testing.assert_array_equal(twice, 5 + 2 * product_)


@INTERPRETS
def test_a_register_left_product_on_tcgen05s_chunks_is_its_shared_memory_product():
    """Interpreted: on tcgen05's chunks, whose product takes no register left operand
    (`takes_register_left`), a register value staged into the issuer's tile and multiplied from there
    (`register_products`), alone and into a sum the walk carries in registers, gives exactly the product of
    the same operands from shared memory, and the carried sum plus it."""
    keys = jax.random.split(jax.random.key(3))
    x = jax.random.randint(keys[0], (64, CHUNK_DEPTH), -4, 4).astype(jnp.bfloat16)
    w = jax.random.randint(keys[1], (CHUNK_DEPTH, 128), -4, 4).astype(jnp.bfloat16)
    ss, staged, carried = (np.asarray(out) for out in _chunks_products(x, w, interpret=INTERPRETED))
    expected = np.asarray(x, np.float32) @ np.asarray(w, np.float32)
    np.testing.assert_array_equal(ss, expected)
    np.testing.assert_array_equal(staged, expected)
    np.testing.assert_array_equal(carried, expected + 1)


@pytest.mark.parametrize("shift", [0, 5, -7])
def test_a_row_shift_takes_each_row_from_its_groups_row_shift_up(shift):
    """A tile's rows shifted within groups of 16 (`Issuer.row_shifted`, interpreted at an H100's facts):
    each row whose source lies in its group holds that row's values, the rest masked, on the host's CPU
    (`on_the_host`)."""
    from dew.nn.kernels.templates import bundles

    tile = np.arange(64 * 32, dtype=np.float32).reshape(64, 32)
    family = bundles.of("sm90", jnp.bfloat16)
    with on_the_host():
        out = np.asarray(family.row_shifted(tile, np.int32(shift), period=16, interpret=True))
    rows = np.arange(64)
    kept = (rows % 16 - shift >= 0) & (rows % 16 - shift < 16)
    np.testing.assert_array_equal(out[kept], tile[rows[kept] - shift])


def test_a_product_issued_within_a_staged_block_is_refused():
    """A warpgroup MMA's or tcgen05's product reads shared memory asynchronously, so one issued while a
    consumer's staged tiles are still unpublished (`Family.staged_together`) is refused, not left to read them
    unfenced."""
    from dew.nn.kernels.templates import mosaic
    from dew.nn.kernels.templates.mosaic_sm90 import WGMMA

    with pytest.raises(ValueError), WGMMA.staged_together():
        mosaic.committed()


@pytest.mark.gpu(reason="compiles a launch under each lowering semantics and runs it on this GPU")
@pytest.mark.generations("sm90", "sm100")
@pytest.mark.parametrize("semantics", [plgpu.LoweringSemantics.Lane, plgpu.LoweringSemantics.Warpgroup],
                         ids=["lane", "warpgroup"])
def test_a_launch_runs_a_flags_branch_and_its_negations_branch_each_alone(semantics):
    """A launch's `pl.when(flag)` and `pl.when(~flag)` each run where their condition holds and there alone,
    under either lowering semantics: post102's warpgroup lowering negated a boolean through ~int(x), -1 or -2,
    which its conversion to a boolean read as true either way, so a persistent kernel ran its inactive tiles'
    branch beside every active tile's (train 967, seen only as 8 bytes spilled)."""
    def body(out_gmem):
        program = jax.lax.axis_index("program")

        @functools.partial(pl.run_scoped, staged=plgpu.SMEM((128,), jnp.int32))
        def _(staged):
            active = program > 0

            @pl.when(active)
            def _():
                staged[...] = jnp.full(staged.shape, 1, jnp.int32)

            @pl.when(~active)
            def _():
                staged[...] = jnp.full(staged.shape, 2, jnp.int32)

            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(staged, out_gmem.at[pl.ds(program * 128, 128)])
            plgpu.wait_smem_to_gmem(0)

    launch = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((256,), jnp.int32), grid=(2,),
                          grid_names=("program",),
                          compiler_params=plgpu.CompilerParams(lowering_semantics=semantics))
    out = ahead_arm(jax.jit(launch), f"a flag's branches: {semantics.name}")()
    np.testing.assert_array_equal(np.asarray(out), np.repeat(np.asarray([2, 1], np.int32), 128))


def test_an_mma_sync_walks_shared_memory_is_what_its_pipeline_scopes_for_its_copies():
    """mma.sync's walk sizes its shared memory as `emit_pipeline` scopes it for the cp.async copies every
    device the family runs on makes (`CpAsync.lifetimes`, read from JAX's own builder, row 6135): a slot of
    each block a stage and no barrier, whatever GPU is assumed (an sm120, whose copies would be TMA's and
    take a barrier a stage, runs no mma.sync on cp.async: `bundles.legal`); a walk of fewer steps than stages
    as many slots as steps, also planned inside a jit's trace over a mesh of its own; two consumers running
    it together, the two barriers they meet at, as the builder scopes them for its thread axis. That the plan
    is the kernel's own allocation the compile gate checks of every launch (`kernel_compile.mosaic_failures`:
    its planned shared bytes against those the compiled kernel asks for)."""
    family = bundles.named("mma.sync", jnp.bfloat16, None)
    streams = family.streams({"lhs": ((64, 32), jnp.bfloat16), "rhs": ((32, 64), jnp.bfloat16)}, 3, 1)

    def walked(steps: int | None = None) -> tuple[list, list]:
        (), (walk,) = family.lifetimes(streams, steps)
        barriers = [int(np.prod(leaf.num_barriers)) for leaf in walk if isinstance(leaf, plgpu.Barrier)]
        return [leaf.shape for leaf in walk if not isinstance(leaf, plgpu.Barrier)], barriers

    for target in ("sm80", "sm120"):
        with generation.assumed(target):
            assert walked() == ([(3, 64, 32), (3, 32, 64)], []), target
    streams = family.streams({"lhs": ((64, 32), jnp.bfloat16), "rhs": ((32, 64), jnp.bfloat16)}, 3, 2)
    assert walked() == ([(3, 64, 32), (3, 32, 64)], [2])
    streams = family.streams({"lhs": ((64, 32), jnp.bfloat16), "rhs": ((32, 64), jnp.bfloat16)}, 3, 1)
    planned = []
    mesh = Mesh(np.array(jax.devices("cpu")[:1]), ("data",))
    with jax.set_mesh(mesh):
        jax.make_jaxpr(lambda: planned.append(walked(2)))()
    assert planned == [([(2, 64, 32), (2, 32, 64)], [])]


@pytest.mark.gpu(reason="a warp's lanes hold values of their own only on the device")
@pytest.mark.parametrize("read", ["uniform", "own"])
def test_a_uniform_read_gives_each_lane_its_warps_lane_zero_value(read):
    """`across.uniform` of a scalar each lane of a warpgroup holds its own of (its thread's index plus 7, as
    a word each lane reads may differ) gives every lane its warp's lane 0's: one shuffle from a constant
    lane. Read as each lane holds it, each lane's own (the control)."""
    from jax.experimental.mosaic.gpu import fragmented_array, utils
    from jax.extend.mlir import ir
    from jax.extend.mlir.dialects import arith, vector

    scalar, lanes = plgpu.Layout.WG_SPLAT(()), plgpu.Layout.WG_STRIDED((128,), vec_size=1)

    def own(ctx):
        del ctx
        i32 = ir.IntegerType.get_signless(32)
        thread = arith.remui(utils.thread_idx(), utils.c(128, i32))
        value = arith.addi(thread, utils.c(7, i32))
        return fragmented_array.FragmentedArray.splat(value, (), is_signed=True)

    def spread(ctx, value):
        # Each thread's register at its own index.
        del ctx
        i32 = ir.IntegerType.get_signless(32)
        held = vector.broadcast(ir.VectorType.get((1,), i32), value.registers.item())
        return fragmented_array.FragmentedArray(
            _registers=np.asarray([held], dtype=object),
            _layout=fragmented_array.WGStridedFragLayout((128,), 1), _is_signed=True)

    def body(out_gmem):
        held = plgpu.inline_mgpu(
            return_type=plgpu.ShapeDtypeStruct((), jnp.dtype(jnp.int32), layout=scalar))(own)()
        read_value = across.uniform(held) if read == "uniform" else held
        each = plgpu.ShapeDtypeStruct((128,), jnp.dtype(jnp.int32), layout=lanes)
        out_gmem[...] = plgpu.inline_mgpu(arg_types=(scalar,), return_type=each)(spread)(read_value)

    kernel = jax.jit(plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((128,), jnp.int32), grid=(1,),
                                  grid_names=("programs",)))
    out = ahead_arm(kernel, f"uniform read {read}")()
    threads = np.arange(128, dtype=np.int32)
    expected = (threads // 32 * 32 if read == "uniform" else threads) + 7
    np.testing.assert_array_equal(np.asarray(out), expected)


ONCE_BUNDLES = [pytest.param("wgmma", marks=INTERPRETS), pytest.param("fed", marks=INTERPRETS), "mxu"]
"""The ways a family walks a once stream: held beside a ring, held by a consumer-fed bundle's loader, and a
TPU core's fold, its kernels interpreted on any host."""


def _walked_once(x, w, *, family, interpret, all_once=False, prologue=False):
    """`x[:, :PERSISTED_DEPTH] @` the sum of `w`'s `PERSISTED_STEPS` step blocks through one plain walk
    (`Transport.run`) on `family`'s roles: `x`'s block a once stream (`Walk.once`), its view's at step zero
    though its index map moves with the step, loaded once outside the ring and read by every step beside
    `w`'s ringed blocks, the walk's sums read out after its last step (`Walk.read`). `all_once`, `w`'s first
    block a once stream too: every step reads the two once blocks, and the walk rotates no ring. `prologue`,
    the walk's prologue reads `w`'s first row from global memory, each step adding it to the walk's carry,
    which the read adds to every row: the sums plus `PERSISTED_STEPS` times that row."""
    rows, columns = family.rows, w.shape[1]
    family = family.specialized(rows, columns)
    shape = (rows, columns)

    from jax.experimental.pallas import tpu as pltpu

    from dew.nn.kernels.templates import mosaic_tpu

    # A TPU core's program writes an output in HBM by a copy from VMEM, a GPU's in place.
    staging = [pltpu.VMEM(shape, jnp.float32)] if isinstance(family, mosaic_tpu.VmemRings) else []

    def body(x_gmem, w_gmem, o_gmem):
        def scope(*staged, streams):
            def read(carry, sums):
                out = staged[0] if staged else o_gmem
                for first, (value,) in family.chunks(sums):
                    if prologue:
                        value = value + carry[None, first:first + value.shape[1]]
                    out[:, pl.ds(first, value.shape[1])] = value
                if staged:
                    pltpu.sync_copy(out, o_gmem)

            once = {'x': (x_gmem, lambda j: (0, j))}
            views = {'w': (w_gmem, lambda j: (j, 0))}
            if all_once:
                once, views = {**once, **views}, {}
            step = _multiplying(family, 'x', 'w', interpret)
            if prologue:
                def step(index, slots, carry, accumulators, *, loaded, multiplied=step):
                    carry, accumulators = multiplied(index, slots, carry, accumulators)
                    return carry + loaded.astype(jnp.float32), accumulators

            walk = mosaic.Walk(views, step, steps=PERSISTED_STEPS, sums=(shape,), read=read, once=once,
                               carry=jnp.zeros((columns,), jnp.float32) if prologue else (),
                               prologue=(lambda: w_gmem[0]) if prologue else None)
            family.run(streams["walked"], walk, interpret, once=streams["once"])

        ringed, held = {'w': ((PERSISTED_DEPTH, columns), w.dtype)}, {'x': ((rows, PERSISTED_DEPTH), x.dtype)}
        if all_once:
            ringed, held = {}, {**held, **ringed}
        family.scoped(scope, *staging, streams={"walked": family.streams(ringed, 2, 1),
                                                "once": family.streams(held, 1, 1)})

    out = jax.ShapeDtypeStruct(shape, jnp.float32)
    return family.kernel(body, out_type=out, grid=(1,), grid_names=("programs",), consumers=1,
                         interpret=interpret, name="walked_once")(x, w)


@pytest.mark.parametrize("bundle", ["wgmma", "fed"])
def test_a_walks_prologue_loads_before_it_waits_on_its_once_streams(bundle):
    """Traced for an H100: a walk with a prologue and a once stream (`Walk.prologue`, `Walk.once`) on
    warpgroup MMA's roles and its consumer-fed bundle's: the consumer's prologue reads global memory before
    it first waits on a barrier, as a prologue's loads are in flight beside the streams' (row 6657); its once
    streams' acquire waits after them."""
    from dew.nn.kernels.templates import mosaic_sm90

    family = {"wgmma": mosaic_sm90.WGMMA, "fed": mosaic_sm90.FED}[bundle]
    x, w = _persisted_operands(family.rows)
    x = x[:family.rows]
    with generation.assumed("sm90"):
        traced = jax.make_jaxpr(lambda x, w: _walked_once(x, w, family=family, interpret=False,
                                                          prologue=True))(x, w)
    order = [equation.primitive.name for equation in equations(traced.jaxpr)
             if equation.primitive.name == "barrier_wait"
             or (equation.primitive.name == "get"
                 and getattr(equation.invars[0].aval, "memory_space", None) == plgpu.GMEM)]
    assert "get" in order and order.index("get") < order.index("barrier_wait"), order


@pytest.mark.parametrize("bundle", ONCE_BUNDLES)
def test_a_walks_once_stream_loads_once_and_every_step_reads_it(bundle):
    """Interpreted: a plain walk whose left operand is a once stream (`Walk.once`), its index map moving with
    the step, on each way a family holds one: warpgroup MMA's producer loading it beside the ring, its
    consumer-fed bundle's loading consumer before the walk (`fed.ConsumerFed.taken_once`), and a TPU core
    folding it into its walked streams (`folded`); every step reads its step-zero block with its own ringed
    block, so the walk sums that block times every step's: exactly, small integers in fp32. (tcgen05's plain
    walks' accumulators are their persist's or their launch's tensor memory, which this one scopes none of;
    mma.sync's pipeline refuses a once stream, `test_a_cp_async_walk_refuses_a_once_stream`.) On the host's
    CPU (`on_the_host`)."""
    family, target, interpret = _once_bundle(bundle)
    x, w = _persisted_operands(family.rows)
    x = x[:family.rows]
    with generation.assumed(target), on_the_host():
        got = np.asarray(_walked_once(x, w, family=family, interpret=interpret))
    blocks = np.asarray(w, np.float32).reshape(PERSISTED_STEPS, PERSISTED_DEPTH, -1).sum(axis=0)
    np.testing.assert_array_equal(got, np.asarray(x, np.float32)[:, :PERSISTED_DEPTH] @ blocks)


@pytest.mark.parametrize("bundle", ONCE_BUNDLES[:2])
def test_a_walks_prologue_value_reaches_every_step_beside_its_once_stream(bundle):
    """Interpreted: the once-stream walk with a prologue (`Walk.prologue`) on a GPU's two ways of holding
    one, the prologue reading `w`'s first row from global memory before the walk waits on its once streams
    (row 6657; a TPU core's walk reads HBM only by a copy, its prologue none): every step adds that row to
    the walk's carry, which its read adds to every row of the sums, so the output is the once-stream walk's
    plus `PERSISTED_STEPS` times that row, exactly in fp32 (base review 584d4776f F1: a prologue whose value
    no step read held only the emitted order)."""
    family, target, interpret = _once_bundle(bundle)
    x, w = _persisted_operands(family.rows)
    x = x[:family.rows]
    with generation.assumed(target):
        got = np.asarray(_walked_once(x, w, family=family, interpret=interpret, prologue=True))
    blocks = np.asarray(w, np.float32).reshape(PERSISTED_STEPS, PERSISTED_DEPTH, -1).sum(axis=0)
    expected = np.asarray(x, np.float32)[:, :PERSISTED_DEPTH] @ blocks
    np.testing.assert_array_equal(got, expected + PERSISTED_STEPS * np.asarray(w, np.float32)[0])


@pytest.mark.parametrize("bundle", ONCE_BUNDLES)
def test_a_walk_whose_every_operand_is_a_once_stream_reads_them_at_each_step(bundle):
    """Interpreted: a plain walk of two operands both once streams, no ring to rotate (a consumer-fed
    bundle's loading consumer one step deep, as a ring family's walk is): every step reads the two step-zero
    blocks, so the walk sums their product once a step, on the host's CPU (`on_the_host`)."""
    family, target, interpret = _once_bundle(bundle)
    x, w = _persisted_operands(family.rows)
    x = x[:family.rows]
    with generation.assumed(target), on_the_host():
        got = np.asarray(_walked_once(x, w, family=family, interpret=interpret, all_once=True))
    first = np.asarray(x, np.float32)[:, :PERSISTED_DEPTH] @ np.asarray(w, np.float32)[:PERSISTED_DEPTH]
    np.testing.assert_array_equal(got, PERSISTED_STEPS * first)


def _once_bundle(bundle: str):
    """`bundle`'s family, the target it is assumed on, and how its kernel is interpreted: a GPU family's by
    Mosaic GPU's interpreter, a TPU core's by Pallas's."""
    from dew.nn.kernels.templates import mosaic_sm90, mosaic_tpu
    if bundle == "mxu":
        return mosaic_tpu.MXU, "v5e", True
    return {"wgmma": mosaic_sm90.WGMMA, "fed": mosaic_sm90.FED}[bundle], "sm90", INTERPRETED


@INTERPRETS
def test_a_cp_async_walk_refuses_a_once_stream():
    """mma.sync's walk refuses a once stream (`CpAsync.folded`): its pipeline would allocate the block a
    slot per stage in the walk's scope, where the plan (`lifetimes`) counts one, so a ring depth the plan
    accepts could overflow an A100's shared memory."""
    family = mosaic_sm80.Mma(jnp.dtype(jnp.bfloat16))
    x, w = _persisted_operands(family.rows)
    with generation.assumed("sm80"), pytest.raises(NotImplementedError):
        _walked_once(x[:family.rows], w, family=family, interpret=INTERPRETED)


def test_a_values_parts_sum_back_to_it_exactly_fp32s_largest_among_them():
    """An fp32 value's three bf16 parts (`product.formed`) sum back to it exactly (`product.whole`), the
    largest finite values among them: fp32's largest, and bf16's largest past by `2^120 - 2^111`, whose
    leading two parts summed first reach `2^128`, past fp32's largest, to infinity (rows 6842, 6866)."""
    largest = float(np.finfo(np.float32).max)
    past = 2.0**128 - 2.0**111
    x = np.asarray([largest, past, -largest, -past, 1.0 + 2.0**-23], np.float32)
    got = ahead_arm(jax.jit(lambda v: product.whole(product.formed(v, 3))), "values parts exact")(x)
    np.testing.assert_array_equal(np.asarray(got), x)


STAGED = (8, 32, 3)
"""A staged output's tile: its rows, a chunk buffer's columns, and its chunks (two buffers)."""


def _staged_launch(waits: tuple[int, ...], stored: tuple[int, ...] = (STAGED[0],) * 2,
                   staging: Callable = stores.staged_chunks, **launch):
    """A kernel storing tiles in turn of an output staged through two chunk buffers, a chunk a wait of
    `waits` a tile (`staging`, `stores.staged_chunks` or its unguarded `stores.staged`, at `waits`: chunk c in
    buffer c % 2), tile t's `stored[t]` rows at its own rows, `STAGED`'s rows and columns a chunk, each its
    columns' indices plus a thousand times its own; `launch` the kernel's other arguments."""
    rows, width, _ = STAGED
    chunks = len(waits)

    def body(out_gmem):
        def scoped(smem):
            for tile, count in enumerate(stored):
                value = jax.lax.broadcasted_iota(jnp.float32, (rows, chunks * width), 1) + 1000.0 * tile
                staging(smem, value, out_gmem, (tile * rows, 0, jnp.int32(count), 0, rows, rows), waits=waits)
            plgpu.wait_smem_to_gmem(0)

        pl.run_scoped(scoped, plgpu.SMEM((2, rows, width), jnp.float32))

    out = jax.ShapeDtypeStruct((len(stored) * rows, chunks * width), jnp.float32)
    return plgpu.kernel(body, out_type=out, **launch)


@pytest.mark.parametrize("loosened", [None, 0, 2], ids=["derived", "first-chunk", "third-chunk"])
def test_a_staged_chunks_buffer_is_rewritten_once_its_last_store_has_read_it(loosened):
    """An output staged through two chunk buffers three chunks a tile, two tiles in turn (`_staged_launch`),
    under the GPU interpreter's race detection: at the derived waits (`stores.buffer_waits`: none left in
    flight before a tile's first chunk, whose buffer the tile before stored last, two before its second, one
    before its third) every buffer is rewritten only once its last TMA store has read it, the output the
    tiles' values; a wait allowing one group more at a tile's first chunk or its third rewrites a buffer a
    store still reads, which the detector names (its second's buffer the first's drain already freed). On
    the host's CPU (`on_the_host`)."""
    from jax._src.pallas.mosaic_gpu.interpret.interpret_pallas_call import reset_gpu_interpret_mode_state

    rows, width, chunks = STAGED
    waits = list(stores.buffer_waits(chunks, 2))
    assert waits == [0, 2, 1]
    if loosened is not None:
        waits[loosened] += 1
    kernel = _staged_launch(tuple(waits), interpret=plgpu.InterpretGPUParams(detect_races=True))
    with on_the_host():
        if loosened is None:
            tiles = 1000.0 * np.repeat([0, 1], rows)[:, None]
            columns = np.arange(chunks * width, dtype=np.float32)
            np.testing.assert_array_equal(np.asarray(kernel()), columns + tiles)
            return
        try:
            with pytest.raises(jax.errors.JaxRuntimeError):
                kernel()
        finally:
            reset_gpu_interpret_mode_state()


SKIPPED = (STAGED[0], 0, STAGED[0])
"""Three tiles' stored rows, the second's share storing none (a ragged last tile's second consumer's)."""

SKIPPED_WAITS = stores.buffer_waits(2, 2)
"""`SKIPPED`'s tiles' waits, two chunks a tile through two buffers: one group left in flight before each, so
a zero-row share staged anyway writes a buffer another tile's store may still read (three chunks drain every
store before a tile's first: base review 68e1f210f F3)."""


def test_a_share_storing_no_rows_stages_no_chunk_between_stored_tiles():
    """Three tiles of an output staged through two chunk buffers at the derived waits, the second tile's share
    storing no rows (`SKIPPED`), under the GPU interpreter's race detection: `stores.staged_chunks` stages
    nothing for it, no race is named and the output holds the first and third tiles' values. The race a
    zero-row share staged anyway makes is the protocol checker's, which counts store groups (the next
    test; base review 5028c8148 F2). On the host's CPU (`on_the_host`)."""
    rows, width, _ = STAGED
    chunks = len(SKIPPED_WAITS)
    assert SKIPPED_WAITS == (1, 1)
    kernel = _staged_launch(SKIPPED_WAITS, SKIPPED, interpret=plgpu.InterpretGPUParams(detect_races=True))
    with on_the_host():
        out = np.asarray(kernel())
    columns = np.broadcast_to(np.arange(chunks * width, dtype=np.float32), (rows, chunks * width))
    np.testing.assert_array_equal(out[:rows], columns)
    np.testing.assert_array_equal(out[2 * rows:], columns + 2000.0)


@pytest.mark.parametrize("staging", [stores.staged_chunks, stores.staged], ids=["skipped", "staged"])
def test_a_share_storing_no_rows_issues_no_write_and_no_group(staging):
    """`SKIPPED`'s launch through the group-aware protocol checker (`ring_protocol`, its stores retired a
    group at a time): skipped, the second tile's share writes no buffer and issues no group, the program
    two tiles' staging, and every interleaving holds; staged anyway (`stores.staged`), its chunks write their
    buffers with no group, so a wait counted in groups leaves the first tile's store reading a buffer as it
    is written again, which the checker finds where the interpreter, ignoring commits, saw only its order
    (base review 3150e8dfc F6)."""
    from ring_protocol import Violation, commits_group, explore, programs

    chunks = len(SKIPPED_WAITS)
    with on_the_host():
        threads, arrivals = programs(_staged_launch(SKIPPED_WAITS, SKIPPED, staging), threads=1)
    [program] = threads
    writes = sum(event.kind == "write" for event in program)
    if staging is stores.staged_chunks:
        groups = sum(map(commits_group, program))
        assert (writes, groups) == (2 * chunks, 2 * chunks), program
        assert explore(threads, arrivals) > 0
        return
    assert writes == 3 * chunks, program
    with pytest.raises(Violation):
        explore(threads, arrivals)


def test_a_staged_outputs_launch_states_its_cost():
    """The staged output's launch (`_staged_launch`, each of its chunks' copies its own store group) costed as
    the toolkit's launches are (`emission.costed`, the estimate XLA and Dew's FLOP count read): it states the
    bytes it is given and no arithmetic, a store group's commit priced as the synchronization it is."""
    rows, width, chunks = STAGED
    stored = 2 * rows * chunks * width * 4
    launch = _staged_launch(stores.buffer_waits(chunks, 2))
    traced = jax.make_jaxpr(emission.costed(launch, Stated(steps=1, bytes_accessed=stored)))()
    [estimate] = [equation.params["cost_estimate"] for equation in equations(traced.jaxpr)
                  if "cost_estimate" in equation.params]
    assert (estimate.bytes_accessed, estimate.flops, estimate.transcendentals) == (stored, 0, 0)
