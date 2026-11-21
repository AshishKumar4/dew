"""The Mosaic GPU toolkit's ring protocol model-checked (`ring_protocol`): the sm90 ring family's persistent
walk (`ring_family.RingFamily.persist`: its producer's loads at a counter running across tiles, its drains,
its consumers' walks with their slots acquired and released, a step's products in flight or not, a rotation
unrolled or a step a trip) and the tcgen05 family's over a gathered ring (its load warps' gathers, its product
warp's commits), traced as the toolkit builds it and explored over every interleaving, within the bounds
below: no deadlock, no aliased parity, no early arrival, no read before its write lands, no overwrite before
its read. And the checker catches each class it names: a flipped parity, a missing arrival, an early release,
each a counterexample trace."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import ring_protocol
from gathered_walks import gathered_kernel, tile_value
from ring_protocol import Event, Violation, _Splat, explore, programs

from dew.nn.kernels import generation
from dew.nn.kernels.templates import mosaic_sm90, rings

BLOCK = (64, 64)

BOUNDS = {"stages": (1, 2, 3, 4), "consumers": (1, 2, 3), "tiles": ((3,), (2, 0, 3), (1, 4)),
          "in_flight": (0, 1), "unrolled": (False, True)}
"""The checked bounds: a ring's slots, its consumer warpgroups (the family's most, `Wgmma.consumers`), each
persistent tile's steps (an empty tile among them, its walk inactive), a step's products in flight past it,
and a rotation unrolled or a step a trip."""

CASES = [case for case in ring_protocol.configurations(**BOUNDS) if case["in_flight"] < case["stages"]]
"""Every configuration of `BOUNDS` a walk takes: fewer steps in flight than its ring's slots."""

STREAMS = ("a", "b")
"""The streams a two-ring walk reads, each step a slot of each (a GEMM's operands), drained together
(`rings.drain`)."""

TWO_RING_CASES = [
    {**case, "streams": STREAMS} for case in ring_protocol.configurations(
        stages=(1, 2, 3), consumers=(1, 2), tiles=((3,), (2, 0, 3)), in_flight=(0, 1),
        unrolled=BOUNDS["unrolled"])
    if case["in_flight"] < case["stages"]]
"""Two rings' walk at the configurations within the bounds a GEMM's product takes."""

SEGMENT_CASES = [
    case for case in ring_protocol.configurations(
        stages=(2, 3), consumers=BOUNDS["consumers"], segments=((6,), (3, 3), (0, 1, 0, 2, 3, 0)),
        in_flight=(0, 1, 2), unrolled=BOUNDS["unrolled"])
    if case["in_flight"] < case["stages"]]
"""One walk's six steps in one segment or adjacent segments, including empty and short segments."""


@functools.cache
def walked(stages: int, consumers: int, tiles: tuple, in_flight: int, unrolled: bool,
           streams: tuple = ("a",), pingpong: bool = False, empty_active: bool = False, joined: bool = False,
           among: bool = False):
    """The ring family's persistent walk at the configuration over a ring of each of `streams`, each step
    reading its slot of each, its two consumers taking alternate tiles whole where `pingpong` (each slot
    released by its tile's owner alone), a tile of no steps active where `empty_active` (its turn taken, its
    walk none), its streams one barrier pair between them where `joined` (`rings.joined`), each step's
    source chosen between two arrays by its tile where `among` (`rings.Among`): each warpgroup's program and
    the barriers' arrivals (`ring_protocol.programs`)."""
    family = mosaic_sm90.WGMMA
    starts = [sum(tiles[:index]) for index in range(len(tiles))]
    names = streams  # the streams' names, not their allocations' keys (a joined pair's one)

    def body(x_gmem, out_gmem):
        def scope(*, streams):
            def schedule(index):
                tile = index[0]
                steps = tile_value(tile, tiles)
                view = (x_gmem, lambda step: (tile_value(tile, starts) + step, 0))
                source = rings.Among(lambda step: tile % 2, (view, view)) if among else view
                views = dict.fromkeys(names, source)
                return (steps > 0) | empty_active, steps, views, tile

            def issue(wg, tile, walk, *, active):
                def step(index, slots, carry, accumulators):
                    for name in names:
                        slots[name][0, 0]
                    return carry, accumulators

                walk(step, None, ())

            def finish(*_, **__):
                pass

            family.persist(streams, (len(tiles),), schedule, issue, finish, consumers=consumers,
                           axis="programs", outputs=(), interpret=False, in_flight=in_flight,
                           unrolled=unrolled, pingpong=pingpong)

        family.scoped(scope, streams=family.streams(dict.fromkeys(streams, (BLOCK, jnp.bfloat16)), stages,
                                                    1 if pingpong else consumers,
                                                    joined=streams if joined else ()))

    kernel = family.kernel(body, out_type=jax.ShapeDtypeStruct((8, 8), jnp.float32), grid=(1,),
                           grid_names=("programs",), consumers=consumers, interpret=False, name="ring_model")
    rows = max(sum(tiles), 1) * BLOCK[0]
    return programs(kernel, jnp.zeros((rows, BLOCK[1]), jnp.bfloat16), threads=consumers + 1)


@pytest.mark.parametrize("case", CASES, ids=[
    f"stages{c['stages']}-consumers{c['consumers']}-tiles{c['tiles']}-inflight{c['in_flight']}"
    f"-{'unrolled' if c['unrolled'] else 'stepped'}" for c in CASES])
def test_the_ring_familys_walk_holds_every_property_in_every_interleaving(case):
    """The ring family's persistent walk at the configuration: every interleaving of its warpgroups and its
    loads' landings explored, each property held (`ring_protocol.explore`)."""
    threads, arrivals = walked(**case)
    assert any(event.kind == "load" for event in threads[-1]), "the producer loads nothing"
    assert explore(threads, arrivals) > 0


@pytest.mark.parametrize("case", TWO_RING_CASES, ids=[
    f"stages{c['stages']}-consumers{c['consumers']}-tiles{c['tiles']}-inflight{c['in_flight']}"
    f"-{'unrolled' if c['unrolled'] else 'stepped'}" for c in TWO_RING_CASES])
def test_a_ring_loading_its_steps_among_sources_holds_every_property_in_every_interleaving(case):
    """Two rings whose every step's source is chosen at run time between two arrays by its tile
    (`rings.Among`, matrices side by side, row 6794): one copy a step, its arrivals as any view's, every
    interleaving explored and each property held, the producer's loads as many as a ring of one source's."""
    threads, arrivals = walked(**case, among=True)
    plain, _ = walked(**case)
    loads = sum(event.kind == "load" for event in threads[-1])
    assert loads == sum(event.kind == "load" for event in plain[-1]) == len(STREAMS) * sum(case["tiles"])
    assert explore(threads, arrivals) > 0


@pytest.mark.parametrize("case", TWO_RING_CASES, ids=[
    f"stages{c['stages']}-consumers{c['consumers']}-tiles{c['tiles']}-inflight{c['in_flight']}"
    f"-{'unrolled' if c['unrolled'] else 'stepped'}" for c in TWO_RING_CASES])
def test_two_rings_of_one_barrier_pair_hold_every_property_in_every_interleaving(case):
    """Two rings joined on one barrier pair (`rings.joined`, a GEMM's operands as CUTLASS's PipelineTmaAsync
    holds them, row 5666): each step's two loads arriving at one load barrier, awaited once and given back
    once; every interleaving explored, each property held, both rings loaded every step, and a step's waits
    and arrivals half of two pairs' (one of each a step a consumer)."""
    threads, arrivals = walked(**case, joined=True)
    apart, _ = walked(**case)
    loads = sum(event.kind == "load" for event in threads[-1])
    assert loads == len(STREAMS) * sum(case["tiles"])
    waits = [sum(event.kind == "wait" for event in thread) for thread in threads[:-1]]
    waited_apart = [sum(event.kind == "wait" for event in thread) for thread in apart[:-1]]
    halved = zip(waits, waited_apart, strict=True)
    assert all(2 * joined <= both for joined, both in halved), (waits, waited_apart)
    assert explore(threads, arrivals) > 0


@pytest.mark.parametrize("case", TWO_RING_CASES, ids=[
    f"stages{c['stages']}-consumers{c['consumers']}-tiles{c['tiles']}-inflight{c['in_flight']}"
    f"-{'unrolled' if c['unrolled'] else 'stepped'}" for c in TWO_RING_CASES])
def test_two_rings_drained_together_hold_every_property_in_every_interleaving(case):
    """Two rings' persistent walk, each step a slot of each and the producer's drain a slot of both at a time
    (`rings.drain`): every interleaving explored, each property held, the producer's last waits each
    ring's last releases."""
    threads, arrivals = walked(**case)
    loads = sum(event.kind == "load" for event in threads[-1])
    assert loads == len(STREAMS) * sum(case["tiles"])
    assert explore(threads, arrivals) > 0


@functools.cache
def segmented(stages: int, consumers: int, segments: tuple, in_flight: int, unrolled: bool):
    """One ring walk's adjacent segments with traced bounds and a nonzero origin."""
    family = mosaic_sm90.WGMMA
    starts = [sum(segments[:index]) for index in range(len(segments))]
    total = sum(segments)

    def body(x_gmem, out_gmem):
        first = jax.lax.axis_index("programs") + 3

        def scope(*, streams):
            rings = family.opened(streams)
            views = {"a": (x_gmem, lambda step: (step - first, 0))}

            def step(index, slots, carry, accumulators):
                slots["a"][0, 0]
                return carry, accumulators

            def consumer(wg):
                family.walk(rings, tuple((step, first + lo, first + lo + count)
                                         for lo, count in zip(starts, segments, strict=True)),
                            None, (), interpret=False, first=first, in_flight=in_flight, unrolled=unrolled)

            family.scheduled(consumers, consumer, lambda: family.stream(rings, views, first, first + total))

        family.scoped(scope, streams=family.streams({"a": (BLOCK, jnp.bfloat16)}, stages, consumers))

    kernel = family.kernel(body, out_type=jax.ShapeDtypeStruct((8, 8), jnp.float32), grid=(1,),
                           grid_names=("programs",), consumers=consumers, interpret=False,
                           name="segmented_ring_model")
    return programs(kernel, jnp.zeros((total * BLOCK[0], BLOCK[1]), jnp.bfloat16), threads=consumers + 1)


@pytest.mark.parametrize("case", SEGMENT_CASES, ids=[
    f"stages{c['stages']}-consumers{c['consumers']}-segments{c['segments']}-inflight{c['in_flight']}"
    f"-{'unrolled' if c['unrolled'] else 'stepped'}" for c in SEGMENT_CASES])
def test_segments_of_one_walk_release_their_slots_before_the_producer_reuses_them(case):
    """Adjacent segments release every step exactly once without deadlock, including their pending tails."""
    threads, arrivals = segmented(**case)
    total = sum(case["segments"])
    assert sum(event.kind == "load" for event in threads[-1]) == total
    for consumer in threads[:-1]:
        assert sum(event.kind == "read" for event in consumer) == total
        assert sum(event.kind == "arrive" for event in consumer) == total
    assert explore(threads, arrivals) > 0


PINGPONG_CASES = [case for case in ring_protocol.configurations(
    stages=(2, 3), consumers=(2,), tiles=((3,), (2, 0, 3), (1, 4), (2, 0, 1, 3), (0, 2), (0,)),
    in_flight=(0, 1), unrolled=BOUNDS["unrolled"], empty_active=(False, True))
    if case["in_flight"] < case["stages"]]
"""Ping-pong's walks: one active tile, an inactive one between two, an odd and an even count of active
tiles, a first tile inactive, so either consumer may take none or the last, and a walk of no steps at all;
each with its tiles of no steps inactive, or active (a turn taken, no step walked)."""


def _owned(tiles: tuple, empty_active: bool = False) -> list[int]:
    """Each consumer's steps under ping-pong: the active tiles' in turn, the first consumer's first."""
    steps = [0, 0]
    for turn, count in enumerate(count for count in tiles if count or empty_active):
        steps[turn % 2] += count
    return steps


@pytest.mark.parametrize("case", PINGPONG_CASES, ids=[
    f"stages{c['stages']}-tiles{c['tiles']}-inflight{c['in_flight']}"
    f"-{'unrolled' if c['unrolled'] else 'stepped'}{'-empty-active' if c['empty_active'] else ''}"
    for c in PINGPONG_CASES])
def test_a_pingpong_walk_alternates_whole_tiles_and_holds_every_property(case):
    """The ring family's walk with its two consumers taking alternate tiles whole (`persist`'s `pingpong`):
    each consumer reads and releases its own tiles' steps alone, the active tiles in turn from the first
    consumer, each slot one release a generation, and every interleaving holds each property (no deadlock,
    no aliased parity, no early arrival, no read before its write lands, no overwrite before its read)."""
    threads, arrivals = walked(**case, pingpong=True)
    owned = _owned(case["tiles"], case["empty_active"])
    released = _released(threads)
    assert [sum(event.kind == "read" for event in consumer) for consumer in threads[:-1]] == owned
    assert [sum(event.kind == "arrive" and event.at[0] in released for event in consumer)
            for consumer in threads[:-1]] == owned
    assert explore(threads, arrivals) > 0


def _released(threads) -> set:
    """The barriers a ring's slots are released at: those the producer waits on before it refills one."""
    return {event.at[0] for event in threads[-1] if event.kind == "wait"}


def _turns(threads) -> set:
    """The barriers the consumers arrive at past the rings' releases (the turns of `Family.meeting`)."""
    released = _released(threads)
    return {event.at[0] for consumer in threads[:-1] for event in consumer
            if event.kind == "arrive" and event.at[0] not in released}


def test_a_pingpong_walk_missing_its_turn_is_caught():
    """The first consumer's first turn arrival dropped: its partner never takes its turn, a deadlock."""
    threads, arrivals = walked(2, 2, (2, 3), 0, unrolled=False, pingpong=True)
    turns = _turns(threads)
    index = next(index for index, event in enumerate(threads[0])
                 if event.kind == "arrive" and event.at[0] in turns)
    with pytest.raises(Violation) as found:
        explore(_mutated(threads, 0, index, []), arrivals)
    assert found.value.kind in ("deadlock", "unfinished")


def test_a_pingpong_ring_awaiting_both_consumers_releases_is_caught():
    """The rings' slots awaiting both consumers' releases, as a cooperative tile's do: under ping-pong the
    tile's partner releases none of its steps, so the producer never refills them, a deadlock."""
    threads, arrivals = walked(2, 2, (2, 3), 0, unrolled=False, pingpong=True)
    released = _released(threads)
    both = {name: 2 if name in released else count for name, count in arrivals.items()}
    with pytest.raises(Violation) as found:
        explore(threads, both)
    assert found.value.kind in ("deadlock", "unfinished")


def _mutated(threads, thread: int, index: int, replaced) -> list:
    """`threads` with thread `thread`'s `index`-th event replaced by `replaced` (a list of events)."""
    mutated = [list(program) for program in threads]
    mutated[thread][index:index + 1] = replaced
    return mutated


def _first(program, kind: str, *, after: int = 0) -> int:
    return next(index for index, event in enumerate(program) if event.kind == kind and index >= after)


def test_a_flipped_parity_is_caught():
    """A consumer's first wait on a load at the phase after, of the other parity: it passes before the load
    lands, or waits a phase that never comes, and the checker gives the interleaving."""
    threads, arrivals = walked(2, 1, (3,), 0, unrolled=False)
    consumer = threads[0]
    index = _first(consumer, "wait")
    event = consumer[index]
    flipped = dataclasses.replace(event, argument=event.argument + 1)
    with pytest.raises(Violation) as found:
        explore(_mutated(threads, 0, index, [flipped]), arrivals)
    assert found.value.kind in ("aliased", "unordered", "deadlock")


def test_a_missing_arrival_is_caught():
    """A consumer's first release dropped: its slot's barrier never completes the phase the producer waits
    on, a deadlock the checker gives."""
    threads, arrivals = walked(2, 1, (5,), 0, unrolled=False)
    index = _first(threads[0], "arrive")
    with pytest.raises(Violation) as found:
        explore(_mutated(threads, 0, index, []), arrivals)
    assert found.value.kind in ("deadlock", "unfinished")


@pytest.mark.parametrize("streams", [("a",), STREAMS])
def test_a_missing_drain_is_caught(streams):
    """The producer's drain of its last ring dropped (`rings.drain`): the last releases of that ring's slots
    never waited for, a slot freed while its release is unseen, which the checker gives."""
    threads, arrivals = walked(2, 1, (5,), 0, unrolled=False, streams=streams)
    producer = threads[-1]
    drained = [index for index, event in enumerate(producer) if event.kind == "wait"][-2:]
    with pytest.raises(Violation) as found:
        explore(_mutated(threads, len(threads) - 1, drained[0], []), arrivals)
    assert found.value.kind == "unobserved"


def test_an_early_release_is_caught():
    """A consumer's release moved before its read of the slot: the producer may refill the slot while the
    consumer reads it, an overwrite before its read the checker gives."""
    threads, arrivals = walked(2, 1, (5,), 0, unrolled=False)
    consumer = threads[0]
    read = _first(consumer, "read")
    release = _first(consumer, "arrive", after=read)
    reordered = list(consumer)
    released = reordered.pop(release)
    reordered.insert(read, released)
    with pytest.raises(Violation) as found:
        explore([reordered, *threads[1:]], arrivals)
    assert found.value.kind == "unordered"


def test_a_loop_whose_trips_the_checker_cannot_read_is_refused():
    """A loop whose trip count reads a value the checker does not evaluate (a table's entry in global memory,
    None to it): the checker refuses it, as it refuses a branch on one, where it ran the loop zero times and
    explored the kernel with its events missing (row 3588)."""
    def kernel(table):
        return jax.lax.while_loop(lambda i: i < table[0], lambda i: i + 1, jnp.int32(0))

    closed = jax.make_jaxpr(kernel)(jnp.zeros((1,), jnp.int32))
    table = ring_protocol.Ref("gmem0", barrier=False, num_arrivals=0)
    with pytest.raises(ValueError):
        ring_protocol._Interpreter(0).run(closed.jaxpr, closed.consts, [table])


def test_the_programs_are_the_toolkits_events():
    """The traced programs are the kernel's own synchronization: the producer's loads one a step, each
    with its slot's barrier, every consumer's waits and releases one a step a stream, and a read each."""
    tiles = (2, 0, 3)
    threads, arrivals = walked(2, 2, tiles, 0, unrolled=False)
    total = sum(tiles)
    producer = threads[-1]
    assert sum(event.kind == "load" for event in producer) == total
    for consumer in threads[:-1]:
        assert sum(event.kind == "read" for event in consumer) == total
        assert sum(event.kind == "arrive" for event in consumer) == total
    assert all(isinstance(event, Event) for program in threads for event in program)
    assert sorted(set(arrivals.values())) == [1, 2]


GATHERED = (32, 64)
"""A gathered step's block: four 8-row tiles (`swizzling.TILE_ROWS`), one more than the issuing warpgroup's
three load warps, so the first issues two (`gathered.Gathers.tiles`); its rows the product's contraction (a
kernel gradient's gathered tokens)."""

PLAIN = (128, 32)
"""The plain stream's step: the product's left operand, a tcgen05 accumulator's 128 rows over the gathered
block's rows."""

ACCUMULATOR = (128, 64)
"""A tile's accumulator in tensor memory, the plain block times the gathered one, which the walk's turns
hold."""


@functools.cache
def gathered_walk(stages: int, tiles: tuple, streams: int = 1, tiled: bool = False):
    """The tcgen05 family's persistent walk over `streams` gathered streams and a plain one the first load
    warp copies whole (`gathered_walks.gathered_kernel`, unstored), its tiles' rows `tiles`: each program,
    the epilogue warpgroup's and each of the issuing warpgroup's warps' (`ring_protocol.programs`), and the
    barriers' arrivals; its rows the tile's at every step where `tiled` (`rings.Rows.tiled`, a gathered
    product's)."""
    starts = [sum(tiles[:index]) for index in range(len(tiles))]
    height, width = GATHERED
    names = tuple(f"x{index}" for index in range(streams))

    def views(operands, tile):
        x_gmem, table_gmem, w_gmem, _ = operands

        def rows(step):
            return rings.Rows(table_gmem, tile * height, (step,), 1, tiled=tiled)

        start = tile_value(tile, starts)
        return {**dict.fromkeys(names, (x_gmem, rows)), "w": (w_gmem, lambda step: (start + step, 0))}

    blocks = {**dict.fromkeys(names, (GATHERED, jnp.bfloat16)), "w": (PLAIN, jnp.bfloat16)}
    kernel = gathered_kernel(blocks, names, tiles, views, ACCUMULATOR, stages=stages, stored=False,
                             out=jax.ShapeDtypeStruct((8, 8), jnp.float32), name="gathered_model")
    operands = (jnp.zeros((len(tiles) * height, width * max(tiles)), jnp.bfloat16),
                jnp.zeros((len(tiles) * height,), jnp.int32),
                jnp.zeros((max(sum(tiles), 1) * PLAIN[0], PLAIN[1]), jnp.bfloat16))
    with generation.assumed("sm100"):
        return programs(kernel, *operands, threads=2)


GATHERED_CASES = [(1, (2,)), (2, (3,)), (2, (1, 0, 2))]
"""A gathered walk's slots and each persistent tile's steps (an empty tile among them)."""


@pytest.mark.parametrize(("stages", "tiles"), GATHERED_CASES, ids=[
    f"stages{stages}-tiles{'-'.join(map(str, tiles))}" for stages, tiles in GATHERED_CASES])
def test_a_gathered_walk_over_the_load_warps_holds_every_property_in_every_interleaving(stages, tiles):
    """The tcgen05 family's walk over a gathered ring as the toolkit builds it, each step's gathers spread
    over the issuing warpgroup's load warps, the first alone arriving at the slot's load barrier expecting
    every tile's copy (`gathered.Gathers`), a plain ring beside it, the product warp's products reading both
    slots until its commit releases them: every interleaving explored, the other warps' copies landing
    before that arrival among them (the barrier's transaction count below zero, the phase complete only once
    the arrival is made and the count is back at zero), each property held: no early or missing arrival, no
    copy into a slot a product still reads."""
    threads, arrivals = gathered_walk(stages, tiles)
    assert explore(threads, arrivals) > 0


def test_the_gathered_programs_are_the_load_warps_own_gathers():
    """The traced programs are the gathered ring's own, a program a warp of the issuing warpgroup: each load
    warp copying its share of each step's four tiles (two, one and one: tile t warp t mod three's), the
    first alone arriving at the slot's load barrier expecting all four; the product warp's tcgen05 product
    reading each step's slot of each ring once its fill completes."""
    steps = 3
    threads, _ = gathered_walk(2, (steps,))
    assert len(threads) == 5  # the epilogue warpgroup, the issuing warpgroup's four warps
    loaders = [program for program in threads if any(event.kind == "copy" for event in program)]
    copies = [sum(event.kind == "copy" for event in program) for program in loaders]
    assert copies == [2 * steps, steps, steps]
    assert [[event.argument for event in program if event.kind == "expect"]
            for program in loaders] == [[4] * steps, [], []]
    # Each step's product reads the plain ring's slot (the first load warp's TMA load's) and the gathered one.
    [plain] = {event.at[0] for event in loaders[0] if event.kind == "load"}
    [gathered] = {event.at[0] for program in loaders for event in program if event.kind == "copy"}
    [products] = [program for program in threads if any(event.kind == "product" for event in program)]
    assert [event.argument for event in products if event.kind == "product"] == [
        ((plain, step % 2), (gathered, step % 2)) for step in range(steps)]


def _first_release(threads) -> tuple[int, int, int]:
    """The product warp of a gathered walk's programs (`gathered_walk`), and the places in its program of its
    first product and of the first commit after it, which releases the product's slots."""
    [at] = [index for index, program in enumerate(threads)
            if any(event.kind == "product" for event in program)]
    program = threads[at]
    product = next(position for position, event in enumerate(program) if event.kind == "product")
    release = next(position for position, event in enumerate(program)
                   if position > product and event.kind == "committed")
    return at, product, release


def test_a_tcgen05_release_committed_before_its_product_is_caught():
    """The gathered walk's programs with the product warp's first commit releasing a slot moved before the
    product that reads it (a release before its product, `CommittedRing.release` ahead of `products`): the
    load warps refill the slot while the product may still read it, which the explorer finds."""
    threads, arrivals = gathered_walk(1, (2,))
    at, product, release = _first_release(threads)
    program, mutated = threads[at], [*threads]
    mutated[at] = [*program[:product], *program[product + 1:release + 1], program[product],
                   *program[release + 1:]]
    with pytest.raises(Violation):
        explore(mutated, arrivals)


def test_a_slot_written_after_its_products_commit_issues_but_before_it_completes_is_caught():
    """The gathered walk's programs with the product warp writing its first product's left slot right after
    the commit that releases it: a commit completes asynchronously (`tcgen05.commit`), so its issue orders
    nothing of its thread's after it, and the write may land while the product still reads the slot, which
    the explorer finds; a thread orders itself after the product only by a wait observing the commit's
    arrival."""
    threads, arrivals = gathered_walk(1, (2,))
    at, product, release = _first_release(threads)
    program, mutated = threads[at], [*threads]
    mutated[at] = [*program[:release + 1], Event("write", program[product].argument[0]),
                   *program[release + 1:]]
    with pytest.raises(Violation):
        explore(mutated, arrivals)


def test_a_write_between_two_commits_is_unordered_with_a_read_after_the_first_commits_wait():
    """A product warp's product, a commit to `A`, a write of `y`, a commit to `B` and its wait on `B`, beside
    a thread that waits on `A` and reads `y`: the write follows `A`'s issue, not the product's completion,
    so the read is unordered with it whichever commit completes first. Where `B` completed first, its whole
    clock stood for the product's completion, and `A`'s then ordered the write before the reader's wait
    (row 8474)."""
    products = [Event("product", (), (("x", None),)), Event("committed", ("A", None)),
                Event("write", ("y", None)), Event("committed", ("B", None)), Event("wait", ("B", None), 0)]
    reader = [Event("wait", ("A", None), 0), Event("read", ("y", None))]
    with pytest.raises(Violation):
        explore([products, reader], {"A": 1, "B": 1})


@pytest.mark.parametrize(("kept", "races"), [(0, False), (1, True)], ids=["drained", "in-flight"])
def test_a_product_wait_retires_whole_issues(kept, races):
    """A thread issuing one warpgroup MMA reading two slots, then rewriting the first after a wait leaving
    `kept` issues in flight: none left, both reads have ended and the write is ordered after them; one left,
    the issue may still read both, which the checker finds, where retiring its operands one by one as groups
    ended the first slot's read (base review 156a4595a F1)."""
    program = [Event("product", (), (("smem", 0), ("smem", 1))), Event("product_wait", (), kept),
               Event("write", ("smem", 0))]
    if races:
        with pytest.raises(Violation):
            explore([program], {})
    else:
        assert explore([program], {}) > 0


@pytest.mark.parametrize(("kept", "races"), [(1, False), (2, True)], ids=["derived", "loosened"])
def test_a_store_wait_retires_whole_groups(kept, races):
    """A thread storing three chunks through two buffers, a chunk's two copies one group (the second
    committing it), then rewriting the first buffer after a wait leaving `kept` groups in flight: one left,
    the first chunk's group has been read and the write is ordered after it; two left, both its copies may
    still read the buffer, which the checker finds, where counting copies one by one retired them (row
    ca3f0ad22 F2)."""
    def chunk(buffer):
        return [Event("store", ("smem", buffer), argument=commits) for commits in (False, True)]

    program = [*chunk(0), *chunk(1), Event("store_wait", (), kept), Event("write", ("smem", 0))]
    if races:
        with pytest.raises(Violation):
            explore([program], {})
    else:
        assert explore([program], {}) > 0


def test_the_load_warps_read_each_steps_entries_a_step_ahead():
    """The gathered walk's load warps (`gathered.Held`): at each step but a tile's last, every loader fills
    the next step's entries of its own tiles into the parity its counter's next takes, after its gathers;
    at each step it awaits its fills (`fill_wait`) and reads its tiles' entries from the parity they were
    filled into, before its gathers, a tile's first step filling its own first."""
    steps = 3
    threads, _ = gathered_walk(2, (steps,))
    loaders = [program for program in threads if any(event.kind == "copy" for event in program)]
    for program in loaders:
        kinds = [event.kind for event in program if event.kind in ("fill", "fill_wait", "read", "copy")]
        tiles = sum(kind == "copy" for kind in kinds) // steps
        step = ["copy"] * tiles
        assert kinds == ["fill", "fill_wait", "read", *step, "fill",
                         "fill_wait", "read", *step, "fill",
                         "fill_wait", "read", *step], kinds
        fills = [event.at[1] for event in program if event.kind == "fill"]
        reads = [event.at[1] for event in program if event.kind == "read"]
        assert fills == reads and fills[0] != fills[1], (fills, reads)


def _other_parity(self, row):
    """`gathered.Held.entry` mutated: row `row`'s entry from the other step's half of the entries."""
    from jax.extend.mlir import ir
    from jax.extend.mlir.dialects import arith, memref

    from dew.nn.kernels.templates import gathered

    index = ir.IndexType.get()
    other = arith.subi(gathered.mgpu_utils.c(1, ir.IntegerType.get_signless(32)), self.parity)
    return memref.load(self.entries, [arith.index_cast(index, other), arith.index_cast(index, row)])


def _other_half_filled(filled):
    """`gathered.filled` mutated: its copies' destination the other step's half of the entries."""
    from jax.extend.mlir import ir
    from jax.extend.mlir.dialects import arith

    from dew.nn.kernels.templates import gathered

    def mutated(entries, table, parity, share):
        other = arith.subi(gathered.mgpu_utils.c(1, ir.IntegerType.get_signless(32)), parity)
        return filled(entries, table, other, share)

    return mutated


def _ahead_other_parity(call):
    """`gathered.Gathers.__call__` mutated where it hands its step's operands on: the parity it reads its
    entries ahead at (`gathered.Held.read`) the other step's."""
    from jax.extend.mlir import ir
    from jax.extend.mlir.dialects import arith

    from dew.nn.kernels.templates import gathered

    def mutated(self, ctx, *operands):
        *given, entries, parity = operands
        other = arith.subi(gathered.mgpu_utils.c(1, ir.IntegerType.get_signless(32)),
                           gathered.splat_value(parity))
        return call(self, ctx, *given, entries, _Splat(other))

    return mutated


@pytest.mark.parametrize("mutation", [None, "no-wait", "no-warp-barrier", "one-in-flight", "other-parity",
                                      "other-half-filled", "ahead-other-parity"])
def test_a_load_warp_reading_its_entries_before_their_fills_land_is_caught(mutation):
    """The gathered walk as the kernel builds it, every interleaving holding; and built with its load warps'
    entries mutated in the kernel, each read from its lowered bodies' ops (`_lowered`, `_lowered_reads`):
    their wait for their entries' cp.async groups removed, or leaving one group in flight
    (`cp_async_wait_group(1)`), or the warp not meeting after it (each lane's copy its own till then, base
    review 760be77fe F4), a step's read may come before the fill it needs lands; their loads reading
    the other step's half of the entries, or their copies filling it, or their gathers handing the read the
    other step's parity, a read no fill of its step wrote. The explorer finds each (row 8657; base reviews
    0991196fb F2 and 9d013a4b9 F1: the checker read the wait, the read and the fill's half from the body's
    fields, so each mutated kernel passed; head review of 0099a4701 F2: it read the entries from a Held of
    its own, so a gather handing on another parity passed)."""
    from dew.nn.kernels.templates import gathered

    wait = gathered.mgpu_nvvm.cp_async_wait_group
    mutated = {"no-wait": (gathered.mgpu_nvvm, "cp_async_wait_group", lambda count: None),
               "no-warp-barrier": (gathered.mgpu_utils, "warp_barrier", lambda: None),
               "one-in-flight": (gathered.mgpu_nvvm, "cp_async_wait_group", lambda count: wait(1)),
               "other-parity": (gathered.Held, "entry", _other_parity),
               "other-half-filled": (gathered, "filled", _other_half_filled(gathered.filled)),
               "ahead-other-parity": (gathered.Gathers, "__call__",
                                      _ahead_other_parity(gathered.Gathers.__call__))}
    gathered_walk.cache_clear()
    with pytest.MonkeyPatch.context() as patch:
        if mutation is not None:
            patch.setattr(*mutated[mutation])
        threads, arrivals = gathered_walk(1, (3,))
    gathered_walk.cache_clear()
    if mutation is None:
        assert explore(threads, arrivals) > 0
        return
    with pytest.raises(Violation):
        explore(threads, arrivals)


def test_a_load_warp_reading_entries_no_fill_wrote_is_caught():
    """The gathered walk's programs with a load warp's fill of its next step's entries left out (the fill
    `gathered.Held.copied_ahead` issues after a step's gathers): the next step reads the other parity's
    entries, which no fill wrote, and the explorer refuses it ("unwritten"), where a read was ordered only
    against writes and passed a parity never filled (base review ae8a242ed F3)."""
    threads, arrivals = gathered_walk(1, (3,))
    [at, *_] = [index for index, program in enumerate(threads)
                if any(event.kind == "fill" for event in program)]
    fills = [position for position, event in enumerate(threads[at]) if event.kind == "fill"]
    mutated = [*threads]
    mutated[at] = [event for position, event in enumerate(threads[at]) if position != fills[1]]
    with pytest.raises(Violation) as found:
        explore(mutated, arrivals)
    assert found.value.kind == "unwritten"


@pytest.mark.parametrize(("stages", "tiles"), [(1, (3,)), (1, (2, 0, 2))], ids=["one-tile", "three-tiles"])
def test_a_tiles_rows_are_read_once_a_tile(stages, tiles):
    """A gathered walk whose rows are the tile's at every step (`rings.Rows.tiled`, a gathered product's):
    each load warp fills its entries at a tile's first step alone and reads them there at every step, no
    step filling the next's (base review ae8a242ed F6), and every interleaving holds."""
    threads, arrivals = gathered_walk(stages, tiles, tiled=True)
    loaders = [program for program in threads if any(event.kind == "copy" for event in program)]
    walked = sum(1 for count in tiles if count)
    for program in loaders:
        assert sum(event.kind == "fill" for event in program) == walked, program
        assert sum(event.kind == "read" for event in program) == sum(tiles), program
    assert explore(threads, arrivals) > 0


@pytest.mark.parametrize(("read", "caught"), [(2, False), (0, True)], ids=["latest", "overwritten"])
def test_a_read_of_entries_takes_their_last_fill(read, caught):
    """One warp fills a location's entries for step 0, waits, fills them again for step 2, waits, and reads
    them as step `read`'s: step 2's read holds, step 0's is refused ("unwritten"), its fill overwritten,
    where any earlier fill of the step matched (base review 3f38bf20d F4)."""
    at = ("entries", 0, 0)
    program = [Event("fill", at, 0), Event("fill_wait"), Event("fill", at, 2), Event("fill_wait"),
               Event("read", at, ("filled", read))]
    if not caught:
        assert explore([program], {}) > 0
        return
    with pytest.raises(Violation) as found:
        explore([program], {})
    assert found.value.kind == "unwritten"


def test_a_load_warp_reading_a_stale_steps_entries_is_caught():
    """The gathered walk's programs with a load warp's fill of its third step's entries left out (the fill
    its second step issues, `gathered.Filled`): the third step reads the parity its first step's entries
    still hold, which a fill wrote and its wait ordered, and the explorer refuses it ("unwritten": no fill of
    that step's), where an untagged read passed a stale step's entries (base review 571a5ee10 F2)."""
    threads, arrivals = gathered_walk(1, (3,))
    [at, *_] = [index for index, program in enumerate(threads)
                if any(event.kind == "fill" for event in program)]
    mutated = [*threads]
    mutated[at] = [event for event in threads[at] if not (event.kind == "fill" and event.argument == 2)]
    assert len(mutated[at]) == len(threads[at]) - 1
    with pytest.raises(Violation) as found:
        explore(mutated, arrivals)
    assert found.value.kind == "unwritten"


def test_a_load_warps_rings_await_their_entries_before_either_gathers():
    """A walk of two gathered rings (`gathered_walk`'s `streams`): each load warp starts both rings' entries
    (`gathered.Started`), then drains its fills once (`gathered.Drained`, its one `fill_wait`) before either
    ring's gathers, so neither ring's wait waits on the other's next-step fill issued after its gathers (base
    reviews 103007773 F4, b68fc2382 F6: each ring's own drain), and every interleaving holds."""
    threads, arrivals = gathered_walk(1, (2,), streams=2)
    loaders = [program for program in threads if any(event.kind == "copy" for event in program)]
    for program in loaders:
        # The program's steps, each from its first wait for entries after a step's gathers: every step's waits
        # all before any of its gathers (base review a1ad2b4ca F6: a check of the first step alone passed
        # steady-state waits in each ring's gathers), the warp's one drain.
        kinds = [event.kind for event in program if event.kind in ("fill_wait", "copy")]
        steps, gathered = [], True
        for kind in kinds:
            if kind == "fill_wait" and gathered:
                steps.append([])
            gathered = kind == "copy"
            steps[-1].append(kind)
        assert len(steps) == 2, kinds
        for step in steps:
            waits = step.index("copy")
            assert waits == 1 and "fill_wait" not in step[waits:], step
    assert explore(threads, arrivals) > 0


def test_a_gathered_load_warp_skipping_its_slots_release_wait_is_caught():
    """The gathered walk's programs with the second load warp's waits for its slots' release left out (the
    waits `Committed.load` makes first, `rings.Ring.free`), the first warp's arrival still in order: its copy
    of a step lands in a slot whose last step's products may still read it, which the explorer finds."""
    threads, arrivals = gathered_walk(1, (2,))
    loaders = [index for index, program in enumerate(threads)
               if any(event.kind == "copy" for event in program)]
    program = threads[loaders[1]]
    last = max(position for position, event in enumerate(program) if event.kind == "copy")
    mutated = [*threads]
    mutated[loaders[1]] = [event for position, event in enumerate(program)
                           if event.kind != "wait" or position > last]
    assert len(mutated[loaders[1]]) < len(program)
    with pytest.raises(Violation):
        explore(mutated, arrivals)


def test_a_gathered_indexs_location_shares_memory_with_what_it_reads_alone():
    """The checker's locator (`ring_protocol._located`) over a ref's indexers as a kernel's trace states
    them, a buffer laid `[role, slot, column]`: a role's slot, `[0, 1]`, located at both indices, its columns
    whole; the role's slots gathered by an array of indices, `[0, [0, 2]]` (JAX broadcasts the role's index
    beside them, one value), sharing memory with those slots alone; a `(2, 2)` ref's rows gathered by a
    repeated index, `[[0, 0]]`, then the view's second row, `[1]`, the ref's row 0 again, with that row
    alone (the base review of 56a716c95, F6: a gathered index raised; of d8f8ab837, F1: a repeated one read
    as one value, which lost the alias, row 8978)."""
    from jax._src.state.indexing import NDIndexer

    ref, shape = ring_protocol.Ref("ring", barrier=False, num_arrivals=0), (2, 4, 8)
    assert ring_protocol._located(ref, [NDIndexer.from_indices_shape((0, 1), shape)]) == ("ring", 0, 1)
    gathered = ring_protocol._located(ref, [NDIndexer.from_indices_shape((0, np.asarray([0, 2], np.int32)),
                                                                         shape)])
    assert [ring_protocol._overlaps(gathered, ("ring", 0, slot)) for slot in range(4)] == [True, False, True,
                                                                                          False]
    assert not ring_protocol._overlaps(gathered, ("ring", 1))
    repeated = [NDIndexer.from_indices_shape((np.asarray([0, 0], np.int32),), (2, 2)),
                NDIndexer.from_indices_shape((1,), (2, 2))]
    assert ring_protocol._overlaps(ring_protocol._located(ref, repeated), ("ring", 0, 1))
    assert not ring_protocol._overlaps(ring_protocol._located(ref, repeated), ("ring", 1))


GATHERS = {
    "a row apart": ((2, 2), (([1],),), ("ring", 0), False),
    "its row": ((2, 2), (([1],),), ("ring", 1), True),
    "a cell apart": ((2, 2), ((0, [1]),), ("ring", 0, 0), False),
    "its cell": ((2, 2), ((0, [1]),), ("ring", 0, 1), True),
    "a picked cell apart": ((2, 2), (([1],), (0, 0)), ("ring", 1, 1), False),
    "its picked cell": ((2, 2), (([1],), (0, 0)), ("ring", 1, 0), True),
    "a rank-two pick apart": ((4,), (([[0, 1], [2, 3]],), (1, 0)), ("ring", 1), False),
    "its rank-two pick": ((4,), (([[0, 1], [2, 3]],), (1, 0)), ("ring", 2), True),
    "an empty gather": ((2, 2), (([],),), ("ring", 0), False),
    "an empty pick": ((2, 2), (([1],), ([],)), ("ring", 1), False),
}
"""Each gather cell's ref shape, its indexers in turn (a list an array of indices), the location the other
warpgroup writes, and whether that write shares memory with the gather's read."""


@pytest.mark.parametrize(("shape", "indexers", "written", "races"), GATHERS.values(), ids=GATHERS.keys())
def test_a_gather_races_only_what_it_reads(shape, indexers, written, races):
    """Two warpgroups, one reading a ref through a gather (`ref.at[[1]]`, its row 1; `ref.at[0, [1]]`, its
    cell `(0, 1)`; `ref.at[[1]].at[0, 0]`, the cell `(1, 0)` its view's entry picks; a `(4,)` ref's
    `ref.at[[[0, 1], [2, 3]]].at[1, 0]`, its element 2; `ref.at[[]]` and `ref.at[[1]].at[[]]`, nothing; its
    location `_located`'s) and the other writing `written` with no barrier between them: the explorer
    takes a write apart from what the gather reads as apart, and catches one of it unordered with the read
    (the base reviews of 71686face, F2: the row gather taken as the ref whole raced the other row; of
    8c1c9efca, F2: the cell gather taken as its row raced the other cell; of 31b5fe85b, F1: the picked cell
    taken as its row raced the other cell of it; of 8a5405c55, F1: the rank-two gather flattened missed its
    element, F2: an empty one raised)."""
    from jax._src.state.indexing import NDIndexer

    ref, transforms = ring_protocol.Ref("ring", barrier=False, num_arrivals=0), []
    for indexer in indexers:
        indices = tuple(np.asarray(index, np.int32) if isinstance(index, list) else index
                        for index in indexer)
        transforms.append(NDIndexer.from_indices_shape(indices, shape))
        shape = transforms[-1].get_indexer_shape()
    read = ring_protocol._located(ref, transforms)
    reader, writer = [Event("read", read)], [Event("write", written)]
    if not races:
        assert explore([reader, writer], {}) > 0
    else:
        with pytest.raises(Violation) as found:
            explore([reader, writer], {})
        assert found.value.kind == "unordered"


def test_a_kernels_gathered_reads_touch_what_they_read():
    """A kernel's reads of its shared tiles through gathers, as the checker's interpreter takes them
    (`_access`, `_located`), at a row the checker does not evaluate (one read from global memory, None to
    it): an empty gather of a `(3, 2)` tile, one of a `(0, 2)` tile and the `(3, 2)` tile's rows 0 and 2
    gathered then picked at that row and no column emit no read, touching no memory (the base reviews of
    8a5405c55, 839bb19de and 369d0ad7b, F2: one raised, the others read their tile whole); its rows 0 and 2
    gathered then picked at that row, or a one-row slice from it, and the first column read where the
    gather's rows are, either one's first cell, for the pick could be either, and never row 1, which the
    gather does not reach (of 839bb19de and 369d0ad7b, F1: each raised; of b0f3db894, F6: the tile whole
    passed)."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import mosaic_gpu as plgpu

    def body(rows_gmem, out_gmem, tile, empty):
        del out_gmem
        row = rows_gmem[0]
        rows = tile.at[2 * jnp.arange(2, dtype=jnp.int32)]
        tile.at[jnp.zeros((0,), jnp.int32)][...]
        empty.at[jnp.zeros((0,), jnp.int32)][...]
        rows.at[row, :0][...]
        rows.at[row, 0][...]
        rows.at[pl.ds(row, 1), 0][...]

    kernel = plgpu.kernel(body, out_type=jax.ShapeDtypeStruct((1,), jnp.float32),
                          scratch_types=[plgpu.SMEM((3, 2), jnp.float32), plgpu.SMEM((0, 2), jnp.float32)])
    (events,), _ = programs(kernel, jnp.zeros((1,), jnp.int32), threads=1)
    (gmem, *picks) = [event.at for event in events if event.kind == "read"]
    assert len(picks) == 2
    for picked in picks:
        (name,) = {block[0] for block in ring_protocol._each(picked)}
        assert gmem[0] != name
        assert [ring_protocol._overlaps(picked, (name, row, 0)) for row in range(3)] == [True, False, True]


@pytest.mark.parametrize("waits", [True, False], ids=["ordered", "unordered"])
def test_a_slot_under_a_roles_index_is_ordered_against_a_read_of_the_roles_whole(waits):
    """A write of a role's ring's slot (`[0, 1]` of a buffer laid `[role, slot]`) and a read of the role's
    every slot (`[0]`): one location under the other (`_located`'s leading indices, `_overlaps`), so the
    read is ordered after the write by the barrier its thread waits on, and caught where it waits on none.
    The role's other slot (`[0, 0]`) is a location apart from the write's, read unordered with it."""
    writer = [Event("write", ("ring", 0, 1)), Event("arrive", ("filled", 1))]
    apart = [Event("read", ("ring", 0, 0))]
    reader = [Event("wait", ("filled", 1), 0)] if waits else []
    reader.append(Event("read", ("ring", 0)))
    if waits:
        assert explore([writer, apart, reader], {"filled": 1}) > 0
    else:
        with pytest.raises(Violation) as found:
            explore([writer, apart, [*reader, Event("wait", ("filled", 1), 0)]], {"filled": 1})
        assert found.value.kind == "unordered"


def test_an_asynchronous_landing_aliasing_an_implicit_wait_is_caught():
    """An asynchronous commit landing past the completion an implicit wait awaits must raise `aliased` even
    when local events between the commit and wait advance in bulk."""
    barrier = ("B", 0)
    program = [
        Event("arrive", barrier), Event("wait", barrier, 0), Event("wait", barrier, 0),
        Event("arrive", barrier), Event("committed", barrier), Event("read", ("x", 0)),
        Event("wait", barrier, None), Event("wait", barrier, 2),
    ]
    with pytest.raises(Violation) as found:
        explore([program], {"B": 1})
    assert found.value.kind == "aliased"
