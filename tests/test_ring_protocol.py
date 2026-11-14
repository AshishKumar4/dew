"""The Mosaic GPU toolkit's ring protocol model-checked (`ring_protocol`): the sm90 ring family's persistent
walk (`mosaic.RingFamily.persist`: its producer's loads at a counter running across tiles, its drains, its
consumers' walks with their slots acquired and released, a step's products in flight or not, a rotation
unrolled or a step a trip), traced as the toolkit builds it and explored over every interleaving, within the
bounds below: no deadlock, no aliased parity, no early arrival, no read before its write lands, no overwrite
before its read. And the checker catches each class it names: a flipped parity, a missing arrival, an early
release, each a counterexample trace."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import pytest
import ring_protocol
from ring_protocol import Event, Violation, explore, programs

from dew.nn.kernels.templates import mosaic_sm90

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
           streams: tuple = ("a",), pingpong: bool = False, empty_active: bool = False):
    """The ring family's persistent walk at the configuration over a ring of each of `streams`, each step
    reading its slot of each, its two consumers taking alternate tiles whole where `pingpong` (each slot
    released by its tile's owner alone), a tile of no steps active where `empty_active` (its turn taken, its
    walk none): each warpgroup's program and the barriers' arrivals (`ring_protocol.programs`)."""
    family = mosaic_sm90.WGMMA
    starts = [sum(tiles[:index]) for index in range(len(tiles))]

    def at(tile, values):
        # A tile's value of `values`, scalars alone (a kernel closes over no array).
        return sum(jnp.where(tile == index, jnp.int32(value), jnp.int32(0))
                   for index, value in enumerate(values))

    def body(x_gmem, out_gmem):
        def scope(*, streams):
            def schedule(index):
                tile = index[0]
                steps = at(tile, tiles)
                views = {name: (x_gmem, lambda step: (at(tile, starts) + step, 0)) for name in streams}
                return (steps > 0) | empty_active, steps, views, tile

            def issue(wg, tile, walk, *, active):
                def step(index, slots, carry, accumulators):
                    for name in streams:
                        slots[name][0, 0]
                    return carry, accumulators

                walk(step, None, ())

            def finish(*_, **__):
                pass

            family.persist(streams, (len(tiles),), schedule, issue, finish, consumers=consumers,
                           axis="programs", outputs=(), interpret=False, in_flight=in_flight,
                           unrolled=unrolled, pingpong=pingpong)

        family.scoped(scope, streams=family.streams(dict.fromkeys(streams, (BLOCK, jnp.bfloat16)), stages,
                                                    1 if pingpong else consumers))

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
