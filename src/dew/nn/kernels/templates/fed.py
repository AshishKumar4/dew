"""A ring family whose consumers feed their own rings, no producer beside them (`ConsumerFed`): the first
consumer issues every ring's loads between its own steps (`Feed`), every consumer walking them."""

import dataclasses
from collections.abc import Callable

import jax
from jax import lax
from jax.experimental import pallas as pl

from dew.nn.kernels.templates import resources
from dew.nn.kernels.templates.mosaic import Ring, RingFamily


@dataclasses.dataclass(frozen=True)
class Feed:
    """A walk's steps `lo` to `hi` (its indices, `walk`'s segment bounds) as a consumer-fed walk reads them
    (`ConsumerFed`), step `index` in the rings' slot of counter `index - first` (`walk`'s `first`, as
    `Slots` takes it): the loading consumer (`loads`, one warpgroup's traced flag) loads each step into
    each of `rings` from its view of `views` at `index - origin` (a plain walk's views take its own
    indices, as a producer's `stream` gives them: `origin` 0; `persist`'s a tile's steps counted from its
    first), the first steps before the walk (`primed`) and each later one once the slot it reuses is given
    back
    (`refilled`, at the given step's counter): by the walk after each step's release, or where `refills`
    is False by the template, at a point of its own (a walk whose turns order its consumers refills past
    the turn that lets the other finish); the loading consumer waits at the walk's end for every slot's
    last release where it `drains` (a plain walk's; `persist` drains after a program's last tile)."""
    rings: dict[str, Ring]
    views: dict
    lo: jax.Array
    hi: jax.Array
    loads: jax.Array
    first: jax.Array | int = 0
    origin: jax.Array | int = 0
    refills: bool = True
    drains: bool = False

    @property
    def stages(self) -> int:
        return min(ring.stages for ring in self.rings.values())

    def load(self, index) -> None:
        for name, ring in self.rings.items():
            view = ring.viewed(self.views[name], index - self.origin)
            ring.load(ring.turn(index - self.first), view)

    def primed(self) -> None:
        """The walk's first steps, a slot each, loaded."""
        for ahead in range(self.stages):
            pl.when(self.loads & (self.lo + ahead < self.hi))(lambda ahead=ahead: self.load(self.lo + ahead))

    def refilled(self, counter) -> None:
        """The step at ring counter `counter`, its slots given back, has them take the walk's step a ring
        ahead where it has one: its load waits until every consumer has given them back (`Ring.load`)."""
        ahead = counter + self.first + self.stages
        pl.when(self.loads & (ahead < self.hi))(lambda: self.load(ahead))

    def drained(self) -> None:
        """The loading consumer's wait for the release of each slot the walk filled last (`Ring.drain`), so
        no slot is reused or freed while another consumer reads it."""
        @pl.when(self.loads)
        def _():
            for ring in self.rings.values():
                ring.drain(self.hi - self.first)


class ConsumerFed[Accumulator](RingFamily[Accumulator]):
    """A ring family's layout without a producer (`RingFamily`'s otherwise), an instruction family's bundle
    taking it first among its bases (`mosaic_sm90.WgmmaFed`): its
    consumers alone run, the first issuing every ring's loads between its own steps (`Feed`; Mosaic GPU
    elects one of its threads to issue each copy), so each holds its program's entry registers, which no
    raise moves (`resources.entry_registers`: two consumers' 255, where a producer's third warpgroup leaves
    them 168 while the GPU's compile ignores setmaxnreg, `resources.RAISES_HONORED`). SETMAXNREG-FFI3-h100:
    attention's dK/dV at two consumers so, 0.732 ms, against 1.588 ms with a producer at 168 registers,
    spilling. Its streams run through `persist`, or a plain walk over a `Feed` (`fed`) every consumer walks
    whole (no `passed`: a consumer's steps are every consumer's); it takes no cluster (a multicast ring's
    loads are each program's producer's)."""
    cluster = 1
    feeds_itself = True
    gathers_rows = False  # a consumer's `Feed` loads its rings, no producer warpgroup to gather a step's rows
    loader = 0
    """The consumer that issues the loads."""

    def registers(self, consumers: int, programs: int = 1) -> int:
        return resources.entry_registers(consumers, programs)


    def warpgroups(self, consumers: int) -> int:
        return consumers

    def roles(self, consumers: int, consumer, producer):
        # Every consumer runs one program: its value the program's, for a scope after the walk's.
        del consumers, producer
        return consumer(lax.axis_index("wg"))

    def stream(self, streams, views: dict, lo, hi, *, adds: tuple | None = None) -> None:
        raise NotImplementedError(f"{self.name}'s consumers feed their own rings (`fed`)")

    def passed(self, streams, lo, hi, *, first: jax.Array | int = 0, between=None) -> None:
        raise NotImplementedError(f"{self.name}'s consumers each walk every step of their rings (`fed`)")

    def walk_own[R](self, streams, views: dict, own: tuple, streamed: tuple, wg,
                    walk: Callable[..., R], *, between=None) -> R:
        # Every consumer walks each streamed step over its own feed: no partner reads a step it skips.
        del own, between
        lo, hi = streamed
        return walk(self.fed(streams, views, lo, hi, wg), lo, hi, lo)

    def fed(self, streams: dict[str, Ring], views: dict, lo, hi, wg, *,
            refills: bool = True) -> "dict[str, Ring] | Feed":
        """`RingFamily.fed`: a `Feed` of steps `lo` to `hi` from `views`, its ring counters from `lo` (the
        walk's `first`), its first steps primed by the loading consumer, drained at its walk's end."""
        feed = Feed(streams, views, lo, hi, wg == self.loader, first=lo, refills=refills, drains=True)
        feed.primed()
        return feed

    def walk[C](self, streams, segments, start: C, shapes: tuple, interpret: bool, *,
                first: jax.Array | int = 0, in_flight: int = 0, initial: tuple | None = None,
                unrolled: bool = False, held_past: tuple[str, ...] = (), finish: Callable | None = None
                ) -> tuple[C, tuple[jax.Array, ...]]:
        if not isinstance(streams, Feed):
            raise NotImplementedError(f"{self.name}'s consumers feed their own rings (`fed`)")
        # The walk reads each step at the ring counter its feed loads it at, both from the feed's `lo`.
        first = streams.first
        if in_flight and len(segments) > 1:
            # A segment's last step's release waits for the walk's end, which the feed's next segment, its
            # loads behind that release, would never reach.
            raise NotImplementedError(f"{self.name}'s fed walk keeps no products in flight across segments")
        walked = self._walked(streams.rings, segments, start, shapes, interpret, first=first,
                              in_flight=in_flight, initial=initial, unrolled=unrolled,
                              freed=streams.refilled if streams.refills else None, held_past=held_past,
                              finish=finish)
        if streams.drains:
            streams.drained()
        return walked

    def _tile_streams(self, rings: dict[str, Ring], counter, steps, views: dict,
                      wg) -> "dict[str, Ring] | Feed":
        # The tile's steps as its loading consumer feeds them, the first ones primed.
        feed = Feed(rings, views, counter, counter + steps, wg == self.loader, origin=counter)
        feed.primed()
        return feed

    def _ended(self, rings: dict[str, Ring], end, wg) -> None:
        # The loading consumer has seen every release but the last ring's worth: it waits for those too, so
        # no slot is reused or freed while another consumer reads it, and every release is seen.
        @pl.when(wg == self.loader)
        def _():
            for ring in rings.values():
                ring.drain(end)
