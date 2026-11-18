"""What a Mosaic GPU program does past its walks: where tiles it scopes after them lie (`scoped`: a scope
of their own, the program's, or over its drained rings' slots), and a second walk its consumers decide to
take over the same steps (`Decided`: a decode's scaled walk where a row's sums passed fp32's largest)."""

import dataclasses
import functools
import operator
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn import derivatives
from dew.nn.kernels.templates import across, bodies


def scoped(family, body, allocations: tuple, streams: dict, after_walks: tuple, *, over_rings: bool):
    """`family.scoped`'s `body` with tiles `after_walks` no walk reads, which `body` takes as `after`:
    `after(fn)` runs `fn(*refs)` of them once a role's walks have ended, in a scope of their own where the
    family scopes its walks apart (`Family.walks_scoped`: the walks' slots reused, as
    `resources.Scope.after` counts them), over its rings' slots where the launch asks it (`over_rings`) and
    the family stages so (`over_rings_scoped`), the program's otherwise."""
    if over_rings and family.stages_over_rings and not family.walks_scoped:
        return over_rings_scoped(family, body, allocations, streams, after_walks)
    if family.walks_scoped:
        # The after-walks scope is the program's own kind (`program_scoped`: collective where its
        # consumers share one).
        def after(fn):
            return family.program_scoped(lambda *refs, streams: fn(*refs), *after_walks, streams=streams)

        return family.program_scoped(functools.partial(body, after=after), *allocations, streams=streams)
    held = len(allocations)

    def program(*refs, streams):
        return body(*refs[:held], streams=streams, after=lambda fn: fn(*refs[held:]))

    return family.program_scoped(program, *allocations, *after_walks, streams=streams)


def over_rings_scoped(family, body, allocations: tuple, streams: dict, after_walks: tuple):
    """`scoped` where the tiles after the walks share the rings' slots (`Family.stages_over_rings`): every
    ring's slots and those tiles one union (as large as the larger, `resources.Scope.over`), the rings'
    barriers apart, and `body` given `after` and `walked`. The order a tile is written in: a consumer's last
    product reads its slot (mma.sync's loads are its own; a warpgroup MMA's awaited before its accumulator
    finishes), then it arrives at `walked`'s barrier, of as many arrivals as a ring's slot has readers; a
    tile is written only after every consumer's arrival, so after every read of every slot, and no load
    lands past a consumer-fed walk's last step. Each consumer passes `walked` once, before its first write
    `after`. A program of one tile alone: a persistent launch (`Transport.persist`) would prime its next
    tile's loads into the union, which no barrier here orders after the stores' reads of its tiles. This
    order is the union's proof obligation, which an interpreted cell witnesses once the interpreter's race
    detection takes a fed ring (row 7853, post110)."""
    names = tuple(streams)
    slots = tuple(streams[name]["smem"] for name in names)
    rest = {name: {key: value for key, value in streams[name].items() if key != "smem"} for name in names}
    readers = streams[names[0]]["consumed"].num_arrivals
    held = len(allocations)

    def program(*refs, streams):
        (ring_slots, staged), passed = refs[held], refs[held + 1]
        rings = {name: {**streams[name], "smem": slot} for name, slot in zip(names, ring_slots, strict=True)}

        def walked():
            plgpu.barrier_arrive(passed)
            plgpu.barrier_wait(passed)

        return body(*refs[:held], streams=rings, after=lambda fn: fn(*staged), walked=walked)

    union = plgpu.RefUnion(slots, tuple(after_walks))
    return family.program_scoped(program, *allocations, union, plgpu.Barrier(num_arrivals=readers),
                                 streams=rest)


@dataclasses.dataclass(frozen=True)
class Decided:
    """A walk its program's consumers decide to take again, over the same steps, after the walk before it
    (`Transport.stream`'s `again`, `Family.walk_own`'s `counted`): each consumer's verdict a word of its own
    (`words`, `Transport.decided_scratch`'s) published at a barrier of the consumers' arrivals, which the
    producer and every consumer wait on, so all of them read the same words and walk again, or none does.
    A program of one warpgroup, which no producer serves (`ConsumerFed`'s one consumer, cp.async's), decides
    alone: its verdict its own, no word or barrier (`words` None). A program of one tile alone: its words
    are written once a launch, and a persistent launch's next tile would write them before a slower reader
    read this tile's (`Walk.persisted` takes no second walk)."""

    words: Any
    barrier: Any
    consumers: int

    def stated(self, wg, verdict: jax.Array) -> jax.Array:
        """Consumer `wg`'s `verdict` (a scalar, uniform over its warpgroup) published, and whether any
        consumer's takes the walk again (`taken`)."""
        if self.words is None:
            return _rare(_uniform(verdict.astype(jnp.int32)) != 0)
        self.words[wg] = verdict.astype(jnp.int32)
        plgpu.barrier_arrive(self.barrier)
        return self.taken()

    def taken(self) -> jax.Array:
        """Whether some consumer's published verdict takes the walk again, read once every consumer's is: a
        rare verdict (`derivatives.Rare`), which a count of the kernel's work reads as not taken."""
        plgpu.barrier_wait(self.barrier)
        published = (self.words[index] for index in range(self.consumers))
        return _rare(_uniform(functools.reduce(operator.or_, published)) != 0)


_rare = derivatives.marked(derivatives.Rare(), lambda verdict: verdict)
"""A decided walk's verdict marked rare (`derivatives.Rare`)."""


def _uniform(verdict: jax.Array) -> jax.Array:
    """A verdict every lane holds alike read warp-uniform (`across.uniform`), so a branch on it is not
    divergent and the copies issued under it address their tiles once, not lane by lane (row 7898); the
    interpreter's its own."""
    return verdict if bodies.interpreting() else across.uniform(verdict)
