"""A program's roles and the buffers between them (`Schedule`): its warpgroup roles (`Role`), each its share
of the registers, and the hand-offs between them (`HandOff`: tensor-core signals, tensor-memory accumulators),
whose barriers count the arrivals of the roles that fill and give back each."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import generation
from dew.nn.kernels.templates import resources
from dew.nn.kernels.templates.bodies import RESIDENT


@dataclasses.dataclass(frozen=True)
class Role:
    """One of a program's roles (`Schedule`): `count` units, warpgroups or (`warp`) warps of the program's
    last warpgroup, each running the body `Schedule.run` binds to `name`; `weight` its warpgroups' share of
    the registers the program's raises divide (`resources.role_registers`), 0 lowering them to setmaxnreg's
    least (the last warpgroup's). Warp roles share their warpgroup, so its least too."""
    name: str
    count: int = 1
    warp: bool = False
    weight: int = 1

    @property
    def arrivals(self) -> int:
        """A barrier's arrivals this role makes at it in a turn: one a warpgroup, where a role's threads
        arrive; one, where a warp's copy or a product's commit does."""
        return 1 if self.warp else self.count


@dataclasses.dataclass(frozen=True)
class HandOff:
    """A buffer between a program's roles (`Schedule`): `shape`, an allocation that states its barriers'
    arrivals (`arrived(produced, released)`, then `scratch(target)` and `opened(*refs)`: tensor-core
    signals, tensor-memory accumulators), filled by the role `producer` and given back by the roles
    `released_by` (none for a signal, which is only filled)."""
    name: str
    shape: Any
    producer: str
    released_by: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class Schedule:
    """A program's roles and the buffers between them: its warpgroup roles in turn, each its `count`
    warpgroups, its register budget raised to its weight's share (`registers`); then one warpgroup at
    setmaxnreg's least, which holds either a role of weight 0 (a register family's producer) or the warp
    roles (tcgen05's load and product warps, each a warp of it). A program of one role runs it on every
    warpgroup and moves no registers. Its hand-offs' barriers count arrivals by the roles that fill and give
    back each (`HandOff`), so no template states an arrival count."""
    roles: tuple[Role, ...]
    handoffs: tuple[HandOff, ...] = ()
    axis: str | None = "wg"
    """The axis a program's warpgroups index along (`kernel`'s); None for a program with none (an sm80
    program's, a TPU core's), whose one role runs at index 0."""

    def __post_init__(self):
        raised = [role for role in self.roles if role.weight]
        lowered = [role for role in self.roles if not role.weight]
        if raised and lowered and not (all(role.warp for role in lowered)
                                        or (len(lowered) == 1 and lowered[0].count == 1)):
            raise ValueError(f"a schedule's last warpgroup holds one role of weight 0 or its warp roles, not "
                             f"{[role.name for role in lowered]}")
        if any(role.warp and role.weight for role in self.roles):
            raise ValueError("a warp role shares its warpgroup's least registers (weight 0)")
        if self.roles[:len(raised)] != tuple(raised):
            raise ValueError("a schedule's raised roles come before its last warpgroup's")

    @property
    def raised(self) -> tuple[Role, ...]:
        return tuple(role for role in self.roles if role.weight)

    def role(self, name: str) -> Role:
        return next(role for role in self.roles if role.name == name)

    def warpgroups(self) -> int:
        """The warpgroups a program of this schedule runs (`kernel`'s `warpgroups`)."""
        return sum(role.count for role in self.roles if not role.warp) + any(role.warp for role in self.roles)

    def registers(self, programs: int = 1) -> dict[str, int]:
        """Each role's registers a thread, `programs` programs held by a multiprocessor at once
        (`resources.role_registers`); a warp role's its warpgroup's."""
        weights = tuple(role.weight for role in self.raised for _ in range(role.count))
        budgets = resources.role_registers(weights + (0,) * (len(self.raised) < len(self.roles)), programs)
        held, first = {}, 0
        for role in self.roles:
            held[role.name] = budgets[first] if role.weight else budgets[-1]
            first += role.count if role.weight else 0
        return held

    def stated(self) -> tuple:
        """Each hand-off's shape with its barriers' arrivals stated by its roles (`HandOff`), in turn."""
        return tuple(handoff.shape.arrived(self.role(handoff.producer).arrivals,
                                           sum(self.role(name).arrivals for name in handoff.released_by))
                     for handoff in self.handoffs)

    def scratch(self, target: generation.GpuTarget) -> tuple:
        """Each hand-off's allocations (`stated`), in turn."""
        return tuple(shape.scratch(target) for shape in self.stated())

    def opened(self, refs: Sequence) -> Mapping[str, object]:
        """Each hand-off opened by name over its scoped `refs` (`scratch`'s, in turn)."""
        return {handoff.name: shape.opened(*(ref if isinstance(ref, tuple) else (ref,)))
                for handoff, shape, ref in zip(self.handoffs, self.stated(), refs, strict=True)}

    def run[R](self, **bodies: Callable[..., R]) -> R | None:
        """Each role's body by its name: on a schedule of one role, `body(index)` on every warpgroup, its
        index along `axis` (the value it returns, a program's whose consumers keep theirs); otherwise each
        raised role's `body(index)` on its warpgroups, its index among them, their registers raised to its
        share at the launch's residency (`registers`, `bodies.RESIDENT`), then the last warpgroup's, lowered
        to the least, its role's `body()` or each warp role's on its warp."""
        if set(bodies) != {role.name for role in self.roles}:
            raise ValueError(f"a schedule of {[role.name for role in self.roles]} given bodies for "
                             f"{sorted(bodies)}")
        if len(self.roles) == 1:
            role = self.roles[0]
            return bodies[role.name](0 if self.axis is None else lax.axis_index(self.axis))
        wg = lax.axis_index(self.axis)
        budgets = self.registers(RESIDENT.get())
        first = 0
        for role in self.raised:
            # The first role's warpgroups are those below its count, each its own index: no bound and no
            # shift is traced for it, so a one-role table builds the kernel a producer beside consumers
            # always built (an attention forward's spill-free registers on an H100 turn on it,
            # test_attention_mosaic's compile gate).
            within = wg < first + role.count if first == 0 else (wg >= first) & (wg < first + role.count)

            @pl.when(within)
            def _(role=role, first=first):
                plgpu.set_max_registers(budgets[role.name], action="increase")
                bodies[role.name](wg if first == 0 else wg - first)

            first += role.count
        lowered = [role for role in self.roles if not role.weight]
        if not lowered:
            return None

        @pl.when(wg == first)
        def _():
            plgpu.set_max_registers(budgets[lowered[0].name], action="decrease")
            if not lowered[0].warp:
                bodies[lowered[0].name]()
                return

            @plgpu.warp_map
            def _(warp):
                for index, role in enumerate(lowered):
                    pl.when(warp == index)(bodies[role.name])
        return None
