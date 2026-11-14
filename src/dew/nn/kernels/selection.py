"""Kernel selection: one selector for every family of ops (`select`), over a typed description of
the call (the op's case, the `C` its `Op[C]`, kernels and requirements share, so a requirement of
another op's case fails the type check) and what each kernel declares it takes (its `SUPPORTS`, ordered
`Requirement`s), the one place a kernel's eligibility is written. A kernel's requirements are evaluated in
their order and the selector stops at the first that refuses (`Kernel.refused`): a requirement may plan the
call every earlier one admits, so none runs on a call an earlier one ruled out.

An op states its preference: its own rule over the call (`Op.preferred`), else the first of its kernels
whose requirements hold, in its fixed order; a cell a measurement found another kernel faster on
(`Op.declared`: the receipt beside it) runs that kernel. The first kernel
whose requirements all hold takes the call, and the route names the preferred kernel's first failed
requirement as its refusal (`generation.route`). A kernel the caller forces
(`generation.chosen`, or a layer's named implementation) that does not take the call raises, naming its
first refusal and every one on request (`KernelRefused.refusals`), as torch's `sdpa_kernel` lists them; a
generic kernel forced so hands the call to the next generic one.

Selection reads only the call's static facts and the target's while a program traces: it adds no
operation to the program, and outside `generation.routes()` it records nothing
(`dew._cache.KERNEL_ROUTES` stays `()`), so a run's trace keys are those of a run that never selects.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Hashable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .generation import Choice, Route


@dataclasses.dataclass(frozen=True)
class Refusal:
    """Why the kernel a target prefers does not take a call, from the call's own facts on that
    target (its shapes, dtypes, mask): a stable `code`, which a kernel's `SUPPORTS` keys its declared
    limits on, and the `detail` a log names. What the host cannot lower is no refusal (`Unlowerable`)."""

    code: str
    detail: str
    row: str | None = None
    """The roadmap or ledger row that tracks lifting the limit (`dew.nn.kernels.selection.Requirement`),
    `INHERENT` where no implementation of the kernel lifts it, None for a refusal no requirement gives
    (an analysis's own, which its op's requirements read)."""

    kind: str = 'capability'
    """A capability, measured preference, or structural handoff, as the selector treats it."""

    def __str__(self) -> str:
        return self.detail


@dataclasses.dataclass(frozen=True, init=False)
class Requirement[C]:
    """One fact a kernel needs of a case `C`, the description its op's selector reads: its `analysis` of a
    case, the refusal a call that fails it gets (its `code` and the detail a log names), or None where the
    case holds it (`generation`'s facts, honouring `assumed`). A limit the code could lift names the `row`
    of the roadmap or ledger that tracks lifting it; one no implementation of the kernel lifts names
    `INHERENT`; every refusal the requirement reports carries them (`check`). Its `kind` is 'capability',
    what the kernel cannot compute; 'measured', where the target measured another kernel faster or never
    qualified this one, which a forced choice overrides; or 'handoff', a call structurally outside what the
    kernel applies to, which it hands to the op's next kernel even when forced (a generic kernel's to the
    next generic one, a join's projections run apart, a decode token with nothing to fuse runs the step
    alone).

    Built from a predicate, `Requirement(code, holds, detail, row, kind)`: it holds where `holds(case)` does,
    its refusal `code` with `detail` (text, or a function of the case); or from an analysis that finds its
    own refusal (`analysed`)."""

    code: str
    analysis: Callable[[C], Refusal | None]
    row: str
    kind: str

    def __init__(self, code: str, holds: Callable[[C], bool] | None = None,
                 detail: str | Callable[[C], str] = "", row: str = "", kind: str = 'capability', *,
                 analysis: Callable[[C], Refusal | None] | None = None) -> None:
        if not row:
            raise ValueError(f"requirement {code!r} names the row that tracks lifting it, or INHERENT")
        if kind not in ('capability', 'measured', 'handoff'):
            raise ValueError(f"requirement {code!r} is of kind {kind!r}, not capability, measured or handoff")
        if holds is not None and analysis is None:
            def predicate(case: C, holds: Callable[[C], bool] = holds) -> Refusal | None:
                return None if holds(case) else Refusal(code, detail(case) if callable(detail) else detail)

            analysis = predicate
        elif (holds is None) == (analysis is None):
            raise ValueError(f"requirement {code!r} is a predicate `holds` or an `analysis`, one of them")
        for name, value in (("code", code), ("analysis", analysis), ("row", row), ("kind", kind)):
            object.__setattr__(self, name, value)

    def holds(self, case: C) -> bool:
        """Whether `case` meets the requirement."""
        return self.analysis(case) is None

    def check(self, case: C) -> Refusal | None:
        """The refusal `case` gets, under the requirement's row and kind, or None when it holds."""
        found = self.analysis(case)
        return None if found is None else dataclasses.replace(found, row=self.row, kind=self.kind)


def analysed[C](code: str, analysis: Callable[[C], Refusal | None], row: str,
                kind: str = 'capability') -> Requirement[C]:
    """The requirement an analysis decides (a mainloop's reason it declines a pass,
    `attention_route.MAINLOOP`): it reports the analysis's own refusal, its code and detail, under its row
    and kind. A function, not a method of `Requirement`: its case in a return position would make the
    requirement's case invariant, and a requirement of a protocol's case (`recurrence_chunk.Templated`)
    would no longer serve its ops'."""
    return Requirement(code, row=row, kind=kind, analysis=analysis)


@dataclasses.dataclass(frozen=True)
class Kernel[C]:
    """A kernel an op may run: `supports`, its requirements in the order a refusal names them, whether it is
    `generic`, a path that takes what the others turn down, and whether it is `reverse_only`: it defines its
    reverse-mode derivative alone (a `jax.custom_vjp`), its backward traced after the call, so forward mode
    takes its tangents from another path computing the same function
    (`dew.nn.attention.forward_differentiable`); the others are JAX's own products, or state their own
    tangents, which forward mode differentiates as written."""

    name: str
    supports: tuple[Requirement[C], ...]
    generic: bool = False
    reverse_only: bool = False

    def refusals(self, case: C, forced: bool = False) -> Iterator[Refusal]:
        """The refusals `case` gets of the kernel's requirements, in their order, each requirement evaluated
        only as the next refusal is asked for (forcing passes over measured preferences): a selector asks
        for the first (`refused`), as a requirement may plan the call its predecessors admit (a GEMM's split
        derivation past its rows, a template's eligibility past 'forked'); a report asks for as many as it
        names."""
        for requirement in self.supports:
            if forced and requirement.kind == 'measured':
                continue
            if (refusal := requirement.check(case)) is not None:
                yield refusal

    def refused(self, case: C, forced: bool = False) -> Refusal | None:
        """The first refusal `case` gets (`refusals`), no requirement after it evaluated; None where every
        requirement holds."""
        return next(self.refusals(case, forced), None)


@runtime_checkable
class Celled(Protocol):
    """A case an op with declared cells reads (`Op.declared`): its `cell`, the scope a receipt measures
    (a template's instruction family, `generation.family`, None outside every family; a standalone
    kernel's device generation, `generation.device_generation`) and the kind of call it is, the object
    its op states it by (a GEMM's `grouped_matmul.Kind`, a norm's rows' dtype, a recurrence's product)."""

    @property
    def cell(self) -> tuple[str | None, Hashable]: ...


@dataclasses.dataclass(frozen=True)
class Op[C]:
    """A family of ops one selector serves: its `choice` (the name `chosen` and routes use), its `kernels`
    in the order the selector tries them after the preferred one, `preferred(case)` the op's own rule for
    the kernel it prefers (else the first of its kernels that takes the case, `select`), `unlowered(kernel)`
    why this host cannot lower the target's kernel, None where it lowers it (`Unlowerable`), and
    `declared`: the cells where
    a measurement found another kernel faster than the op's preference (a template's old route, owner,
    bible da64ae5; a standalone kernel against XLA's program), each cell (`Celled.cell`: the scope the
    receipt measured, a template's instruction family or a device's generation, and the kind of call) to
    that kernel (one of `kernels`, the object itself) and the receipt. A declared cell runs its kernel where
    it takes the call; any other call, and a declared one its kernel cannot run, the op's order."""

    choice: Choice
    kernels: tuple[Kernel[C], ...]
    preferred: Callable[[C], str] | None = None
    unlowered: Callable[[str], str | None] = lambda kernel: None
    # A cell's kind is its op's own object (`Celled.cell`); a mapping's keys are invariant, so each op's
    # table states its own kind and the field takes any.
    declared: Mapping[tuple[str | None, Any], tuple[Kernel[C], str]] = dataclasses.field(
        default_factory=dict, compare=False, hash=False)
    named_from: Mapping[tuple[Choice, str], str] = dataclasses.field(
        default_factory=dict, compare=False, hash=False)
    """Another op's named choice that names this one's kernel, unless this call or op names its own.
    Entries are ordered, as the op's requirements and candidates are."""

    def __post_init__(self) -> None:
        # Each declared cell runs one of the op's kernels, in a scope a device generation or its instruction
        # family states: one that names another, or no scope, matches no call.
        from . import generation

        scopes = {*generation.TARGETS, *map(generation.family, generation.TARGETS)}
        for (scope, _), (kernel, _) in self.declared.items():
            if kernel not in self.kernels or scope not in scopes:
                raise ValueError(f"{self.choice} declares {kernel.name!r} in {scope!r}, which matches no "
                                 f"call: its kernels are {', '.join(each.name for each in self.kernels)}")

    def kernel(self, name: str) -> Kernel[C]:
        for kernel in self.kernels:
            if kernel.name == name:
                return kernel
        raise ValueError(f"{self.choice} has no kernel {name!r}; its kernels are "
                         f"{', '.join(kernel.name for kernel in self.kernels)}")

    def bound[**P, R](self, call: Callable[P, R]) -> Callable[P, R]:
        """Keep this op's named choice when `call` runs later, such as a pullback outside its forward's
        scope. With no named choice, the later call keeps deciding in the scope where it runs."""
        requested = _named(self, None)
        if requested is None:
            return call

        @functools.wraps(call)
        def scoped(*args: P.args, **kwargs: P.kwargs) -> R:
            from . import generation

            with generation.chosen(**{self.choice: requested}):
                return call(*args, **kwargs)

        return scoped


class Unlowerable(RuntimeError):
    """A kernel the target runs that this host cannot lower for it (`Op.unlowered`: a step traced for an
    assumed GPU on a host without the kernel's library): no kernel lowers in its place, as the program would
    not be the GPU's (row 4616), so a compile ahead refuses the program, naming it. The op's `choice`, the
    `kernel`, `why`, and the `program` a compile ahead refuses for it (`refusing`)."""

    def __init__(self, choice: str, kernel: str, why: str, program: str | None = None) -> None:
        super().__init__(f"{choice} kernel {kernel!r} does not lower on this host: {why}"
                         + ("" if program is None else f" (program {program!r} not compiled ahead)"))
        self.choice, self.kernel, self.why, self.program = choice, kernel, why, program

    def refusing(self, program: str) -> Unlowerable:
        """This refusal as a compile ahead states it: `program`, the program it refuses."""
        return Unlowerable(self.choice, self.kernel, self.why, program)


class KernelRefused(ValueError):
    """A forced kernel that does not take a call: the op's `choice`, the `kernel` chosen, its first
    refusal, which its message names, and the rest of the kernel's refusals as `refusals` reads them."""

    def __init__(self, choice: str, kernel: str, first: Refusal, rest: Iterator[Refusal]) -> None:
        super().__init__(f"{choice} kernel {kernel!r} was chosen and does not take this call: "
                         f"{first.code}: {first.detail}")
        self.choice, self.kernel, self.first = choice, kernel, first
        self._rest, self._refusals = rest, None

    @property
    def refusals(self) -> list[Refusal]:
        """Every refusal of the call, in the kernel's order: its first, then the requirements after it,
        evaluated on this first request (`Kernel.refusals`)."""
        if self._refusals is None:
            self._refusals = [self.first, *self._rest]
        return self._refusals


def forced(choice: Choice) -> str | None:
    """The kernel `generation.chosen` names for `choice` within its block, or None."""
    from dew import _cache

    return dict(_cache.KERNEL_CHOICES.value).get(choice)


def _named(op: Op, named: str | None) -> str | None:
    if (requested := named or forced(op.choice)) is not None:
        return requested
    return next((kernel for (parent, choice), kernel in op.named_from.items() if forced(parent) == choice),
                None)


def select[C](op: Op[C], case: C, named: str | None = None, *, preference: tuple[str, ...] = ()) -> Route:
    """The kernel `op` runs for `case` on the target, reported as its route: `named` (or the kernel
    `chosen` names) where it takes the case, else the op's preference (`preferred`, or the first of its
    kernels that takes the case), else the first of the op's kernels that does. A named kernel that does
    not take the case raises with its first refusal, its measured requirements aside, unless that refusal
    is a handoff, whose call it hands to the op's next kernels (a generic one to the next generic
    ones). A call's soft `preference`, where neither `named` nor `chosen` forces a kernel, is its own order
    of the op's kernels: the first of them that takes the case, the first passing what the targets measured
    of it as a named kernel does, and its refusal reported (a norm's per-call implementation, an explicit
    xla's decode step). `Op.named_from` declares a named dependency where neither the call nor this op
    names its own kernel. The route states what forced it (`Route.forced`). A kernel this host cannot lower
    for the target raises (`Unlowerable`)."""
    from . import generation

    kernel, preferred, refusal, insisted = _decided(op, case, named, preference)
    return generation.route(op.choice, kernel=_lowered(op, kernel).name, preferred=preferred,
                            refusal=refusal, forced=insisted)


def selected[C](op: Op[C], case: C, named: str | None = None, *, preference: tuple[str, ...] = ()
                ) -> Kernel[C]:
    """The kernel `select` runs for `case`, reporting nothing: what a caller reads ahead of the dispatch
    that reports it (the kernel a call that built its mask names, a projection's choice between XLA's
    program and the GEMM template's launches, which select their own), raising as `select` does. A
    declared cell's kernel is read without the kernel the op would prefer there, which only a report names
    (`_old_route`): deriving that one's plan for no report cost a decode's trace a second a projection."""
    return _lowered(op, _decided(op, case, named, preference, reported=False)[0])


def _lowered[C](op: Op[C], kernel: Kernel[C]) -> Kernel[C]:
    """`kernel`, which this host lowers for the target, else `Unlowerable` (`Op.unlowered`)."""
    if (why := op.unlowered(kernel.name)) is not None:
        raise Unlowerable(op.choice, kernel.name, why)
    return kernel


def _decided[C](op: Op[C], case: C, named: str | None, preference: tuple[str, ...], *,
                reported: bool = True) -> tuple[Kernel[C], str, Refusal | None, str | None]:
    """`select`'s kernel for `case`, the kernel preferred, the preferred one's refusal and the kernel that
    forced the choice (the call, its own `chosen`, then `named_from`), None where nothing did; a declared
    cell's preference only where `reported` (`selected` reads the kernel alone)."""
    insisted = _named(op, named)
    soft = insisted is None and bool(preference)
    if soft:
        # The call's own order, exactly: its first the call's pick, the rest its fallbacks in turn.
        preferred = preference[0]
        candidates = [op.kernel(name) for name in preference]
        first = candidates[0]
    else:
        old = None if insisted is not None else _old_route(op, case, reported=reported)
        if old is not None:
            return (*old, None)
        preferred = insisted or _preferred(op, case)
        first = op.kernel(preferred)
        # An insisted kernel hands a call a handoff turns down first, to the op's next kernels in turn; a
        # generic one to the next generic ones.
        handing = insisted is not None and not first.generic
        rest = op.kernels[op.kernels.index(first) + 1:] if handing else op.kernels
        candidates = [first, *(kernel for kernel in rest if kernel is not first
                               and (insisted is None or not first.generic or kernel.generic))]
    refusal = None
    for kernel in candidates:
        # A kernel a call names or picks passes what the targets measured of it; its requirements are
        # evaluated up to the first that refuses, which decides (a handoff's call is outside the kernel, so
        # it goes on to the next).
        refusals = kernel.refusals(case, forced=insisted is not None or (soft and kernel is first))
        found = next(refusals, None)
        if found is None:
            return kernel, preferred, refusal, insisted
        if kernel is first:
            refusal = found
            if insisted is not None and found.kind != 'handoff':
                raise KernelRefused(op.choice, kernel.name, found, refusals)
    raise ValueError(f"no {op.choice} kernel takes this call ({refusal})")


def _old_route[C](op: Op[C], case: C, *, reported: bool = True
                  ) -> tuple[Kernel[C], str, Refusal | None] | None:
    """The kernel a declared cell runs (`Op.declared`): its kernel where it takes the call, the kernel the
    op would prefer otherwise, and that one's refusal naming the receipt (where `reported`: else the
    declared kernel as its own preference); None where the call's cell is not declared or its kernel cannot
    run it."""
    if not op.declared or not isinstance(case, Celled):
        return None
    cell = case.cell
    if cell not in op.declared:
        return None
    kernel, receipt = op.declared[cell]
    name = kernel.name
    if kernel.refused(case) is not None:
        return None
    preferred = _preferred(op, case) if reported else name
    if preferred == name:
        return kernel, name, None
    return kernel, preferred, Refusal(
        'measured', f"{name} measured faster than {preferred} on this cell "
                    f"({', '.join(str(part) for part in cell if part)}): {receipt}",
        'bible da64ae5', kind='measured')


def _preferred[C](op: Op[C], case: C) -> str:
    """The kernel `op` prefers for `case`: its measured preference where it states one; else the first
    non-generic kernel that takes it, else the first of them, whose refusal the route then names."""
    if op.preferred is not None:
        return op.preferred(case)
    native = [kernel for kernel in op.kernels if not kernel.generic and kernel.refused(case) is None]
    return (native or [kernel for kernel in op.kernels if not kernel.generic] or list(op.kernels))[0].name
