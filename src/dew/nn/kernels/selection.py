"""Kernel selection: one selector for every family of ops (`select`), over a typed description of
the call (the op's case, the `C` its `Op[C]`, kernels and requirements share, so a requirement of
another op's case fails the type check) and what each kernel declares it takes (its `SUPPORTS`, ordered
`Requirement`s), the one place a kernel's eligibility is written.

An op states its preference: its own rule over the call (`Op.preferred`), else the first of its kernels
whose requirements hold, in its fixed order; a cell a measurement found another kernel faster on
(`Op.declared`: the receipt beside it) runs that kernel. The first kernel
whose requirements all hold takes the call, and the route names the preferred kernel's first failed
requirement as its refusal (`generation.route`). A kernel the caller forces
(`generation.chosen`, or a layer's named implementation) that does not take the call raises, listing
every requirement it fails, as torch's `sdpa_kernel` does; a generic kernel forced so hands the call to
the next generic one.

Selection reads only the call's static facts and the target's while a program traces: it adds no
operation to the program, and outside `generation.routes()` it records nothing
(`dew.cache.KERNEL_ROUTES` stays `()`), so a run's trace keys are those of a run that never selects.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .generation import Route


@dataclasses.dataclass(frozen=True)
class Refusal:
    """Why the kernel a target prefers does not take a call, from the call's own facts on that
    target (its shapes, dtypes, mask): a stable `code`, which a kernel's `SUPPORTS` keys its declared
    limits on, and the `detail` a log names. What the host cannot run is no refusal (`Route.ran`)."""

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


@dataclasses.dataclass(frozen=True)
class Requirement[C]:
    """One fact a kernel needs of a case `C`, the description its op's selector reads: `code` names
    it in a refusal, `holds(case)` tests it on the target (`generation`'s facts, honouring
    `assumed`), and `detail` (text, or a function of the case) says why a call fails it. A limit
    the code could lift names the `row` of the roadmap or ledger that tracks lifting it; one no
    implementation of the kernel lifts names `INHERENT`. Its
    `kind` is 'capability', what the kernel cannot compute; 'measured', where the target measured
    another kernel faster or never qualified this one, which a forced choice overrides; or
    'handoff', a call structurally outside what the kernel applies to, which it hands to the op's next
    kernel even when forced (a generic kernel's to the next generic one, a join's projections run apart, a
    decode token with nothing to fuse runs the step alone)."""

    code: str
    holds: Callable[[C], bool]
    detail: str | Callable[[C], str]
    row: str
    kind: str = 'capability'
    analysis: Callable[[C], Refusal | None] | None = None
    """An analysis whose own refusal the requirement reports (`analysed`), stamped with its row and kind."""

    def __post_init__(self) -> None:
        if not self.row:
            raise ValueError(f"requirement {self.code!r} names the row that tracks lifting it, or INHERENT")
        if self.kind not in ('capability', 'measured', 'handoff'):
            raise ValueError(f"requirement {self.code!r} is of kind {self.kind!r}, not capability, measured "
                             "or handoff")

    def check(self, case: C) -> Refusal | None:
        """The failed requirement's data, or None when it holds."""
        if self.analysis is not None:
            found = self.analysis(case)
            return None if found is None else dataclasses.replace(found, row=self.row, kind=self.kind)
        if self.holds(case):
            return None
        return Refusal(self.code, self.detail(case) if callable(self.detail) else self.detail,
                       self.row, self.kind)


def analysed[C](code: str, analysis: Callable[[C], Refusal | None], row: str,
                kind: str = 'capability') -> Requirement[C]:
    """The requirement an analysis decides (a mainloop's reason it declines a pass,
    `attention_route.MAINLOOP`): it holds where the analysis finds no refusal, and reports the analysis's
    own refusal, its code and detail, under this requirement's row and kind. A function, not a method of
    `Requirement`: its case in a return position would make the requirement's case invariant, and a
    requirement of a protocol's case (`recurrence_chunk.Templated`) would no longer serve its ops'."""
    return Requirement(code, lambda case: analysis(case) is None, lambda case: str(analysis(case)), row, kind,
                       analysis)


@dataclasses.dataclass(frozen=True)
class Kernel[C]:
    """A kernel an op may run: `supports`, its requirements in the order a refusal names them, and
    whether it is `generic`, a path that takes what the others turn down."""

    name: str
    supports: tuple[Requirement[C], ...]
    generic: bool = False

    def failed(self, case: C, forced: bool = False) -> list[Refusal]:
        """The requirements `case` fails, evaluated once, in order; forcing overrides measured preferences."""
        failures = []
        for requirement in self.supports:
            if forced and requirement.kind == 'measured':
                continue
            refusal = requirement.check(case)
            if refusal is not None:
                failures.append(refusal)
        return failures


@runtime_checkable
class Celled(Protocol):
    """A case an op with declared cells reads (`Op.declared`): its `cell`, the scope a receipt measures
    (a template's instruction family, `generation.family`, None outside every family; a standalone
    kernel's device generation, `generation.device_generation`) and the kind of call it is (a GEMM's
    'grouped forward', a recurrence's product, a norm's 'bfloat16')."""

    @property
    def cell(self) -> tuple[str | None, str]: ...


@dataclasses.dataclass(frozen=True)
class Op[C]:
    """A family of ops one selector serves: its `choice` (the name `chosen` and routes use), its `kernels`
    in the order the selector tries them after the preferred one, `preferred(case)` the op's own rule for
    the kernel it prefers (else the first of its kernels that takes the case, `select`), `hosted(kernel)`
    what this host runs for the target's kernel (`generation.gpu_placed`), and `declared`: the cells where
    a measurement found another kernel faster than the op's preference (a template's old route, owner,
    bible da64ae5; a standalone kernel against XLA's program), each cell (`Celled.cell`: the scope the
    receipt measured, a template's instruction family or a device's generation, and the kind of call) to
    that kernel and the receipt. A declared cell runs its kernel where it takes the call; any other call,
    and a declared one its kernel cannot run, the op's order."""

    choice: str
    kernels: tuple[Kernel[C], ...]
    preferred: Callable[[C], str] | None = None
    hosted: Callable[[str], str] = lambda kernel: kernel
    declared: Mapping[tuple[str | None, str], tuple[str, str]] = dataclasses.field(
        default_factory=dict, compare=False, hash=False)

    def kernel(self, name: str) -> Kernel[C]:
        for kernel in self.kernels:
            if kernel.name == name:
                return kernel
        raise ValueError(f"{self.choice} has no kernel {name!r}; its kernels are "
                         f"{', '.join(kernel.name for kernel in self.kernels)}")


class KernelRefused(ValueError):
    """A forced kernel that does not take a call: the op's `choice`, the `kernel` chosen, and the
    `refusals` of every requirement the call fails, in order."""

    def __init__(self, choice: str, kernel: str, refusals: list[Refusal]) -> None:
        super().__init__(f"{choice} kernel {kernel!r} was chosen and does not take this call: "
                         + "; ".join(f"{refusal.code}: {refusal.detail}" for refusal in refusals))
        self.choice, self.kernel, self.refusals = choice, kernel, refusals


def forced(choice: str) -> str | None:
    """The kernel `generation.chosen` names for `choice` within its block, or None."""
    from dew import cache

    return dict(cache.KERNEL_CHOICES.value).get(choice)


def select[C](op: Op[C], case: C, named: str | None = None, *, preference: tuple[str, ...] = ()) -> Route:
    """The kernel `op` runs for `case` on the target, reported as its route: `named` (or the kernel
    `chosen` names) where it takes the case, else the op's preference (`preferred`, or the first of its
    kernels that takes the case), else the first of the op's kernels that does. A named kernel that does
    not take the case raises with every requirement it fails, its measured ones aside, and hands a case
    only its handoff requirements turn down to the op's next kernels (a generic one to the next generic
    ones). A call's soft `preference`, where neither `named` nor `chosen` forces a kernel, is its own order
    of the op's kernels: the first of them that takes the case, the first passing what the targets measured
    of it as a named kernel does, and its refusal reported (a norm's per-call implementation, an explicit
    xla's decode step). The route states what forced it (`Route.forced`)."""
    from . import generation

    kernel, preferred, refusal, insisted = _decided(op, case, named, preference)
    return generation.route(op.choice, kernel=kernel.name, preferred=preferred, refusal=refusal,
                            ran=op.hosted(kernel.name), forced=insisted)


def selected[C](op: Op[C], case: C, named: str | None = None, *, preference: tuple[str, ...] = ()
                ) -> Kernel[C]:
    """The kernel `select` runs for `case`, reporting nothing: what a caller reads ahead of the dispatch
    that reports it (the kernel a call that built its mask names, a projection's choice between XLA's
    program and the GEMM template's launches, which select their own), raising as `select` does."""
    return _decided(op, case, named, preference)[0]


def _decided[C](op: Op[C], case: C, named: str | None,
                preference: tuple[str, ...]) -> tuple[Kernel[C], str, Refusal | None, str | None]:
    """`select`'s kernel for `case`, the kernel preferred, the preferred one's refusal and the kernel that
    forced the choice (`named`, else `chosen`'s), None where nothing did."""
    insisted = named or forced(op.choice)
    soft = insisted is None and bool(preference)
    if soft:
        # The call's own order, exactly: its first the call's pick, the rest its fallbacks in turn.
        preferred = preference[0]
        candidates = [op.kernel(name) for name in preference]
        first = candidates[0]
    else:
        old = None if insisted is not None else _old_route(op, case)
        if old is not None:
            return (*old, None)
        preferred = insisted or _preferred(op, case)
        first = op.kernel(preferred)
        # An insisted kernel hands a call only its handoffs turn down, to the op's next kernels in turn; a
        # generic one to the next generic ones.
        handing = insisted is not None and not first.generic
        rest = op.kernels[op.kernels.index(first) + 1:] if handing else op.kernels
        candidates = [first, *(kernel for kernel in rest if kernel is not first
                               and (insisted is None or not first.generic or kernel.generic))]
    refusal = None
    for kernel in candidates:
        # A kernel a call names or picks passes what the targets measured of it.
        failed = kernel.failed(case, forced=insisted is not None or (soft and kernel is first))
        if not failed:
            return kernel, preferred, refusal, insisted
        if kernel is first:
            refusal = failed[0]
            if insisted is not None and any(requirement.kind != 'handoff' for requirement in failed):
                raise KernelRefused(op.choice, kernel.name, failed)
    raise ValueError(f"no {op.choice} kernel takes this call ({refusal})")


def _old_route[C](op: Op[C], case: C) -> tuple[Kernel[C], str, Refusal | None] | None:
    """The kernel a declared cell runs (`Op.declared`): its kernel where it takes the call, the kernel the
    op would prefer otherwise, and that one's refusal naming the receipt; None where the call's cell is
    not declared or its kernel cannot run it."""
    if not op.declared or not isinstance(case, Celled):
        return None
    cell = case.cell
    if cell not in op.declared:
        return None
    name, receipt = op.declared[cell]
    kernel = op.kernel(name)
    if kernel.failed(case):
        return None
    preferred = _preferred(op, case)
    if preferred == name:
        return kernel, name, None
    return kernel, preferred, Refusal(
        'measured', f"{name} measured faster than {preferred} on this cell "
                    f"({', '.join(part for part in cell if part)}): {receipt}",
        'bible da64ae5', kind='measured')


def _preferred[C](op: Op[C], case: C) -> str:
    """The kernel `op` prefers for `case`: its measured preference where it states one; else the first
    non-generic kernel that takes it, else the first of them, whose refusal the route then names."""
    if op.preferred is not None:
        return op.preferred(case)
    native = [kernel for kernel in op.kernels if not kernel.generic and not kernel.failed(case)]
    return (native or [kernel for kernel in op.kernels if not kernel.generic] or list(op.kernels))[0].name
