"""Kernel selection: one selector for every family of ops (`select`), over a typed description of
the call (the op's case, the `C` its `Op[C]`, kernels and requirements share, so a requirement of
another op's case fails the type check) and what each kernel declares it takes (its `SUPPORTS`, ordered
`Requirement`s), the one place a kernel's eligibility is written.

An op that measures states its preference: the target's measured kernel (its row in
`generation.TARGETS`, read through `measured_kernel`) comes first, then the op's own order; any other
op prefers the first of its kernels whose requirements hold, in its fixed order. The first kernel
whose requirements all hold takes the call, and the route names the preferred kernel's first failed
requirement as its refusal (`generation.route`). A kernel the caller forces
(`generation.chosen`, or a layer's named implementation) that does not take the call raises, listing
every requirement it fails, as torch's `sdpa_kernel` does; a generic kernel forced so hands the call to
the next generic one.

Selection reads only the call's static facts and the target's while a program traces: it adds no
operation to the program, and outside `generation.routes()` it records nothing
(`dew.cache.KERNEL_ROUTES` stays `()`), so a run's trace keys are those of a run that never selects.
"""

import dataclasses
from collections.abc import Callable

from . import generation
from .generation import Refusal, Route


@dataclasses.dataclass(frozen=True)
class Requirement[C]:
    """One fact a kernel needs of a case `C`, the description its op's selector reads: `code` names
    it in a refusal, `holds(case)` tests it on the target (`generation`'s facts, honouring
    `assumed`), and `detail` (text, or a function of the case) says why a call fails it. A limit
    the code could lift names the `row` of the roadmap or ledger that tracks lifting it; one no
    implementation of the kernel lifts names `INHERENT`. Its
    `kind` is 'capability', what the kernel cannot compute; 'measured', where the target measured
    another kernel faster or never qualified this one, which a forced choice overrides; or
    'handoff', what a generic kernel hands to the next generic one even when forced."""

    code: str
    holds: Callable[[C], bool]
    detail: str | Callable[[C], str]
    row: str
    kind: str = 'capability'

    def __post_init__(self) -> None:
        if not self.row:
            raise ValueError(f"requirement {self.code!r} names the row that tracks lifting it, or INHERENT")
        if self.kind not in ('capability', 'measured', 'handoff'):
            raise ValueError(f"requirement {self.code!r} is of kind {self.kind!r}, not capability, measured "
                             "or handoff")

    def refusal(self, case: C) -> Refusal:
        """This requirement's refusal of `case`, carrying the row that tracks it or `INHERENT`."""
        return Refusal(self.code, self.detail(case) if callable(self.detail) else self.detail, self.row)


@dataclasses.dataclass(frozen=True)
class Kernel[C]:
    """A kernel an op may run: `supports`, its requirements in the order a refusal names them, and
    whether it is `generic`, a path that takes what the others turn down."""

    name: str
    supports: tuple[Requirement[C], ...]
    generic: bool = False

    def failed(self, case: C, forced: bool = False) -> list[Requirement[C]]:
        """The requirements `case` fails, in order; a `forced` kernel's measured ones hold."""
        return [requirement for requirement in self.supports
                if not (forced and requirement.kind == 'measured') and not requirement.holds(case)]


@dataclasses.dataclass(frozen=True)
class Op[C]:
    """A family of ops one selector serves: its `choice` (the name `TARGETS` rows, `chosen` and routes
    use), its `kernels` in the order the selector tries them after the preferred one, `preferred(case)`
    the kernel the target measured fastest for the case where the op measures (else the first of its
    kernels that takes the case, `select`), and `hosted(kernel)` what this host runs for the target's
    kernel (`generation.gpu_placed`)."""

    choice: str
    kernels: tuple[Kernel[C], ...]
    preferred: Callable[[C], str] | None = None
    hosted: Callable[[str], str] = lambda kernel: kernel

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


def select[C](op: Op[C], case: C, named: str | None = None, *, insist: bool = True) -> Route:
    """The kernel `op` runs for `case` on the target, reported as its route: `named` (or the kernel
    `chosen` names) where it takes the case, else the op's preference (`preferred`, or the first of its
    kernels that takes the case), else the first of the op's kernels that does. A named kernel that does
    not take the case raises with every requirement it fails, its measured ones aside; a generic one
    hands a case its handoff requirements turn down to the op's next generic kernel. Without `insist`,
    `named` is the call's own preference, before `chosen`'s, and one that does not take the case hands
    it to the op's other kernels, its refusal reported (a norm's or a SiLU product's per-call
    implementation)."""
    kernel, preferred, refusal = _decided(op, case, named, insist)
    return generation.route(op.choice, kernel=kernel.name, preferred=preferred, refusal=refusal,
                            ran=op.hosted(kernel.name))


def selected[C](op: Op[C], case: C, named: str | None = None, *, insist: bool = True) -> Kernel[C]:
    """The kernel `select` runs for `case`, reporting nothing: what a caller reads ahead of the dispatch
    that reports it (the kernel a call that built its mask names, a projection's choice between XLA's
    program and the GEMM template's launches, which select their own), raising as `select` does."""
    return _decided(op, case, named, insist)[0]


def _decided[C](op: Op[C], case: C, named: str | None,
                insist: bool) -> tuple[Kernel[C], str, Refusal | None]:
    """`select`'s kernel for `case`, the kernel preferred and the preferred one's refusal."""
    if named is not None and not insist:
        preferred, insisted = named, None
    else:
        insisted = named or forced(op.choice)
        preferred = insisted or _preferred(op, case)
    first = op.kernel(preferred)
    candidates = [first, *(kernel for kernel in op.kernels if kernel is not first
                           and (insisted is None or (first.generic and kernel.generic)))]
    refusal = None
    for kernel in candidates:
        failed = kernel.failed(case, forced=insisted is not None)
        if not failed:
            return kernel, preferred, refusal
        if kernel is first:
            refusal = failed[0].refusal(case)
            if insisted is not None and any(requirement.kind != 'handoff' for requirement in failed):
                raise KernelRefused(op.choice, kernel.name,
                                    [requirement.refusal(case) for requirement in failed])
    raise ValueError(f"no {op.choice} kernel takes this call ({refusal})")


def _preferred[C](op: Op[C], case: C) -> str:
    """The kernel `op` prefers for `case`: its measured preference where it states one; else the first
    non-generic kernel that takes it, else the first of them, whose refusal the route then names."""
    if op.preferred is not None:
        return op.preferred(case)
    native = [kernel for kernel in op.kernels if not kernel.generic and not kernel.failed(case)]
    return (native or [kernel for kernel in op.kernels if not kernel.generic] or list(op.kernels))[0].name
