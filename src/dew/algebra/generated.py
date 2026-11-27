"""The opt-in seam between a template's written component and the body the generator emitted and checked for
it (docs/design/algebra.md §6.11, rows 3272 and 6109): a site asks for its declared component's generated body
(`body`) and, given one, calls it in place of its written math. Under the default kernel choice
(`generation.Choice.GENERATED`, "written") the site's own math runs and its program is the written one; under
"generated" a call whose case its generated module declares (policy, target, operand shapes and dtypes, and
the site's own static program, `when`) runs the checked body, and any other call its written math. Every
answer under "generated" is counted (`taken`), a fallback by its reason, so a run or a measurement can tell a
generated body ran from a written one that stood in for it. The generated modules import nothing of the
registry, rules or extractor, and nor does this seam."""

from __future__ import annotations

import collections
import contextlib
import logging
from collections.abc import Callable, Iterator, Sequence
from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np

from dew.algebra import _generated
from dew.nn.kernels import generation

_log = logging.getLogger(__name__)

GENERATED = "generated"
"""The outcome of a call that runs its generated body (`taken`)."""

_COUNTS: list[collections.Counter[tuple[str, str]]] = []


@contextlib.contextmanager
def taken() -> Iterator[collections.Counter[tuple[str, str]]]:
    """Within the block, each answer `body` gives under the "generated" choice, counted by `(qualified,
    outcome)`: `GENERATED`, or the reason the written body stood in. Counted as a call traces (a jitted
    program's body answers once, as it is traced)."""
    counts: collections.Counter[tuple[str, str]] = collections.Counter()
    _COUNTS.append(counts)
    try:
        yield counts
    finally:
        _COUNTS.remove(counts)


def _counted(qualified: str, outcome: str) -> None:
    for counts in _COUNTS:
        counts[qualified, outcome] += 1
    if outcome != GENERATED:
        _log.info("the written body of %s stands in for its generated one: %s", qualified, outcome)


def enabled() -> bool:
    """Whether the run asks for generated bodies (`generation.Choice.GENERATED`): a site whose operands or
    static program cost a trace's work to state asks this first."""
    return generation.chosen_kernel(generation.Choice.GENERATED, "written") == GENERATED


def body(qualified: str, operands: Sequence, *, when: bool = True) -> Callable | None:
    """The generated body of the component `qualified` (its declaring module and name) for a call over
    `operands` (its flat ABI, as declared), where the "generated" choice is in effect, `when` holds (the
    site's static program is the declared one) and the call's case is declared: run it as the call. None
    otherwise, the site's written math to run; under "generated" that fallback is counted (`taken`)."""
    if not enabled():
        return None
    if not when:
        _counted(qualified, "another static program")
        return None
    module, name = qualified.rsplit(".", 1)
    derived = _derived(module)
    forms = getattr(derived, f"_{name}_forms")
    policy = "accurate" if generation.numerics().forms == "accurate" else "default"
    typed = [jax.ShapeDtypeStruct(np.shape(operand), jnp.result_type(operand)) for operand in operands]
    key = (policy, generation.device_generation(), _generated.case_of(typed))
    if key not in forms:
        _counted(qualified, f"no declared case {key}")
        return None
    _counted(qualified, GENERATED)
    function = getattr(derived, name)
    return lambda *values: function(*values, policy=key[0], target=key[1])


def _derived(module: str) -> ModuleType:
    """The generated module of the declaring `module`, imported where named (a static import, which a code
    digest reads: `dew._code`); a module no site's component is declared in has none here."""
    if module == "dew.algebra.component_cases":
        from dew.algebra import component_cases_derived

        return component_cases_derived
    raise ValueError(f"no generated module of {module!r} serves a site")
