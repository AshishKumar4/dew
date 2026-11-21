"""Dew's algebraic rewrite of its own programs (docs/design/algebra.md): a jaxpr held as an e-graph of terms
(`terms`), rules applied in named sets (`rules`), and each value's cheapest form under a device's prices
(`cost`) evaluated in place of the program's (`program.rewritten`).

Every outer program and the resident training step jit their function rewritten (`rewritten`, §6.7), under
the switch the gate opens per arm (`passes`); each rewrite joins the one kernel report
(`dew.nn.kernels.templates.report.rewrites`). Its saturation stops at `SATURATION`'s limits (§6.6).

Its derivation of a template's walk forms from its declaration (`derive`: a `Reduction` to its `Derived`
init, step, merge, finish and mapped outputs) and of a recurrence's chunk forms and adjoint from its token
update (`scan`), each shared piece in `forms`.

Each submodule and name is loaded where it is first read (PEP 562's module `__getattr__`), never with the
package: importing one module of it (`dew.algebra.bits`, `dew.algebra.terms`) runs this file, and an eager
import here loaded the cost model and with it the kernels' templates (their resources, emission and report)
into every process that read any of it, so the test cache keyed every such test on them (3,891 rows moved on
one change to the templates' resources, train 1107 to 1108)."""

from __future__ import annotations

import functools
import importlib
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from jax.extend.core import Jaxpr

    from dew.algebra import cost, derive, error, forms, program, rules, scan, terms
    from dew.algebra.derive import Reduction
    from dew.algebra.forms import Derived, Refused
    from dew.algebra.program import AlgebraReport, Rewrite, Saturation, passes
    from dew.algebra.scan import Scan

    SATURATION: Saturation
    rewritten: functools.partial[Callable]
    rewritten_program: functools.partial[Jaxpr]

_SUBMODULES = frozenset({"cost", "derive", "error", "forms", "program", "rules", "scan", "terms"})
"""The submodules the package names (`__all__`), each imported where it is first read."""

_NAMED = {"Reduction": "derive", "Derived": "forms", "Refused": "forms", "AlgebraReport": "program",
          "Rewrite": "program", "Saturation": "program", "passes": "program", "Scan": "scan"}
"""Each name the package re-exports, by the submodule that defines it."""


class _Limited(NamedTuple):
    """The engine's limits (§6.6) and the rewrite at them (`_limited`)."""

    SATURATION: Saturation
    rewritten: functools.partial[Callable]
    rewritten_program: functools.partial[Jaxpr]


@functools.cache
def _limited() -> _Limited:
    """The engine's limits (§6.6) and the rewrite at them. `SATURATION`: above every compile-time fixpoint
    measured, none of the 756 windows of a UNet's fsdp2 x tensor2 matrix cell and benchmark_step's cpu-smoke
    steps reached either limit, the largest using under a quarter of each, and none gained at four times
    both (row 4010). `rewritten`: `program.rewritten` at them, a function with its program rewritten.
    `rewritten_program`: `program.rewritten_program` at them, a jit's own trace of a program, rewritten."""
    program = importlib.import_module(f"{__name__}.program")
    saturation = program.Saturation(growth=64, depth=4)
    return _Limited(saturation, functools.partial(program.rewritten, saturation=saturation),
                    functools.partial(program.rewritten_program, saturation=saturation))


def __getattr__(name: str) -> ModuleType | type | Saturation | functools.partial | Callable[..., Iterator]:
    """`name` of the package, loaded where it is first read and held from then on, so each read after the
    first is the same object (the module's own attribute)."""
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    if name in _NAMED:
        found = getattr(importlib.import_module(f"{__name__}.{_NAMED[name]}"), name)
    elif name in _Limited._fields:
        found = getattr(_limited(), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = found
    return found


__all__ = ["SATURATION", "AlgebraReport", "Derived", "Reduction", "Refused", "Rewrite", "Saturation", "Scan",
           "cost", "derive", "error", "forms", "passes", "program", "rewritten", "rewritten_program", "rules",
           "scan", "terms"]
