"""Dew's algebraic rewrite of its own programs (docs/design/algebra.md): a jaxpr held as an e-graph of terms
(`terms`), rules applied in named sets (`rules`), and each value's cheapest form under a device's prices
(`cost`) evaluated in place of the program's (`program.rewritten`).

Every outer program and the resident training step jit their function rewritten (`rewritten`, §6.7), under
the switch the gate opens per arm (`passes`); each rewrite joins the one kernel report
(`dew.nn.kernels.templates.report.rewrites`). Its saturation stops at `SATURATION`'s limits (§6.6).

Its derivation of a template's walk forms from its declaration (`derive`: a `Reduction` to its `Derived`
init, step, merge, finish and mapped outputs) and of a recurrence's chunk forms and adjoint from its token
update (`scan`), each shared piece in `forms`."""

import functools

from dew.algebra import cost, derive, error, forms, program, rules, scan, terms
from dew.algebra.derive import Reduction
from dew.algebra.forms import Derived, Refused
from dew.algebra.program import AlgebraReport, Rewrite, Saturation, passes
from dew.algebra.scan import Scan

SATURATION = Saturation(growth=64, depth=4)
"""The limits the engine runs at until they are measured (§6.6)."""
rewritten = functools.partial(program.rewritten, saturation=SATURATION)
"""`program.rewritten` at `SATURATION`: a function with its program rewritten."""
rewritten_program = functools.partial(program.rewritten_program, saturation=SATURATION)
"""`program.rewritten_program` at `SATURATION`: a jit's own trace of a program, rewritten."""

__all__ = ["SATURATION", "AlgebraReport", "Derived", "Reduction", "Refused", "Rewrite", "Saturation", "Scan",
           "cost", "derive", "error", "forms", "passes", "program", "rewritten", "rewritten_program", "rules",
           "scan", "terms"]
