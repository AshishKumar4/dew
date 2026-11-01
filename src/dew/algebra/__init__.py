"""Dew's algebraic rewrite of its own programs (docs/design/algebra.md): a jaxpr held as an e-graph of terms
(`terms`), rules applied in named sets (`rules`), and each value's cheapest form under a device's prices
(`cost`) evaluated in place of the program's (`program.rewritten`).

The engine stands alone in this cut: no program Dew compiles passes through it yet (the `OuterProgram` seam,
§6.7), the report joins no kernel report, `dot_general` and control-flow bodies are leaves, and its
saturation limits (`SATURATION`) are set, not yet measured on the eval's programs (§6.6). Its derivation
of a template's walk forms from its declaration (`derive`: a `Reduction` to its `Derived` init, step, merge,
finish and mapped outputs) and of a recurrence's chunk forms and adjoint from its token update (`scan`),
each shared piece in `forms`."""

import functools

from dew.algebra import cost, derive, forms, program, rules, scan, terms
from dew.algebra.derive import Reduction
from dew.algebra.forms import Derived, Refused
from dew.algebra.program import AlgebraReport, Rewrite, Saturation
from dew.algebra.scan import Scan

SATURATION = Saturation(growth=64, depth=4)
"""The limits the engine runs at until they are measured (§6.6)."""
rewritten = functools.partial(program.rewritten, saturation=SATURATION)
"""`program.rewritten` at `SATURATION`: a function with its program rewritten."""

__all__ = ["SATURATION", "AlgebraReport", "Derived", "Reduction", "Refused", "Rewrite", "Saturation", "Scan", "cost", "derive", "forms", "program", "rewritten", "rules", "scan", "terms"]
