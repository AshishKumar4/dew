"""Dew's algebraic rewrite of its own programs (docs/design/algebra.md): a jaxpr held as an e-graph of terms
(`terms`), rules applied in named sets (`rules`), and each value's cheapest form under a device's prices
(`cost`) evaluated in place of the program's (`program.rewritten`). Its derivation
of a template's walk forms from its declaration (`derive`: a `Reduction` to its `Derived` init, step, merge,
finish and mapped outputs) and of a recurrence's chunk forms and adjoint from its token update (`scan`),
each shared piece in `forms`."""

from dew.algebra import cost, derive, forms, program, rules, scan, terms
from dew.algebra.derive import Reduction
from dew.algebra.forms import Derived, Refused
from dew.algebra.program import AlgebraReport, Saturation, rewritten
from dew.algebra.scan import Scan
from dew.nn.precision import PROTECTED

__all__ = ["PROTECTED", "AlgebraReport", "Derived", "Reduction", "Refused", "Saturation", "Scan", "cost", "derive", "forms", "program", "rewritten", "rules", "scan", "terms"]
