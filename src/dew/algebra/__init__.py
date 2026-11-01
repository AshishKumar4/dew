"""Dew's algebraic rewrite of its own programs (docs/design/algebra.md): a jaxpr held as an e-graph of terms
(`terms`), rules applied in named sets (`rules`), and each value's cheapest form under a device's prices
(`cost`) evaluated in place of the program's (`program.rewritten`)."""

from dew.algebra import cost, program, rules, terms
from dew.algebra.program import AlgebraReport, Saturation, rewritten
from dew.nn.precision import PROTECTED

__all__ = ["PROTECTED", "AlgebraReport", "Saturation", "cost", "program", "rewritten", "rules", "terms"]
