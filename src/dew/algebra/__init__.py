"""Dew's algebraic rewrite of its own programs (docs/design/algebra.md): a jaxpr held as an e-graph of terms
(`terms`), rules applied in named sets (`rules`), and each value's cheapest form under a device's prices
(`cost`) evaluated in place of the program's (`program.rewritten`).

The engine stands alone in this cut: no program Dew compiles passes through it yet (the `OuterProgram` seam,
§6.7), the report joins no kernel report, `dot_general` and control-flow bodies are leaves, and its
saturation limits (`SATURATION`) are set, not yet measured on the eval's programs (§6.6)."""

import functools

from dew.algebra import cost, program, rules, terms
from dew.algebra.program import AlgebraReport, Rewrite, Saturation

SATURATION = Saturation(growth=64, depth=4)
"""The limits the engine runs at until they are measured (§6.6)."""

rewritten = functools.partial(program.rewritten, saturation=SATURATION)
"""`program.rewritten` at `SATURATION`: a function with its program rewritten."""

__all__ = ["SATURATION", "AlgebraReport", "Rewrite", "Saturation", "cost", "program", "rewritten", "rules",
           "terms"]
