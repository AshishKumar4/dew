"""Dew's algebraic rewrite of its own programs (docs/design/algebra.md): a jaxpr held as an e-graph of terms
(`terms`), rules applied in named sets (`rules`), and each value's cheapest form under a device's prices
(`cost`) evaluated in place of the program's (`program.rewritten`).

Every outer program and the resident training step jit their function rewritten (`rewritten`, §6.7), under
the switch the gate opens per arm (`passes`); each rewrite joins the one kernel report
(`dew.nn.kernels.templates.report.rewrites`). Its saturation stops at `SATURATION`'s limits (§6.6)."""

import functools

from dew.algebra import cost, program, rules, terms
from dew.algebra.program import AlgebraReport, Rewrite, Saturation, passes

SATURATION = Saturation(growth=64, depth=4)
"""The limits the engine runs at until they are measured (§6.6)."""

rewritten = functools.partial(program.rewritten, saturation=SATURATION)
"""`program.rewritten` at `SATURATION`: a function with its program rewritten."""

__all__ = ["SATURATION", "AlgebraReport", "Rewrite", "Saturation", "cost", "passes", "program", "rewritten",
           "rules", "terms"]
