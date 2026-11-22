"""A native Mosaic GPU kernel body's arithmetic: an algebra's own functions run over its MLIR values (`Word`),
so a kernel that keeps a schedule of its own emits the arithmetic the portable program runs, one statement of
it (the compensated scan's renormalized sums, `templates.scan`; the recurrence's own pairs' band terms and
write gates, `templates.recurrence.own_pairs`)."""

import dataclasses

from jax.extend.mlir import ir
from jax.extend.mlir.dialects import arith


@dataclasses.dataclass(frozen=True)
class Word:
    """An fp32 MLIR value whose `+`, `-` and `*` emit its kernel's own addition, subtraction and
    multiplication (`arith.addf`, `arith.subf`, `arith.mulf`, each rounded alone, none contracted): what an
    algebra's function reads where a native kernel's body runs it (`precision.renormalized_sum`,
    `recurrence_algebra.band_term`, `recurrence_algebra.gated`)."""

    value: ir.Value

    def __add__(self, other: "Word") -> "Word":
        return Word(arith.addf(self.value, other.value))

    def __sub__(self, other: "Word") -> "Word":
        return Word(arith.subf(self.value, other.value))

    def __mul__(self, other: "Word") -> "Word":
        return Word(arith.mulf(self.value, other.value))
