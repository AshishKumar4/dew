"""Reading a traced program (a jaxpr) as the platform it is lowered for runs it.

`dew.nn.flops` counts a program's work as an accelerator runs it, and `dew.training.compiled_program` reads
the program XLA compiles for a platform; each meets the platform choices (`jax.lax.platform_dependent`) a
model's numerics make (`dew.nn.precision`), and this is where they agree on which branch a platform runs.
"""

from __future__ import annotations

from collections.abc import Sequence


def platform_branch(platforms: Sequence[Sequence[str] | None], lowering: str) -> int:
    """The index of the branch of a platform choice that a program lowered for `lowering` runs: of a
    `cond` whose `branches_platforms` name each branch's platforms (None for its default), the first
    whose platforms hold `lowering`, or the default. JAX keeps that branch for each platform it lowers for
    (`_cond_lowering`, jax/_src/lax/control_flow/conditionals.py)."""
    return next(index for index, names in enumerate(platforms) if names is None or lowering in names)
