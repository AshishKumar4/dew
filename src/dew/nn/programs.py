"""Reading a traced program (a jaxpr) as the platform it is lowered for runs it.

`dew.nn.flops` counts a program's work as an accelerator runs it, and `dew.training.compiled_program` reads
the program XLA compiles for a platform; each meets the platform choices (`jax.lax.platform_dependent`) a
model's numerics make (`dew.nn.precision`), and this is where they agree on which branch a platform runs.
"""

from __future__ import annotations

from collections.abc import Sequence

CALLS = frozenset({"pjit", "jit", "closed_call", "core_call", "remat2", "checkpoint", "custom_jvp_call",
                   "custom_vjp_call", "marked_call"})
"""The primitives that run a program on their operands, one for one, and return its results. A `jit`
lowers to a call that XLA's `CallInliner` inlines (`_pjit_lowering`, jax/_src/pjit.py;
`GpuCompiler::RunPreSPMDPartitionerPasses`, xla/service/gpu/gpu_compiler.cc), and a checkpoint
(`_remat_lowering`, jax/_src/ad_checkpoint.py), a custom derivative's primal
(`_custom_jvp_vjp_call_lowering`, jax/_src/custom_derivatives.py) and a marked span
(`dew.nn.derivatives.marked_call_p`, its body inlined) lower to their program in the caller, so XLA sees no
call. A result that is the program's own parameter is its operand's buffer, and one that is a
cheap value of another is recomputed where it is read, as it would be without the call."""


def platform_branch(platforms: Sequence[Sequence[str] | None], lowering: str) -> int:
    """The index of the branch of a platform choice that a program lowered for `lowering` runs: of a
    `cond` whose `branches_platforms` name each branch's platforms (None for its default), the first
    whose platforms hold `lowering`, or the default. JAX keeps that branch for each platform it lowers for
    (`_cond_lowering`, jax/_src/lax/control_flow/conditionals.py)."""
    return next(index for index, names in enumerate(platforms) if names is None or lowering in names)
