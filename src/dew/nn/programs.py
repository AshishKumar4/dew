"""Reading a traced program (a jaxpr) as the platform it is lowered for runs it.

`dew.nn.flops` counts a program's work as an accelerator runs it, `dew.training.compiled_program` reads
the program XLA compiles for a platform, and `dew.interop.sequence_causality` walks a source's program for
what each position reads; each meets the platform choices (`jax.lax.platform_dependent`) a model's numerics
make (`dew.nn.precision`) and the programs an equation runs within it, and this is where they agree on
which branch a platform runs and which bodies XLA sees.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.extend.core as jcore

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


def opaque(equation: jcore.JaxprEqn) -> bool:
    """Whether XLA sees nothing of `equation`'s body: a native Pallas kernel's, one custom call whose
    results are its outputs alone, its body's values in registers and shared memory, its products
    Triton's (`pallas_call_lowering`, jax/_src/pallas/triton/pallas_call_registration.py). A program
    `dew.training.compiled_program.compiled` reads holds no interpreted one: it is its interpreter's
    program (`dew.training.compiled_program.interpreter`)."""
    return equation.primitive.name == "pallas_call"


def bodies(equation: jcore.JaxprEqn) -> list[jcore.Jaxpr]:
    """The programs `equation` runs within it: a call's, a loop's, a branch's or a checkpoint's. A
    native Pallas kernel runs none XLA sees (`opaque`)."""
    if opaque(equation):
        return []
    found = []
    for value in equation.params.values():
        found.extend(each for each in (value if isinstance(value, tuple | list) else (value,))
                     if isinstance(each, jcore.Jaxpr))
    return found
