"""Reading a traced program (a jaxpr) as the platform it is lowered for runs it.

`dew.nn.flops` counts a program's work as an accelerator runs it, `dew.training.compiled_program` reads
the program XLA compiles for a platform, and `dew.interop.sequence_causality` walks a source's program for
what each position reads; each meets the platform choices (`jax.lax.platform_dependent`) a model's numerics
make (`dew.nn.precision`) and the programs an equation runs within it, and this is where they agree on
which branch a platform runs and which bodies XLA sees.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import TypeGuard

import jax
import jax.extend.core as jcore
from jax.interpreters import ad

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


@functools.cache
def _jax_own() -> Mapping[jcore.Primitive, str]:
    """JAX's own operations by their objects: every primitive `jax.lax`, `jax.lax.linalg` and
    `jax.extend.core.primitives` export (the random operations' `random_seed`, `random_split`,
    `random_fold_in`, `random_bits` and `random_gamma` among them, and the sum JAX's derivatives add with,
    `add_any`), and the three its public random functions bind that no module of JAX's exports
    (`random_wrap`, `random_unwrap`, `random_clone`), read off a trace of them once asked."""
    def keyed(seed_words):
        # A key of its own implementation, whichever the process's default is (`jax_default_prng_impl`).
        key = jax.random.wrap_key_data(seed_words, impl="threefry2x32")
        return jax.random.key_data(jax.random.clone(key))

    traced = jax.make_jaxpr(keyed)(jax.ShapeDtypeStruct((2,), jax.numpy.uint32)).jaxpr
    exported = (primitive for module in (jax.lax, jax.lax.linalg, jcore.primitives)
                for primitive in vars(module).values() if isinstance(primitive, jcore.Primitive))
    return {primitive: primitive.name
            for primitive in (*exported, *(equation.primitive for equation in traced.eqns), ad.add_jaxvals_p)}


def jax_name(primitive) -> str | None:
    """`primitive`'s name where it is JAX's own operation (`_jax_own`), None for any other: a primitive
    of another library or of a user's takes any name, whatever it computes, so a reader that knows an
    operation by its name knows it by its object. A call (`CALLS`) runs the program it holds, read as
    such by its name."""
    return _jax_own().get(primitive)


def computed(value) -> TypeGuard[jcore.Var]:
    """Whether `value` is a program's variable, not a literal it reads."""
    return isinstance(value, jcore.Var) and not isinstance(value, jcore.Literal)


def inlined(body: jcore.Jaxpr, operands: Sequence[jcore.Var | jcore.Literal],
            consts: Sequence = ()) -> tuple[list[jcore.JaxprEqn], list[jcore.Var | jcore.Literal]]:
    """`body`'s equations, each value its own fresh variable, read over `operands` and its constants over
    `consts`, and its results: a program run in its caller as one of its own, as one body traced once may
    be run more than once. A constant is the caller's variable or literal where `consts` gives one (a
    constant the caller holds among its own, `dew.algebra.program`'s span opened), else a literal of the
    value given (a program compiled whole, `dew.training.compiled_program`)."""
    local: dict[jcore.Var, jcore.Var | jcore.Literal] = dict(zip(body.invars, operands, strict=True))
    for var, value in zip(body.constvars, consts, strict=True):
        local[var] = value if isinstance(value, jcore.Var | jcore.Literal) else jcore.Literal(value, var.aval)

    def read(value):
        return local.get(value, value) if computed(value) else value

    equations = []
    for equation in body.eqns:
        fresh = [out if isinstance(out, jcore.DropVar) else jcore.Var(out.aval) for out in equation.outvars]
        equations.append(equation.replace(invars=[read(value) for value in equation.invars], outvars=fresh))
        local.update(zip(equation.outvars, fresh, strict=True))
    return equations, [read(value) for value in body.outvars]


def called(equation: jcore.JaxprEqn) -> jcore.Jaxpr | None:
    """The program a call (`CALLS`) runs on `equation`'s operands one for one, returning its results, else
    None: every reader of a call through its program reads it here (`dew.training.compiled_program`'s
    `compiled` and `gradients_read`, staging's flattening, the kernel split's `_inlined`, the float64
    terms)."""
    found = bodies(equation) if equation.primitive.name in CALLS else []
    if len(found) == 1 and (len(found[0].invars), len(found[0].outvars)) == (
            len(equation.invars), len(equation.outvars)):
        return found[0]
    return None
