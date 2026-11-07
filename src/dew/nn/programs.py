"""Reading a traced program (a jaxpr) as the platform it is lowered for runs it.

`dew.nn.flops` counts a program's work as an accelerator runs it, `dew.training.compiled_program` reads
the program XLA compiles for a platform, and `dew.interop.sequence_causality` walks a source's program for
what each position reads; each meets the platform choices (`jax.lax.platform_dependent`) a model's numerics
make (`dew.nn.precision`) and the programs an equation runs within it, and this is where they agree on
which branch a platform runs and which bodies XLA sees.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator, Mapping, Sequence
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


ENTRYWISE = frozenset({
    "abs", "acos", "acosh", "add", "add_any", "and", "asin", "asinh", "atan", "atan2", "atanh", "bessel_i0e",
    "bessel_i1e", "cbrt", "ceil", "clamp", "clz", "complex", "conj", "convert_element_type", "copy", "copy_p",
    "cos", "cosh", "digamma", "div", "eq", "erf", "erf_inv", "erfc", "exp", "exp2", "expm1", "floor", "ge",
    "gt", "igamma", "igammac", "imag", "integer_pow", "is_finite", "le", "lgamma", "log", "log1p", "log2",
    "logistic", "lt", "max", "min", "mul", "ne", "neg", "nextafter", "not", "or", "polygamma",
    "population_count", "pow", "real", "reduce_precision", "rem", "round", "rsqrt", "select_n", "shift_left",
    "shift_right_arithmetic", "shift_right_logical", "sign", "sin", "sinh", "sqrt", "square", "stop_gradient",
    "sub", "tan", "tanh", "threefry2x32", "xor", "zeta",
})
"""The operations whose every output entry reads its operands' entries at its own index (each operand of
the output's shape, or a scalar) and nothing else, by JAX's name (`jax_name`): the one vocabulary the
analyses of a program's entries read (`templates.layout`, `dew.interop.sequence_causality`)."""

KEYS = frozenset({"random_bits", "random_fold_in", "random_split", "random_unwrap", "random_wrap"})
"""The operations on PRNG keys, each output entry its own key's: entrywise over a key array's positions,
but not over a value's (a key's data extends its shape, and `random_bits` takes the shape it fills), so an
analysis of positions takes them beside `ENTRYWISE` and one of values' entries does not."""

REDUCTIONS = frozenset({"argmax", "argmin", "reduce_and", "reduce_max", "reduce_min", "reduce_or",
                        "reduce_prod", "reduce_sum", "reduce_xor"})
"""The operations that reduce the axes their `axes` parameter names."""

CUMULATIVE = frozenset({"cumlogsumexp", "cummax", "cummin", "cumprod", "cumsum"})
"""The operations that scan along the axis their `axis` parameter names."""

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



def equations(jaxpr: jcore.Jaxpr) -> Iterator[jcore.JaxprEqn]:
    """Every equation of `jaxpr` and of every program within it, at any depth (`jax.extend.core.subjaxprs`):
    a call's, a loop's, a branch's, a custom derivative's, a kernel's."""
    yield from jaxpr.eqns
    for inner in jcore.subjaxprs(jaxpr):
        yield from equations(inner)


@functools.cache
def _jax_own() -> Mapping[jcore.Primitive, str]:
    """JAX's own operations by their objects: every primitive `jax.lax`, `jax.lax.linalg` and
    `jax.extend.core.primitives` export (the random operations' `random_seed`, `random_split`,
    `random_fold_in`, `random_bits` and `random_gamma` among them, and the sum JAX's derivatives add with,
    `add_any`), and those its public functions bind that no module of JAX's exports, read off a trace of
    them once asked: its random functions' `random_wrap`, `random_unwrap` and `random_clone`, a ragged
    product's `ragged_dot_general`, a copy's `copy_p`, and its callbacks' and foreign calls'
    `pure_callback`, `io_callback`, `debug_callback` and `ffi_call`."""
    def keyed(seed_words):
        # A key of its own implementation, whichever the process's default is (`jax_default_prng_impl`).
        key = jax.random.wrap_key_data(seed_words, impl="threefry2x32")
        return jax.random.key_data(jax.random.clone(key))

    def bound(rows, groups):
        shape = jax.ShapeDtypeStruct((), jax.numpy.float32)
        products = jax.lax.ragged_dot(rows, rows.T[None], groups)
        jax.debug.callback(lambda value: None, products)
        called = jax.pure_callback(lambda value: value, shape, products.sum())
        written = io_callback(lambda value: value, shape, called)
        foreign = jax.ffi.ffi_call("dew_named", shape)(written)
        return jax.numpy.copy(foreign)

    from jax.experimental import io_callback

    float32 = jax.numpy.float32
    traced = jax.make_jaxpr(keyed)(jax.ShapeDtypeStruct((2,), jax.numpy.uint32)).jaxpr
    held = jax.make_jaxpr(bound)(jax.ShapeDtypeStruct((2, 2), float32),
                                 jax.ShapeDtypeStruct((1,), jax.numpy.int32)).jaxpr
    exported = (primitive for module in (jax.lax, jax.lax.linalg, jcore.primitives)
                for primitive in vars(module).values() if isinstance(primitive, jcore.Primitive))
    return {primitive: primitive.name
            for primitive in (*exported, *(equation.primitive for equation in (*traced.eqns, *held.eqns)),
                              ad.add_jaxvals_p)}


def jax_name(primitive) -> str | None:
    """`primitive`'s name where it is JAX's own operation (`_jax_own`), None for any other: a primitive
    of another library or of a user's takes any name, whatever it computes, so a reader that knows an
    operation by its name knows it by its object. A call (`CALLS`) runs the program it holds, read as
    such by its name."""
    return _jax_own().get(primitive)


HOLDERS = CALLS | {"pallas_call", "shard_map"}
"""The primitives a reader knows by their names though no module of JAX's exports them: the calls
(`CALLS`), a Pallas kernel's (`opaque`) and a manual mesh's, each running a program it holds."""


def named(primitive) -> str:
    """The name a reader of a program knows `primitive` by: JAX's own operation's (`jax_name`), a
    program holder's (`HOLDERS`), else none (""): a primitive of another library or of a user's takes any
    name, whatever it computes, so no reader reads one as the JAX operation its name spells (row 4094: a
    user's `convert_element_type` read as a cast raised KeyError on its params)."""
    own = jax_name(primitive)
    if own is not None:
        return own
    return primitive.name if primitive.name in HOLDERS else ""


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
