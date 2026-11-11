"""Reading a traced program (a jaxpr) as the platform it is lowered for runs it.

`dew.nn.flops` counts a program's work as an accelerator runs it, `dew.training.compiled_program` reads
the program XLA compiles for a platform, and `dew.interop.sequence_causality` walks a source's program for
what each position reads; each meets the platform choices (`jax.lax.platform_dependent`) a model's numerics
make (`dew.nn.precision`) and the programs an equation runs within it, and this is where they agree on
which branch a platform runs and which bodies XLA sees.
"""

from __future__ import annotations

import contextlib
import enum
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import TypeGuard

import jax
import jax.extend.core as jcore
import jax.numpy as jnp
from jax._src import mesh as mesh_lib
from jax._src.pallas.mpmd import mpmd_map_p
from jax.experimental.pallas import pallas_call_p
from jax.extend.backend import backends


def holes[Tree, Leaf](tree: Tree, selected: Callable[[Leaf], bool],
                      is_leaf: Callable[[Leaf], bool] | None = None
                      ) -> tuple[list[Leaf], Callable[[Sequence[Leaf]], Tree]]:
    """The leaves of `tree` (a call's arguments, `(args, kwargs)`) that `selected` keeps, in order, and
    `fill(values)`, `tree` with those leaves replaced by `values` in their order and every other as it is:
    what a trace of a function of some of its arguments (its arrays) binds and what it rebuilds."""
    leaves, structure = jax.tree.flatten(tree, is_leaf=is_leaf)
    places = [index for index, leaf in enumerate(leaves) if selected(leaf)]

    def fill(values: Sequence[Leaf]) -> Tree:
        filled = list(leaves)
        for index, value in zip(places, values, strict=True):
            filled[index] = value
        return jax.tree.unflatten(structure, filled)

    return [leaves[index] for index in places], fill


def platform_branch(platforms: Sequence[Sequence[str] | None], lowering: str) -> int:
    """The index of the branch of a platform choice that a program lowered for `lowering` runs: of a
    `cond` whose `branches_platforms` name each branch's platforms (None for its default), the first
    whose platforms hold `lowering`, or the default. JAX keeps that branch for each platform it lowers for
    (`_cond_lowering`, jax/_src/lax/control_flow/conditionals.py)."""
    return next(index for index, names in enumerate(platforms) if names is None or lowering in names)


NATIVE_KERNELS = (pallas_call_p, mpmd_map_p)
"""The primitives a Pallas kernel binds: a grid's call (`pallas_call`), and a core map's (`pallas.core_map`,
which a Mosaic GPU kernel, `plgpu.kernel`, runs through), each one custom call where it runs natively
(`pallas_call_lowering`, jax/_src/pallas/triton/pallas_call_registration.py; `_mpmd_map_lowering`,
jax/_src/pallas/mpmd.py)."""


def opaque(equation: jcore.JaxprEqn) -> bool:
    """Whether XLA sees nothing of `equation`'s body: a native Pallas kernel's (`NATIVE_KERNELS`), one
    custom call whose results are its outputs alone, its body's values in registers and shared memory,
    its products Triton's or Mosaic GPU's. A program `dew.training.compiled_program.compiled` reads holds
    no interpreted grid call: it is its interpreter's program (`dew.training.compiled_program.interpreter`).
    An interpreted core map is no native kernel: JAX lowers it through its interpreter
    (`_mpmd_map_fallback_lowering`), so its body is read as XLA runs it."""
    if equation.primitive is mpmd_map_p:
        return not equation.params["interpret"]
    return equation.primitive is pallas_call_p


def bodies(equation: jcore.JaxprEqn) -> list[jcore.Jaxpr]:
    """The programs `equation` runs within it (JAX's own finder, `jax.extend.core.jaxprs_in_params`): a
    call's, a loop's, a branch's or a checkpoint's. A native Pallas kernel runs none XLA sees (`opaque`)."""
    if opaque(equation):
        return []
    return list(jcore.jaxprs_in_params(equation.params))



def with_bodies(params: Mapping, rebuilt: Callable[[jcore.Jaxpr], jcore.Jaxpr]) -> dict:
    """`params` with each program within them (`bodies`' finder's: a parameter that is a program, or a tuple
    holding them, a branch's) replaced by `rebuilt(program)`, every other parameter as it is, and a tuple
    itself where `rebuilt` returns each of its programs as it was: what a program's equation retraced at
    other shapes binds, and what a pass that rewrites bodies keeps where it changes none (a program's
    identity is what JAX shares its lowering by)."""
    def replaced(value):
        if isinstance(value, jcore.Jaxpr):
            return rebuilt(value)
        if isinstance(value, tuple) and any(isinstance(each, jcore.Jaxpr) for each in value):
            made = tuple(rebuilt(each) if isinstance(each, jcore.Jaxpr) else each for each in value)
            return value if all(new is old for new, old in zip(made, value, strict=True)) else made
        return value

    return {key: replaced(value) for key, value in params.items()}


def equations(jaxpr: jcore.Jaxpr) -> Iterator[jcore.JaxprEqn]:
    """Every equation of `jaxpr` and of every program within it, at any depth (`jax.extend.core.subjaxprs`):
    a call's, a loop's, a branch's, a custom derivative's, a kernel's."""
    yield from jaxpr.eqns
    for inner in jcore.subjaxprs(jaxpr):
        yield from equations(inner)


def computed(value) -> TypeGuard[jcore.Var]:
    """Whether `value` is a program's variable, not a literal it reads."""
    return isinstance(value, jcore.Var) and not isinstance(value, jcore.Literal)


class RaggedLayout(enum.Enum):
    """A grouped product's layout as `jax.lax.ragged_dot_general` states it (`ragged_layout`)."""

    ROWS = "rows"
    """Rows grouped and their depth contracted: `[rows, depth]` by `[groups, depth, width]`."""
    ROWS_TRANSPOSED = "rows transposed"
    """The same over matrices held `[groups, width, depth]`."""
    GROUPS = "groups"
    """Each group's rows contracted: `[rows, depth]` by `[rows, width]` into `[groups, depth, width]`, a
    grouped matrix's gradient."""


def ragged_layout(lhs_rank: int, rhs_rank: int, numbers) -> RaggedLayout | None:
    """Which grouped product a `ragged_dot_general` of operands of those ranks under `numbers` (its
    `RaggedDotDimensionNumbers`) is, its left operand's rows ragged and no batch axes; None for any other.
    The one reading of those dimensions: the algebra's fusion takes a product's form from it
    (`dew.algebra.fusion`), and a CPU's pairwise sum what it blocks (`dew.nn.pairwise`)."""
    (contracted, against), batch = (tuple(map(tuple, pair)) for pair in numbers.dot_dimension_numbers)
    grouped = tuple(numbers.rhs_group_dimensions)
    if batch != ((), ()) or lhs_rank != 2 or tuple(numbers.lhs_ragged_dimensions) != (0,):
        return None
    if grouped == (0,) and rhs_rank == 3 and contracted == (1,) and against in ((1,), (2,)):
        return RaggedLayout.ROWS if against == (1,) else RaggedLayout.ROWS_TRANSPOSED
    if grouped == () and rhs_rank == 2 and contracted == (0,) and against == (0,):
        return RaggedLayout.GROUPS
    return None


def read_by(jaxpr: jcore.Jaxpr, sinks: Iterable) -> set[jcore.Var]:
    """The variables of `jaxpr` that `sinks` read, themselves included: each sink's own, and every operand of
    an equation one of whose results is read. A literal among `sinks` reads nothing. Effects are not
    followed: what a value depends on, not what a program must still run (`live`)."""
    read = {value for value in sinks if computed(value)}
    for equation in reversed(jaxpr.eqns):
        if any(var in read for var in equation.outvars):
            read.update(value for value in equation.invars if computed(value))
    return read


def live(equations: Sequence[jcore.JaxprEqn], outvars: Sequence) -> list[jcore.JaxprEqn]:
    """`equations`, in order, without those nothing reads: the ones `outvars` read (`read_by`'s walk) and
    every equation with an effect, kept whatever reads it."""
    wanted = {value for value in outvars if computed(value)}
    kept: list[jcore.JaxprEqn] = []
    for equation in reversed(equations):
        if equation.effects or any(var in wanted for var in equation.outvars):
            kept.append(equation)
            wanted.update(value for value in equation.invars if computed(value))
    return kept[::-1]


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
    """The program `equation` runs on its operands one for one, returning its results, else None: every
    reader of a call through its program reads it here (`dew.training.compiled_program`'s `compiled` and
    `gradients_read`, the kernel split's `_inlined`, the float64 terms, the algebra's normal forms). A call
    is a primitive JAX lowers to its program in the caller, which JAX states where it defines one
    (`Primitive.inlined_call`): a `jit`'s, which XLA's `CallInliner` inlines (`_pjit_lowering`,
    jax/_src/pjit.py; `GpuCompiler::RunPreSPMDPartitionerPasses`, xla/service/gpu/gpu_compiler.cc), an
    evaluated jaxpr's, a checkpoint's (`_remat_lowering`, jax/_src/ad_checkpoint.py), a custom derivative's
    primal (`_custom_jvp_vjp_call_lowering`, jax/_src/custom_derivatives.py), a custom batching rule's
    (`custom_vmap_p`'s lowering, jax/_src/custom_batching.py) and a marked span
    (`dew.nn.derivatives.marked_call_p`). XLA sees no call: a result that is the program's own parameter is
    its operand's buffer, and one that is a cheap value of another is recomputed where it is read, as it
    would be without the call. Its constants are its own (`consts`); a custom rule's leading constants
    (`num_consts`) are among the operands its program reads, one for one. A loop's body runs as many times as
    it steps and a branch's only where it is taken, so neither is a call."""
    body = _program(equation.primitive, equation.params)
    if body is not None and (len(body.invars), len(body.outvars)) == (
            len(equation.invars), len(equation.outvars)):
        return body
    return None


def _program(primitive: jcore.Primitive, params: Mapping) -> jcore.Jaxpr | None:
    """The one program a call's primitive (`Primitive.inlined_call`) holds in `params`, None for any other
    primitive."""
    if not primitive.inlined_call:
        return None
    found = list(jcore.jaxprs_in_params(params))
    return found[0] if len(found) == 1 else None


def purely_called(equation: jcore.JaxprEqn) -> jcore.Jaxpr | None:
    """The program `equation` runs (`called`) where its call does nothing else, else None: what a reader that
    runs a call in its caller reads (staging's `flattened`, the pairwise sums' `dew.nn.precision._paired`).
    JAX states which calls are pure where it defines them (`Primitive.transparent_call`: a `jit`'s and an
    evaluated jaxpr's; an optax helper's `jit` over a whole tree, `jnp.einsum`'s). A checkpoint's, a custom
    derivative's and a custom batching rule's call do more than their program, a rule of their own, and a
    marked span is a boundary the algebra pass reads (`dew.nn.derivatives.marked`: a protected span's inside
    no rule touches), so each keeps its rule only as a call."""
    return called(equation) if equation.primitive.transparent_call else None


UNROUNDED: dict[jcore.Primitive, Callable] = {}
"""Each primitive whose implementation rounds to a format of its own whatever its operands' dtype, to the
operation it rounds: a function of its operands that computes at their width. JAX's own primitives compute at
the dtype they are given, so none is here; one of Dew's that fixes a format states its operation where it is
defined, as it states its derivative there (`dew.nn.precision.correctly_rounded_divide_p`: the quotient).
What evaluates a traced program wider than it was traced reads this in place of the primitive (a float64
replay, the rounding model's tapped run): bound on wider operands, the primitive itself would still round
to its format."""


def own_derivative(primitive: jcore.Primitive) -> bool:
    """Whether `primitive` is a call whose derivative is a rule of its own (`jax.custom_jvp`'s,
    `jax.custom_vjp`'s): its cotangents and tangents are the rule's, not its program's."""
    return primitive in (jcore.primitives.custom_jvp_call_p, jcore.primitives.custom_vjp_call_p)


def _of(equation: jcore.JaxprEqn, primitive: jcore.Primitive) -> Mapping:
    """`equation`'s parameters, where it binds `primitive`: a reader of one control-flow operation's programs
    is asked of that operation alone, and another's equation is refused by its name."""
    if equation.primitive is not primitive:
        raise TypeError(f"{equation.primitive.name} is not {primitive.name}, whose programs were asked of it")
    return equation.params


def scan_body(equation: jcore.JaxprEqn) -> jcore.Jaxpr:
    """The program a scan's equation (`jax.lax.scan_p`) runs each step."""
    return _of(equation, jax.lax.scan_p)["jaxpr"]


def loop_programs(equation: jcore.JaxprEqn) -> tuple[jcore.Jaxpr, jcore.Jaxpr]:
    """A while loop's equation's (`jax.lax.while_p`) predicate and body."""
    params = _of(equation, jax.lax.while_p)
    return params["cond_jaxpr"], params["body_jaxpr"]


def branches(equation: jcore.JaxprEqn) -> tuple[jcore.Jaxpr, ...]:
    """A conditional's equation's (`jax.lax.cond_p`: `cond`, `switch`, a platform choice) branches, in the
    order its index picks them."""
    return tuple(_of(equation, jax.lax.cond_p)["branches"])


def control_flow(primitive: jcore.Primitive) -> bool:
    """Whether `primitive` is JAX's control flow (a scan, a while loop, a conditional): its programs run a
    step at a time or where they are taken (`scan_body`, `loop_programs`, `branches`), never as a call's."""
    return primitive in (jax.lax.scan_p, jax.lax.while_p, jax.lax.cond_p)


def entrywise(primitive: jcore.Primitive) -> bool:
    """Whether `primitive`'s every output entry reads its operands' entries at its own index (each operand of
    the output's shape, or a scalar) and nothing else: what JAX states where it defines an operation
    (`Primitive.elementwise`: its unary and n-ary operations, a cast, a selection, a clamp, the sum its
    derivatives add with). The one vocabulary the analyses of a program's entries read (`templates.layout`,
    `dew.interop.sequence_causality`, the algebra's fusion). A primitive of another library's or a user's
    states nothing and is none, whatever its name."""
    return primitive.elementwise


def positionwise(primitive: jcore.Primitive) -> bool:
    """Whether `primitive` maps over its operands' positions, each output position reading its operands' at
    that position and nothing else, whatever it does within one: an entrywise operation (`entrywise`), and
    one JAX batches by leaving the mapped axis where it found it, which it states where it registers that
    rule (`Primitive.positionwise`), so an added axis anywhere changes nothing of it. The operations on PRNG
    keys are such (a key's data extends its shape, and `random_bits` takes the shape it fills past the
    keys'), which are entrywise over a key array's positions and not over a value's."""
    return primitive.elementwise or primitive.positionwise


def rearranges(primitive: jcore.Primitive) -> bool:
    """Whether `primitive` only moves entries, each output entry one operand's entry or a fill, with no
    arithmetic and no rounding (a reshape, a squeeze, a broadcast, a transpose, a reversal, a slice, a
    gather, a concatenation, a padding, a split): JAX states it where it defines each
    (`Primitive.rearranges`)."""
    return primitive.rearranges


def reduction(primitive: jcore.Primitive) -> bool:
    """Whether `primitive` reduces the axes its `axes` parameter names (a sum, a product, an extremum or its
    place, a bitwise fold): JAX states it where it registers a reducer (`Primitive.reduces`)."""
    return primitive.reduces


def scatter(primitive: jcore.Primitive) -> bool:
    """Whether `primitive` writes updates into its first operand at indices, replacing or combining (a
    scatter, and its add, subtract, multiply, minimum and maximum): JAX states it where it registers the
    scatters' batching (`Primitive.scatters`)."""
    return primitive.scatters


def cumulative(primitive: jcore.Primitive) -> bool:
    """Whether `primitive` scans along the axis its `axis` parameter names (a cumulative sum, product,
    extremum or log-sum-exp): JAX states it where it defines the cumulative reductions
    (`Primitive.cumulative`)."""
    return primitive.cumulative


def host_evaluated(primitive, operands: Sequence, params: Mapping) -> list | None:
    """`primitive`'s outputs over concrete `operands`, computed as this host computes a constant: at compile
    time, on its CPU, outside any mesh a program traces under and without the outputs' stated shardings
    (`on_host`); None where an operand is an array this process cannot read whole (one laid across another
    process's devices), or an operand or output is of an extended dtype (a PRNG key, which no host array
    holds), or the primitive runs per device of a mesh it names (a shard_map's body sees one shard, and the
    host holds no mesh to run it over), none a constant of this host. A call is its program, interpreted
    (`called`'s): a constant has no derivative, and a custom derivative's bind takes its rules as functions,
    not the equation's jaxprs. The one evaluator of a program's constants, the compile point's and a compiled
    program's plan alike."""
    if "mesh" in params or any((isinstance(value, jax.Array) and not value.is_fully_addressable)
                               or _extended(value) for value in operands):
        return None
    params = {name: None if name in ("sharding", "out_sharding") else value for name, value in params.items()}
    body = _program(primitive, params)
    with on_host():
        if body is not None:
            made = jcore.jaxpr_as_fun(body)(*operands)
        else:
            made = primitive.bind(*operands, **params)
    made = list(made) if primitive.multiple_results else [made]
    return None if any(map(_extended, made)) else made


def _extended(value) -> bool:
    """Whether `value` is of an extended dtype (a PRNG key's)."""
    return jnp.issubdtype(getattr(value, "dtype", None) or jnp.result_type(value), jax.dtypes.extended)


@contextlib.contextmanager
def on_host() -> Iterator[None]:
    """Compile-time evaluation on this host, no mesh in context, abstract or concrete: a step
    traced under its devices' mesh (`jax.set_mesh`, a compile-only TPU topology's, a multi-process one's)
    would place an eager primitive's operands on devices that run nothing here. `jax.set_mesh(None)` does
    the same swaps but refuses to run under a trace, which a program's rewrite may be."""
    abstract = mesh_lib.abstract_mesh_context_manager.swap_local(mesh_lib.empty_abstract_mesh)
    concrete = mesh_lib.device_context.swap_local(mesh_lib.empty_concrete_mesh)
    try:
        # This process's own device: its CPU, or, where it runs no CPU backend (`JAX_PLATFORMS=cuda`), its
        # first. Its own: the pool's first device may be another process's, which this one cannot read back.
        host = jax.local_devices(backend="cpu" if "cpu" in backends() else None)[0]
        with jax.ensure_compile_time_eval(), jax.default_device(host):
            yield
    finally:
        mesh_lib.device_context.set_local(concrete)
        mesh_lib.abstract_mesh_context_manager.set_local(abstract)
