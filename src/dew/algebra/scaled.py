"""A scaled value (docs/design/algebra.md §6.12): a float carried as its significand, in [1, 2) or zero, of
its own format (float32, or float64 where a truth runs the same form), and its power of two, an int32, so that
a product, a quotient or a sum whose factors reach past the format's range, or below its least normal, holds
every intermediate within it. Its one rounding into the format is its landing
(`landed`, `applied`), the value's own; an intermediate leaves the normal range only where the value does.

The construct is the engine's: any form whose terms' factors span more than the format is written over it,
the merge of finished attention states its first (`dew.algebra.means`). Each value carries the bound its
exponent is proven within (`Scaled.reach`), from its operands' as it is formed; a form whose bound passes
`REACH` is refused (`forms.Refused`), never emitted.

The construct is values alone: no program differentiates through a scaled value. A float tangent of a
scaled value is lost wherever it leaves the format (read in its significand's units, `dx 2^-e` overflowed
where a tangent passed its primal by the format's range and flushed where it fell below; read in its
value's, a coefficient's tangent flushed beside the terms it cancels against: the degenerate sweep,
`tests/test_algebra_scaled_edges.py`), so a law over scaled values states its derivatives as one of its
own operations whose derivative is that operation again, its tangents its float operands' and its values'
(`dew.algebra.means.ratio_p`). Its primitives (`split_p`, `applied_p`) carry no derivative rule: a program
differentiating one is refused by JAX, never silently differentiated through `frexp`. A subnormal operand
is read from its bits, its value kept where XLA:CPU's arithmetic flushes it (`scaled`), and a zero is +0:
no scaled value, nor a float the construct lands (`applied`, `landed`: a flushed landing's too), holds a
zero's sign, which no law over them reads (a ratio lands its zero +0, `means.Ratio`)."""

from __future__ import annotations

import dataclasses
import enum
import math
import types
from collections.abc import Callable, Sequence
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src import dispatch
from jax.extend import core
from jax.interpreters import batching, mlir

from dew.algebra.forms import Refused
from dew.nn.derivatives import contracted
from dew.nn.precision import correctly_rounded_divide, two_product, two_sum


@dataclasses.dataclass(frozen=True)
class Format:
    """A float format's constants the construct reads (`format_of`)."""

    least: int
    """Its least normal exponent: `2^least` its least normal value."""
    largest: int
    """Its largest exponent."""
    mantissa: int
    """Its stored significand bits."""
    word: type
    """The unsigned integer of its width, its bits'."""
    reach: int
    """The bound a value's own exponent is within as a scaled value's (`scaled`): a normal's from `least` to
    `largest`, a subnormal's down to `least - mantissa`, one past for the significand's place (float32's
    150)."""
    shift: int
    """The most a sum aligns a significand down by (`add`), twice its significand's bits and 16
    (float32's 64): a significand in [1, 2) shifted further lies below `2^(1 - shift)`, inside half a unit in
    the last place of any significand of at least `2^(-shift/2 - 6)` it is added to, so the sum rounds as it
    would without the shift, a significand that small being one that cancelled to nothing the format holds
    beside its other operand; shifted by `2^-shift` it stays a normal value."""


def format_of(dtype) -> Format:
    """`dtype`'s constants (`Format`), from its `finfo`."""
    limits = jnp.finfo(dtype)
    return Format(least=limits.minexp, largest=limits.maxexp - 1, mantissa=limits.nmant,
                  word=jnp.dtype(f"uint{limits.bits}").type, reach=-limits.minexp + limits.nmant + 1,
                  shift=2 * (limits.nmant + 1) + 16)


REACH = 1 << 24
"""The largest exponent bound a scaled value may reach, held as it is formed (`_held`). A float's is its
format's `reach`; a product's or a quotient's its operands' sum and two; a sum's the larger, the format's
`shift` and its significand's bits and two, renormalized after a cancellation; a total's of `n` the operand's,
`shift`, its bits and three and `log2 n`. A form's bound is thus at most its format's `reach` times its
leaves' count and `shift` and the bits and three times its operations', additive in both: a merge's term in
float32, of under thirty operations over a few hundred states, stays under 2^15. Held at 2^24, the int32 sum
or difference of any two bounded exponents is exact; a form past it is refused, its exponents' arithmetic no
longer exact."""


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class Scaled:
    """A float value as `significand * 2**exponent`: `significand` of its format, in [1, 2) or 0 (its sign
    the value's), `exponent` int32 (any, where the significand is 0), each of the value's shape, `reach` the
    bound `|exponent| <= reach` it is proven within, static."""

    significand: jax.Array
    exponent: jax.Array
    reach: int

    def tree_flatten(self):
        return (self.significand, self.exponent), self.reach

    @classmethod
    def tree_unflatten(cls, reach: int, children) -> Scaled:
        significand, exponent = children
        return cls(significand, exponent, reach)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.significand.shape

    @property
    def format(self) -> Format:
        return format_of(self.significand.dtype)


def _held(reach: int) -> int:
    """`reach`, refused past `REACH` (`forms.Refused`): the bound every exponent the form computes is proven
    within, beyond which their int32 sums would no longer be exact."""
    if reach > REACH:
        raise Refused(f"a scaled form's exponents reach 2^{reach}, past the {REACH} their int32 sums hold "
                      f"exactly")
    return reach


def power2(exponent: jax.Array, dtype) -> jax.Array:
    """`2.0 ** exponent` in `dtype` for an int32 exponent within its normal range, exactly, from its bits:
    the one owner of an exact power of two (a template's among its callers)."""
    form = format_of(dtype)
    return lax.bitcast_convert_type((exponent + form.largest).astype(form.word) << form.mantissa, dtype)


def scaled_by(x: jax.Array, exponent: jax.Array, reach: int) -> jax.Array:
    """`x * 2**exponent` as products by normal powers of two of the exponent's sign in turn (as many as
    `|exponent| <= reach` takes, or the format's `saturating` where that is less), each exact but the one
    where the product leaves the normal range: the value moves one way, so no intermediate passes the range
    where the result does not. An exponent past `saturating` lands any finite `x` past the format, 0 or
    infinite as it is, and is taken at it: a law's factor held exactly to 2^-65536 (`means.FLOOR`) would
    otherwise unroll 521 products a landing, every compile of the law the slower (a merge kernel's, a split
    decode's)."""
    form = format_of(x.dtype)
    saturating = (form.largest + 1) - (form.least - form.mantissa) + 2
    exponent = jnp.clip(exponent, -saturating, saturating)
    for _ in range(max(1, math.ceil(min(reach, saturating) / -form.least))):
        part = jnp.clip(exponent, form.least, form.largest)
        x = x * power2(part, x.dtype)
        exponent = exponent - part
    return x


def _split_value(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`x`'s significand in [1, 2), its sign kept, and its exponent (`jnp.frexp`'s, one less): a subnormal's
    read from its bits, its fraction field (a normal float of the format, exactly) times the least
    subnormal's power, so its exponent lies beneath the least normal's (within its format's `reach`) and a
    process whose float arithmetic flushes a subnormal operand (XLA's CPU) keeps its value (outputs `2^-127`
    and `2^-125` merged to `2^-126`, the first dropped, the merge P0's head review, F2); a zero's +0 at
    exponent 0, and a non-finite value its own significand. Only `frexp`, a bitcast and an integer's
    conversion, which every lowering of the construct's operations takes (a Triton kernel's, `merged`)."""
    form = format_of(x.dtype)
    bits = lax.bitcast_convert_type(x, jnp.dtype(f"int{jnp.finfo(x.dtype).bits}"))
    field = bits & ((1 << form.mantissa) - 1)
    subnormal = ((bits >> form.mantissa) & (2 * form.largest + 1) == 0) & (field != 0)
    fraction, exponent = jnp.frexp(jnp.where(subnormal, field.astype(x.dtype), x))
    fraction = jnp.where(subnormal & (bits < 0), -fraction, fraction)
    exponent = jnp.where(subnormal, exponent + form.least - form.mantissa, exponent)
    return (jnp.where(fraction == 0, jnp.zeros_like(x), 2 * fraction),
            jnp.where(fraction == 0, 0, exponent - 1).astype(jnp.int32))


def _valued(parameter) -> bool:
    """Whether a primitive's `parameter` compares and hashes by its value, one key for every call that means
    it: a scalar, a string, an enum's member, a dtype or a type, None, a tuple of such, a frozen dataclass's
    instance, or a module's function (one object a process). A partial, a closure, a function made inside
    another or a mutable container is a new key at each call that makes it."""
    if parameter is None or isinstance(parameter, (bool, int, float, complex, str, enum.Enum, np.generic,
                                                   np.dtype, type)):
        return True
    if isinstance(parameter, tuple):
        return all(map(_valued, parameter))
    if dataclasses.is_dataclass(parameter):
        # The parameters `dataclasses` kept on the nearest class it made, beside its fields; each field a
        # value too (a law's form holding the units a call passed, `means.Ratio`).
        frozen = next(vars(made)["__dataclass_params__"] for made in type(parameter).__mro__
                      if "__dataclass_params__" in vars(made)).frozen
        fields = dataclasses.fields(parameter)
        return frozen and all(_valued(getattr(parameter, field.name)) for field in fields)
    return isinstance(parameter, types.FunctionType) and "<locals>" not in parameter.__qualname__


class _Primitive(core.Primitive):
    """A primitive of the construct's (`primitive`), its parameters each a value (`_valued`), held at bind:
    JAX keys an eager call's compiled program and a trace's caches on a primitive's parameters, so a
    parameter made fresh at each call (a partial of the call's own state) compiles every call again and
    holds every executable (eager merges measured 2026-11-16: 8 MiB and 5.6 s a call, armada's
    containers killed at 12 GiB)."""

    def bind(self, *arguments, **parameters):
        made = [name for name, parameter in parameters.items() if not _valued(parameter)]
        if made:
            raise TypeError(f"{self.name}: its parameters {made} are no values (`scaled._valued`): each "
                            f"would key a compiled program of its own at every call; pass a module's "
                            f"function or a frozen dataclass's instance")
        return super().bind(*arguments, **parameters)


MADE: dict[core.Primitive, tuple[Callable, bool]] = {}
"""Every operation of the construct's own (`primitive`), the law's in `dew.algebra.means` among them: its
value, the function its default lowering runs, and whether it returns several values, by the primitive,
so a lowering of a launch that runs the construct's functions (a kernel template's) holds each as it is
made."""


def _alike(value: Callable) -> Callable:
    """`value` (a primitive's computation, `primitive`'s) over its operands made to vary alike: each made to
    vary along every manual axis (`jax.shard_map`'s) any of them varies along (`jax.lax.pcast`), as JAX's own
    operations do before they compute, so the operations its computation binds see operands that vary
    alike, as its result type states (`varying`). Under `shard_map`, a quotient of a replicated value by
    each device's own was refused as it lowered (the merge P0's base review 8, F1: "div requires varying
    manual axes to match"); outside one, every operand varies along none, and nothing changes."""
    def computed(*operands, **params):
        axes = frozenset().union(*(jax.typeof(operand).mat.varying for operand in operands))
        operands = [operand if jax.typeof(operand).mat.varying == axes else jax.lax.pcast(
            operand, tuple(sorted(axes - jax.typeof(operand).mat.varying)), to="varying")
            for operand in operands]
        return value(*operands, **params)
    return computed


def primitive(name: str, value: Callable, typed: Callable, batched: Callable | None = None, *,
              several: bool = False) -> core.Primitive:
    """An operation of its own of the construct's (`split_p`, `applied_p`, and the law's in
    `dew.algebra.means`): computed by `value` (its lowering, and eagerly one dispatch of it, as JAX's own
    primitives run), typed by `typed`, batched by `batched` (entrywise, by broadcasting, where none is
    named), returning `several` values; a derivative and a transpose, where it has them, registered beside
    it (`means.ratio_p`). Its parameters are values, held at bind (`_Primitive`); it is recorded with its
    value (`MADE`)."""
    made = _Primitive(name)
    made.multiple_results = several
    made.def_impl(partial(dispatch.apply_primitive, made))
    made.def_abstract_eval(typed)
    mlir.register_lowering(made, mlir.lower_fun(_alike(value), multiple_results=several))
    if batched is None:
        batching.defbroadcasting(made)
        made.elementwise = True
    else:
        batching.primitive_batchers[made] = batched
    MADE[made] = (value, several)
    return made


def along_axis(bound: Callable[..., core.Primitive]) -> Callable:
    """The batching rule of a primitive whose parameter `axis` names its operands' reduced axis: each
    operand's batch axis moved first, `axis` one further, each result's batch axis first."""
    def batched(arguments, dimensions, *, axis: int, **params):
        paired = list(zip(arguments, dimensions, strict=True))
        size = next(argument.shape[dimension] for argument, dimension in paired if dimension is not None)
        fronted = [batching.bdim_at_front(argument, dimension, size) for argument, dimension in paired]
        made = bound()
        batched_values = made.bind(*fronted, axis=axis + 1, **params)
        return batched_values, (0,) * len(batched_values) if made.multiple_results else 0
    return batched


split_p = primitive(
    "scaled_split", _split_value,
    lambda x: (x.update(), x.update(dtype=np.dtype(np.int32), weak_type=False)),
    # Entrywise: a batch's significands and exponents are its entries', along its own batch axis.
    lambda arguments, dimensions: (split_p.bind(*arguments), (dimensions[0], dimensions[0])), several=True)
"""A float array's significands and exponents (`_split_value`), an operation of its own, a value's alone (the
module's account)."""


def _split(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return split_p.bind(x)


def _float(x: jax.typing.ArrayLike) -> jax.Array:
    """`x` in its float format: float64 as it is, any other as float32 (a 16-bit float widened exactly)."""
    x = jnp.asarray(x)
    return x if x.dtype == jnp.float64 else x.astype(jnp.float32)


def scaled(x: jax.typing.ArrayLike) -> Scaled:
    """`x`, a float array, as a scaled value of its format (`Scaled`, `_float`), exact: a subnormal `x` read
    from its bits (`_split_value`)."""
    x = _float(x)
    significand, exponent = _split(x)
    return Scaled(significand, exponent, format_of(x.dtype).reach)


def normalized(significand: jax.Array, exponent: jax.Array, reach: int) -> Scaled:
    """`significand * 2**exponent` with its significand moved back into [1, 2) (exact: a power of two), its
    exponent bound `reach`."""
    moved, by = _split(significand)
    # A zero's exponent 0: no power a product or its transpose applies to it passes the range (`_chain`).
    return Scaled(moved, jnp.where(moved == 0, 0, exponent + by), _held(reach))


def base2(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`x log2(e)`, a natural exponent's base-2 one, as its rounded value and its residual, exact together:
    the constant split into two floats of `x`'s format, the larger normalized first so a tiny `x`'s
    residual stays normal, and its product with `x` split exactly (`two_product`: float32's over truncated
    halves, which no contraction into a fused multiply-add changes and no large `x` overflows; float64's by
    Dekker's splitter, past which no float32 value's difference reaches). The residual's product is rounded
    apart from its sum (`lax.optimization_barrier`), so no compiler's contraction into a fused multiply-add
    moves it with the fusion it lands in."""
    factor, power = np.frexp(math.log2(math.e))
    x = x * np.asarray(2.0 ** int(power), x.dtype)
    high = np.asarray(factor, x.dtype)
    low = np.asarray(factor - float(high), x.dtype)
    rounded, error = two_product(x, high, bit_split=x.dtype == jnp.float32)
    return rounded, error + lax.optimization_barrier(x * low)


def one_like(x: jax.Array) -> Scaled:
    """The scaled value 1 at each entry of `x`, of its format, its shape and its sharding (an explicit mesh's
    type, which a value built from a shape alone lacks)."""
    return Scaled(jnp.ones_like(x), jnp.zeros_like(x, dtype=jnp.int32), 0)


def mul(a: Scaled, b: Scaled) -> Scaled:
    """`a * b`: the significands' product (within [1, 4)), the exponents' sum."""
    return normalized(a.significand * b.significand, a.exponent + b.exponent, a.reach + b.reach + 2)


def div(a: Scaled, b: Scaled) -> Scaled:
    """`a / b` for `b` nonzero: the significands' correctly rounded quotient (within (1/2, 2)), the
    exponents' difference. A divisor in [1, 2) has a normal reciprocal, so no compiler's rewrite of the
    division flushes it. Correctly rounded in float32 (`precision.correctly_rounded_divide`); a float64 run
    of the form (a truth's) divides as XLA does, within two of its roundings."""
    quotient = (correctly_rounded_divide(a.significand, b.significand) if a.significand.dtype == jnp.float32
                else a.significand / b.significand)
    return normalized(quotient, a.exponent - b.exponent, a.reach + b.reach + 2)


def neg(a: Scaled) -> Scaled:
    """`-a`."""
    return Scaled(-a.significand, a.exponent, a.reach)


def where(condition: jax.Array, a: Scaled, b: Scaled) -> Scaled:
    """`a` where `condition`, else `b`."""
    return Scaled(jnp.where(condition, a.significand, b.significand),
                  jnp.where(condition, a.exponent, b.exponent), max(a.reach, b.reach))


def _aligned(a: Scaled, to: jax.Array) -> jax.Array:
    """`a`'s significand at the exponent `to`, at least its own: shifted down by their difference, at most
    its format's `shift`, exact (`Format.shift`'s account)."""
    shift = a.format.shift
    return a.significand * power2(jnp.clip(a.exponent - to, -shift, 0), a.significand.dtype)


def add(a: Scaled, b: Scaled) -> Scaled:
    """`a + b`: each significand aligned to the larger exponent of a nonzero operand (`_aligned`), their sum
    (within (-4, 4)) renormalized. A sum that cancels keeps what its operands' significands hold: their
    difference is exact (Sterbenz) where the exponents are one, so V4's equal and opposite terms leave
    nothing."""
    exponent = jnp.where(a.significand == 0, b.exponent,
                         jnp.where(b.significand == 0, a.exponent, jnp.maximum(a.exponent, b.exponent)))
    return normalized(_aligned(a, exponent) + _aligned(b, exponent), exponent,
                       max(a.reach, b.reach) + a.format.shift + a.format.mantissa + 2)


def sub(a: Scaled, b: Scaled) -> Scaled:
    """`a - b` (`add`)."""
    return add(a, neg(b))


@contracted("test_the_convex_mean_holds_every_order")
def applied(x: jax.typing.ArrayLike, a: Scaled) -> jax.Array:
    """`x * a` as a float of `a`'s format: `x` times `a`'s significand, then by normal powers of two of its
    exponent's sign in turn (`_chain`). Where `a` is at least 1 every factor is at least 1, and where it is
    below 1 every factor is at most 1 (its significand halved, its exponent one more), so the product moves
    one way from `x` to the result, and no intermediate passes the range where the result does not: the one
    rounding into the format is the result's; a zero is +0 (the module's account). Bilinear in `x` and `a`'s
    significand (`applied_p`), so a tangent or a cotangent multiplied by a scaled coefficient transposes
    through it, the same chain."""
    x, significand, exponent = jnp.broadcast_arrays(jnp.asarray(x, a.significand.dtype), a.significand,
                                                    a.exponent)
    return applied_p.bind(x, significand, exponent, reach=a.reach)


def landed(a: Scaled) -> jax.Array:
    """`a` as a float of its format (`applied` to one): its one rounding into the format, its own value's."""
    return applied(jnp.ones_like(a.significand), a)


def _chain(x: jax.Array, significand: jax.Array, exponent: jax.Array, reach: int) -> jax.Array:
    """`applied`'s product: the factors of one direction (`applied`'s account), the powers of two first and
    the significand last, so its transpose takes a cotangent by the significand first: a zero coefficient's
    cotangent is zero before any power reaches it."""
    growing = exponent >= 0
    last = jnp.where(growing, significand, 0.5 * significand)
    rest = jnp.where(growing, exponent, exponent + 1)
    product = scaled_by(x, rest, reach + 1) * last
    # A zero +0, its sign none (the module's account): a flushed landing's or a zero factor's alike.
    return jnp.where(product == 0, jnp.zeros_like(product), product)


applied_p = primitive(
    "scaled_applied", _chain,
    lambda x, significand, exponent, *, reach: entrywise(x, significand, integers=(exponent,)))
"""`x * significand * 2^exponent` entrywise (`_chain`, `reach` the exponent's bound), an operation of its own,
a value's alone (the module's account)."""


def entrywise(x, *others, integers=()):
    """An entrywise operation's result type over operands of `x`'s rank (`applied_p`'s), float operands
    `others` and integer ones `integers` (an exponent): their broadcast shape (a batch leaves an operand it
    does not map of unit axes, `batching.defbroadcasting`), `x`'s dtype, sharding and memory, varying along
    every manual axis any operand varies along, an exponent's among them (`varying`: the merge P0's base
    review 9, F1, a product of replicated values by each device's own exponent typed as replicated), weak
    where every float operand is, as JAX's own entrywise operations type theirs."""
    typed = x.update(shape=jnp.broadcast_shapes(*(operand.shape for operand in (x, *others, *integers))),
                     weak_type=all(operand.weak_type for operand in (x, *others)))
    return varying(typed, x, *others, *integers)


def varying(typed, *read):
    """`typed`, a result's type, varying along every manual axis (`jax.shard_map`'s) any operand it `read`s
    varies along: as JAX's own operations make an operand that does not vary vary before they compute
    (`pvary`), the value of an operation of the construct's own varies along their union, which its type
    states (the merge P0's base review 8, F1: a merge's result typed by its outputs alone dropped its totals'
    varying axis, and a scan carrying it was refused). A result typed by an operand of no mesh (a constant its
    program closes over) takes the mesh its varying operands lie on, as JAX's own operations read their
    operands' one mesh."""
    axes = frozenset().union(*(operand.mat.varying for operand in read))
    if not axes:
        return typed
    if typed.sharding.mesh.empty:
        mesh = next(operand.sharding.mesh for operand in read if operand.mat.varying)
        typed = typed.update(sharding=typed.sharding.update(mesh=mesh))
    return typed.update(manual_axis_type=typed.mat.update(varying=axes))


def compensated(terms: Sequence[jax.Array]) -> tuple[jax.Array, jax.Array]:
    """`terms`' sum (each an array of one shape), compensated, in pairs: a tree over their order, each
    pair's sum exact (`precision.two_sum`, a span the algebra pass keeps as written) and its error carried
    with its halves' errors, so a sum of `n` terms is `log2 n` dependent steps (a kernel's program, the
    decode merge's: a slice order's `n` steps were its span, the attention lane's GH200 window) and the
    rounded sum and its partial sums' exact errors' sum, added once, are within `u |sum|` and the rounding
    of the errors' own sum of the exact sum of `terms` (Sum2's bound, which holds in any order of its
    additions). Linear in `terms`: a tangent's sum transposes through it. The terms are materialized first
    (`lax.optimization_barrier`), so a compiled sum does not add a product unrounded beside its rounded
    opposite. An infinite partial sum, which only terms of one sign summing past the format make, leaves the
    errors NaN; a sum of values reads them where the sum is finite (`dew.algebra.means`). Given apart, never
    stacked: a kernel's body (Pallas Triton's) stacks no more than two arrays."""
    parts = list(lax.optimization_barrier(tuple(terms)))
    errors = [jnp.zeros_like(parts[0])] * len(parts)
    while len(parts) > 1:
        paired, carried = [], []
        for at in range(0, len(parts) - 1, 2):
            summed, error = two_sum(parts[at], parts[at + 1])
            paired.append(summed)
            carried.append(error + errors[at] + errors[at + 1])
        if len(parts) % 2:
            paired.append(parts[-1])
            carried.append(errors[-1])
        parts, errors = paired, carried
    return parts[0], errors[0]
