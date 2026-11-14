"""The magnitude table's every claim (docs/design/algebra.md §6.3, `terms._bounded`) checked against what each
equation computes, for every primitive of the language at every dtype it takes (float16 to float64, every
signed and unsigned integer width, booleans, and the mixed-dtype equations: conversions between any two, a
selection by a boolean, a product accumulating wider), under declared operand bounds.

Integers are proven: an 8-bit dtype's every operand value (each pair, a binary equation's), a 16-bit one's
every value (a unary equation's), and at 16 to 64 bits each corner of the declared box (its ends, their
neighbours, zero and one, each signed). Every integer claim is either the dtype's whole range, which holds
of every value it holds, or the real arithmetic's bound below that range, where no result wraps; there each
transfer (a sum, a difference, a product, a negation, an absolute value, a maximum, a selection, a power, a
reduction's sum, a contraction) is monotone or bilinear in its operands over the box, so its extremes lie at
the box's corners, where the check reads them (a reduction's groups and a contraction's terms each one
corner repeated, its extreme).

Floats are checked differentially, not proven: each transfer's analytic justification is its rule's
(`terms._real_bound`); the draws (the bound's edges, the 24 powers of two under it, their neighbours, zero,
the least subnormal and normal, and draws between) cover the rounding at the edges where a rule's slack is
spent.

The bound a region states is a power of two, `|x| <= 2**b`, the program's reach the narrowest float's
largest one: a bound at or under the output dtype's reach holds every output; above it a float's may round
to infinity, which no reordering reads (`rules.dominated`)."""

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from dew.algebra import terms

FLOATS = (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64)
INTEGERS = (jnp.int8, jnp.uint8, jnp.int16, jnp.uint16, jnp.int32, jnp.uint32, jnp.int64, jnp.uint64)
DTYPES = (*FLOATS, *INTEGERS, jnp.bool_)
BOUNDS = (None, -3, 0, 2, 6, 7, 15, 31)
"""Each operand's declared bound (`|x| <= 2**b`), None for none: of a unary equation's operand each, of a
binary one's each pair of `PAIRED`."""
PAIRED = (None, -3, 2, 7, 31)

UNARY = {
    "neg": lax.neg, "abs": lax.abs, "sign": lax.sign, "floor": lax.floor, "ceil": lax.ceil,
    "round": lax.round, "exp": lax.exp, "exp2": lax.exp2, "tanh": lax.tanh, "logistic": lax.logistic,
    "sin": lax.sin, "cos": lax.cos, "erf": lax.erf, "square": lax.square, "is_finite": lax.is_finite,
    "not": lax.bitwise_not, "integer_pow": lambda x: lax.integer_pow(x, 3),
    "reduce_sum": lambda x: jnp.sum(x.reshape(-1, 4), axis=1),
    "reduce_max": lambda x: jnp.max(x.reshape(-1, 4), axis=1),
    "reduce_min": lambda x: jnp.min(x.reshape(-1, 4), axis=1), "reshape": lambda x: x.reshape(-1, 2),
    **{f"to {jnp.dtype(target).name}": (lambda x, target=target: x.astype(target)) for target in DTYPES},
}
BINARY = {
    "add": lax.add, "sub": lax.sub, "mul": lax.mul, "max": lax.max, "min": lax.min, "and": lax.bitwise_and,
    "or": lax.bitwise_or, "xor": lax.bitwise_xor, "lt": lax.lt, "eq": lax.eq,
    "select_n": lambda x, y: lax.select(x > y, x, y),
    "dot_general": lambda x, y: jnp.einsum("ik,ik->i", x.reshape(-1, 4), y.reshape(-1, 4)),
    "dot_general wider": lambda x, y: lax.dot_general(x.reshape(-1, 4), y.reshape(-1, 4),
                                                      (((1,), (1,)), ((0,), (0,))),
                                                      preferred_element_type=_wider(x.dtype)),
}


def _wider(dtype):
    """The dtype a product of `dtype` operands accumulates in wider: a 16-bit float's float32, an 8- or
    16-bit integer's int32, else its own."""
    dtype = jnp.dtype(dtype)
    if dtype in (jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16)):
        return jnp.float32
    if jnp.issubdtype(dtype, jnp.integer) and dtype.itemsize < 4:
        return jnp.int32 if jnp.issubdtype(dtype, jnp.signedinteger) else jnp.uint32
    return dtype


def _admits(name: str, dtypes) -> bool:
    """Whether the primitive `name` takes operands of `dtypes` (a bitwise one integers or booleans, a float
    function floats)."""
    floating = all(jnp.issubdtype(dtype, jnp.floating) for dtype in dtypes)
    boolean = any(jnp.dtype(dtype) == jnp.bool_ for dtype in dtypes)
    if name in ("not", "and", "or", "xor"):
        return not any(jnp.issubdtype(dtype, jnp.floating) for dtype in dtypes)
    if name in ("floor", "ceil", "round", "exp", "exp2", "tanh", "logistic", "sin", "cos", "erf",
                "is_finite"):
        return floating
    if boolean:
        return name.startswith("to ") or name in ("reshape", "lt", "eq", "max", "min", "select_n")
    return not (name == "abs" and any(jnp.issubdtype(dtype, jnp.unsignedinteger) for dtype in dtypes))


def _values(dtype, bound: int | None, *, arity: int) -> np.ndarray:
    """The operand values of `dtype` within `2**bound` a check reads: a boolean's, an 8-bit integer's each
    one, a 16-bit integer's each one for a unary equation, any wider one's box corners (`_corners`), a
    float's edges (the bound, the 24 powers of two under it, each one's neighbours, zero, the least
    subnormal and normal) and 64 draws between, each with its negation; with no bound, the dtype's own
    range."""
    dtype = jnp.dtype(dtype)
    if dtype == jnp.bool_:
        return np.asarray([False, True] if bound is None or bound >= 0 else [False])
    if jnp.issubdtype(dtype, jnp.integer):
        integers = jnp.iinfo(dtype)
        if integers.bits == 8 or (integers.bits == 16 and arity == 1):
            every = np.arange(int(integers.min), int(integers.max) + 1)
            return (every if bound is None else every[np.abs(every) <= 2 ** bound]).astype(dtype)
        return _corners(dtype, bound)
    floats = jnp.finfo(dtype)
    top = float(floats.max) if bound is None else min(2.0 ** bound, float(floats.max))
    highest = math.frexp(top)[1] - 1
    powers = [2.0 ** k for k in range(highest - 24, highest + 1)]
    edges = np.asarray([*powers, top, float(floats.smallest_subnormal), float(floats.tiny)], dtype)
    # Each edge's neighbours, its bits' one apart (every edge positive).
    unsigned = np.dtype(f"uint{8 * dtype.itemsize}")
    bits = edges.view(unsigned)
    neighbours = [np.asarray(bits + 1, unsigned).view(dtype), np.asarray(bits - 1, unsigned).view(dtype)]
    # Scaled after the draw: float64's whole range is past the generator's span.
    draws = (np.random.default_rng(0).uniform(-1.0, 1.0, 64) * top).astype(dtype)
    found = np.concatenate([edges, *neighbours, draws, np.zeros(1, dtype)]).astype(np.float64)
    found = np.concatenate([found, -found])
    return found[np.isfinite(found) & (np.abs(found) <= top)].astype(dtype)


def _corners(dtype, bound: int | None) -> np.ndarray:
    """An integer dtype's box of values within `2**bound` by its corners and its transfers' critical points:
    its ends, their neighbours inside it, zero (where an absolute value, a square and an even power take
    their least) and one, each signed where the dtype holds it (in Python's integers, exact at 64 bits)."""
    integers = jnp.iinfo(dtype)
    largest = 2 ** bound if bound is not None and bound >= 0 else 0
    high = int(integers.max) if bound is None else min(largest, int(integers.max))
    low = int(integers.min) if bound is None else max(-high, int(integers.min))
    picked = {value for end in (low, high) for value in (end, end - 1, end + 1)} | {0, 1, -1}
    return np.asarray(sorted(value for value in picked if low <= value <= high), dtype)


def _claimed(fn, operands, bounds) -> int | None:
    """The bound the region states for `fn`'s output over `operands` declared at `bounds`."""
    closed = jax.make_jaxpr(fn)(*operands)
    declared = {index: 2.0 ** bound for index, bound in enumerate(bounds) if bound is not None}
    region = terms.region(closed, declared)
    return region.magnitudes.get(terms.key(region.outputs[0])) if region.outputs[0] is not None else None


def _reach(dtype) -> float:
    """The output dtype's largest power of two, past which a float's claim may be infinite."""
    dtype = jnp.dtype(dtype)
    if jnp.issubdtype(dtype, jnp.floating):
        return math.frexp(float(jnp.finfo(dtype).max))[1] - 1
    return math.inf


CASES = [(name, dtypes, bounds) for table, arity in ((UNARY, 1), (BINARY, 2))
         for name in table for dtypes in itertools.product(DTYPES, repeat=arity)
         if len(set(dtypes)) == 1 and _admits(name, dtypes)
         for bounds in itertools.product(BOUNDS if arity == 1 else PAIRED, repeat=arity)]


@pytest.mark.parametrize(("name", "dtypes", "bounds"), CASES,
                         ids=[f"{name}-{jnp.dtype(dtypes[0]).name}-{bounds}"
                              for name, dtypes, bounds in CASES])
def test_every_stated_bound_holds_every_output_its_operands_reach(name, dtypes, bounds):
    """The bound the region states for one equation of the language holds each of its outputs over every
    operand value its declared bounds admit (the module's reading): a bound at or under the output dtype's
    reach is each finite output's; one above it claims nothing a float's rounding to infinity breaks."""
    fn = (UNARY if len(dtypes) == 1 else BINARY)[name]
    with jax.enable_x64(new_val=True):
        made = jax.eval_shape(fn, *(jax.ShapeDtypeStruct((4,), dtype) for dtype in dtypes))
    # x64 wherever an operand or an output is 64-bit: a conversion to float64 of a 32-bit operand included.
    outputs = [leaf.dtype for leaf in jax.tree.leaves(made)]
    wide = any(jnp.dtype(dtype).itemsize == 8 for dtype in (*dtypes, *outputs))
    with jax.enable_x64(new_val=wide):
        columns = [_values(dtype, bound, arity=len(dtypes))
                   for dtype, bound in zip(dtypes, bounds, strict=True)]
        grid = [np.ravel(axis) for axis in np.meshgrid(*columns, indexing="ij")]
        # Each value four times in a row: a reduction's group, a contraction's terms, one corner.
        operands = [jnp.asarray(np.repeat(axis, 4), dtype) for axis, dtype in zip(grid, dtypes, strict=True)]
        claimed = _claimed(fn, operands, bounds)
        if claimed is None:
            return
        out = np.asarray(fn(*operands))
    magnitude = np.abs(out.astype(np.complex128 if np.iscomplexobj(out) else np.float64))
    if jnp.issubdtype(out.dtype, jnp.integer):
        # Exactly, in Python's integers (a 64-bit integer's magnitude is no float64's).
        exceeding = [int(value) for value in out.ravel() if abs(int(value)) > 2 ** claimed]
        assert not exceeding, f"{name} claims 2**{claimed} and makes {exceeding[:4]}"
        return
    finite = np.isfinite(magnitude)
    if claimed <= _reach(out.dtype):
        assert finite.all(), f"{name} claims 2**{claimed} and makes {out[~finite][:4]}"
    assert (magnitude[finite] <= 2.0 ** claimed).all(), (
        f"{name} claims 2**{claimed} and makes {out[finite][magnitude[finite] > 2.0 ** claimed][:4]}")


SIGNS = [(target, source, bound) for target in DTYPES for source in DTYPES
         if jnp.issubdtype(source, jnp.signedinteger) or jnp.issubdtype(source, jnp.floating)
         for bound in (0, 7, 31)]


@pytest.mark.parametrize(("target", "source", "bound"), SIGNS,
                         ids=[f"{jnp.dtype(source).name}-to-{jnp.dtype(target).name}-{bound}"
                              for target, source, bound in SIGNS])
def test_every_kept_sign_holds_every_value_its_operand_reaches(target, source, bound):
    """A value declared at most 0 (in `(-2**bound, 0)`) converted to `target`, and exponentiated, is at most
    1 in the reals (2 as computed) wherever the region states it is (the conversion keeps its sign,
    `terms._keeps_sign`), over every
    nonpositive value of `source` the module reads: a narrowing integer's wrap, an unsigned or boolean
    target's, a float's to an integer, each where it claims nothing."""
    wide = jnp.dtype(source).itemsize == 8 or jnp.dtype(target).itemsize == 8
    real = jnp.float64 if wide else jnp.float32

    def exponentiated(x):
        # The converted value read as a float (an integer's sign kept) and exponentiated.
        return jnp.exp(x.astype(target).astype(real))

    with jax.enable_x64(new_val=wide):
        values = _values(source, bound, arity=1)
        operand = jnp.asarray(np.repeat(values[values <= 0], 4), source)
        closed = jax.make_jaxpr(exponentiated)(operand)
        region = terms.region(closed, {0: (-(2.0 ** bound), 0.0)})
        if region.magnitudes.get(terms.key(region.outputs[0])) != terms.UNIT + 1:
            return
        converted = np.asarray(operand.astype(target)).astype(np.float64)
    positive = converted[converted > 0]
    assert not positive.size, f"{jnp.dtype(target).name} claims the sign of {positive[:4]} kept"


FLOORS = (-10, -3, 0, 4)
"""Each divisor's declared floor (`y >= 2**l`), its bound `2**(l + 6)` above it; beside them each dtype's
least normal power and the floor whose interval ends at its largest (`_apart_cases`)."""
EPSILON = 2.0 ** -10


def _grouped(y):
    return y.reshape(-1, 4)


def _signed(value):
    """A value at most 1 in magnitude where `value` is at least 0, which the region states where it reads
    `value` so, and infinite where `value` is -2 or under: a sign the region claims falsely shows as a bound
    an infinity breaks."""
    return 1.0 / (jnp.maximum(value + 1.0, -1.0) + 1.0)


def _negative_zero(x):
    """A value at least 0 as the region reads it, a negative zero wherever `x` is above 0."""
    return lax.neg(jnp.minimum(x, 0.0))


APART = {
    "quotient": lambda x, y: x / y,
    "reciprocal root": lambda x, y: x * lax.rsqrt(y),
    "root": lambda x, y: x / lax.sqrt(y),
    "root of a magnitude": lambda x, y: lax.sqrt(lax.abs(x)) * y,
    "logarithm": lambda x, y: x * lax.log(y),
    "logarithm of two": lambda x, y: x * lax.log2(y),
    "power": lambda x, y: lax.pow(y, x),
    "negative integer power": lambda x, y: x * lax.integer_pow(y, -2),
    "odd negative integer power": lambda x, y: x * lax.integer_pow(y, -3),
    "over a negation": lambda x, y: x / lax.neg(y),
    "over a product": lambda x, y: x / (y * y),
    "over a square": lambda x, y: x / lax.square(y),
    "over an integer power": lambda x, y: x / lax.integer_pow(y, 3),
    "over a reciprocal": lambda x, y: x / lax.integer_pow(y, -1),
    "over a reciprocal root": lambda x, y: x / lax.rsqrt(y),
    "over an absolute value": lambda x, y: x / lax.abs(lax.neg(y)),
    "over a square and a constant": lambda x, y: y / (lax.square(x) + EPSILON),
    "over an absolute value and a constant": lambda x, y: y / (lax.abs(x) + EPSILON),
    "over a logistic and a constant": lambda x, y: y / (lax.logistic(x) + EPSILON),
    "over an exponential": lambda x, y: y / lax.exp(jnp.maximum(jnp.minimum(x, 8.0), -8.0)),
    "over an exponential of two": lambda x, y: y / lax.exp2(jnp.maximum(jnp.minimum(x, 8.0), -8.0)),
    "over a maximum with a constant": lambda x, y: y / jnp.maximum(x, EPSILON),
    "over a maximum of negatives": lambda x, y: x / lax.max(lax.neg(y), lax.neg(y + y)),
    "over a minimum": lambda x, y: x / lax.min(y, 2.0 * y),
    "over a minimum with a negative": lambda x, y: x / jnp.minimum(lax.neg(y), x),
    "over a selection": lambda x, y: x / lax.select(x > 0, y, y + y),
    "over a quotient": lambda x, y: x / (y / (lax.abs(x) + 1.0)),
    "over a sum along an axis": lambda x, y: _grouped(x) / jnp.sum(_grouped(y), axis=1, keepdims=True),
    "over a least along an axis": lambda x, y: _grouped(x) / jnp.min(_grouped(y), axis=1, keepdims=True),
    "over a largest along an axis": lambda x, y: _grouped(x) / jnp.max(_grouped(y), axis=1, keepdims=True),
    "over a transposed layout": lambda x, y: _grouped(x) / jnp.squeeze(_grouped(y).T[None], 0).T,
    "over a broadcast of a broadcast": lambda x, y: _grouped(x)[:, None] / jnp.broadcast_to(
        jnp.broadcast_to(jnp.min(_grouped(y), axis=1, keepdims=True), _grouped(y).shape)[:, None],
        (y.size // 4, 2, 4)),
    "over a difference": lambda x, y: x / (y - lax.neg(y)),
    "normalized": lambda x, y: _grouped(x) * lax.rsqrt(
        jnp.sum(lax.square(_grouped(x)), axis=1, keepdims=True) / 4.0 + EPSILON),
    "over a narrower float": lambda x, y: x / y.astype(jnp.float16).astype(x.dtype),
    # A value that may have overflowed: nothing that shrinks it is bounded, and no floor is made of it.
    "a square that may overflow, divided": lambda x, y: lax.square(lax.square(y)) / 32768.0,
    "a square that may overflow, rooted": lambda x, y: lax.sqrt(lax.square(lax.square(y))),
    "a square that may overflow, its logarithm": lambda x, y: lax.log(lax.square(lax.square(y))),
    "a square that may overflow, to a half": lambda x, y: lax.pow(lax.square(lax.square(y)), 0.5 + 0.0 * x),
    "over a quotient by a square that may overflow": lambda x, y: x / (y / lax.square(lax.square(y))),
    "over a reciprocal root of a square that may overflow": lambda x, y: x / lax.rsqrt(
        lax.square(lax.square(y))),
    # A possible overflow outlives a conversion to a wider float, and so does the NaN two make.
    "a narrow square that may overflow, widened and rooted": lambda x, y: lax.sqrt(
        lax.square(y.astype(jnp.float16)).astype(y.dtype)),
    "a square that may overflow, through an integer, divided": lambda x, y: (
        lax.square(lax.square(y)) * EPSILON).astype(jnp.int32).astype(y.dtype) / y,
    "over a maximum with a difference that may be no number": lambda x, y: x / jnp.maximum(
        lax.square(lax.square(y)) - lax.square(lax.square(y)), 1.0),
    # A root and a logarithm of a value that may be under zero are no numbers; a difference of one side's
    # values may be zero.
    "a root of a signed value": lambda x, y: y * lax.sqrt(x),
    "a reciprocal root of a signed value": lambda x, y: y * lax.rsqrt(x),
    "a logarithm of a signed value": lambda x, y: y * lax.log(x),
    "over a difference of one side's values": lambda x, y: x / (y + y - y),
    # A sign the region reads of each operation that makes or keeps one (`_signed`).
    "the sign of a quotient": lambda x, y: y * _signed(lax.abs(x) / y),
    "the sign of a reciprocal root": lambda x, y: y * _signed(lax.rsqrt(y)),
    "the sign of a reciprocal": lambda x, y: y * _signed(lax.integer_pow(y, -1)),
    "the sign of a root": lambda x, y: y * _signed(lax.sqrt(lax.abs(x))),
    "the sign of a product": lambda x, y: y * _signed(lax.abs(x) * y),
    "the sign of a negated negation": lambda x, y: y * _signed(lax.neg(lax.neg(y))),
    "the sign of a minimum": lambda x, y: y * _signed(lax.min(lax.abs(x), y)),
    "the sign of a sum along an axis": lambda x, y: _grouped(y) * _signed(
        jnp.sum(lax.abs(_grouped(x)), axis=1, keepdims=True)),
    # A zero is at least 0 with either sign, and its reciprocal takes its sign: no sign is read through one.
    "the sign of a quotient by a zero": lambda x, y: y * _signed(y / _negative_zero(x)),
    "the sign of a reciprocal root of a zero": lambda x, y: y * _signed(lax.rsqrt(_negative_zero(x))),
    "the sign of a reciprocal of a zero": lambda x, y: y * _signed(lax.integer_pow(_negative_zero(x), -1)),
    "the sign of a cubed reciprocal of a zero": lambda x, y: y * _signed(
        lax.integer_pow(_negative_zero(x), -3)),
}
"""Programs of a signed `x` within its bound and a `y` at least its floor whose bound holds only where the
value divided by, rooted or read by a logarithm is held away from zero, a number and finite: each
operation `_apart_bound` bounds, each rule that carries a floor to a quotient (`terms._apart`), a value that
may have overflowed under each equation that reads it, and each sign the region claims."""
def _fourth_overflows(dtype, top: int) -> bool:
    """Whether `square(square(y))` of a `y` up to `2**top` may overflow `dtype`."""
    return 4 * top >= int(jnp.finfo(dtype).maxexp)


OVERFLOWING = {
    **dict.fromkeys((
        "a square that may overflow, divided", "a square that may overflow, rooted",
        "a square that may overflow, its logarithm", "a square that may overflow, to a half",
        "over a reciprocal root of a square that may overflow",
        "over a maximum with a difference that may be no number",
        "a square that may overflow, through an integer, divided"), _fourth_overflows),
    # Its divisor's floor is `y`'s (`2**(top - 6)`) less the fourth power's bound, kept as a normal power of
    # the dtype alone: float16 holds none under `2**-14`.
    "over a quotient by a square that may overflow": lambda dtype, top: _fourth_overflows(dtype, top) or (
        top - 6 - 4 * top < int(jnp.finfo(dtype).minexp)),
    # float16's square, whatever dtype reads it after.
    "a narrow square that may overflow, widened and rooted":
        lambda dtype, top: 2 * top >= int(jnp.finfo(jnp.float16).maxexp),
}
"""The programs no bound is stated of where a value of theirs may have overflowed, by whether it may for a
`y` up to `2**top` in a dtype: what reads an infinity, or the NaN two make, is bounded by nothing; nor is a
quotient by a value whose composed floor the dtype holds no normal power of."""
NEVER = {"the sign of a quotient by a zero", "the sign of a reciprocal root of a zero",
         "the sign of a reciprocal of a zero", "the sign of a cubed reciprocal of a zero",
         "a root of a signed value", "a reciprocal root of a signed value", "a logarithm of a signed value",
         "over a difference of one side's values"}
"""The programs no bound is stated of at all: a sign read through a reciprocal of a value that may be a zero
of either sign, a root or a logarithm of a value that may be under zero, a quotient by a difference that may
be zero."""
APART_DTYPES = (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64)


def _apart_cases():
    for name in APART:
        for dtype in APART_DTYPES:
            floats = jnp.finfo(dtype)
            least, largest = int(floats.minexp), int(floats.maxexp) - 1
            for floor in (*FLOORS, least, largest - 6):
                for bound in (-3, 2, 7, largest):
                    yield name, dtype, floor, bound


APART_CASES = list(_apart_cases())


@pytest.mark.parametrize(("name", "dtype", "floor", "bound"), APART_CASES,
                         ids=[f"{name}-{jnp.dtype(dtype).name}-{floor}-{bound}"
                              for name, dtype, floor, bound in APART_CASES])
def test_every_bound_stated_from_a_floor_holds_every_output_its_operands_reach(name, dtype, floor, bound):
    """A quotient's, a root's, a logarithm's and a power's bound, stated where the region holds the operand
    it needs away from zero and finite (a declared interval above zero, and what each equation keeps of
    one), holds each output over a signed `x` within `2**bound`, zeros of both signs among them, and every
    `y` from `2**floor` to `2**(floor + 6)`, their edges and neighbours among them, from each dtype's least
    normal power to its largest; the region states one for every program but `NEVER`'s, in every dtype,
    at a floor of `2**-3` and a bound of 4, so no bound holds by being absent; and it states none of
    `NEVER`'s anywhere, nor of `OVERFLOWING`'s where a value of theirs may have overflowed."""
    fn = APART[name]
    with jax.enable_x64(new_val=jnp.dtype(dtype).itemsize == 8):
        signed = _values(dtype, bound, arity=2)
        signed = np.concatenate([signed, -np.zeros(1, signed.dtype)])
        positive = _values(dtype, floor + 6, arity=2).astype(np.float64)
        positive = positive[positive >= 2.0 ** floor].astype(dtype)
        grid = [np.ravel(axis) for axis in np.meshgrid(signed, positive, indexing="ij")]
        operands = [jnp.asarray(np.repeat(axis, 4), dtype) for axis in grid]
        closed = jax.make_jaxpr(fn)(*operands)
        region = terms.region(closed, {0: 2.0 ** bound, 1: (2.0 ** floor, 2.0 ** (floor + 6))})
        claimed = region.magnitudes.get(terms.key(region.outputs[0]))
        unbounded = name in NEVER or (name in OVERFLOWING and OVERFLOWING[name](dtype, floor + 6))
        if (floor, bound) == (-3, 2):
            assert (claimed is None) == unbounded, (name, claimed)
        if unbounded:
            assert claimed is None, (name, claimed)
        if claimed is None:
            return
        out = np.asarray(fn(*operands)).astype(np.float64)
    finite = np.isfinite(out)
    if claimed <= _reach(dtype):
        assert finite.all(), f"{name} claims 2**{claimed} and makes {out[~finite][:4]}"
    # A claim past float64's own largest power holds every finite float64.
    limit = 2.0 ** claimed if claimed < 1024 else np.inf
    assert (np.abs(out[finite]) <= limit).all(), (
        f"{name} claims 2**{claimed} and makes {out[finite][np.abs(out[finite]) > limit][:4]}")


@pytest.mark.parametrize("reduction", [lax.reduce_max, lax.reduce_min, lax.reduce_sum],
                         ids=["max", "min", "sum"])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32])
def test_a_reduction_of_no_element_keeps_nothing_of_its_operand(reduction, dtype):
    """A maximum, a minimum and a sum along an axis of no element are their identities (an infinity of
    either sign, a zero), whatever their operand's declared interval: the region states no bound of a
    program a magnitude, a sign or a floor of the operand's would bound, which is infinite at each."""
    def fn(y):
        reduced = reduction(y.reshape(1, 0), (1,))
        return _signed(reduced) + 1.0 / lax.abs(reduced) + lax.abs(reduced)

    operand = jnp.zeros((0,), dtype)
    region = terms.region(jax.make_jaxpr(fn)(operand), {0: (1.0, 1.0)})
    assert not np.isfinite(np.asarray(fn(operand))).any()
    assert region.magnitudes.get(terms.key(region.outputs[0])) is None


@pytest.mark.parametrize("value", [7.999999999999999, 8.0, 8.000000000000002, 2.0 ** -1022, 1e-320],
                         ids=["under a power", "a power", "over a power", "the least normal", "a subnormal"])
def test_a_constants_and_a_declared_intervals_powers_are_exact(value):
    """A constant's bound and floor, and a declared interval's, are its exponent's exactly: the float under 8
    is under `2**3` and at least `2**2`, and the one over it is over `2**3`, where a rounded logarithm reads
    both as 8; so `x * c`, `x / c` and `x / y` are bounded by powers that hold them, and a quotient by a
    normal float is bounded."""
    constant = np.float64(value)
    with jax.enable_x64(new_val=True):
        ones, normal = jnp.ones(4, jnp.float64), value >= 2.0 ** -1022
        for fn, operands, declared, bounded in (
                (lambda x: x * constant, (ones,), {0: 1.0}, True),
                (lambda x: x / constant, (ones,), {0: 1.0}, normal),
                (lambda x, y: x / y, (ones, jnp.full(4, constant)), {0: 1.0, 1: (value, value)}, normal)):
            region = terms.region(jax.make_jaxpr(fn)(*operands), declared)
            claimed = region.magnitudes.get(terms.key(region.outputs[0]))
            assert (claimed is not None) == bounded
            if bounded:
                assert float(np.max(np.abs(np.asarray(fn(*operands))))) <= 2.0 ** claimed, claimed
