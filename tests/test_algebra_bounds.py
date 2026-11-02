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
    wide = any(jnp.dtype(dtype).itemsize == 8 for dtype in dtypes)
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
