"""A signed sum of contractions of products and quotients of float arrays, each formed apart from its
binades (`scaled_contraction_p`, rows 2654 and 12388).

`contract(form, *arrays)` is `sum_t sign_t einsum(groups_t -> out)` (`Form.terms`): a term's group the
product of its operands, each to the power `+1` or `-1`, each operand over its own subscript
(`Form.subscripts`). Each operand is split into a significand in `[1, 2)` and an integer exponent from its
bits (`_split`, so a subnormal keeps its value where a process flushes subnormal arithmetic); each group is
held at its exponent's largest over the letters it sums (`Form.out` lacks them), so every operand a
contraction reads lies within `[2^-n, 2^n]` of one, `n` its factors' count; and the terms are joined at
their exponents' largest before the one power of two that lands the result (`power2`). So a sum of values
near the format's largest, a quotient of `2^126` by a total of `2^62` moved by `2^62`, or a product rule's
`+2^200` and `-2^200` parts, form no float past the format beside the factor that brings it back.

Its derivative at every order, forward and reverse, is the primitive again: along an operand to `+1` the
same term with its tangent in its place; along one to `-1`, `-dx / x^2`, the term with the tangent to `+1`
and the operand to `-1` twice (`_jvp`, every term of the product rule in one sum, landed once); and the
transpose along a linear operand the sum of its terms' other groups contracted with the cotangent onto its
letters (`_transpose`). No order takes a quotient, a power or a sum outside it, so none forms `total^-2`
(2^-200 at a mass of 2^100, zero) or a cotangent's sum past the format before the factor that brings it
back (the reference's mean, `attention.held_mean`).

A sum whose terms cancel across more binades than the format holds (`2^127 - 2^127 + 2^-30`, a group held
at `2^127`) loses the small part, as any float sum at one exponent does; holding it takes a banded exact sum
(`dew.algebra.windows`), row 12452.
"""

from __future__ import annotations

import dataclasses
import string
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax._src import api_util
from jax._src.interpreters import partial_eval as pe
from jax.extend import core as jcore
from jax.interpreters import ad, batching, mlir

from dew.nn.float_bits import float_bits
from dew.nn.precision import at_least_fp32, exact_float32, power2

Factor = tuple[int, int]
"""An operand's index and its power, `+1` or `-1`."""

Term = tuple[int, tuple[tuple[Factor, ...], ...]]
"""A term's sign and its groups, each a product of its factors."""

PRECISION = exact_float32(None)
"""The einsums' in float32: fp32 products at fp32 (`exact_float32`), as the reference's own contractions."""

LEAST = jnp.iinfo(jnp.int32).min // 4
"""An exponent beneath every float's, a zero's in a largest (`_held`, `_value`)."""


@dataclasses.dataclass(frozen=True)
class Form:
    """A sum's shape: each operand's subscript, each term (`Term`, its factors naming operands) and the
    result's subscript `out`; with `divisor` terms, the sum over them (over the letters `divided`, a subset
    of `out`'s) divides it, both formed apart from their binades and divided before they land, so a quotient
    whose numerator and divisor pass the format lands within it (a zero divisor read as one). Every letter
    has one extent wherever it appears, and none twice in one subscript."""

    subscripts: tuple[str, ...]
    terms: tuple[Term, ...]
    out: str
    divisor: tuple[Term, ...] = ()
    divided: str = ""

    def letters(self, group: tuple[Factor, ...]) -> str:
        """`group`'s letters, each once, in the order its factors' subscripts name them."""
        return "".join(dict.fromkeys("".join(self.subscripts[index] for index, _ in group)))


def _laid(array: jax.Array, subscript: str, letters: str) -> jax.Array:
    """`array` over `subscript` laid over `letters`, a letter it lacks of extent one."""
    ordered = [letter for letter in letters if letter in subscript]
    moved = jnp.transpose(array, [subscript.index(letter) for letter in ordered])
    return moved.reshape([array.shape[subscript.index(letter)] if letter in subscript else 1
                          for letter in letters])


def _split(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """`x` as its significand in `[1, 2)` (signed; zero a zero's, a non-finite `x` itself) and its exponent
    `floor(log2 |x|)` (int32; zero's and a non-finite's 0), read from its bits (`float_bits`): a subnormal's
    by its leading bit, where a process flushing subnormal arithmetic reads none of it."""
    limits = jnp.finfo(x.dtype)
    sign, biased, fraction, normalized, exponent = float_bits(x)
    unit = (limits.maxexp - 1) << limits.nmant
    significand = jax.lax.bitcast_convert_type(sign | unit | normalized, x.dtype)
    zero = (biased == 0) & (fraction == 0)
    finite = biased < (2 * limits.maxexp - 1)
    significand = jnp.where(zero, jnp.zeros((), x.dtype), jnp.where(finite, significand, x))
    return significand, jnp.where(zero | ~finite, 0, exponent).astype(jnp.int32)


def _held(form: Form, group: tuple[Factor, ...], arrays: Sequence[jax.Array], dtype):
    """A group's product as its significand within `[2^-n, 2^n]` of one at its exponent's largest over the
    letters it sums, and that largest over the letters it keeps (`out`'s, the group's order)."""
    letters = form.letters(group)
    significand, exponent = jnp.ones((), dtype), jnp.int32(0)
    for index, power in group:
        part, binade = _split(_laid(arrays[index].astype(dtype), form.subscripts[index], letters))
        significand = significand * part if power > 0 else significand / part
        exponent = exponent + binade if power > 0 else exponent - binade
    summed = tuple(axis for axis, letter in enumerate(letters) if letter not in form.out)
    largest = jnp.max(jnp.where(significand != 0, exponent, LEAST), axis=summed, keepdims=True,
                      initial=LEAST)
    largest = jnp.where(largest == LEAST, 0, largest)
    return letters, power2(significand, exponent - largest), jnp.squeeze(largest, summed)


def _dtype(arrays) -> jnp.dtype:
    """The sum's format: its operands', at least float32."""
    return at_least_fp32(jnp.result_type(*(array.dtype for array in arrays)))


def _sum(arrays, form: Form, terms: tuple[Term, ...], out: str, dtype):
    """`terms`' signed sum over `out` unlanded: its significands' sum and the exponent it lies at, each
    term's groups held apart from their binades (`_held`) and contracted, the terms joined at their
    exponents' largest."""
    precision = PRECISION if dtype == jnp.float32 else jax.lax.Precision.HIGHEST
    shaped = Form(form.subscripts, terms, out)
    parts = []
    for sign, groups in terms:
        held = [_held(shaped, group, arrays, dtype) for group in groups]
        contracted = jnp.einsum(",".join(letters for letters, _, _ in held) + "->" + out,
                                *(significand for _, significand, _ in held), precision=precision,
                                preferred_element_type=dtype)
        scale = sum((_laid(largest, "".join(letter for letter in letters if letter in out), out)
                     for letters, _, largest in held), jnp.zeros((1,) * len(out), jnp.int32))
        parts.append((sign * contracted, jnp.where(contracted != 0, scale, LEAST)))
    joined = parts[0][1]
    for _, scale in parts[1:]:
        joined = jnp.maximum(joined, scale)
    joined = jnp.where(joined == LEAST, 0, joined)
    landed = sum((power2(contracted, scale - joined) for contracted, scale in parts), jnp.zeros((), dtype))
    return landed, joined


def _value(*arrays: jax.Array, form: Form) -> jax.Array:
    """The sum (the module's law) in its operands' format, at least float32 (`_dtype`), its numerator over
    its divisor (`Form.divisor`) divided before one power of two lands it (`power2`)."""
    extents = _extents(arrays, form)
    dtype = _dtype(arrays)
    if not form.terms:
        return jnp.zeros([extents[letter] for letter in form.out], dtype)
    total, joined = _sum(arrays, form, form.terms, form.out, dtype)
    if form.divisor:
        below, exponent = _sum(arrays, form, form.divisor, form.divided, dtype)
        none = below == 0
        total = total / _laid(jnp.where(none, 1, below), form.divided, form.out)
        joined = joined - _laid(jnp.where(none, 0, exponent), form.divided, form.out)
    return power2(total, joined)


def _extents(arrays, form: Form) -> dict[str, int]:
    """Each letter's extent, one wherever it appears; a subscript naming a letter twice, or of another rank
    than its operand's, refused."""
    extents: dict[str, int] = {}
    for subscript, array in zip(form.subscripts, arrays, strict=True):
        if len(set(subscript)) != len(subscript) or len(subscript) != len(array.shape):
            raise ValueError(f"a scaled contraction's subscript {subscript!r} of an operand {array.shape}")
        for letter, extent in zip(subscript, array.shape, strict=True):
            if extents.setdefault(letter, extent) != extent:
                raise ValueError(f"letter {letter!r} of extents {extents[letter]} and {extent}")
    if not set(form.divided) <= set(form.out):
        raise ValueError(f"a divisor over {form.divided!r} beyond the result's {form.out!r}")
    return extents


scaled_contraction_p = jcore.Primitive("scaled_contraction")


@scaled_contraction_p.def_abstract_eval
def _abstract(*arrays, form: Form):
    """The sum's type as JAX types its value (`_value`): its shape and format, and its operands' shardings,
    weak types and manual axes as JAX's own rules carry them."""
    def value(*operands):
        return _value(*operands, form=form)

    (typed,) = pe.abstract_eval_fun(value, *arrays, debug_info=api_util.debug_info(
        "scaled_contraction", value, arrays, {}))
    return typed


scaled_contraction_p.def_impl(lambda *arrays, form: _value(*arrays, form=form))
mlir.register_lowering(scaled_contraction_p, mlir.lower_fun(_value, multiple_results=False))


def contract(form: Form, *arrays: jax.Array) -> jax.Array:
    """`form`'s sum of `arrays` (the module's law), at every order the primitive again."""
    return scaled_contraction_p.bind(*arrays, form=form)


def _moved(terms: tuple[Term, ...], moving: dict, at: dict[int, int]) -> list[Term]:
    """`terms`' product rule along the moving operands: along one to `+1` its tangent in its place; along
    one to `-1`, `-dx / x^2`, its tangent to `+1` and the operand to `-1` twice."""
    moved = []
    for sign, groups in terms:
        for g, group in enumerate(groups):
            for f, (index, power) in enumerate(group):
                if index not in moving:
                    continue
                factors = (((at[index], 1),) if power > 0 else ((at[index], 1), (index, -1), (index, -1)))
                changed = (*group[:f], *factors, *group[f + 1:])
                moved.append((sign if power > 0 else -sign, (*groups[:g], changed, *groups[g + 1:])))
    return moved


def _jvp(primals, tangents, *, form: Form):
    """The sum's tangent, one sum of the product rule's terms (`_moved`, `ad.Zero` a still operand's) landed
    once; a quotient's `(dN - out dD) / D` over its own divisor, the result `out` an operand of the
    divisor's moved terms, so its numerator's and divisor's parts cancel within the one sum."""
    out = contract(form, *primals)
    moving = {index: tangent for index, tangent in enumerate(tangents) if type(tangent) is not ad.Zero}
    if not moving:
        return out, ad.Zero(jax.typeof(out).to_tangent_aval())
    at = {index: len(primals) + place for place, index in enumerate(moving)}
    terms = _moved(form.terms, moving, at)
    operands = [*primals, *moving.values()]
    subscripts = [*form.subscripts, *(form.subscripts[index] for index in moving)]
    if form.divisor:
        below = _moved(form.divisor, moving, at)
        if below:
            operands.append(out)
            subscripts.append(form.out)
            terms += [(-sign, (*groups, ((len(operands) - 1, 1),))) for sign, groups in below]
    if not terms:
        return out, ad.Zero(jax.typeof(out).to_tangent_aval())
    tangent = Form(tuple(subscripts), tuple(terms), form.out, form.divisor, form.divided)
    return out, contract(tangent, *operands)


def _transpose(cotangent, *arrays, form: Form):
    """Each linear operand's cotangent (an `ad.UndefinedPrimal`, to `+1` in each term that reads it, as
    `_jvp` places it): the sum over those terms of their other factors contracted with the cotangent over
    `out`, onto its letters; a letter none of them reads taken by a unit operand along it."""
    out: list[jax.Array | None] = [None] * len(arrays)
    if type(cotangent) is ad.Zero:
        return out
    known = [index for index, array in enumerate(arrays) if not ad.is_undefined_primal(array)]
    place = {index: at for at, index in enumerate(known)}
    below: tuple[Factor, ...] = ()
    shared = [arrays[index] for index in known]
    shared_subscripts = [form.subscripts[index] for index in known]
    if form.divisor:
        if any(index not in place for _, groups in form.divisor for group in groups for index, _ in group):
            raise NotImplementedError("a scaled contraction is linear in no divisor's operand")
        divisor = tuple((sign, tuple(tuple((place[index], power) for index, power in group)
                                     for group in groups)) for sign, groups in form.divisor)
        total = contract(Form(tuple(shared_subscripts), divisor, form.divided), *shared)
        shared.append(jnp.where(total == 0, jnp.ones((), total.dtype), total))
        shared_subscripts.append(form.divided)
        below = ((len(shared) - 1, -1),)
    for linear, array in enumerate(arrays):
        if not ad.is_undefined_primal(array):
            continue
        subscript = form.subscripts[linear]
        operands, subscripts = list(shared), list(shared_subscripts)
        operands.append(cotangent)
        subscripts.append(form.out)
        pulled = len(operands) - 1
        terms = []
        for sign, groups in form.terms:
            found = [(g, f) for g, group in enumerate(groups) for f, (index, _) in enumerate(group)
                     if index == linear]
            if not found:
                continue
            ((g, f),) = found
            if groups[g][f][1] < 0:
                raise NotImplementedError("a scaled contraction is linear in no divisor")
            rest = tuple(tuple((place[index], power) for index, power in group if index != linear)
                         for group in (*groups[:g], tuple(x for i, x in enumerate(groups[g]) if i != f),
                                       *groups[g + 1:]))
            read = set(form.out).union(*(form.subscripts[index] for group in groups for index, _ in group
                                         if index != linear))
            unread = "".join(letter for letter in subscript if letter not in read)
            extra: tuple[Factor, ...] = ()
            if unread:
                operands.append(jnp.ones([array.aval.shape[subscript.index(letter)] for letter in unread],
                                         array.aval.dtype))
                subscripts.append(unread)
                extra = ((len(operands) - 1, 1),)
            terms.append((sign, (*(group for group in rest if group), ((pulled, 1), *below, *extra))))
        if not terms:
            # An operand no term reads (a sum's operand its own transpose carried along): no cotangent.
            continue
        landed = contract(Form(tuple(subscripts), tuple(terms), subscript), *operands)
        out[linear] = landed.astype(array.aval.dtype)
    return out


ad.primitive_jvps[scaled_contraction_p] = _jvp
ad.primitive_transposes[scaled_contraction_p] = _transpose


def _batched(arrays, dims, *, form: Form):
    """The sum of a batch: a fresh letter on each batched operand (its batch axis moved first) and on the
    result; an unbatched operand shared along it, and a term that reads no batched operand taking a unit
    operand along it, so each term spans the result's letters."""
    used = set(form.out).union(*form.subscripts)
    letter = next((each for each in string.ascii_letters if each not in used), None)
    if letter is None:
        raise ValueError("a scaled contraction batched past its letters")
    size = next(array.shape[dim] for array, dim in zip(arrays, dims, strict=True) if dim is not None)
    operands, subscripts = [], []
    for array, dim, subscript in zip(arrays, dims, form.subscripts, strict=True):
        operands.append(array if dim is None else jnp.moveaxis(array, dim, 0))
        subscripts.append(subscript if dim is None else letter + subscript)
    batched = {index for index, dim in enumerate(dims) if dim is not None}

    def spanned(terms: tuple[Term, ...]) -> tuple[Term, ...]:
        laid = []
        for sign, groups in terms:
            if not any(index in batched for group in groups for index, _ in group):
                if len(operands) == len(arrays):
                    operands.append(jnp.ones((size,), _dtype(arrays)))
                    subscripts.append(letter)
                groups = (*groups, ((len(arrays), 1),))
            laid.append((sign, groups))
        return tuple(laid)

    terms = spanned(form.terms)
    divisor = spanned(form.divisor) if form.divisor else ()
    divided = letter + form.divided if form.divisor else ""
    return contract(Form(tuple(subscripts), terms, letter + form.out, divisor, divided), *operands), 0


batching.primitive_batchers[scaled_contraction_p] = _batched
