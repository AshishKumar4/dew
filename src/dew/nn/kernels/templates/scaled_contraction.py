"""A contraction of products and quotients of float arrays, each formed apart from its binades
(`scaled_contraction_p`, rows 2654 and 12388).

`contract(form, *arrays)` is `einsum(groups -> out)`, a group the product of its arrays, each to the power
`+1` or `-1` (`Form.groups`), its letters each array's own subscript's. Each group is formed as a mantissa
and an integer exponent apart (`jnp.frexp`), then held at its exponent's largest over the letters it sums
(`Form.out` lacks them), so every operand the contraction reads lies within `[2^-n, 2^n]` of one, `n` its
arrays' count: the einsum's products and sums stay within the format wherever its result does, and the
exponents join it once, last (`power2`). So a sum of values near the format's largest, a quotient by a
total of 2^100, or a direction 2^-227 of the values it moves, forms no float apart from the sum it is in.

Its derivative at every order, forward and reverse, is the primitive again: along an array to the power
`+1` the same contraction of its tangent in its place; along one to `-1`, `-dx / x^2`, the contraction with
the tangent and the array beside it again (`_jvp`); and the transpose along a linear array the contraction
of the others with the cotangent onto its letters (`_transpose`). No order takes a quotient, a power or a
sum outside it, so none forms `total^-2` (2^-200 at a mass of 2^100, zero) or a cotangent's sum past the
format before the factor that brings it back (the reference's mean, `attention.held_mean`).
"""

from __future__ import annotations

import dataclasses
import string
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax.extend import core as jcore
from jax.interpreters import ad, batching, mlir

from dew.nn.precision import at_least_fp32, exact_float32, power2

Term = tuple[str, int]
"""An array's subscript and its power, `+1` or `-1`."""

PRECISION = exact_float32(None)
"""The einsum's in float32: fp32 products at fp32 (`exact_float32`), as the reference's own contractions."""

LEAST = jnp.iinfo(jnp.int32).min // 4
"""An exponent beneath every float's, a zero's in a group's largest (`_held`)."""


@dataclasses.dataclass(frozen=True)
class Form:
    """A contraction's shape: each group's terms (an array's subscript and power, `Term`), in its operands'
    order, and the result's subscript `out`. Every letter has one extent wherever it appears."""

    groups: tuple[tuple[Term, ...], ...]
    out: str

    def letters(self, group: tuple[Term, ...]) -> str:
        """`group`'s letters, each once, in the order its terms name them."""
        return "".join(dict.fromkeys("".join(subscript for subscript, _ in group)))


def _laid(array: jax.Array, subscript: str, letters: str) -> jax.Array:
    """`array` over `subscript` laid over `letters`, a letter it lacks of extent one."""
    ordered = [letter for letter in letters if letter in subscript]
    moved = jnp.transpose(array, [subscript.index(letter) for letter in ordered])
    return moved.reshape([array.shape[subscript.index(letter)] if letter in subscript else 1
                          for letter in letters])


def _held(group: tuple[Term, ...], arrays: Sequence[jax.Array], letters: str, out: str, dtype):
    """A group's product as its mantissa within `[2^-n, 2^n]` of one at its exponent's largest over the
    letters it sums, and that largest over the letters it keeps (`out`'s, its own order)."""
    mantissa, exponent = jnp.ones((), dtype), jnp.int32(0)
    for (subscript, power), array in zip(group, arrays, strict=True):
        fraction, binade = jnp.frexp(_laid(array.astype(dtype), subscript, letters))
        mantissa = mantissa * fraction if power > 0 else mantissa / fraction
        exponent = exponent + binade if power > 0 else exponent - binade
    summed = tuple(axis for axis, letter in enumerate(letters) if letter not in out)
    largest = jnp.max(jnp.where(mantissa != 0, exponent, LEAST), axis=summed, keepdims=True, initial=LEAST)
    largest = jnp.where(largest == LEAST, 0, largest)
    return power2(mantissa, exponent - largest), jnp.squeeze(largest, summed)


def _dtype(arrays) -> jnp.dtype:
    """The contraction's format: its operands', at least float32."""
    return at_least_fp32(jnp.result_type(*(array.dtype for array in arrays)))


def _value(*arrays: jax.Array, form: Form) -> jax.Array:
    """The contraction (the module's law) in its operands' format, at least float32 (`_dtype`): each group
    held apart from its binades (`_held`), contracted, the groups' exponents joined last (`power2`)."""
    dtype = _dtype(arrays)
    operands, scale, at = [], 0, 0
    subscripts = []
    for group in form.groups:
        letters = form.letters(group)
        held, largest = _held(group, arrays[at:at + len(group)], letters, form.out, dtype)
        at += len(group)
        operands.append(held)
        subscripts.append(letters)
        kept = "".join(letter for letter in letters if letter in form.out)
        scale = scale + _laid(largest, kept, form.out)
    precision = PRECISION if dtype == jnp.float32 else jax.lax.Precision.HIGHEST
    contracted = jnp.einsum(",".join(subscripts) + "->" + form.out, *operands, precision=precision,
                            preferred_element_type=dtype)
    return power2(contracted, scale)


def _extents(arrays, form: Form) -> dict[str, int]:
    """Each letter's extent, one wherever it appears."""
    extents: dict[str, int] = {}
    terms = [term for group in form.groups for term in group]
    for (subscript, _), array in zip(terms, arrays, strict=True):
        if len(subscript) != len(array.shape):
            raise ValueError(f"a {subscript!r} term of rank {len(array.shape)}")
        for letter, extent in zip(subscript, array.shape, strict=True):
            if extents.setdefault(letter, extent) != extent:
                raise ValueError(f"letter {letter!r} of extents {extents[letter]} and {extent}")
    return extents


scaled_contraction_p = jcore.Primitive("scaled_contraction")


@scaled_contraction_p.def_abstract_eval
def _abstract(*arrays, form: Form):
    extents = _extents(arrays, form)
    return arrays[0].update(shape=tuple(extents[letter] for letter in form.out), dtype=_dtype(arrays))


scaled_contraction_p.def_impl(lambda *arrays, form: _value(*arrays, form=form))
mlir.register_lowering(scaled_contraction_p, mlir.lower_fun(_value, multiple_results=False))


def contract(form: Form, *arrays: jax.Array) -> jax.Array:
    """`form`'s contraction of `arrays` (the module's law), at every order the primitive again."""
    return scaled_contraction_p.bind(*arrays, form=form)


def _replaced(form: Form, index: int, terms: tuple[Term, ...]) -> Form:
    """`form` with its `index`-th term (in operands' order) replaced by `terms` within its group."""
    groups, start = [], 0
    for group in form.groups:
        local = index - start
        groups.append((*group[:local], *terms, *group[local + 1:]) if 0 <= local < len(group) else group)
        start += len(group)
    return Form(tuple(groups), form.out)


def _term(form: Form, index: int) -> Term:
    return [term for group in form.groups for term in group][index]


def _jvp(primals, tangents, *, form: Form):
    """The contraction's tangent, a contraction a moving array (`ad.Zero` none): along one to `+1` its tangent
    in its place; along one to `-1`, `-dx / x^2`, its tangent to `+1` and the array to `-1` beside it."""
    out = contract(form, *primals)
    moved = []
    for index, tangent in enumerate(tangents):
        if type(tangent) is ad.Zero:
            continue
        subscript, power = _term(form, index)
        if power > 0:
            moved.append(contract(form, *primals[:index], tangent, *primals[index + 1:]))
        else:
            twice = _replaced(form, index, ((subscript, 1), (subscript, -1), (subscript, -1)))
            moved.append(-contract(twice, *primals[:index], tangent, primals[index], primals[index],
                                   *primals[index + 1:]))
    if not moved:
        return out, ad.Zero(jax.typeof(out).to_tangent_aval())
    total = moved[0]
    for each in moved[1:]:
        total = total + each
    return out, total


def _transpose(cotangent, *arrays, form: Form):
    """The cotangent of the one linear array (an `ad.UndefinedPrimal`, to `+1`, as `_jvp` places it): the
    contraction of its group's other terms, the other groups and the cotangent over `out`, onto its
    letters."""
    linear = [index for index, array in enumerate(arrays) if ad.is_undefined_primal(array)]
    if len(linear) != 1:
        raise NotImplementedError("a scaled contraction is linear in one array at a time")
    (index,) = linear
    subscript, power = _term(form, index)
    if power < 0:
        raise NotImplementedError("a scaled contraction is linear in no divisor")
    if type(cotangent) is ad.Zero:
        return [None] * len(arrays)
    groups, operands, at = [], [], 0
    for group in form.groups:
        here = [(term, array) for term, array in zip(group, arrays[at:at + len(group)], strict=True)]
        kept = [(term, array) for offset, (term, array) in enumerate(here) if at + offset != index]
        at += len(group)
        if kept:
            groups.append(tuple(term for term, _ in kept))
            operands.extend(array for _, array in kept)
    groups.append(((form.out, 1),))
    operands.append(cotangent)
    pulled = contract(Form(tuple(groups), subscript), *operands)
    out: list[jax.Array | None] = [None] * len(arrays)
    out[index] = pulled.astype(arrays[index].aval.dtype)
    return out


ad.primitive_jvps[scaled_contraction_p] = _jvp
ad.primitive_transposes[scaled_contraction_p] = _transpose


def _batched(arrays, dims, *, form: Form):
    """The contraction of a batch: a fresh letter on each batched array (its batch axis moved first) and on
    the result; an unbatched array shared along it."""
    used = set(form.out).union(*(subscript for group in form.groups for subscript, _ in group))
    letter = next(each for each in string.ascii_letters if each not in used)
    groups, operands, at = [], [], 0
    for group in form.groups:
        terms = []
        for term, array, dim in zip(group, arrays[at:at + len(group)], dims[at:at + len(group)], strict=True):
            subscript, power = term
            if dim is not None:
                array = jnp.moveaxis(array, dim, 0)
                subscript = letter + subscript
            terms.append((subscript, power))
            operands.append(array)
        at += len(group)
        groups.append(tuple(terms))
    return contract(Form(tuple(groups), letter + form.out), *operands), 0


batching.primitive_batchers[scaled_contraction_p] = _batched
